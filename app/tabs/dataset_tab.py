from __future__ import annotations

import glob
import json
import random
import re
import shutil
import threading
import queue
from collections import OrderedDict
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from app.backend_bridge import get_runtime_class_names
from app.components import (
    CONSOLE_PANE_WEIGHT,
    HEADER_PROGRESS_LENGTH,
    MAIN_PANE_WEIGHT,
    PATH_ENTRY_WIDTH,
    CollapsibleGroup,
    ConsoleLogPanel,
    ModeHeader,
    ScrollPanel,
)
from app.config_manager import load_settings, save_settings, synchronize_shared_settings
from app.live_model_package import LivePackage, resolve_live_package
from app.ui_platform import get_sidebar_width
from backend.config_threshold_optimization import load_multipliers_from_json


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_STATUSES = {"PASS", "FAIL"}
PASS_MARKERS = {"", "0", "false", "no", "none", "ok", "pass"}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class _HoverTooltip:
    """Small hover tooltip used by setting information icons."""

    def __init__(self, widget, title, message, delay_ms=350):
        self.widget = widget
        self.title = title
        self.message = message
        self.delay_ms = delay_ms
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None or not self.widget.winfo_exists():
            return

        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        try:
            window.wm_attributes("-topmost", True)
        except tk.TclError:
            pass

        frame = ttk.Frame(window, padding=(10, 8), relief="solid", borderwidth=1)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=self.title, font=("TkDefaultFont", 9, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            frame,
            text=self.message,
            justify=tk.LEFT,
            wraplength=360,
        ).pack(anchor="w")

        window.update_idletasks()
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
        y = self.widget.winfo_rooty()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        width = window.winfo_reqwidth()
        height = window.winfo_reqheight()
        if x + width > screen_width - 8:
            x = max(8, self.widget.winfo_rootx() - width - 8)
        if y + height > screen_height - 8:
            y = max(8, screen_height - height - 8)
        window.wm_geometry(f"+{x}+{y}")
        self._window = window

    def _hide(self, _event=None):
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


def _info_button(parent, title, message):
    """Create a compact right-aligned information icon with a hover tooltip."""
    button = ttk.Label(parent, text="ⓘ", cursor="question_arrow", anchor="center")
    _HoverTooltip(button, title, message)
    return button


def class_names_for_dataset_checkpoint(checkpoint_path: str | Path | None, fallback: list[str] | None = None) -> list[str]:
    """Return class names matching the selected checkpoint when package labels exist."""
    names = list(fallback or get_runtime_class_names())
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else None
    labels_path = checkpoint.parent / "labels.json" if checkpoint else None
    if labels_path and labels_path.is_file():
        try:
            payload = json.loads(labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid labels.json beside checkpoint: {labels_path}") from exc
        label_names = payload.get("class_names") if isinstance(payload, dict) else None
        if not isinstance(label_names, list) or not label_names:
            raise ValueError(f"labels.json must contain non-empty class_names list: {labels_path}")
        mapping = payload.get("class_to_index") or {}
        for index, name in enumerate(label_names):
            if mapping and str(mapping.get(name)) != str(index):
                raise ValueError(f"labels.json class_to_index order mismatch for {name}: {labels_path}")
        names = [str(name) for name in label_names]
    if "Pass" not in names:
        names.append("Pass")
    return names


@dataclass
class ReviewItem:
    source_path: Path | None
    source_kind: str
    filename: str
    proposed_class: str
    program: str
    lot: str
    side: str
    needs_review: bool = False
    review_reason: str = ""
    source_reference: str = ""
    predicted_class: str | None = None
    predicted_confidence: float | None = None
    output_filename: str | None = None
    override_class: str = ""
    excluded: bool = False


class DatasetTab(ttk.Frame):
    """Review JSON-log and NG-folder images before copying them into a dataset."""

    def __init__(self, parent, scroll_targets, status_bar, shared_console):
        super().__init__(parent, padding=10)
        self.scroll_targets = scroll_targets
        self.status_bar = status_bar
        self.shared_console = shared_console
        self.class_names = class_names_for_dataset_checkpoint(None)
        self._class_name_lookup = {name.casefold(): name for name in self.class_names}

        self.items: list[ReviewItem] = []
        self.row_widgets: list[dict] = []
        self.thumbnail_refs: list[ImageTk.PhotoImage] = []
        self._thumbnail_cache: OrderedDict[str, ImageTk.PhotoImage | None] = OrderedDict()
        self._thumbnail_cache_limit = 150
        self._filtered_items: list[ReviewItem] = []
        self._page_size = 50
        self._page_index = 0
        self._render_generation = 0
        self._scan_thread: threading.Thread | None = None
        self._scan_results: queue.Queue = queue.Queue()
        self._prediction_thread: threading.Thread | None = None
        self._prediction_restart_requested = False
        self._prediction_stop_event = threading.Event()
        self._prediction_results: queue.Queue = queue.Queue()
        self._build_thread: threading.Thread | None = None
        self._build_results: queue.Queue = queue.Queue()
        self._build_in_progress = False
        self._agreement_sample_ids: set[int] = set()
        self._prediction_available = False
        self._saved_filter_mode = "All"
        self.log_visible = False

        self.json_source_var = tk.StringVar()
        self.ng_source_var = tk.StringVar()
        self.ok_source_var = tk.StringVar()
        self.destination_var = tk.StringVar()
        self.promoted_package_var = tk.StringVar()
        self._dataset_package: LivePackage | None = None
        self.summary_var = tk.StringVar(value="No images scanned.")
        self.prediction_progress_text = tk.StringVar(value="Idle")
        self.filter_var = tk.StringVar(value="All")
        self.prediction_progress_var = tk.DoubleVar(value=0.0)

        self._build()
        self._load_saved_settings()
        self.logger.log("[Ready] Dataset tab initialized.")

    def _build(self):
        sidebar_width = get_sidebar_width(self)
        ModeHeader(
            self,
            title="Dataset",
            buttons_config=[
                ("Scan", self._start_scan),
                ("Build", self._add_to_dataset),
                ("Stop", self._stop_prediction),
                ("Resume", self._resume_prediction),
                ("Clear", self._clear_log),
            ],
            center_builder=self._build_progress_header,
            right_width=sidebar_width,
        )

        self.workspace_container = ttk.Frame(self)
        self.workspace_container.pack(fill=tk.BOTH, expand=True)

        # Match the other tabs: the left workspace owns the resizable console,
        # while the right settings panel stays fixed and full-height.
        self.content_split = ttk.Panedwindow(self.workspace_container, orient=tk.VERTICAL)
        self.content_split.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Review is the main expandable workspace on the left.
        review_panel = ttk.Frame(self.content_split)

        # Sources and Destination uses the same fixed-width settings sidebar
        # style as the other tabs, positioned on the right.
        self.source_panel = ScrollPanel(
            self.workspace_container,
            width=sidebar_width,
            global_scroll_targets=self.scroll_targets,
        )
        self.source_panel.configure(width=sidebar_width)
        self.source_panel.pack_propagate(False)
        self.source_panel.pack(side=tk.RIGHT, fill=tk.Y)

        inputs_group = CollapsibleGroup(
            self.source_panel.inner,
            "Inputs",
            expanded=True,
        )
        inputs_group.pack(fill=tk.X)
        inputs_group.body.columnconfigure(0, weight=1)

        ng_group = ttk.LabelFrame(inputs_group.body, text="NG Input", padding=8)
        ng_group.pack(fill=tk.X, pady=(0, 8))
        ng_group.columnconfigure(1, weight=1)
        self._build_path_row(
            ng_group,
            0,
            "NG Image Folder",
            self.ng_source_var,
            self._browse_ng_source,
            hint=(
                r"Example: <model>\<LotNo>\<Side>\ScreenShort\<image>"
                "\nAll image subfolders are scanned recursively. "
                "The defect class is read from the second-to-last underscore section (Open)."
            ),
        )

        good_group = ttk.LabelFrame(inputs_group.body, text="Good Input", padding=8)
        good_group.pack(fill=tk.X, pady=(0, 8))
        good_group.columnconfigure(1, weight=1)
        self._build_path_row(
            good_group,
            0,
            "JSON Log",
            self.json_source_var,
            self._browse_json_source,
            hint=(
                r"Example: <model>_JSON\<LotNo>\<json file>"
                "\nSelect the root folder containing JEI .json files; all subfolders are scanned recursively."
            ),
        )
        self._build_path_row(
            good_group,
            1,
            "Good Image Folder",
            self.ok_source_var,
            self._browse_ok_source,
            hint=(
                r"Example: <model>_<Side>\<LotNo>\<folder>\<image>"
                "\nWith JSON selected, the folder name (Camera01.jpg), parent lot (LOT001), "
                "and Top/Bottom path must match the JSON fields. Without JSON, every image below this folder is scanned as Pass."
            ),
        )

        model_group = ttk.LabelFrame(inputs_group.body, text="Model", padding=8)
        model_group.pack(fill=tk.X, pady=(0, 8))
        model_group.columnconfigure(1, weight=1)
        self._build_path_row(
            model_group,
            0,
            "Promoted Package",
            self.promoted_package_var,
            self._browse_promoted_package,
            hint=(
                r"Example: D:\Models\JEI_V1\production_package"
                "\nSelect the folder containing:"
                "\n• model.pt"
                "\n• labels.json"
                "\n• thresholds.json"
                "\n• model_metadata.json"
                "\n• defect_multipliers.json (optional)"
            ),
        )

        output_group = ttk.LabelFrame(inputs_group.body, text="Output", padding=8)
        output_group.pack(fill=tk.X)
        output_group.columnconfigure(1, weight=1)
        self._build_path_row(
            output_group,
            0,
            "Output Folder",
            self.destination_var,
            self._browse_destination,
            hint=(
                "The build creates one subfolder per final class, such as Pass/, Open/, and Short/."
            ),
        )

        self.summary_frame = ttk.Frame(review_panel)
        self.summary_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(self.summary_frame, text="Review", style="SectionHeader.TLabel").pack(side=tk.LEFT)
        ttk.Label(self.summary_frame, textvariable=self.summary_var, style="SectionNote.TLabel").pack(
            side=tk.RIGHT
        )

        filter_frame = ttk.Frame(review_panel)
        filter_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(filter_frame, text="View:").pack(side=tk.LEFT, padx=(0, 6))
        self.filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            state="readonly",
            width=28,
        )
        self.filter_combo.pack(side=tk.LEFT)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)
        self.spot_check_button = ttk.Button(
            filter_frame, text="Spot-check sample", command=self._sample_agreements
        )
        self.spot_check_button.pack(side=tk.LEFT, padx=(8, 0))
        _info_button(
            filter_frame,
            "Spot-check Sample",
            "Selects a random sample of images where the proposed class agrees with the model prediction, "
            "so you can manually verify apparently correct results without reviewing every image.",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self._set_filter_options(predictions_available=False)

        pagination = ttk.Frame(review_panel)
        pagination.pack(fill=tk.X, pady=(0, 6))
        self.prev_page_button = ttk.Button(
            pagination, text="Previous", command=self._previous_page, state=tk.DISABLED
        )
        self.prev_page_button.pack(side=tk.LEFT)
        self.page_var = tk.StringVar(value="Page 0 of 0")
        ttk.Label(pagination, textvariable=self.page_var).pack(side=tk.LEFT, padx=10)
        self.next_page_button = ttk.Button(
            pagination, text="Next", command=self._next_page, state=tk.DISABLED
        )
        self.next_page_button.pack(side=tk.LEFT)
        style = ttk.Style(self)
        style.configure("ReviewDisagreement.TFrame", background="#ffe1e1")

        self.table_outer = ttk.Frame(review_panel)
        self.table_outer.pack(fill=tk.BOTH, expand=True)
        self.table_outer.rowconfigure(1, weight=1)
        self.table_outer.columnconfigure(0, weight=1)
        self.table_outer.columnconfigure(1, weight=0)
        table_outer = self.table_outer

        header = ttk.Frame(table_outer, padding=(5, 5))
        header.grid(row=0, column=0, sticky="ew")
        self._configure_review_columns(header)
        headings = (
            "Thumbnail",
            "Filename",
            "Proposed class",
            "Model prediction",
            "Override",
            "Exclude",
            "Status",
        )
        for column, text in enumerate(headings):
            ttk.Label(header, text=text, style="TableHeader.TLabel").grid(
                row=0, column=column, sticky="w", padx=4
            )
        ttk.Frame(table_outer, width=18).grid(row=0, column=1, sticky="ns")

        body_frame = ttk.Frame(table_outer)
        body_frame.grid(row=1, column=0, sticky="nsew")
        body_frame.rowconfigure(0, weight=1)
        body_frame.columnconfigure(0, weight=1)

        self.review_canvas = tk.Canvas(body_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_outer, orient=tk.VERTICAL, command=self.review_canvas.yview)
        self.review_canvas.configure(yscrollcommand=scrollbar.set)
        self.review_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")

        self.review_body = ttk.Frame(self.review_canvas)
        self.review_window = self.review_canvas.create_window(
            (0, 0), window=self.review_body, anchor="nw"
        )
        self._configure_review_columns(self.review_body)
        self.review_body.bind(
            "<Configure>",
            lambda _event: self.review_canvas.configure(scrollregion=self.review_canvas.bbox("all")),
        )
        self.review_canvas.bind(
            "<Configure>",
            lambda event: self.review_canvas.itemconfigure(self.review_window, width=event.width),
        )
        self.scroll_targets.append(self.review_canvas)

        self.log_panel = ConsoleLogPanel(self.content_split, height=8)
        self.content_split.add(review_panel, weight=MAIN_PANE_WEIGHT)
        self.log_visible = False
        self.logger = self.log_panel.logger
        self.shared_console.attach(self.logger)

    def _build_progress_header(self, parent):
        ttk.Label(
            parent,
            textvariable=self.prediction_progress_text,
            style="SectionNote.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.prediction_progress = ttk.Progressbar(
            parent,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.prediction_progress_var,
            length=HEADER_PROGRESS_LENGTH,
        )
        self.prediction_progress.grid(row=0, column=1, sticky="w")

    @staticmethod
    def _configure_review_columns(frame):
        widths = (80, 140, 125, 135, 120, 70, 250)
        for index, width in enumerate(widths):
            frame.columnconfigure(index, minsize=width, weight=1 if index == 6 else 0)

    @staticmethod
    def _build_path_row(parent, row, label, variable, command, hint=None):
        ttk.Label(parent, text=f"{label}:").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=(4, 2)
        )
        ttk.Entry(parent, textvariable=variable, width=PATH_ENTRY_WIDTH).grid(
            row=row, column=1, sticky="ew", padx=(0, 12), pady=(4, 2)
        )
        ttk.Button(parent, text="...", width=3, command=command).grid(
            row=row, column=2, padx=(8, 0), pady=(4, 2)
        )
        if hint:
            _info_button(
                parent,
                f"{label} Hint",
                hint,
            ).grid(row=row, column=3, sticky="e", padx=(8, 0), pady=(4, 2))

    def _browse_json_source(self):
        selected = filedialog.askdirectory(title="Select folder containing JEI JSON logs")
        if selected:
            self.json_source_var.set(selected)
            self._save_settings()

    def _browse_ng_source(self):
        selected = filedialog.askdirectory(title="Select NG image folder")
        if selected:
            self.ng_source_var.set(selected)
            self._save_settings()

    def _browse_ok_source(self):
        selected = filedialog.askdirectory(title="Select OK image folder")
        if selected:
            self.ok_source_var.set(selected)
            self._save_settings()

    def _browse_promoted_package(self):
        selected = filedialog.askdirectory(
            title="Select Promoted Model Package",
        )
        if not selected:
            return
        self.promoted_package_var.set(selected)
        if not self._load_promoted_package(selected, show_dialog=True):
            return
        self._save_settings()

    def _load_promoted_package(self, package_path: str, show_dialog: bool = False) -> bool:
        try:
            package = resolve_live_package(package_path)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self._dataset_package = None
            self.logger.log(f"[Package Error] {message}")
            if show_dialog:
                messagebox.showwarning("Dataset", message)
            return False

        self._dataset_package = package
        self._refresh_class_names_from_package(package)
        self.logger.log("[Package] Loaded promoted package for Dataset prediction.")
        for line in package.summary.splitlines():
            self.logger.log(f"[Package] {line}")
        self.logger.log(f"[Package] Classes: {', '.join(package.class_names)}")
        self.logger.log(f"[Package] Active thresholds: {len(package.active_thresholds)}")
        return True

    def _resolve_dataset_model_settings(self) -> dict:
        package_path = self.promoted_package_var.get().strip()
        if not package_path:
            raise ValueError("Promoted Package required.")
        package = resolve_live_package(package_path)
        self._dataset_package = package
        self._refresh_class_names_from_package(package)
        return {
            "package": package,
            "checkpoint_path": package.checkpoint_path,
            "class_names": list(package.class_names),
            "active_thresholds": dict(package.active_thresholds),
            "multiplier_path": package.multiplier_path,
            "image_size": package.image_size,
        }

    def _refresh_class_names_from_package(self, package: LivePackage):
        class_names = list(package.class_names)
        if class_names == self.class_names:
            return
        self.class_names = class_names
        self._class_name_lookup = {name.casefold(): name for name in self.class_names}
        self.logger.log(f"[Model] Dataset class list: {', '.join(self.class_names)}")

    def _browse_destination(self):
        selected = filedialog.askdirectory(title="Select dataset destination root")
        if selected:
            self.destination_var.set(selected)
            self._save_settings()

    def _start_scan(self):
        if self._scan_thread and self._scan_thread.is_alive():
            self.logger.log("[Warning] A source scan is already running.")
            return
        if self._prediction_thread and self._prediction_thread.is_alive():
            self.logger.log("[Warning] Triage prediction is still running.")
            return
        if not self.json_source_var.get().strip() and not self.ng_source_var.get().strip() and not self.ok_source_var.get().strip():
            messagebox.showwarning("Dataset", "Select at least one source before scanning.")
            return
        try:
            self._resolve_dataset_model_settings()
        except Exception as exc:
            messagebox.showwarning(
                "Dataset",
                str(exc).strip() or exc.__class__.__name__,
            )
            return

        # Read every Tk variable on the UI thread before starting background work.
        # Accessing StringVar or calling widget.after() from a worker can hang Tk on Windows.
        json_source = self.json_source_var.get().strip()
        ng_source = self.ng_source_var.get().strip()
        ok_source = self.ok_source_var.get().strip()

        self._reset_review_table(log_action=False)
        self._save_settings()
        self.summary_var.set("Scanning sources...")
        self.logger.log("[Scan] Starting dataset source scan.")
        self._scan_thread = threading.Thread(
            target=self._scan_sources_worker,
            args=(json_source, ng_source, ok_source),
            daemon=True,
        )
        self._scan_thread.start()
        self.after(50, self._poll_scan_results)

    def _scan_sources_worker(self, json_source: str, ng_source: str, ok_source: str):
        collected: list[ReviewItem] = []
        try:
            if json_source:
                collected.extend(
                    self._scan_json_source(
                        json_source,
                        ok_base_dir=Path(ok_source) if ok_source else None,
                    )
                )
            if ng_source:
                collected.extend(self._scan_ng_source(Path(ng_source)))
            # When JSON is present, the OK folder is the wider JEI image base used
            # to resolve JSON imagePath/imagePosition records. Scanning every file
            # under it would add unverified and duplicate Pass rows. Preserve the
            # standalone all-images behavior only when no JSON source is selected.
            if ok_source and not json_source:
                collected.extend(self._scan_ok_source(Path(ok_source)))
            self._scan_results.put(("success", collected))
        except Exception as exc:
            self._scan_results.put(("error", str(exc)))

    def _poll_scan_results(self):
        completed = False
        while True:
            try:
                result_type, payload = self._scan_results.get_nowait()
            except queue.Empty:
                break

            if result_type == "log":
                self.logger.log(payload)
            elif result_type == "success":
                self._finish_scan(payload)
                completed = True
                break
            elif result_type == "error":
                self._scan_failed(payload)
                completed = True
                break

        if completed:
            return
        if self._scan_thread and self._scan_thread.is_alive():
            self.after(50, self._poll_scan_results)
        else:
            # Give the worker's final queue write one extra UI cycle.
            self.after(50, self._poll_scan_results_once)

    def _poll_scan_results_once(self):
        self._poll_scan_results()

    def _scan_json_source(self, source_text: str, ok_base_dir: Path | None = None) -> list[ReviewItem]:
        json_files = self._expand_json_source(source_text)
        self._thread_log(f"[JSON] Found {len(json_files)} JSON file(s).")
        results: list[ReviewItem] = []
        dir_cache = self._build_jei_ok_directory_cache(ok_base_dir) if ok_base_dir else {}

        for json_path in json_files:
            try:
                with json_path.open("r", encoding="utf-8-sig") as handle:
                    payload = json.load(handle)
            except Exception as exc:
                results.append(
                    ReviewItem(
                        source_path=None,
                        source_kind="JSON",
                        filename=json_path.name,
                        proposed_class="",
                        program="Unknown",
                        lot="Unknown",
                        side="Unknown",
                        needs_review=True,
                        review_reason=f"JSON could not be read: {exc}",
                        source_reference=str(json_path),
                    )
                )
                continue

            if not isinstance(payload, list):
                results.append(
                    ReviewItem(
                        source_path=None,
                        source_kind="JSON",
                        filename=json_path.name,
                        proposed_class="",
                        program="Unknown",
                        lot="Unknown",
                        side="Unknown",
                        needs_review=True,
                        review_reason="JSON root is not a list of records",
                        source_reference=str(json_path),
                    )
                )
                continue

            for record_index, record in enumerate(payload, start=1):
                results.extend(
                    self._items_from_json_record(
                        json_path, record_index, record, dir_cache,
                        use_jei_ok_lookup=bool(ok_base_dir),
                    )
                )
        return results

    def _build_jei_ok_directory_cache(self, base_dir: Path) -> dict[tuple[str, str, str], Path]:
        cache: dict[tuple[str, str, str], Path] = {}
        if not base_dir.is_dir():
            self._thread_log(f"[OK] Base folder not found: {base_dir}")
            return cache

        for folder in base_dir.rglob("*"):
            if not folder.is_dir():
                continue
            folder_name = folder.name.strip().lower()
            if not (folder_name.endswith(".jpg") or folder_name.endswith(".png")):
                continue
            try:
                if not any(child.is_file() for child in folder.iterdir()):
                    continue
            except OSError:
                continue

            lot_name = folder.parent.name.strip().lower()
            path_lower = str(folder).lower()
            side = "bottom" if "bottom" in path_lower else "top"
            cache[(side, lot_name, folder_name)] = folder

        self._thread_log(f"[OK] Cached {len(cache)} JEI image folder(s) under {base_dir}.")
        return cache

    @staticmethod
    def _expand_json_source(source_text: str) -> list[Path]:
        source = Path(source_text).expanduser()
        if source.is_dir():
            return sorted(source.rglob("*.json"), key=lambda path: str(path).lower())
        return sorted(
            (Path(match) for match in glob.glob(source_text, recursive=True)),
            key=lambda path: str(path).lower(),
        )

    def _items_from_json_record(
        self, json_path, record_index, record, dir_cache, use_jei_ok_lookup=False
    ):
        reference = f"{json_path} record {record_index}"
        if not isinstance(record, dict):
            return [
                ReviewItem(
                    None, "JSON", f"record-{record_index}", "", "Unknown", "Unknown", "Unknown",
                    True, "Record is not an object", reference
                )
            ]

        proposed_class, record_reasons = self._validated_record_class(record)

        # The verified-OK JEI workflow intentionally accepts only records where
        # both fields explicitly say PASS, matching get_images_from_json.py.
        if use_jei_ok_lookup:
            defective = str(record.get("Defective", "")).strip().upper()
            status = str(record.get("status", "")).strip().upper()
            if defective != "PASS" or status != "PASS":
                return []
            proposed_class = "Pass"
            record_reasons = []

        images = record.get("Images")
        if not isinstance(images, list) or not images:
            return [
                ReviewItem(
                    None, "JSON", f"record-{record_index}", proposed_class,
                    self._clean_component(record.get("ProgramName"), "Unknown"),
                    self._clean_component(record.get("Lot"), "Unknown"),
                    self._clean_component(record.get("Side"), "Unknown"),
                    True, self._join_reasons(record_reasons, "Images is missing or empty"), reference
                )
            ]

        items = []
        for image_index, image_info in enumerate(images, start=1):
            reasons = list(record_reasons)
            if not isinstance(image_info, dict):
                items.append(
                    ReviewItem(
                        None, "JSON", f"record-{record_index}-image-{image_index}", proposed_class,
                        self._clean_component(record.get("ProgramName"), "Unknown"),
                        self._clean_component(record.get("Lot"), "Unknown"),
                        self._clean_component(record.get("Side"), "Unknown"),
                        True, self._join_reasons(reasons, "Image entry is not an object"), reference
                    )
                )
                continue

            if use_jei_ok_lookup:
                image_path, path_reason = self._resolve_jei_ok_positioned_image(
                    image_info, dir_cache
                )
            else:
                image_path, path_reason = self._resolve_positioned_image(
                    json_path, image_info.get("imagePath"), image_info.get("imagePosition"), dir_cache
                )
            if path_reason:
                reasons.append(path_reason)
            filename = image_path.name if image_path else str(image_info.get("imagePath") or f"image-{image_index}")
            program = self._clean_component(
                image_info.get("ProgramName", record.get("ProgramName")), "Unknown"
            )
            lot = self._clean_component(image_info.get("Lot", record.get("Lot")), "Unknown")
            side = self._clean_component(image_info.get("Side", record.get("Side")), "Unknown")
            output_filename = None
            if use_jei_ok_lookup and image_path is not None:
                image_name_stem = Path(str(image_info.get("imageName", image_path.stem))).stem
                output_filename = self._clean_component(
                    f"{program}_{side}_{lot}_{image_name_stem}", "image"
                ) + image_path.suffix
            items.append(
                ReviewItem(
                    source_path=image_path,
                    source_kind="JSON",
                    filename=filename,
                    proposed_class=proposed_class,
                    program=program,
                    lot=lot,
                    side=side,
                    needs_review=bool(reasons),
                    review_reason="; ".join(reasons),
                    source_reference=f"{reference}, image {image_index}",
                    output_filename=output_filename,
                )
            )
        return items

    def _canonical_class_name(self, value: str) -> str:
        """Return the model's capitalization for a class name when it matches case-insensitively."""
        text = str(value).strip()
        if not text:
            return text
        return self._class_name_lookup.get(text.casefold(), text)

    def _validated_record_class(self, record: dict) -> tuple[str, list[str]]:
        reasons = []
        defective = record.get("Defective")
        status = record.get("status")
        status_text = str(status).strip().upper() if status is not None else ""
        defective_text = str(defective).strip() if defective is not None else ""
        canonical_defective = self._canonical_class_name(defective_text)

        if "Defective" not in record:
            reasons.append("Defective field is missing")
        if "status" not in record:
            reasons.append("status field is missing")
        if status_text not in EXPECTED_STATUSES:
            reasons.append(f"Unexpected status value: {status!r}")

        if status_text == "PASS":
            proposed = "Pass"
            if defective_text.lower() not in PASS_MARKERS:
                reasons.append("PASS status conflicts with Defective value")
        elif status_text == "FAIL":
            proposed = canonical_defective
            if defective_text.lower() in PASS_MARKERS:
                reasons.append("FAIL status has no valid defect class")
            elif canonical_defective not in self.class_names:
                reasons.append(f"Defective class is not in model class list: {defective_text!r}")
        else:
            proposed = canonical_defective if canonical_defective in self.class_names else ""

        return proposed, reasons


    @staticmethod
    def _resolve_jei_ok_positioned_image(image_info, dir_cache):
        required = ("ProgramName", "Lot", "Side", "imagePosition", "imageName", "imagePath")
        missing = [name for name in required if not image_info.get(name)]
        if missing:
            return None, "Missing required JEI field(s): " + ", ".join(missing)

        image_folder_name = str(image_info["imagePath"]).replace("/", "\\").split("\\")[-1].strip().lower()
        side = str(image_info["Side"]).strip().lower()
        lot = str(image_info["Lot"]).strip().lower()
        folder = dir_cache.get((side, lot, image_folder_name))
        if folder is None:
            return None, f"No OK folder matched side={side}, lot={lot}, imagePath={image_folder_name}"

        try:
            position = int(image_info["imagePosition"])
        except (TypeError, ValueError):
            return None, f"Invalid imagePosition: {image_info.get('imagePosition')!r}"
        if position < 1:
            return None, f"imagePosition must be 1-based, got {position}"

        try:
            files = sorted((path for path in folder.iterdir() if path.is_file()), key=lambda path: path.name)
        except OSError as exc:
            return None, f"Cannot read image folder {folder}: {exc}"
        if position > len(files):
            return None, f"imagePosition {position} exceeds {len(files)} files in {folder}"

        selected = files[position - 1]
        if selected.suffix.lower() not in IMAGE_EXTENSIONS:
            return selected, f"Resolved file is not a supported image: {selected.name}"
        return selected, ""

    @staticmethod
    def _resolve_positioned_image(json_path, image_path_value, position_value, dir_cache):
        if not image_path_value:
            return None, "imagePath is missing"
        folder = Path(str(image_path_value)).expanduser()
        if not folder.is_absolute():
            folder = (json_path.parent / folder).resolve()
        if folder.is_file():
            return folder, "imagePath points to a file; expected a folder"
        if not folder.is_dir():
            return None, f"Image folder not found: {folder}"
        try:
            position = int(position_value)
        except (TypeError, ValueError):
            return None, f"Invalid imagePosition: {position_value!r}"
        if position < 1:
            return None, f"imagePosition must be 1-based, got {position}"

        if folder not in dir_cache:
            dir_cache[folder] = sorted(
                (path for path in folder.iterdir() if path.is_file()),
                key=lambda path: path.name.lower(),
            )
        files = dir_cache[folder]
        if position > len(files):
            return None, f"imagePosition {position} exceeds {len(files)} files in {folder}"
        selected = files[position - 1]
        if selected.suffix.lower() not in IMAGE_EXTENSIONS:
            return selected, f"Resolved file is not a supported image: {selected.name}"
        return selected, ""


    def _scan_ok_source(self, root: Path) -> list[ReviewItem]:
        if not root.is_dir():
            return [
                ReviewItem(None, "OK folder", root.name or str(root), "Pass", "OK", "Unknown", "Unknown",
                           True, f"OK source folder not found: {root}", str(root))
            ]
        images = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: str(path).lower(),
        )
        self._thread_log(f"[OK] Found {len(images)} image(s) under {root}.")
        results = []
        for path in images:
            relative_parent = path.parent.relative_to(root)
            parts = relative_parent.parts
            results.append(ReviewItem(
                source_path=path, source_kind="OK folder", filename=path.name, proposed_class="Pass",
                program=self._clean_component(root.name, "OK"),
                lot=self._clean_component(parts[0] if parts else "Unknown", "Unknown"),
                side=self._clean_component(parts[-1] if parts else "Unknown", "Unknown"),
                source_reference=str(path),
            ))
        return results

    def _scan_ng_source(self, root: Path) -> list[ReviewItem]:
        if not root.is_dir():
            return [
                ReviewItem(None, "NG folder", root.name or str(root), "", "NG", "Unknown", "Unknown",
                           True, f"NG source folder not found: {root}", str(root))
            ]
        images = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: str(path).lower(),
        )
        self._thread_log(f"[NG] Found {len(images)} image(s) under {root}.")
        results = []
        for path in images:
            proposed = self._canonical_class_name(self.extract_defect_name(path.name))
            needs_review = proposed == "Unknown" or proposed not in self.class_names
            reason = ""
            if proposed == "Unknown":
                reason = "Filename shape does not contain a usable second-to-last underscore segment"
            elif proposed not in self.class_names:
                reason = f"Parsed class is not in model class list: {proposed!r}"
            relative_parent = path.parent.relative_to(root)
            parts = relative_parent.parts
            program = self._clean_component(root.name, "NG")
            lot = self._clean_component(parts[0] if parts else path.parent.name, "Unknown")
            side = self._clean_component(parts[-1] if parts else "Unknown", "Unknown")
            results.append(
                ReviewItem(path, "NG folder", path.name, proposed, program, lot, side,
                           needs_review, reason, str(path))
            )
        return results

    @staticmethod
    def extract_defect_name(filename: str) -> str:
        """Return the second-to-last underscore-separated filename segment.

        Unexpected shapes, including fewer than two underscore-separated segments or
        an empty defect segment, return ``Unknown`` and require manual review.
        """
        parts = Path(filename).stem.split("_")
        if len(parts) < 2:
            return "Unknown"
        defect_name = parts[-2].strip()
        return defect_name or "Unknown"

    def _finish_scan(self, items: list[ReviewItem]):
        self.items = items
        for item in self.items:
            item.excluded = item.needs_review or item.source_path is None
        usable = sum(item.source_path is not None for item in items)
        self._set_review_summary()
        self.logger.log(
            f"[Scan] Proposed {usable} image(s); {sum(item.needs_review for item in items)} row(s) flagged for review."
        )
        self._page_index = 0
        self._apply_review_filter(log=False)
        self._start_triage_predictions()

    def _set_review_summary(self):
        total = len(self.items)
        usable = sum(item.source_path is not None for item in self.items)
        flagged = sum(item.needs_review for item in self.items)
        disagreements = sum(self._agreement_status(item) == "disagree" for item in self.items)
        predicted = sum(item.predicted_class is not None for item in self.items)
        no_prediction = sum(
            item.source_path is not None and item.predicted_class is None
            for item in self.items
        )
        if predicted:
            self.summary_var.set(
                f"{total} row(s), {disagreements} disagreement(s), "
                f"{flagged} need review, {no_prediction} without prediction"
            )
        else:
            self.summary_var.set(f"{total} row(s), {usable} image(s), {flagged} need review")

    def _start_triage_predictions(self):
        if self._prediction_thread and self._prediction_thread.is_alive():
            self._prediction_restart_requested = True
            return
        self._prediction_restart_requested = False
        self._prediction_stop_event.clear()

        try:
            package_settings = self._resolve_dataset_model_settings()
        except Exception as exc:
            self._prediction_available = False
            self.prediction_progress_var.set(0.0)
            self.prediction_progress_text.set("Prediction unavailable")
            self._set_filter_options(predictions_available=False)
            self._refresh_prediction_cells()
            self._set_review_summary()
            self.logger.log(
                f"[Triage] Select a valid promoted package in the Dataset tab; "
                f"prediction was skipped: {str(exc).strip() or exc.__class__.__name__}"
            )
            return

        start = self._page_index * self._page_size
        end = start + self._page_size
        current_page = [
            item for item in self._filtered_items[start:end]
            if item.source_path is not None and item.predicted_class is None
        ]
        remaining = [
            item for item in self.items
            if item.source_path is not None
            and item.predicted_class is None
            and item not in current_page
        ]
        predictable = current_page + remaining
        if not predictable:
            self._refresh_prediction_cells()
            self.logger.log("[Triage] All resolved images already have predictions.")
            return

        self.prediction_progress_var.set(0.0)
        self.prediction_progress_text.set(f"Predicting 0/{len(predictable)}")
        self.summary_var.set(
            f"Predicting current page first ({len(current_page)} image(s)); "
            f"{len(remaining)} queued in background..."
        )
        package = package_settings["package"]
        self.logger.log(
            f"[Triage] Predicting current page first, then remaining images, "
            f"with promoted package: {package.package_dir}"
        )
        self._prediction_thread = threading.Thread(
            target=self._triage_predictions_worker,
            args=(package_settings, predictable, len(current_page)),
            daemon=True,
        )
        self._prediction_thread.start()
        self.after(50, self._poll_prediction_results)

    def _triage_predictions_worker(
        self, package_settings: dict, items: list[ReviewItem], foreground_count: int,
    ):
        try:
            from backend.inference_service import load_inference_model, predict_image

            class_names = list(package_settings["class_names"])
            multiplier_path = package_settings.get("multiplier_path")
            model_kwargs = {
                "checkpoint_path": package_settings["checkpoint_path"],
                "class_names": class_names,
                "active_thresholds": package_settings.get("active_thresholds") or {},
                "image_size": package_settings.get("image_size"),
            }
            if multiplier_path:
                model_kwargs["defect_multipliers"] = load_multipliers_from_json(
                    multiplier_path, class_names
                )
            else:
                # Match Live: a package without multipliers uses neutral values,
                # never an unrelated deployed multiplier file.
                model_kwargs["defect_multipliers"] = {
                    name: 0.0 for name in class_names if name != "Pass"
                }
            session = load_inference_model(**model_kwargs)
        except Exception as exc:
            self._prediction_results.put(("error", str(exc)))
            return

        total = len(items)
        for index, item in enumerate(items, start=1):
            if self._prediction_stop_event.is_set():
                self._prediction_results.put(("stopped", (index - 1, total)))
                return
            try:
                result = predict_image(session, item.source_path, class_names=class_names)
                item.predicted_class = result.predicted_class
                item.predicted_confidence = result.confidence
            except Exception as exc:
                item.predicted_class = None
                item.predicted_confidence = None
                self._prediction_results.put(
                    ("log", f"[Warning] Triage prediction failed for {item.filename}: {exc}")
                )
            self._prediction_results.put(("item", (item, index, total)))
            if foreground_count and index == foreground_count and index < total:
                self._prediction_results.put(("foreground_complete", (index, total)))
        self._prediction_results.put(("complete", None))

    def _poll_prediction_results(self):
        completed = False
        while True:
            try:
                result_type, payload = self._prediction_results.get_nowait()
            except queue.Empty:
                break
            if result_type == "log":
                self.logger.log(payload)
            elif result_type == "item":
                item, index, total = payload
                progress = index / max(total, 1) * 100.0
                self.prediction_progress_var.set(progress)
                self.prediction_progress_text.set(f"Predicting {index}/{total}")
                self._refresh_visible_prediction_item(item)
                self._set_review_summary()
            elif result_type == "foreground_complete":
                index, total = payload
                self.summary_var.set(
                    f"Current page prediction complete. Processing {total - index} remaining image(s) in background..."
                )
                self.logger.log(
                    f"[Triage] Current page is ready; continuing {total - index} image(s) in background."
                )
            elif result_type == "error":
                self._prediction_failed(str(payload))
                completed = True
                break
            elif result_type == "stopped":
                processed, total = payload
                self._finish_stopped_prediction(processed, total)
                completed = True
                break
            elif result_type == "complete":
                self._finish_triage_predictions()
                completed = True
                break
        if not completed and self._prediction_thread and self._prediction_thread.is_alive():
            self.after(50, self._poll_prediction_results)
        elif not completed:
            self.after(50, self._poll_prediction_results_once)

    def _poll_prediction_results_once(self):
        try:
            result_type, payload = self._prediction_results.get_nowait()
        except queue.Empty:
            self._prediction_failed("Prediction worker stopped without returning a result.")
            return
        if result_type == "error":
            self._prediction_failed(str(payload))
        elif result_type == "stopped":
            processed, total = payload
            self._finish_stopped_prediction(processed, total)
        elif result_type == "complete":
            self._finish_triage_predictions()
        else:
            self._poll_prediction_results()


    def _refresh_visible_prediction_item(self, changed_item: ReviewItem):
        for row_data in self.row_widgets:
            if row_data["item"] is not changed_item:
                continue
            if changed_item.predicted_class and changed_item.predicted_confidence is not None:
                text = f"{changed_item.predicted_class} ({changed_item.predicted_confidence:.1%})"
            else:
                text = "Prediction failed"
            row_data["prediction"].configure(text=text)
            row_style, status_text = self._status_for_item(changed_item)
            row_data["row"].configure(style=row_style)
            row_data["status"].configure(text=status_text)
            break

    def _stop_prediction(self):
        """Request a safe stop after the image currently being processed."""
        if not (self._prediction_thread and self._prediction_thread.is_alive()):
            self.logger.log("[Triage] No prediction is currently running.")
            return
        if self._prediction_stop_event.is_set():
            self.logger.log("[Triage] Prediction stop is already pending.")
            return
        self._prediction_restart_requested = False
        self._prediction_stop_event.set()
        self.prediction_progress_text.set("Stopping prediction...")
        self.summary_var.set("Stopping after the current image. Existing predictions will be kept.")
        self.logger.log("[Triage] Stop requested; waiting for the current image to finish.")

    def _resume_prediction(self):
        """Resume prediction for images that do not yet have a prediction."""
        if self._prediction_thread and self._prediction_thread.is_alive():
            self.logger.log("[Triage] Prediction is already running.")
            return
        remaining = sum(
            item.source_path is not None and item.predicted_class is None
            for item in self.items
        )
        if remaining <= 0:
            self.logger.log("[Triage] There are no remaining images to predict.")
            self.prediction_progress_text.set("Prediction complete")
            self._set_review_summary()
            return
        self._prediction_stop_event.clear()
        self.summary_var.set(f"Resuming prediction for {remaining} remaining image(s)...")
        self.logger.log(f"[Triage] Resuming prediction for {remaining} remaining image(s).")
        self._start_triage_predictions()

    def _finish_stopped_prediction(self, processed: int, total: int):
        self._prediction_thread = None
        self._prediction_restart_requested = False
        self._prediction_available = any(item.predicted_class for item in self.items)
        self.prediction_progress_var.set(processed / max(total, 1) * 100.0)
        self.prediction_progress_text.set(f"Prediction stopped {processed}/{total}")
        self._refresh_prediction_cells()
        self._set_filter_options(predictions_available=self._prediction_available)
        self._apply_review_filter(log=False)
        remaining = sum(
            item.source_path is not None and item.predicted_class is None
            for item in self.items
        )
        self.summary_var.set(
            f"Prediction stopped. {processed} processed in this run; {remaining} image(s) remain."
        )
        self.logger.log(
            f"[Triage] Prediction stopped safely: {processed}/{total} processed; "
            f"{remaining} image(s) remain. Click Resume to continue."
        )

    def _finish_triage_predictions(self):
        self._prediction_thread = None
        restart_prediction = self._prediction_restart_requested
        self._prediction_restart_requested = False
        self._prediction_available = True
        self.prediction_progress_var.set(100.0)
        self.prediction_progress_text.set("Prediction complete")
        self._refresh_prediction_cells()
        disagreements = sum(self._agreement_status(item) == "disagree" for item in self.items)
        no_prediction = sum(self._agreement_status(item) == "no_prediction" for item in self.items)
        flagged = sum(item.needs_review for item in self.items)
        self._set_filter_options(predictions_available=True, select_priority=True)
        if self.filter_var.get() == "Agreements (spot-check)":
            self._select_agreement_sample(log=False)
        self._apply_review_filter()
        self._set_review_summary()
        self.logger.log(
            f"[Triage] Prediction complete: {disagreements} disagreement(s), "
            f"{no_prediction} without prediction."
        )
        if restart_prediction:
            self.after_idle(self._start_triage_predictions)

    def _prediction_failed(self, message: str):
        self._prediction_thread = None
        restart_prediction = self._prediction_restart_requested
        self._prediction_restart_requested = False
        self._prediction_available = False
        self.prediction_progress_var.set(0.0)
        self.prediction_progress_text.set("Prediction failed")
        self._set_filter_options(predictions_available=False)
        self._refresh_prediction_cells()
        self._apply_review_filter()
        self.logger.log(f"[Warning] Triage prediction was skipped: {message}")
        if restart_prediction:
            self.after_idle(self._start_triage_predictions)

    @staticmethod
    def _agreement_status(item: ReviewItem) -> str:
        if not item.predicted_class or not item.proposed_class:
            return "no_prediction"
        if item.predicted_class == item.proposed_class:
            return "agree"
        return "disagree"

    def _set_filter_options(self, predictions_available: bool, select_priority: bool = False):
        if predictions_available:
            values = (
                "Priority (Disagreements + Needs review)",
                "All",
                "Disagreements only",
                "Needs review only",
                "Agreements (spot-check)",
            )
        else:
            values = ("All", "Needs review only")
        self.filter_combo.configure(values=values)

        saved = self._saved_filter_mode
        if select_priority and saved not in values:
            selected = "Priority (Disagreements + Needs review)"
        elif saved in values:
            selected = saved
        else:
            selected = values[0]
        self.filter_var.set(selected)
        self.spot_check_button.configure(
            state=tk.NORMAL if predictions_available else tk.DISABLED
        )

    def _on_filter_changed(self, _event=None):
        self._saved_filter_mode = self.filter_var.get()
        if self.filter_var.get() == "Agreements (spot-check)" and not self._agreement_sample_ids:
            self._select_agreement_sample(log=False)
        self._save_settings()
        self._apply_review_filter()

    def _select_agreement_sample(self, log: bool = True):
        agreement_ids = [
            id(item) for item in self.items if self._agreement_status(item) == "agree"
        ]
        if not agreement_ids:
            self._agreement_sample_ids.clear()
            if log:
                self.logger.log("[Triage] No agreement rows are available for spot-checking.")
            return
        sample_size = max(1, round(len(agreement_ids) * 0.10))
        self._agreement_sample_ids = set(random.sample(agreement_ids, sample_size))
        if log:
            self.logger.log(
                f"[Triage] Selected {sample_size} of {len(agreement_ids)} agreement row(s) "
                "for spot-checking."
            )

    def _sample_agreements(self):
        self._select_agreement_sample(log=True)
        self.filter_var.set("Agreements (spot-check)")
        self._saved_filter_mode = self.filter_var.get()
        self._save_settings()
        self._apply_review_filter()

    def _item_matches_filter(self, item: ReviewItem) -> bool:
        mode = self.filter_var.get()
        agreement = self._agreement_status(item)
        return (
            mode == "All"
            or (mode == "Needs review only" and item.needs_review)
            or (mode == "Disagreements only" and agreement == "disagree")
            or (
                mode == "Priority (Disagreements + Needs review)"
                and (item.needs_review or agreement == "disagree")
            )
            or (
                mode == "Agreements (spot-check)"
                and agreement == "agree"
                and id(item) in self._agreement_sample_ids
            )
        )

    def _apply_review_filter(self, log: bool = True):
        self._filtered_items = [item for item in self.items if self._item_matches_filter(item)]
        page_count = self._page_count()
        if page_count == 0:
            self._page_index = 0
        else:
            self._page_index = min(self._page_index, page_count - 1)
        self._render_current_page()
        if self.items and log:
            self.logger.log(
                f"[Filter] Showing {len(self._filtered_items)} of {len(self.items)} review row(s)."
            )

    def _page_count(self) -> int:
        if not self._filtered_items:
            return 0
        return (len(self._filtered_items) + self._page_size - 1) // self._page_size

    def _previous_page(self):
        if self._page_index > 0:
            self._page_index -= 1
            self._render_current_page()
            if not (self._prediction_thread and self._prediction_thread.is_alive()):
                self._start_triage_predictions()

    def _next_page(self):
        if self._page_index + 1 < self._page_count():
            self._page_index += 1
            self._render_current_page()
            if not (self._prediction_thread and self._prediction_thread.is_alive()):
                self._start_triage_predictions()

    def _render_current_page(self):
        self._render_generation += 1
        generation = self._render_generation
        for child in self.review_body.winfo_children():
            child.destroy()
        self.row_widgets.clear()
        self.thumbnail_refs.clear()
        self.review_canvas.yview_moveto(0)

        page_count = self._page_count()
        if page_count == 0:
            self.page_var.set("Page 0 of 0")
            self.prev_page_button.configure(state=tk.DISABLED)
            self.next_page_button.configure(state=tk.DISABLED)
            return

        start = self._page_index * self._page_size
        end = min(start + self._page_size, len(self._filtered_items))
        page_items = self._filtered_items[start:end]
        self.page_var.set(f"Page {self._page_index + 1} of {page_count}")
        self.prev_page_button.configure(
            state=tk.NORMAL if self._page_index > 0 else tk.DISABLED
        )
        self.next_page_button.configure(
            state=tk.NORMAL if self._page_index + 1 < page_count else tk.DISABLED
        )
        self._render_page_batch(page_items, 0, generation)

    def _render_page_batch(self, page_items: list[ReviewItem], offset: int, generation: int):
        if generation != self._render_generation:
            return
        batch_end = min(offset + 10, len(page_items))
        for item in page_items[offset:batch_end]:
            self._append_review_row(item, load_thumbnail=False)
        if batch_end < len(page_items):
            self.after(1, lambda: self._render_page_batch(page_items, batch_end, generation))
        else:
            self.after_idle(lambda: self._load_visible_thumbnails(generation, 0))

    def _load_visible_thumbnails(self, generation: int, offset: int):
        if generation != self._render_generation:
            return
        batch_end = min(offset + 8, len(self.row_widgets))
        for row_data in self.row_widgets[offset:batch_end]:
            self._populate_row_thumbnail(row_data)
        if batch_end < len(self.row_widgets):
            self.after(1, lambda: self._load_visible_thumbnails(generation, batch_end))

    def _status_for_item(self, item: ReviewItem) -> tuple[str, str]:
        """Return the row style and one of the four review statuses."""
        if item.needs_review:
            reason = item.review_reason or "scan issue requires manual review"
            return "ReviewWarning.TFrame", f"Needs review: {reason}"

        agreement = self._agreement_status(item)
        if agreement == "disagree":
            return "ReviewDisagreement.TFrame", "Needs review: class disagreement"
        if agreement == "agree":
            return "TFrame", "Model agrees"
        return "TFrame", "Awaiting prediction"

    def _refresh_prediction_cells(self):
        for row_data in self.row_widgets:
            item = row_data["item"]
            if item.predicted_class and item.predicted_confidence is not None:
                prediction_text = f"{item.predicted_class} ({item.predicted_confidence:.1%})"
            else:
                prediction_text = "Pending..."
            row_data["prediction"].configure(text=prediction_text)

            row_style, status_text = self._status_for_item(item)
            row_data["row"].configure(style=row_style)
            row_data["status"].configure(text=status_text)

    def _scan_failed(self, message):
        self.summary_var.set("Scan failed.")
        self.logger.log(f"[Error] Dataset scan failed: {message}")
        messagebox.showerror("Dataset scan failed", message)

    def _append_review_row(self, item: ReviewItem, load_thumbnail: bool = True):
        row_index = len(self.row_widgets)
        style = "ReviewWarning.TFrame" if item.needs_review else "TFrame"
        row = ttk.Frame(self.review_body, padding=(4, 5), style=style)
        row.grid(row=row_index, column=0, columnspan=7, sticky="ew")
        self._configure_review_columns(row)

        thumbnail_label = ttk.Label(
            row, text="Loading...", anchor="center", cursor="hand2"
        )
        thumbnail_label.grid(row=0, column=0, sticky="ew", padx=4)
        thumbnail_label.bind(
            "<Button-1>", lambda _event, i=item: self._open_image_preview(i)
        )

        ttk.Label(row, text=item.filename, wraplength=135).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(row, text=item.proposed_class or "—").grid(row=0, column=2, sticky="w", padx=4)
        prediction_text = (
            f"{item.predicted_class} ({item.predicted_confidence:.1%})"
            if item.predicted_class and item.predicted_confidence is not None
            else "No prediction" if self._prediction_available else "Pending..."
        )
        prediction_label = ttk.Label(row, text=prediction_text)
        prediction_label.grid(row=0, column=3, sticky="w", padx=4)

        override_var = tk.StringVar(value=item.override_class)
        override = ttk.Combobox(
            row, textvariable=override_var, values=[""] + self.class_names, state="readonly", width=18
        )
        override.grid(row=0, column=4, sticky="ew", padx=4)

        exclude_var = tk.BooleanVar(value=item.excluded)
        exclude_button = ttk.Checkbutton(
            row, variable=exclude_var,
            command=lambda i=item, ex=exclude_var: self._set_item_excluded(i, ex.get()),
        )
        exclude_button.grid(row=0, column=5, padx=4)

        row_style, status_text = self._status_for_item(item)
        row.configure(style=row_style)
        status_label = ttk.Label(row, text=status_text, wraplength=245)
        status_label.grid(row=0, column=6, sticky="w", padx=4)

        override.bind(
            "<<ComboboxSelected>>",
            lambda _event, i=item, ov=override_var, ex=exclude_var, label=status_label: self._resolve_row(
                i, ov, ex, label
            ),
        )
        row_data = {
            "item": item,
            "row": row,
            "thumbnail": thumbnail_label,
            "prediction": prediction_label,
            "override": override_var,
            "exclude": exclude_var,
            "status": status_label,
            "photo": None,
        }
        self.row_widgets.append(row_data)
        if load_thumbnail:
            self._populate_row_thumbnail(row_data)

    def _set_item_excluded(self, item: ReviewItem, excluded: bool):
        item.excluded = bool(excluded)

    def _resolve_row(self, item, override_var, exclude_var, status_label):
        item.override_class = override_var.get().strip()
        if item.override_class:
            item.excluded = item.source_path is None
            exclude_var.set(item.excluded)
            if item.source_path is None:
                status_label.configure(text="Cannot include: source image was not resolved")
            else:
                status_label.configure(text="Resolved by manual override")

    def _populate_row_thumbnail(self, row_data):
        label = row_data["thumbnail"]
        try:
            if not label.winfo_exists():
                return
            photo = self._load_thumbnail(row_data["item"].source_path)
            if photo is None:
                label.configure(image="", text="No preview")
                row_data["photo"] = None
            else:
                label.configure(image=photo, text="")
                row_data["photo"] = photo
                self.thumbnail_refs.append(photo)
        except tk.TclError as exc:
            label.configure(image="", text="No preview")
            row_data["photo"] = None
            self.logger.log(f"[Warning] Thumbnail bitmap allocation failed: {exc}")

    def _load_thumbnail(self, path: Path | None):
        if path is None or not path.is_file():
            return None
        key = str(path.resolve())
        if key in self._thumbnail_cache:
            cached = self._thumbnail_cache.pop(key)
            self._thumbnail_cache[key] = cached
            return cached
        try:
            with Image.open(path) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((88, 64), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(preview)
        except (OSError, ValueError, tk.TclError):
            photo = None
        self._thumbnail_cache[key] = photo
        while len(self._thumbnail_cache) > self._thumbnail_cache_limit:
            self._thumbnail_cache.popitem(last=False)
        return photo


    def _open_image_preview(self, item: ReviewItem):
        path = item.source_path
        if path is None or not path.is_file():
            messagebox.showwarning("Image preview", "The original image is unavailable.")
            return
        try:
            with Image.open(path) as image:
                original = ImageOps.exif_transpose(image).convert("RGB").copy()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Image preview", f"Could not open image:\n{exc}")
            return

        window = tk.Toplevel(self)
        window.title(item.filename)
        window.geometry("500x500")
        window.minsize(400, 400)
        toolbar = ttk.Frame(window, padding=(8, 6))
        toolbar.pack(fill=tk.X)
        canvas = tk.Canvas(window, background="#202020", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        state = {"scale": 1.0, "photo": None}

        def redraw(_event=None):
            available_w = max(canvas.winfo_width(), 1)
            available_h = max(canvas.winfo_height(), 1)
            width = max(1, int(original.width * state["scale"]))
            height = max(1, int(original.height * state["scale"]))
            shown = original.resize((width, height), Image.Resampling.LANCZOS)
            state["photo"] = ImageTk.PhotoImage(shown)
            canvas.delete("all")
            canvas.create_image(available_w // 2, available_h // 2, image=state["photo"], anchor="center")
            canvas.configure(scrollregion=(0, 0, max(width, available_w), max(height, available_h)))
            zoom_var.set(f"{state['scale'] * 100:.0f}%")

        def fit():
            window.update_idletasks()
            cw, ch = max(canvas.winfo_width() - 20, 1), max(canvas.winfo_height() - 20, 1)
            state["scale"] = min(cw / original.width, ch / original.height, 1.0)
            redraw()

        def zoom(factor):
            state["scale"] = min(8.0, max(0.05, state["scale"] * factor))
            redraw()

        zoom_var = tk.StringVar(value="100%")
        ttk.Button(toolbar, text="−", width=3, command=lambda: zoom(0.8)).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=zoom_var, width=7, anchor="center").pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="+", width=3, command=lambda: zoom(1.25)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Fit", command=fit).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="100%", command=lambda: (state.update(scale=1.0), redraw())).pack(side=tk.LEFT, padx=4)
        ttk.Label(toolbar, text=f"{original.width} × {original.height}").pack(side=tk.RIGHT)
        canvas.bind("<MouseWheel>", lambda event: zoom(1.15 if event.delta > 0 else 1 / 1.15))
        canvas.bind("<Configure>", lambda _event: redraw())
        window.after_idle(redraw)

    def _add_to_dataset(self):
        destination_text = self.destination_var.get().strip()
        if not destination_text:
            messagebox.showwarning("Dataset", "Select a destination root first.")
            return
        destination = Path(destination_text).expanduser()
        selected_rows = []
        unresolved = []

        for item in self.items:
            if item.excluded:
                continue
            final_class = item.override_class.strip() or item.proposed_class
            if item.source_path is None or not item.source_path.is_file():
                unresolved.append(f"{item.filename}: source image is unavailable")
                continue
            if not final_class or final_class not in self.class_names:
                unresolved.append(f"{item.filename}: choose a valid class override")
                continue
            if item.needs_review and not item.override_class.strip():
                unresolved.append(f"{item.filename}: flagged rows require an explicit override")
                continue
            selected_rows.append((item, final_class))

        if unresolved:
            messagebox.showwarning(
                "Review required",
                "Some included rows are unresolved:\n\n" + "\n".join(unresolved[:12]),
            )
            self.logger.log(f"[Warning] Build cancelled; {len(unresolved)} included row(s) remain unresolved.")
            return
        if not selected_rows:
            messagebox.showinfo("Dataset", "No reviewed images are selected for inclusion.")
            return
        if not messagebox.askyesno(
            "Build Dataset",
            f"Copy {len(selected_rows)} reviewed image(s) into:\n{destination}?",
        ):
            return

        if self._build_in_progress:
            messagebox.showinfo("Build Dataset", "A dataset build is already running.")
            return
        self._build_in_progress = True
        self.prediction_progress_var.set(0.0)
        self.prediction_progress_text.set(f"Building 0/{len(selected_rows)}")
        self.summary_var.set(f"Building dataset: 0 of {len(selected_rows)} image(s)...")
        self.logger.log(f"[Dataset] Starting background copy of {len(selected_rows)} image(s).")
        self._build_thread = threading.Thread(
            target=self._build_dataset_worker,
            args=(destination, selected_rows),
            daemon=True,
        )
        self._build_thread.start()
        self.after(50, self._poll_build_results)


    def _build_dataset_worker(self, destination: Path, selected_rows):
        copied = 0
        failed = 0
        total = len(selected_rows)
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for index, (item, class_name) in enumerate(selected_rows, start=1):
                try:
                    class_dir = destination / self._clean_component(class_name, "Unknown")
                    class_dir.mkdir(parents=True, exist_ok=True)
                    output_name = self._traceable_filename(item)
                    output_path = self._unique_destination(class_dir / output_name)
                    shutil.copy2(item.source_path, output_path)
                    copied += 1
                except Exception as exc:
                    failed += 1
                    self._build_results.put(
                        ("log", f"[Error] Could not copy {item.filename}: {exc}")
                    )
                if index == total or index % 25 == 0:
                    self._build_results.put(
                        ("progress", (index, total, copied, failed))
                    )
            self._build_results.put(("complete", (copied, failed)))
        except Exception as exc:
            self._build_results.put(("error", str(exc)))

    def _poll_build_results(self):
        completed = False
        while True:
            try:
                result_type, payload = self._build_results.get_nowait()
            except queue.Empty:
                break
            if result_type == "log":
                self.logger.log(payload)
            elif result_type == "progress":
                index, total, copied, failed = payload
                self.prediction_progress_var.set(index / max(total, 1) * 100.0)
                self.prediction_progress_text.set(f"Building {index}/{total}")
                self.summary_var.set(
                    f"Building dataset: {index}/{total}; copied {copied}, failed {failed}"
                )
            elif result_type == "error":
                self._finish_dataset_build(error=str(payload))
                completed = True
                break
            elif result_type == "complete":
                copied, failed = payload
                self._finish_dataset_build(copied=copied, failed=failed)
                completed = True
                break
        if not completed:
            self.after(75, self._poll_build_results)

    def _finish_dataset_build(self, copied=0, failed=0, error: str | None = None):
        self._build_in_progress = False
        self._build_thread = None
        if error:
            self.prediction_progress_text.set("Build failed")
            self.summary_var.set("Dataset build failed.")
            self.logger.log(f"[Error] Dataset build failed: {error}")
            messagebox.showerror("Build Dataset", error)
            return
        self.prediction_progress_var.set(100.0)
        self.prediction_progress_text.set("Build complete")
        self.summary_var.set(f"Dataset build complete: copied {copied}, failed {failed}")
        self.logger.log(f"[Dataset] Copied {copied} image(s); {failed} failed.")
        self._save_settings()
        messagebox.showinfo("Dataset", f"Copied: {copied}\nFailed: {failed}")

    def _traceable_filename(self, item: ReviewItem) -> str:
        if item.output_filename:
            return item.output_filename
        original = self._clean_component(item.filename, "image")
        prefix = "_".join(
            [
                self._clean_component(item.program, "Unknown"),
                self._clean_component(item.lot, "Unknown"),
                self._clean_component(item.side, "Unknown"),
            ]
        )
        return f"{prefix}_{original}"

    @staticmethod
    def _unique_destination(path: Path) -> Path:
        if not path.exists():
            return path
        counter = 2
        while True:
            candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    @staticmethod
    def _clean_component(value, fallback):
        text = str(value).strip() if value is not None else ""
        text = INVALID_FILENAME_CHARS.sub("-", text).strip(" ._")
        return text or fallback

    @staticmethod
    def _join_reasons(reasons, extra):
        return "; ".join([*reasons, extra])

    def _thread_log(self, message):
        # Worker threads communicate through a queue; only the UI thread touches Tk.
        self._scan_results.put(("log", str(message)))

    def _clear_log(self):
        """Clear console output without removing review rows or predictions."""
        self.logger.clear()

    def _reset_review_table(self, log_action=True):
        """Internal reset used when a new source scan replaces the current table."""
        for child in self.review_body.winfo_children():
            child.destroy()
        self.items.clear()
        self.row_widgets.clear()
        self.thumbnail_refs.clear()
        self._thumbnail_cache.clear()
        self._filtered_items.clear()
        self._page_index = 0
        self._render_generation += 1
        self.page_var.set("Page 0 of 0")
        self.prev_page_button.configure(state=tk.DISABLED)
        self.next_page_button.configure(state=tk.DISABLED)
        self._agreement_sample_ids.clear()
        self._prediction_available = False
        self.prediction_progress_var.set(0.0)
        self.prediction_progress_text.set("Idle")
        self._set_filter_options(predictions_available=False)
        self.summary_var.set("No images scanned.")
        if log_action:
            self.logger.log("[Review] Cleared dataset review table.")

    def _toggle_log_panel(self):
        """Show or hide the console pane when the shared Console Log button is used."""
        panes = tuple(str(pane) for pane in self.content_split.panes())
        log_path = str(self.log_panel)

        if log_path in panes:
            self.content_split.forget(self.log_panel)
            self.log_visible = False
            self.logger.log("[UI] Console Log hidden.")
        else:
            self.content_split.add(self.log_panel, weight=CONSOLE_PANE_WEIGHT)
            self.log_visible = True
            self.update_idletasks()
            try:
                total_height = max(self.content_split.winfo_height(), 1)
                self.content_split.sashpos(0, max(total_height - 180, total_height // 2))
            except tk.TclError:
                pass
            self.logger.log("[UI] Console Log displayed.")

    def _load_saved_settings(self):
        settings = load_settings().get("dataset_tab", {})
        self.json_source_var.set(settings.get("json_log_source", ""))
        self.ng_source_var.set(settings.get("ng_folder_source", ""))
        self.ok_source_var.set(settings.get("ok_folder_source", ""))
        self.destination_var.set(settings.get("destination_root", ""))
        self.promoted_package_var.set(
            settings.get("promoted_package", settings.get("promoted_package_path", ""))
        )
        saved_package = self.promoted_package_var.get().strip()
        if saved_package:
            self._load_promoted_package(saved_package, show_dialog=False)
        self._saved_filter_mode = settings.get("review_filter_mode", "All")
        self.filter_var.set(self._saved_filter_mode)
        self._set_filter_options(predictions_available=False)

    def _save_settings(self):
        config = load_settings()
        config["dataset_tab"] = {
            "json_log_source": self.json_source_var.get().strip(),
            "ng_folder_source": self.ng_source_var.get().strip(),
            "ok_folder_source": self.ok_source_var.get().strip(),
            "destination_root": self.destination_var.get().strip(),
            "promoted_package": self.promoted_package_var.get().strip(),
            "review_filter_mode": self._saved_filter_mode,
        }
        synchronize_shared_settings(config, "dataset_tab")
        save_settings(config)
