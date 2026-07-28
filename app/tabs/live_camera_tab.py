import csv
import json
import threading
import time
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk

from pathlib import Path
DESKTOP_DIR = str(Path.home() / "Desktop")

from app import components
from app.live_model_package import resolve_live_package
from app.live_runtime import BoundedDropQueue, DuplicateFileTracker, RecentResultBuffer, can_start_worker, format_inference_error, release_camera_handle
from app.config_manager import load_settings, save_settings, synchronize_shared_settings
from app.ui_platform import get_sidebar_width
    

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


class LiveCameraTab(ttk.Frame):
    def __init__(self, parent, global_scroll_targets, status_bar, shared_console):
        super().__init__(parent, padding=10)
        self.scroll_targets = global_scroll_targets
        self.status_bar = status_bar
        self.shared_console = shared_console
        self.logger = None
        self.prediction_cards = None
        self.preview = None
        self.history_table = None
        self.path_entries = {}
        self.runtime_form = None
        self._processed_paths = set()

        self._worker_thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._session_results = []
        self._processed_total = 0
        self._recent_results = RecentResultBuffer(maxlen=200)
        self._pending_paths = BoundedDropQueue(maxsize=32)
        self._duplicate_tracker = DuplicateFileTracker()
        self._camera_handle = None
        self._live_package = None
        self.package_status_vars = {
            "Status": tk.StringVar(value="Not selected"),
            "Model": tk.StringVar(value="--"),
            "Model Version": tk.StringVar(value="--"),
            "Classes": tk.StringVar(value="--"),
            "Thresholds": tk.StringVar(value="--"),
            "Multipliers": tk.StringVar(value="--"),
        }
        self.live_progress = tk.DoubleVar(value=0.0)
        self.live_progress_text = tk.StringVar(value="Idle")

        self.log_visible = False
        self.log_overlay_frame = None
        self.toggle_log_btn = None

        self._build_tab_layout()
        self._load_saved_settings()

    def _build_tab_layout(self):
        sidebar_width = get_sidebar_width(self)
        components.ModeHeader(
            self,
            buttons_config=[
                ("Start", self._start_prediction),
                ("Pause", self._pause_prediction),
                ("Stop", self._stop_prediction),
                ("Clear", self._clear_session),
            ],
            title="Live",
            center_builder=self._build_progress_header,
            right_width=sidebar_width,
        )

        self.workspace_container = ttk.Frame(self)
        self.workspace_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        config_panel = components.ScrollPanel(
            self.workspace_container,
            width=sidebar_width,
            global_scroll_targets=self.scroll_targets,
        )
        config_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.display_panel = ttk.Panedwindow(self.workspace_container, orient=tk.VERTICAL)
        self.display_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        upper_pane = ttk.Frame(self.display_panel)

        image_panel = ttk.LabelFrame(upper_pane, text="Input Preview", padding=8)
        image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.preview = components.ImagePreview(
            image_panel,
            empty_text="No image loaded\nSelect promoted package, then press Start.",
            max_size=(900, 560),
        )
        self.preview.pack(fill=tk.BOTH, expand=True)

        telemetry_panel = ttk.Frame(upper_pane, width=240)
        telemetry_panel.pack(side=tk.RIGHT, fill=tk.Y)
        telemetry_panel.pack_propagate(False)

        self.prediction_panel = ttk.LabelFrame(telemetry_panel, text="Current Prediction", padding=12)
        self.prediction_panel.pack(fill=tk.BOTH, expand=True, pady=4)

        self.prediction_cards = components.SummaryCards(
            self.prediction_panel,
            [
                ("Class", "--"),
                ("Confidence", "--"),
            ],
            columns=1,
        )

        if "Class" in self.prediction_cards.value_labels:
            self.prediction_cards.value_labels["Class"].config(
                font=("TkDefaultFont", 22, "bold"),
                width=14,
                anchor="w",
            )
        if "Confidence" in self.prediction_cards.value_labels:
            self.prediction_cards.value_labels["Confidence"].config(
                font=("TkDefaultFont", 10, "normal"),
                width=14,
                anchor="w",
            )

        self.display_panel.add(upper_pane, weight=3)

        from app.styles import COLORS

        self.log_overlay_frame = tk.Frame(
            self.display_panel,
            bg=COLORS["panel"],
            highlightbackground="#555555",
            highlightthickness=1,
        )

        lower_notebook = ttk.Notebook(self.log_overlay_frame)
        lower_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        history_frame = ttk.Frame(lower_notebook, padding=4)
        lower_notebook.add(history_frame, text=" Recent Predictions Table ")
        self.history_table = components.SimpleTable(
            history_frame,
            columns=("Time", "File", "Prediction", "Confidence", "Saved"),
            rows=[],
        )

        log_frame = ttk.Frame(lower_notebook, padding=4)
        lower_notebook.add(log_frame, text=" Console Log ")

        self.logger = components.LogWidget(log_frame, height=8)
        self.logger.pack(fill=tk.BOTH, expand=True)
        self.shared_console.attach(self.logger)
        self.logger.log("Prediction workspace initialized.")

        inputs_group = components.EntryButtonGroup(
            config_panel.inner,
            "Inputs",
            [
                ("Promoted Package", "..."),
                ("Input Folder", "..."),
                ("Output Folder", "..."),
            ],
            self._on_browse_path,
            required_labels={"Promoted Package", "Input Folder", "Output Folder"},
        )
        self.path_entries = inputs_group.entries
        self._add_input_hints(inputs_group.body)
        self._build_package_status_panel(config_panel.inner, sidebar_width)

        self.runtime_form = components.DataDrivenForm(
            config_panel.inner,
            "Runtime Settings",
            [
                [("Image Processing Delay (ms)", "500")],
                [("Device", ["CPU", "GPU"])],
                [("Result Format", ["JSON", "CSV", "JSON + CSV"])],
            ],
        )
        # Keep direct references because grid_info() can be empty after grid_remove().
        self._result_format_widgets = tuple(self.runtime_form.body.grid_slaves(row=2))

        self._live_multipliers = {}

    def _add_input_hints(self, parent):
        hints = {
            "Promoted Package": (
                r"Example: D:\Models\JEI_V1\production_package"
                "\nSelect the folder containing:"
                "\n• model.pt"
                "\n• labels.json"
                "\n• thresholds.json"
                "\n• model_metadata.json"
                "\n• defect_multipliers.json (optional)"
            ),
            "Input Folder": (
                "Select the folder watched for new images."
                "\nImages added to this folder are queued for live prediction."
            ),
            "Output Folder": (
                "Select where live prediction results are saved."
                "\nThe app writes the selected result format for each processed image."
            ),
        }
        for row, (label, message) in enumerate(hints.items()):
            _info_button(parent, f"{label} Hint", message).grid(
                row=row,
                column=3,
                sticky="e",
                padx=(8, 0),
                pady=4,
            )

    def _build_package_status_panel(self, parent, sidebar_width):
        panel = ttk.LabelFrame(parent, text="Package Status", padding=8)
        panel.pack(fill=tk.X, pady=(0, 8))
        panel.columnconfigure(1, weight=1)
        wrap = max(220, sidebar_width - 150)
        for row, (label, variable) in enumerate(self.package_status_vars.items()):
            ttk.Label(panel, text=f"{label}:").grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=2)
            ttk.Label(
                panel,
                textvariable=variable,
                wraplength=wrap,
                justify="left",
            ).grid(row=row, column=1, sticky="ew", pady=2)

    def _build_progress_header(self, parent):
        ttk.Label(
            parent,
            textvariable=self.live_progress_text,
            style="SectionNote.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.live_progressbar = ttk.Progressbar(
            parent,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.live_progress,
            length=components.HEADER_PROGRESS_LENGTH,
        )
        self.live_progressbar.grid(row=0, column=1, sticky="w")

    def _rebuild_live_multipliers(self, class_names):
        defect_only_classes = [c for c in class_names if str(c).lower() != "pass"]
        if list(self._live_multipliers.keys()) == defect_only_classes:
            return
        self._live_multipliers = {defect_name: 0.0 for defect_name in defect_only_classes}

    def _set_package_status(self, package=None, error: str | None = None):
        if error:
            values = {
                "Status": f"Invalid - {components.display_text(error, 'Unknown package validation error')}",
                "Model": "--",
                "Model Version": "--",
                "Classes": "--",
                "Thresholds": "--",
                "Multipliers": "--",
            }
        elif package is None:
            values = {
                "Status": "Not selected",
                "Model": "--",
                "Model Version": "--",
                "Classes": "--",
                "Thresholds": "--",
                "Multipliers": "--",
            }
        else:
            threshold_count = len(package.active_thresholds or {})
            multiplier_text = Path(package.multiplier_path).name if package.multiplier_path else "Not provided"
            values = {
                "Status": "Valid",
                "Model": package.model_name,
                "Model Version": package.model_version,
                "Classes": ", ".join(package.class_names),
                "Thresholds": f"{threshold_count} active",
                "Multipliers": multiplier_text,
            }
        for key, value in values.items():
            self.package_status_vars[key].set(value)


    def _load_saved_settings(self):
        settings = load_settings().get("live_tab", {})
        self._set_entry_value("Promoted Package", settings.get("promoted_package", settings.get("promoted_package_path", "")))
        self._set_entry_value("Input Folder", settings.get("input_source", ""))
        self._set_entry_value("Output Folder", settings.get("save_result_folder", ""))

        for field, default, key in (
            ("Device", "CPU", "device"),
            ("Result Format", "JSON", "result_format"),
        ):
            widget = self.runtime_form.entries.get(field)
            value = settings.get(key, default)
            if widget is not None and value in tuple(widget.cget("values")):
                widget.set(value)

        delay_widget = self.runtime_form.entries.get("Image Processing Delay (ms)")
        if delay_widget is not None:
            delay_widget.delete(0, tk.END)
            delay_widget.insert(0, settings.get("delay_between_images_ms", settings.get("polling_interval_ms", "500")))
        self._update_mode_specific_settings()
        device = self.runtime_form.entries.get("Device")
        if device is not None:
            device.bind("<<ComboboxSelected>>", self._on_device_changed, add="+")
        if device is not None:
            self.status_bar.set_device(device.get())


    def _on_device_changed(self, _event=None):
        device = self.runtime_form.entries.get("Device")
        if device is not None:
            self.status_bar.set_device(device.get())
        self._save_settings()

    def _update_mode_specific_settings(self):
        """Show Result Format only for JSON Input mode."""
        mode = "Watcher Mode"
        row_widgets = getattr(self, "_result_format_widgets", ())
        if mode == "JSON Input":
            for widget in row_widgets:
                widget.grid()
        else:
            for widget in row_widgets:
                widget.grid_remove()

    def _save_settings(self):
        config = load_settings()
        config["live_tab"] = {
            "promoted_package": self._get_entry_value("Promoted Package"),
            "input_source": self._get_entry_value("Input Folder"),
            "save_result_folder": self._get_entry_value("Output Folder"),
            "input_mode": "Watcher Mode",
            "delay_between_images_ms": self.runtime_form.entries["Image Processing Delay (ms)"].get().strip() or "500",
            "device": self.runtime_form.entries["Device"].get() or "CPU",
            "result_format": self.runtime_form.entries["Result Format"].get() or "JSON",
        }
        synchronize_shared_settings(config, "live_tab")
        save_settings(config)

    def _set_entry_value(self, label, value):
        entry = self.path_entries.get(label)
        if entry is None:
            return
        entry.delete(0, tk.END)
        entry.insert(0, value)

    def _get_entry_value(self, label):
        entry = self.path_entries.get(label)
        return entry.get().strip() if entry is not None else ""

    def _on_browse_path(self, target_field):
        if target_field == "Promoted Package":
            selected_path = filedialog.askdirectory(
                title="Select Promoted Model Package",
                initialdir=DESKTOP_DIR,
            )
        else:
            selected_path = filedialog.askdirectory(
                title=f"Select {target_field}",
                initialdir=DESKTOP_DIR,
            )

        if not selected_path:
            return

        self._set_entry_value(target_field, selected_path)
        if target_field == "Promoted Package":
            if not self._load_promoted_package(selected_path):
                return
        self._save_settings()
        self.logger.log(f"[Path] {target_field} set to: {selected_path}")

    def _load_promoted_package(self, package_path):
        try:
            package = resolve_live_package(package_path)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self._live_package = None
            self._set_package_status(error=message)
            self.logger.log(f"[Package Error] {message}")
            return False
        self._live_package = package
        self._rebuild_live_multipliers(package.class_names)
        self._set_package_status(package=package)
        self.logger.log("[Package] Loaded promoted package.")
        for line in package.summary.splitlines():
            self.logger.log(f"[Package] {line}")
        return True

    def _resolve_live_model_settings(self):
        package_path = self._get_entry_value("Promoted Package")
        if not package_path:
            raise ValueError("Promoted Package required.")
        package = resolve_live_package(package_path)
        return {
            "package": package,
            "checkpoint_path": package.checkpoint_path,
            "class_names": package.class_names,
            "active_thresholds": package.active_thresholds,
            "multiplier_path": package.multiplier_path,
            "image_size": package.image_size,
            "model_version": package.model_version,
        }

    def _start_prediction(self):
        if not can_start_worker(self._worker_thread):
            if self._pause_event.is_set():
                self._pause_event.clear()
                self.logger.log("[Execution] Prediction resumed.")
            else:
                self.logger.log("[Execution] Prediction already running.")
            return
            
        try:
            live_settings = self._resolve_live_model_settings()
        except Exception as exc:
            self.logger.log(f"[Package Error] {str(exc).strip() or exc.__class__.__name__}")
            return
        multiplier_path = live_settings["multiplier_path"]
        input_source = self._get_entry_value("Input Folder")
        if multiplier_path and not Path(multiplier_path).is_file():
            self.logger.log(f"[Error] Promoted package multiplier file not found: {multiplier_path}")
            return
        if not input_source:
            self.logger.log("[Error] Input Folder required.")
            return
        if not self._get_entry_value("Output Folder"):
            self.logger.log("[Error] Output Folder required.")
            return

        self._save_settings()
        
        self._live_package = live_settings["package"]
        self._rebuild_live_multipliers(live_settings["class_names"])
        self._set_package_status(package=live_settings["package"])
        
        self._stop_event.clear()
        self._pause_event.clear()
        self._set_live_progress(0, "Starting")
        self._session_results = []
        self._processed_total = 0
        self._recent_results.clear()
        self._pending_paths.clear()
        self._duplicate_tracker.clear()
        self.logger.log("[Execution] Starting live prediction session...")
        self._worker_thread = threading.Thread(
            target=self._run_prediction_session,
            daemon=True,
        )
        self._worker_thread.start()

    def _pause_prediction(self):
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.logger.log("[Execution] Nothing running to pause.")
            return
        self._pause_event.set()
        self.logger.log("[Execution] Pause requested.")

    def _stop_prediction(self):
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.logger.log("[Execution] Nothing running to stop.")
            return
        self._stop_event.set()
        self._pause_event.clear()
        self._set_live_progress(0, "Stopping")
        self._pending_paths.clear()
        self._release_camera_resources()
        self.logger.log("[Execution] Stop requested.")

    def _clear_session(self):
        self._stop_event.set()
        self._pause_event.clear()
        self.shared_console.clear()
        self.logger.log("[Execution] Session log cleared.")
        self.history_table.clear()
        self.preview.clear()
        self._update_prediction_ui("--", "--")
        self._set_live_progress(0, "Idle")
        self._session_results = []
        self._processed_total = 0
        self._recent_results.clear()
        self._pending_paths.clear()
        self._duplicate_tracker.clear()

    def _run_prediction_session(self):
        try:
            live_settings = self._resolve_live_model_settings()
        except Exception as exc:
            self.after(0, lambda msg=str(exc): self.logger.log(f"[Package Error] {msg}"))
            return
        checkpoint_path = live_settings["checkpoint_path"]
        input_source = Path(self._get_entry_value("Input Folder"))
        output_dir = self._get_entry_value("Output Folder")
        input_mode = "Watcher Mode"
        result_format = self.runtime_form.entries["Result Format"].get().strip() or "JSON"
        model_version = live_settings["model_version"]
        device_name = self.runtime_form.entries["Device"].get().strip() or "CPU"
        image_delay_ms = self.runtime_form.entries["Image Processing Delay (ms)"].get().strip() or "500"

        multiplier_text = live_settings["multiplier_path"]
        if multiplier_text:
            try:
                current_multipliers = self._read_multiplier_file(Path(multiplier_text), class_names=live_settings["class_names"])
            except Exception as exc:
                self.after(0, lambda msg=str(exc): self.logger.log(f"[Error] Could not read multiplier file: {msg}"))
                return
        else:
            current_multipliers = {name: 0.0 for name in self._live_multipliers}

        try:
            image_delay = max(0, int(image_delay_ms)) / 1000.0
        except ValueError:
            image_delay = 0.5

        # Folder rescans stay responsive. This is separate from the user-controlled
        # delay that is applied after each successful prediction.
        folder_rescan_delay = 0.5

        try:
            from app.backend_bridge import ensure_backend_paths
            ensure_backend_paths()
            from backend.inference_service import iter_image_paths, load_inference_model, predict_image

            session = load_inference_model(
                checkpoint_path=checkpoint_path,
                class_names=live_settings["class_names"],
                device_override=device_name,
                defect_multipliers=current_multipliers,
                active_thresholds=live_settings["active_thresholds"],
                image_size=live_settings["image_size"],
            )
            self.after(0, lambda: self.logger.log(f"[Model] Loaded checkpoint on {session.device}."))

            if input_mode == "JSON Input":
                image_paths = self._load_json_image_paths(input_source)
                self.after(0, lambda: self.logger.log(f"[Input] JSON contains {len(image_paths)} image(s)."))
                for image_path in image_paths:
                    processed = self._process_one_image(
                        session,
                        image_path,
                        output_dir,
                        result_format,
                        current_multipliers,
                        source_root=None,
                        model_version=model_version,
                    )
                    if not processed and self._stop_event.is_set():
                        break
                    if processed and image_delay > 0 and self._stop_event.wait(image_delay):
                        break
                self.after(0, lambda: self.logger.log(f"[Execution] JSON prediction complete. Processed {self._processed_total} image(s)."))
                return

            if not input_source.is_dir():
                raise NotADirectoryError(f"Watcher folder not found: {input_source}")

            self._processed_paths = set()
            self.after(0, lambda: self.logger.log(f"[Watcher] Watching folder: {input_source}"))
            while not self._stop_event.is_set():
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.1)

                found_new = False
                for image_path in iter_image_paths(input_source):
                    if self._stop_event.is_set():
                        break
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)
                    try:
                        should_process = self._duplicate_tracker.should_process(image_path)
                    except OSError as exc:
                        self.after(0, lambda p=str(image_path), msg=str(exc): self.logger.log(f"[Warning] File skipped: {p} ({msg})"))
                        continue
                    if not should_process:
                        continue
                    found_new = True
                    processed = self._process_one_image(
                        session,
                        image_path,
                        output_dir,
                        result_format,
                        current_multipliers,
                        source_root=input_source,
                        model_version=model_version,
                    )
                    if not processed and not self._stop_event.is_set():
                        self._duplicate_tracker.forget(image_path)
                    if processed and image_delay > 0 and self._stop_event.wait(image_delay):
                        break
                if not found_new and not self._stop_event.is_set():
                    self._stop_event.wait(folder_rescan_delay)

            self.after(0, lambda: self.logger.log(f"[Watcher] Stopped. Processed {self._processed_total} image(s)."))
        except Exception as exc:
            self.after(0, lambda msg=str(exc): self.logger.log(f"[Error] {msg}"))
        finally:
            self._pending_paths.clear()
            self._release_camera_resources()
            self.after(0, self._on_live_worker_finished)

    def _on_live_worker_finished(self):
        self._worker_thread = None
        self._set_live_progress(0, "Idle")
        self.logger.log("[Execution] Live worker stopped.")

    def _release_camera_resources(self):
        release_camera_handle(self._camera_handle)
        self._camera_handle = None

    def _load_json_image_paths(self, json_path):
        if not json_path.is_file():
            raise FileNotFoundError(f"JSON input not found: {json_path}")
        request = json.loads(json_path.read_text(encoding="utf-8"))
        records = request if isinstance(request, list) else [request]
        image_paths = []
        for record in records:
            if not isinstance(record, dict):
                continue
            images = record.get("Images", record.get("images", []))
            for image in images:
                if isinstance(image, str):
                    value = image
                elif isinstance(image, dict):
                    value = image.get("preparedImagePath") or image.get("image_path") or image.get("path")
                else:
                    value = None
                if value:
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        candidate = json_path.parent / candidate
                    image_paths.append(candidate.resolve())
        if not image_paths:
            raise ValueError("JSON input does not contain any preparedImagePath values.")
        return image_paths

    def _process_one_image(self, session, image_path, output_dir, result_format, multipliers, source_root=None, model_version=""):
        if self._stop_event.is_set():
            return False
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.1)

        started_at = time.perf_counter()
        timestamp = datetime.now()
        try:
            from backend.inference_service import predict_image
            result = predict_image(session, image_path, defect_multipliers=multipliers)
        except Exception as exc:
            self.after(0, lambda msg=format_inference_error(image_path, exc): self.logger.log(msg))
            return False

        processing_time_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        self._processed_total = int(getattr(self, "_processed_total", 0)) + 1
        self._session_results.append(result)
        if len(self._session_results) > 200:
            del self._session_results[:-200]
        self._recent_results.append(result)
        saved_path = None

        # Watcher Mode always copies the original image into the predicted class folder.
        # JSON Input only writes the selected JSON/CSV result export.
        if output_dir and source_root is not None:
            saved_path = self._save_prediction_image(result, output_dir, source_root=source_root)

        if output_dir and source_root is not None:
            self._append_watcher_csv(
                output_dir=output_dir,
                source_root=source_root,
                result=result,
                saved_path=saved_path,
                timestamp=timestamp,
                processing_time_ms=processing_time_ms,
                model_version=model_version,
            )
        elif output_dir:
            self._write_result_exports(output_dir, result_format)

        current = self._processed_total
        self._set_live_progress(100, f"Processed {current} image(s)")
        self.after(0, lambda prediction=result, index=current, saved=saved_path: self._handle_prediction_result(prediction, index, index, bool(saved)))
        return True

    def _set_live_progress(self, percent, text):
        if hasattr(self, "live_progress"):
            self.live_progress.set(float(percent))
        if hasattr(self, "live_progress_text"):
            self.live_progress_text.set(components.display_text(text, "Idle"))

    def _save_prediction_image(self, result, output_dir, source_root=None):
        source_path = Path(result.image_path)
        if not source_path.is_file():
            return None

        # Watcher output mirrors every folder below the selected input root,
        # then adds one prediction folder before the image filename.
        # Example:
        #   input/Line_A/Lot_01/image.png
        #   output/Line_A/Lot_01/Open_Circuit/image.png
        relative_parent = Path()
        if source_root is not None:
            try:
                relative_parent = source_path.resolve().relative_to(Path(source_root).resolve()).parent
            except (ValueError, OSError):
                relative_parent = Path()

        prediction_folder = self._safe_folder_name(result.predicted_class)
        destination_dir = Path(output_dir) / relative_parent / prediction_folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_path.name
        counter = 1
        while destination_path.exists():
            destination_path = destination_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
            counter += 1

        shutil.copy2(source_path, destination_path)
        return destination_path

    def _append_watcher_csv(self, output_dir, source_root, result, saved_path, timestamp, processing_time_ms, model_version):
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        csv_path = output_root / "inspection_results.csv"

        source_path = Path(result.image_path)
        try:
            relative_path = source_path.resolve().relative_to(Path(source_root).resolve())
        except (ValueError, OSError):
            relative_path = Path(source_path.name)

        output_relative_path = ""
        if saved_path:
            try:
                output_relative_path = Path(saved_path).resolve().relative_to(output_root.resolve()).as_posix()
            except (ValueError, OSError):
                output_relative_path = str(saved_path)

        row = {
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "RelativePath": relative_path.as_posix(),
            "OutputPath": output_relative_path,
            "Prediction": result.predicted_class,
            "Confidence": round(result.confidence * 100.0, 2),
            "Status": "Success",
            "ProcessingTimeMs": processing_time_ms,
            "ModelVersion": model_version,
        }
        fieldnames = list(row.keys())
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _safe_folder_name(name):
        value = str(name or "Unknown").strip() or "Unknown"
        for character in '<>:"/\\|?*':
            value = value.replace(character, "_")
        return value.rstrip(" .") or "Unknown"

    def _result_rows(self):
        return [
            {
                "preparedImagePath": Path(result.image_path).name,
                "DefectiveScore": round(result.confidence * 100, 1),
                "DefectiveName": result.predicted_class,
            }
            for result in self._session_results
        ]

    def _write_result_exports(self, output_dir, result_format):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        rows = self._result_rows()
        normalized = result_format.upper()
        if "JSON" in normalized:
            json_file = output_path / "live_results.json"
            json_file.write_text(json.dumps({"Results": rows}, indent=4), encoding="utf-8")
        if "CSV" in normalized:
            csv_file = output_path / "live_results.csv"
            with csv_file.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=["preparedImagePath", "DefectiveScore", "DefectiveName"])
                writer.writeheader()
                writer.writerows(rows)

    def _handle_prediction_result(self, result, current_index, total_count, saved=False):
        try:
            self.preview.show_image(result.image_path)
        except Exception as exc:
            self.logger.log(f"[Warning] Preview failed for {result.image_path}: {exc}")

        self._update_prediction_ui(result.predicted_class, f"{result.confidence:.2%}")
        self.history_table.append_row((
            datetime.now().strftime("%H:%M:%S"),
            Path(result.image_path).name,
            result.predicted_class,
            f"{result.confidence:.2%}",
            "Yes" if saved else "No",
        ))
        self._trim_history_table(max_rows=200)
        self.logger.log(f"[Prediction] {current_index} - {Path(result.image_path).name}: {result.predicted_class} ({result.confidence:.2%})")

    def _trim_history_table(self, max_rows=200):
        if not self.history_table:
            return
        children = list(self.history_table.tree.get_children())
        overflow = len(children) - int(max_rows)
        for item_id in children[:max(0, overflow)]:
            self.history_table.tree.delete(item_id)

    def _update_prediction_ui(self, class_val, confidence_val):
        self.prediction_cards.set_value("Class", class_val)
        self.prediction_cards.set_value("Confidence", confidence_val)

        style = ttk.Style()

        if class_val.lower() == "pass":
            bg_color = "#2e7d32"
            fg_color = "#ffffff"
            style.configure("PassCard.TFrame", background=bg_color, bordercolor="#cccccc", relief="solid", borderwidth=1)
            for child in self.prediction_cards.winfo_children():
                child.configure(style="PassCard.TFrame")
                for label in child.winfo_children():
                    if isinstance(label, tk.Label):
                        label.configure(bg=bg_color, fg=fg_color)

        elif class_val == "--":
            try:
                from ..styles import COLORS
            except ImportError:
                from styles import COLORS

            for child in self.prediction_cards.winfo_children():
                child.configure(style="Card.TFrame")
                for label in child.winfo_children():
                    if isinstance(label, tk.Label):
                        label.configure(bg=COLORS["panel"])
                        if label == self.prediction_cards.value_labels.get("Class"):
                            label.configure(fg=COLORS["text"])
                        else:
                            label.configure(fg=COLORS["muted"])
        else:
            bg_color = "#c62828"
            fg_color = "#ffffff"
            style.configure("DefectCard.TFrame", background=bg_color, bordercolor="#cccccc", relief="solid", borderwidth=1)
            for child in self.prediction_cards.winfo_children():
                child.configure(style="DefectCard.TFrame")
                for label in child.winfo_children():
                    if isinstance(label, tk.Label):
                        label.configure(bg=bg_color, fg=fg_color)

    def _toggle_log_panel(self):
        if self.log_visible:
            self.display_panel.forget(self.log_overlay_frame)
            self.log_visible = False
            self.logger.log("[UI] Console Log hidden.")
        else:
            self.display_panel.add(self.log_overlay_frame, weight=components.CONSOLE_PANE_WEIGHT)
            self.log_visible = True
            self.logger.log("[UI] Console Log displayed.")

    def _read_multiplier_file(self, json_path, class_names=None):
        """Read and validate per-defect multipliers from the selected JSON file."""
        json_path = Path(json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"Multiplier file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Multiplier JSON root must be an object.")

        class_names = class_names or list(self._live_multipliers.keys())
        allowed = {str(name).strip().casefold() for name in class_names}
        unexpected = [key for key in data if str(key).strip().casefold() not in allowed]
        if unexpected:
            raise ValueError(f"Multiplier class not present in labels: {unexpected[0]}")
        lookup = {str(key).strip().casefold(): value for key, value in data.items()}
        multipliers = {}
        for class_name in self._live_multipliers:
            key = class_name.strip().casefold()
            try:
                value = float(lookup.get(key, 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid multiplier for '{class_name}'.") from exc
            multipliers[class_name] = max(-0.999, min(0.999, value))
        return multipliers
