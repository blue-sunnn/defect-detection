import copy
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pathlib import Path
DESKTOP_DIR = str(Path.home() / "Desktop")

from app.backend_bridge import build_runtime_paths, generate_threshold_recommendations, generate_multiplier_recommendations, run_training_job, run_validation_job, run_test_job
from app.checkpoint_registry import (
    gate_result,
    load_registry,
    promoted_checkpoints,
    register_checkpoint,
    register_promoted_package,
    rollback_checkpoint,
    update_checkpoint_metadata,
)
from app.components import (
    CONSOLE_PANE_WEIGHT,
    HEADER_PROGRESS_LENGTH,
    PATH_ENTRY_WIDTH,
    CollapsibleGroup,
    ModeHeader,
    PlainFormGroup,
    ScrollPanel,
    SimpleTable,
    SliderRow,
    display_text,
    enable_debounced_matplotlib_resize,
)
from app.config_manager import load_settings, save_settings, synchronize_shared_settings
from app.constants import (
    COLORS,
    CONFIG_PHASE_1,
    CONFIG_PHASE_1_STG1,
    CONFIG_PHASE_1_STG2,
    CONFIG_PHASE_2_STG1,
    CONFIG_PHASE_2_STG2,
)
from app.training_state_recovery import recover_training_state
from app.promotion_gate import (
    OVERRIDABLE_PROMOTION_RULES,
    evaluate_promotion_gate,
    failed_rule_messages,
)
from app.promotion_package import build_promotion_package
from app.ui_platform import get_figure_dpi, get_sidebar_width
from backend.threshold_persistence import restore_recommended, save_threshold_config, set_active_threshold
from backend.multiplier_persistence import restore_recommended_multiplier, save_multiplier_config, set_active_multiplier


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


SETTING_HINTS = {
    "Training Dataset": "Select the dataset root used for training. It can contain train/ and val/ folders, or class folders that the app splits using Training Split (%).",
    "Fixed Validation Dataset": "Select the locked validation baseline used for promotion checks. Changing it changes the comparison baseline for future models.",
    "Output Folder": "Select where training artifacts, checkpoints, reports, and promoted packages are written.",
    "Pretrained Checkpoint": (
        "Select an existing model package or checkpoint. It is optional when Phase 1 is enabled, "
        "but required when Phase 1 is disabled and Phase 2 transfer learning is enabled."
    ),
    "Phase 2 Dataset": "Select the transfer-learning dataset used when Phase 2 is enabled. Its class names must match the Phase 1 model.",
    "Grad-CAM Output Folder": "Select where Grad-CAM visual explanation images are exported when Grad-CAM export is enabled.",
    "Model Name": "Sets the model family/name stored in training artifacts and promotion metadata.",
    "Model Version": "Sets the version label stored with checkpoints and promoted packages.",
    "Image Width": "Resizes input images to this width before training and evaluation.",
    "Image Height": "Resizes input images to this height before training and evaluation.",
    "Device": "Chooses whether training runs on GPU or CPU. GPU is faster when supported by the local environment.",
    "Batch Size": "Sets how many images are processed per optimizer step. Larger batches use more memory.",
    "Training Split (%)": "Sets the train/validation split when the dataset root does not already contain train/ and val/ folders.",
    "Optimizer": "Chooses the update algorithm for this training stage.",
    "Learning Rate": "Controls the optimizer step size for this training stage.",
    "Epochs": "Sets how many passes this stage makes over the training data.",
    "Dropout Rate": "Controls how much random feature dropout is applied to reduce overfitting.",
    "Focal Loss Gamma": "Increases focus on difficult or misclassified examples. Higher values put more weight on hard samples.",
    "Label Smoothing": "Softens target labels during training to reduce overconfidence.",
    "Crop Min Scale": "Sets the smallest random crop scale used during augmentation.",
    "Crop Max Scale": "Sets the largest random crop scale used during augmentation.",
    "Horizontal Flip": "Sets the probability of randomly flipping training images left-to-right.",
    "Vertical Flip": "Sets the probability of randomly flipping training images top-to-bottom.",
    "Brightness": "Sets the amount of random brightness variation applied during augmentation.",
    "Contrast": "Sets the amount of random contrast variation applied during augmentation.",
    "Saturation": "Sets the amount of random color saturation variation applied during augmentation.",
    "Rotation Degrees": "Sets the maximum random rotation applied to training images. Use a value that matches how much the inspected part can realistically rotate in production.",
    "Export Grad-CAM result images": "Exports Grad-CAM images after validation so model attention can be reviewed visually.",
    "LR Factor": "Multiplies the learning rate by this value when the scheduler reduces it.",
    "LR Patience": "Sets how many validation checks can stall before the scheduler lowers the learning rate.",
    "Minimum LR": "Sets the lowest learning rate the scheduler is allowed to use.",
    "Early Stop Patience": "Stops training after this many validation checks without meaningful improvement.",
}


class TrainingTab(ttk.Frame):
    def __init__(self, parent, scroll_targets, status_bar, shared_console):
        super().__init__(parent, padding=10)
        self.scroll_targets = scroll_targets
        self.status_bar = status_bar
        self.shared_console = shared_console
        self.training_progress = tk.DoubleVar(value=0.0)
        self.training_progress_text = tk.StringVar(value="Idle")
        self.training_stop_event = None
        self.training_event_queue = queue.Queue()
        self.metric_history = []
        self.log_visible = False
        self.console_panel = None
        self.path_entries = {}
        self.general_entries = {}
        self.general_combos = {}
        self.phase1_form = None
        self.stage1_form = None
        self.stage2_form = None
        self.phase2_stage1_form = None
        self.phase2_stage2_form = None
        self.run_phase1_var = tk.BooleanVar(value=True)
        self.run_phase2_var = tk.BooleanVar(value=False)
        self.lr_scheduler_var = tk.BooleanVar(value=True)
        self.export_grad_cam_var = tk.BooleanVar(value=False)
        self.unfreeze_entry = None
        self.phase2_unfreeze_entry = None
        self.loss_form = None
        self.augmentation_form = None
        self.augmentation_entries = {}
        self.augmentation_sliders = {}
        self.training_control_form = None
        self._worker_thread = None
        self._latest_checkpoint_record = None
        self._post_training_busy = False
        self.gate_state_var = tk.StringVar(value="No checkpoint validated yet")
        self.gate_details_var = tk.StringVar(value="Training completion will run the fixed validation gate automatically.")
        self.evaluation_status_var = tk.StringVar(value="Evaluation not run")
        self.evaluation_details_var = tk.StringVar(value="Automatic evaluation will run after checkpoint registration.")
        self.threshold_status_var = tk.StringVar(value="Thresholds disabled until evaluation predictions exist.")
        self.threshold_unsaved_var = tk.StringVar(value="")
        self.threshold_rows = {}
        self.threshold_config = None
        self.threshold_saved_config = None
        self.threshold_config_path = None
        self._updating_threshold_rows = False
        self.multiplier_status_var = tk.StringVar(value="Multipliers disabled until evaluation predictions exist.")
        self.multiplier_unsaved_var = tk.StringVar(value="")
        self.multiplier_rows = {}
        self.multiplier_config = None
        self.multiplier_saved_config = None
        self.multiplier_config_path = None
        self.multiplier_deployed_path = None
        self._updating_multiplier_rows = False
        self.recovered_state = None
        self.rollback_tree = None
        self._build()
        self._load_saved_settings()
        self._recover_latest_training_state()

    def _build(self):
        sidebar_width = get_sidebar_width(self)
        header_btns = [
            ("Start", self._start_training),
            ("Stop", self._stop_training),
            ("Clear", self._clear_logs),
            ("Promote", self._export_model),
        ]
        ModeHeader(
            self,
            title="Training",
            buttons_config=header_btns,
            center_builder=self._build_progress_header,
            right_width=sidebar_width,
        )

        right = ScrollPanel(
            self,
            width=sidebar_width,
            global_scroll_targets=self.scroll_targets,
        )
        right.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 0))

        self.display_panel = ttk.Panedwindow(self, orient=tk.VERTICAL)
        self.display_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        top_pane = ttk.Frame(self.display_panel)
        self.display_panel.add(top_pane, weight=4)

        self._build_console_panel()
        self._build_metrics(top_pane)
        self._build_config_panel(right.inner)
        self.logger.log("[Ready] Training tab initialized.")

    def _build_progress_header(self, parent):
        ttk.Label(
            parent,
            textvariable=self.training_progress_text,
            style="SectionNote.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.pb = ttk.Progressbar(
            parent,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.training_progress,
            length=HEADER_PROGRESS_LENGTH,
        )
        self.pb.grid(row=0, column=1, sticky="w")

    def _build_metrics(self, parent):
        self.training_main_notebook = ttk.Notebook(parent)
        self.training_main_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        metrics_page = ttk.Frame(self.training_main_notebook, padding=4)
        readiness_page = ttk.Frame(self.training_main_notebook, padding=0)
        self.training_main_notebook.add(metrics_page, text=" Training Progress ")
        self.training_main_notebook.add(
            readiness_page,
            text=" Evaluation, Thresholds & Promotion ",
        )

        readiness_scroll = ScrollPanel(
            readiness_page,
            width=1,
            global_scroll_targets=self.scroll_targets,
        )
        readiness_scroll.pack(fill=tk.BOTH, expand=True)

        mf = ttk.LabelFrame(metrics_page, text="Live Metrics", padding=8)
        mf.pack(fill=tk.BOTH, expand=True)

        # Give both charts more room so titles and axis labels do not overlap.
        self.metrics_fig = Figure(
            figsize=(13.8, 4.8),
            dpi=get_figure_dpi(self, base_dpi=96),
        )
        self.metrics_fig.patch.set_facecolor(COLORS["panel"])

        self.ax_loss = self.metrics_fig.add_subplot(121)
        self.ax_acc = self.metrics_fig.add_subplot(122)

        for axis in (self.ax_loss, self.ax_acc):
            axis.set_facecolor(COLORS["panel"])
            axis.tick_params(labelsize=8)
            axis.set_xlabel("Epoch", fontsize=8)
            axis.grid(True, linestyle="--", linewidth=1, alpha=0.45)

        self.ax_loss.set_title("Training & Validation Loss", fontsize=13)
        self.ax_loss.set_ylabel("Loss", fontsize=8)

        self.ax_acc.set_title("Model Performance Metrics", fontsize=13)
        self.ax_acc.set_ylabel("Rate / Probability", fontsize=8)
        self.ax_acc.set_ylim(0, 1.05)

        self.metrics_fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.16,
            top=0.88,
            wspace=0.32,
        )

        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=mf)
        enable_debounced_matplotlib_resize(self.metrics_canvas)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._build_model_readiness(readiness_scroll.inner)

    def _build_model_readiness(self, parent):
        content = ttk.Frame(parent)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        intro = ttk.Label(
            content,
            text=(
                "Training completion runs fixed validation automatically. Review evaluation, "
                "save active thresholds/multipliers, then promote."
            ),
            style="SectionNote.TLabel",
            wraplength=980,
        )
        intro.pack(fill=tk.X, pady=(0, 10))

        evaluation_card = ttk.LabelFrame(content, text="Phase 3 - Evaluation", padding=12)
        evaluation_card.pack(fill=tk.X, pady=(0, 10))
        evaluation_card.columnconfigure(1, weight=1)
        ttk.Label(evaluation_card, text="Status:", style="SectionNote.TLabel").grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 10),
            pady=(0, 6),
        )
        ttk.Label(
            evaluation_card,
            textvariable=self.evaluation_status_var,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(evaluation_card, text="Details:", style="SectionNote.TLabel").grid(
            row=1,
            column=0,
            sticky="nw",
            padx=(0, 10),
        )
        ttk.Label(
            evaluation_card,
            textvariable=self.evaluation_details_var,
            wraplength=980,
            justify=tk.LEFT,
        ).grid(row=1, column=1, sticky="ew")

        threshold_card = ttk.LabelFrame(content, text="Phase 4A - Thresholds", padding=12)
        threshold_card.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            threshold_card,
            textvariable=self.threshold_status_var,
            wraplength=980,
            justify=tk.LEFT,
        ).pack(anchor="w")
        ttk.Label(
            threshold_card,
            textvariable=self.threshold_unsaved_var,
            foreground=COLORS["warning"],
        ).pack(anchor="w", pady=(2, 6))
        self.threshold_rows_frame = ttk.Frame(threshold_card)
        self.threshold_rows_frame.pack(fill=tk.X)
        self._build_recommendation_table_header(self.threshold_rows_frame)
        actions = ttk.Frame(threshold_card)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.threshold_restore_all_button = ttk.Button(
            actions,
            text="Restore Recommended",
            command=self._restore_all_thresholds,
            state=tk.DISABLED,
        )
        self.threshold_restore_all_button.pack(side=tk.LEFT)
        _info_button(
            actions,
            "Restore Recommended Thresholds",
            "Restores the automatically calculated confidence thresholds from the latest evaluation. "
            "Any unsaved manual threshold edits will be overwritten.",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self.threshold_save_button = ttk.Button(
            actions,
            text="Save Active",
            command=self._save_active_thresholds,
            state=tk.DISABLED,
        )
        self.threshold_save_button.pack(side=tk.LEFT, padx=(6, 0))

        multiplier_card = ttk.LabelFrame(content, text="Phase 4B - Multipliers", padding=12)
        multiplier_card.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            multiplier_card,
            textvariable=self.multiplier_status_var,
            wraplength=980,
            justify=tk.LEFT,
        ).pack(anchor="w")
        ttk.Label(
            multiplier_card,
            textvariable=self.multiplier_unsaved_var,
            foreground=COLORS["warning"],
        ).pack(anchor="w", pady=(2, 6))
        self.multiplier_rows_frame = ttk.Frame(multiplier_card)
        self.multiplier_rows_frame.pack(fill=tk.X)
        self._build_recommendation_table_header(self.multiplier_rows_frame)
        mactions = ttk.Frame(multiplier_card)
        mactions.pack(fill=tk.X, pady=(8, 0))
        self.multiplier_restore_all_button = ttk.Button(
            mactions,
            text="Restore Recommended",
            command=self._restore_all_multipliers,
            state=tk.DISABLED,
        )
        self.multiplier_restore_all_button.pack(side=tk.LEFT)
        _info_button(
            mactions,
            "Restore Recommended Multipliers",
            "Restores the recommended defect multipliers generated from the latest evaluation results. "
            "Any unsaved manual multiplier edits will be overwritten.",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self.multiplier_save_button = ttk.Button(
            mactions,
            text="Save Active",
            command=self._save_active_multipliers,
            state=tk.DISABLED,
        )
        self.multiplier_save_button.pack(side=tk.LEFT, padx=(6, 0))

        gate_card = ttk.LabelFrame(content, text="Phase 5 - Promotion", padding=12)
        gate_card.pack(fill=tk.X, pady=(0, 10))
        gate_card.columnconfigure(1, weight=1)
        ttk.Label(gate_card, text="Gate:", style="SectionNote.TLabel").grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 10),
            pady=(0, 6),
        )
        ttk.Label(
            gate_card,
            textvariable=self.gate_state_var,
            font=("TkDefaultFont", 12, "bold"),
        ).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        _info_button(
            gate_card,
            "Promotion Gate",
            "Shows whether the current model satisfies the configured promotion requirements. "
            "Re-run Gate recalculates the result using the latest evaluation metrics, active thresholds, and multipliers.",
        ).grid(row=0, column=2, sticky="ne", padx=(8, 0), pady=(0, 6))
        ttk.Label(gate_card, text="Details:", style="SectionNote.TLabel").grid(
            row=1,
            column=0,
            sticky="nw",
            padx=(0, 10),
        )
        ttk.Label(
            gate_card,
            textvariable=self.gate_details_var,
            wraplength=980,
            justify=tk.LEFT,
        ).grid(row=1, column=1, sticky="ew")
        gate_actions = ttk.Frame(gate_card)
        gate_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(
            gate_actions,
            text="Re-run Gate",
            command=self._rerun_validation_gate,
        ).pack(side=tk.LEFT)
        ttk.Button(
            gate_actions,
            text="Promote",
            command=self._export_model,
        ).pack(side=tk.LEFT, padx=(6, 0))

        history_frame = ttk.LabelFrame(content, text="Promoted Checkpoints", padding=12)
        history_frame.pack(fill=tk.BOTH, expand=True)
        self.rollback_tree = ttk.Treeview(
            history_frame,
            columns=("version", "accuracy", "escape", "over_reject", "promoted", "state"),
            show="headings",
            height=6,
        )
        for key, title, width in [
            ("version", "Model Version", 180),
            ("accuracy", "Accuracy", 100),
            ("escape", "Escape Rate", 100),
            ("over_reject", "Over Reject", 100),
            ("promoted", "Promotion Date", 180),
            ("state", "State", 160),
        ]:
            self.rollback_tree.heading(key, text=title)
            self.rollback_tree.column(key, width=width, anchor="center")
        self.rollback_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(
            history_frame,
            text="Rollback",
            command=self._rollback_selected,
        ).pack(anchor="e", pady=(6, 0))
        self._refresh_registry_ui()

    def _build_recommendation_table_header(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 2))
        for column, (text, width, weight) in enumerate(self._recommendation_columns()):
            header.columnconfigure(column, minsize=width, weight=weight)
            ttk.Label(
                header,
                text=text,
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=0, column=column, sticky="ew", padx=(0, 4))

    @staticmethod
    def _recommendation_columns():
        return (
            ("Class", 125, 0),
            ("Recommended", 105, 0),
            ("Active", 85, 0),
            ("Mode", 70, 0),
            ("Status / Warning", 240, 1),
            ("", 90, 0),
        )

    def _configure_recommendation_row(self, row):
        for column, (_text, width, weight) in enumerate(self._recommendation_columns()):
            row.columnconfigure(column, minsize=width, weight=weight)

    def _show_model_readiness(self):
        notebook = getattr(self, "training_main_notebook", None)
        if notebook is not None:
            notebook.select(1)

    def _add_metrics_epoch_row(
        self,
        epoch,
        train_loss,
        val_loss,
        accuracy,
        escape_rate,
        false_alarm,
    ):
        if not hasattr(self, "last_metrics_table"):
            return

        self.last_metrics_table.append_row(
            (
                epoch,
                f"{float(train_loss):.4f}",
                f"{float(val_loss):.4f}",
                f"{float(accuracy):.2%}",
                f"{float(escape_rate):.2%}",
                f"{float(false_alarm):.2%}",
            )
        )

    def _reset_live_metrics(self):
        self.metric_history = []
        if hasattr(self, "last_metrics_table"):
            self.last_metrics_table.clear()
        self._redraw_live_metrics()

    def _redraw_live_metrics(self):
        if not hasattr(self, "metrics_canvas"):
            return
        self.ax_loss.clear()
        self.ax_acc.clear()
        for axis in (self.ax_loss, self.ax_acc):
            axis.set_facecolor(COLORS["panel"])
            axis.tick_params(labelsize=8)
            axis.set_xlabel("Epoch", fontsize=8)
            axis.grid(True, linestyle="--", linewidth=1, alpha=0.45)
        self.ax_loss.set_title("Training & Validation Loss", fontsize=13)
        self.ax_loss.set_ylabel("Loss", fontsize=8)
        self.ax_acc.set_title("Model Performance Metrics", fontsize=13)
        self.ax_acc.set_ylabel("Rate / Probability", fontsize=8)
        self.ax_acc.set_ylim(0, 1.05)

        if self.metric_history:
            epochs = list(range(1, len(self.metric_history) + 1))
            self.ax_loss.plot(epochs, [m["train_loss"] for m in self.metric_history], color="#1f77b4", linewidth=1.8, label="Train Loss")
            self.ax_loss.plot(epochs, [m["val_loss"] for m in self.metric_history], color="#ff7f0e", linewidth=1.8, label="Val Loss")
            self.ax_acc.plot(epochs, [m["val_acc"] for m in self.metric_history], color="#2ca02c", linewidth=1.8, label="Accuracy")
            self.ax_acc.plot(epochs, [m["escape_rate"] for m in self.metric_history], color="#d62728", linewidth=1.8, label="Escape Rate")
            self.ax_acc.plot(epochs, [m["false_alarm_rate"] for m in self.metric_history], color="#9467bd", linewidth=1.8, label="False Alarm")
            self.ax_loss.legend(fontsize=8)
            self.ax_acc.legend(fontsize=8)
            self.ax_loss.set_xlim(0, max(2, len(epochs) + 1))
            self.ax_acc.set_xlim(0, max(2, len(epochs) + 1))

        self.metrics_fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.16,
            top=0.88,
            wspace=0.32,
        )
        self.metrics_canvas.draw_idle()

    def _handle_training_event(self, event):
        if event.get("type") != "epoch":
            return
        self.metric_history.append(event)
        self._add_metrics_epoch_row(
            epoch=event.get("completed_epochs", len(self.metric_history)),
            train_loss=event["train_loss"],
            val_loss=event["val_loss"],
            accuracy=event["val_acc"],
            escape_rate=event["escape_rate"],
            false_alarm=event["false_alarm_rate"],
        )
        self._redraw_live_metrics()
        if str(self.pb.cget("mode")) != "determinate":
            self.pb.stop()
            self.pb.configure(mode="determinate")
        progress = float(event.get("progress", 0.0))
        self.training_progress.set(progress)
        self.training_progress_text.set(
            f"{event.get('phase', 'Training')} {event.get('epoch')}/{event.get('stage_epochs')} "
            f"({event.get('completed_epochs')}/{event.get('total_epochs')} epochs)"
        )

    def _poll_training_events(self):
        while True:
            try:
                event = self.training_event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_training_event(event)
        if self._worker_thread and self._worker_thread.is_alive():
            self.after(500, self._poll_training_events)

    def _build_console_panel(self):
        self.console_panel = tk.Frame(
            self.display_panel,
            bg=COLORS["panel"],
            highlightbackground="#555555",
            highlightthickness=1,
        )

        self.metrics_console_notebook = ttk.Notebook(self.console_panel)
        self.metrics_console_notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=4,
            pady=4,
        )

        metrics_frame = ttk.Frame(self.metrics_console_notebook, padding=4)
        self.metrics_console_notebook.add(metrics_frame, text=" Last Metrics Epoch ")

        self.last_metrics_table = SimpleTable(
            metrics_frame,
            columns=(
                "Epoch",
                "Train Loss",
                "Val Loss",
                "Accuracy",
                "Escape Rate",
                "False Alarm",
            ),
            rows=[],
        )

        log_frame = ttk.Frame(self.metrics_console_notebook, padding=4)
        self.metrics_console_notebook.add(log_frame, text=" Console Log ")

        from app import components

        self.logger = components.LogWidget(log_frame, height=8)
        self.logger.pack(fill=tk.BOTH, expand=True)
        self.shared_console.attach(self.logger)

    def _build_config_panel(self, container):
        ds_group = CollapsibleGroup(container, "Inputs", expanded=True)

        # Split Output Path and Save Model Path were removed from the UI.
        required_path_labels = {"Training Dataset", "Fixed Validation Dataset"}
        ds_group.body.columnconfigure(0, weight=1)

        required_group = ttk.LabelFrame(ds_group.body, text="Required", padding=8)
        required_group.pack(fill=tk.X, pady=(0, 8))
        required_group.columnconfigure(3, weight=0)
        self._build_training_path_rows(
            required_group,
            [
                "Training Dataset",
                "Fixed Validation Dataset",
                "Output Folder",
            ],
            required_path_labels,
        )

        model_group = ttk.LabelFrame(ds_group.body, text="Optional Model Source", padding=8)
        model_group.pack(fill=tk.X, pady=(0, 8))
        model_group.columnconfigure(3, weight=0)
        self._build_training_path_rows(model_group, ["Pretrained Checkpoint"], required_path_labels)

        phase2_group = ttk.LabelFrame(ds_group.body, text="Phase 2", padding=8)
        phase2_group.pack(fill=tk.X, pady=(0, 8))
        phase2_group.columnconfigure(3, weight=0)
        self._build_training_path_rows(
            phase2_group,
            ["Phase 2 Dataset"],
            required_path_labels,
        )

        grad_cam_group = ttk.LabelFrame(ds_group.body, text="Grad-CAM Output Folder", padding=8)
        grad_cam_group.pack(fill=tk.X)
        grad_cam_group.columnconfigure(3, weight=0)
        self._build_training_path_rows(grad_cam_group, ["Grad-CAM Output Folder"], required_path_labels)
        ttk.Checkbutton(
            grad_cam_group,
            text="Export Grad-CAM result images",
            variable=self.export_grad_cam_var,
        ).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            padx=(0, 12),
            pady=(0, 4),
        )
        _info_button(
            grad_cam_group,
            "Export Grad-CAM result images Hint",
            SETTING_HINTS["Export Grad-CAM result images"],
        ).grid(row=1, column=3, sticky="e", padx=(8, 0), pady=(0, 4))
        ttk.Label(ds_group.body, text="* required", style="SectionNote.TLabel").pack(anchor="w", pady=(6, 0))

        gen_group = CollapsibleGroup(
            container,
            "Runtime Settings",
            expanded=True,
        )

        gen_group.body.columnconfigure(0, weight=1)
        gen_group.body.columnconfigure(1, weight=1)

        # Row 1
        self._add_general_entry(
            gen_group.body,
            "Model Name",
            "EfficientNetV2S",
            row=0,
            column=0,
        )
        self._add_general_entry(
            gen_group.body,
            "Model Version",
            "v1",
            row=0,
            column=1,
        )

        # Row 2
        self._add_general_entry(
            gen_group.body,
            "Image Width",
            "384",
            row=1,
            column=0,
        )
        self._add_general_entry(
            gen_group.body,
            "Image Height",
            "384",
            row=1,
            column=1,
        )

        self._add_general_combo(
            gen_group.body,
            "Device",
            ["GPU", "CPU"],
            "GPU",
            row=2,
            column=0,
        )
        self._add_general_combo(
            gen_group.body,
            "Batch Size",
            ["8", "16", "32", "64"],
            "16",
            row=3,
            column=0,
        )
        self._add_general_entry(
            gen_group.body,
            "Training Split (%)",
            "80",
            row=3,
            column=1,
        )

        adv = CollapsibleGroup(container, "Advanced Settings", expanded=False)

        p1_toggle_row = ttk.Frame(adv.body)
        p1_toggle_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            p1_toggle_row,
            text="Enable Phase 1  (RUN_PHASE_1)",
            variable=self.run_phase1_var,
        ).pack(side=tk.LEFT)
        _info_button(
            p1_toggle_row,
            "Enable Phase 1",
            "Phase 1 trains the model using the selected training dataset. Disable it only when you intend "
            "to skip this training phase and start from an existing pretrained model.",
        ).pack(side=tk.RIGHT, padx=(8, 0))

        self.phase1_form = PlainFormGroup(adv.body, "Phase 1 — Head Tuning", CONFIG_PHASE_1, field_width=12)
        self._add_form_entry_hints(self.phase1_form)

        phase1_ft_frame = ttk.LabelFrame(adv.body, text="Phase 1 — Core Model Fine-Tuning", padding=8)
        phase1_ft_frame.pack(fill=tk.X, pady=(0, 10))

        unfreeze_row = ttk.Frame(phase1_ft_frame)
        unfreeze_row.pack(fill=tk.X, pady=4)
        ttk.Label(unfreeze_row, text="Unfreeze Parameters:", width=20).pack(
            side=tk.LEFT,
            padx=(0, 6),
        )
        self.unfreeze_entry = ttk.Entry(unfreeze_row, width=8)
        self.unfreeze_entry.insert(0, "400")
        self.unfreeze_entry.pack(side=tk.LEFT)
        _info_button(
            unfreeze_row,
            "Phase 1 Unfreeze Parameters",
            "Controls how many model parameters are allowed to update during Phase 1 fine-tuning. "
            "A larger value adapts more of the model but needs more data and may increase overfitting.",
        ).pack(side=tk.RIGHT, padx=(8, 0))

        ttk.Separator(phase1_ft_frame, orient="horizontal").pack(fill=tk.X, pady=8)
        self.stage1_form = PlainFormGroup(phase1_ft_frame, "Stage 1", CONFIG_PHASE_1_STG1, field_width=12)
        self.stage2_form = PlainFormGroup(phase1_ft_frame, "Stage 2", CONFIG_PHASE_1_STG2, field_width=12)

        phase2_frame = ttk.LabelFrame(adv.body, text="Phase 2 — Transfer Learning", padding=8)
        phase2_frame.pack(fill=tk.X, pady=(0, 10))

        phase2_toggle_row = ttk.Frame(phase2_frame)
        phase2_toggle_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            phase2_toggle_row,
            text="Enable Phase 2  (RUN_PHASE_2_TRANSFER)",
            variable=self.run_phase2_var,
        ).pack(side=tk.LEFT)
        _info_button(
            phase2_toggle_row,
            "Enable Phase 2",
            "Phase 2 performs transfer learning using a pretrained model. Enable it when adapting an existing "
            "model to a new dataset. A pretrained model is required when Phase 1 is disabled.",
        ).pack(side=tk.RIGHT, padx=(8, 0))

        phase2_unfreeze_row = ttk.Frame(phase2_frame)
        phase2_unfreeze_row.pack(fill=tk.X, pady=4)
        ttk.Label(phase2_unfreeze_row, text="Unfreeze Parameters:", width=20).pack(
            side=tk.LEFT,
            padx=(0, 6),
        )
        self.phase2_unfreeze_entry = ttk.Entry(phase2_unfreeze_row, width=8)
        self.phase2_unfreeze_entry.insert(0, "5")
        self.phase2_unfreeze_entry.pack(side=tk.LEFT)
        _info_button(
            phase2_unfreeze_row,
            "Phase 2 Unfreeze Parameters",
            "Controls how many model parameters are allowed to update during transfer learning. "
            "Smaller values preserve more pretrained features; larger values adapt more of the model.",
        ).pack(side=tk.RIGHT, padx=(8, 0))

        ttk.Separator(phase2_frame, orient="horizontal").pack(fill=tk.X, pady=8)
        self.phase2_stage1_form = PlainFormGroup(phase2_frame, "Stage 1", CONFIG_PHASE_2_STG1, field_width=12)
        self.phase2_stage2_form = PlainFormGroup(phase2_frame, "Stage 2", CONFIG_PHASE_2_STG2, field_width=12)

        self.loss_form = PlainFormGroup(
            adv.body,
            "Loss & Regularization",
            [
                [("Dropout Rate", "0.5")],
                [("Focal Loss Gamma", "2.0")],
                [("Label Smoothing", "0.1")],
            ],
            field_width=12,
        )
        self._add_form_entry_hints(self.loss_form)
        self.augmentation_form = ttk.LabelFrame(adv.body, text="Augmentation Policy", padding=8)
        self.augmentation_form.pack(fill=tk.X, pady=(0, 10))
        _info_button(
            self.augmentation_form,
            "Augmentation Policy",
            "Controls random image transformations applied during training. Stronger augmentation can improve "
            "robustness, but unrealistic transformations may reduce accuracy and increase training time.",
        ).pack(anchor="e", pady=(0, 4))
        for label, from_value, to_value, initial_value in [
            ("Crop Min Scale", 0.0, 1.0, 0.85),
            ("Crop Max Scale", 0.0, 1.0, 1.0),
            ("Horizontal Flip", 0.0, 1.0, 0.5),
            ("Vertical Flip", 0.0, 1.0, 0.5),
            ("Brightness", 0.0, 1.0, 0.1),
            ("Contrast", 0.0, 1.0, 0.1),
            ("Saturation", 0.0, 1.0, 0.1),
        ]:
            self.augmentation_sliders[label] = SliderRow(
                self.augmentation_form,
                label,
                from_=from_value,
                to=to_value,
                initial_value=initial_value,
            )
            self._add_packed_setting_hint(self.augmentation_sliders[label], label)

        rotation_row = ttk.Frame(self.augmentation_form)
        rotation_row.pack(fill=tk.X, pady=4)
        ttk.Label(rotation_row, text="Rotation Degrees", width=16).pack(side=tk.LEFT)
        rotation_entry = ttk.Entry(rotation_row, width=12)
        rotation_entry.insert(0, "180")
        rotation_entry.pack(side=tk.LEFT, padx=8)
        _info_button(
            rotation_row,
            "Rotation Degrees",
            "Sets the maximum random rotation applied to training images. Use a value that matches how much "
            "the inspected part can realistically rotate in production.",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self.augmentation_entries["Rotation Degrees"] = rotation_entry

        self.training_control_form = PlainFormGroup(adv.body, "Training Control", [], field_width=12)
        self.training_control_form.columnconfigure(2, weight=1)
        ttk.Checkbutton(
            self.training_control_form,
            text="Enable LR Scheduler",
            variable=self.lr_scheduler_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        _info_button(
            self.training_control_form,
            "Enable LR Scheduler",
            "Automatically reduces the learning rate when validation improvement stalls. This can improve "
            "convergence and training stability without changing the initial learning rate.",
        ).grid(row=0, column=2, sticky="e", padx=(8, 0), pady=(0, 6))
        for row, (label, value) in enumerate(
            [
                ("LR Factor", "0.5"),
                ("LR Patience", "3"),
                ("Minimum LR", "1e-7"),
                ("Early Stop Patience", "12"),
            ],
            start=1,
        ):
            ttk.Label(self.training_control_form, text=f"{label}:").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=4,
            )
            entry = ttk.Entry(self.training_control_form, width=12)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="w", pady=4, padx=(0, 10))
            self.training_control_form.entries[label] = entry
            self._grid_setting_hint(self.training_control_form, label, row, column=2)

    def _build_training_path_rows(self, parent, labels, required_path_labels, start_row=0):
        parent.columnconfigure(1, weight=1)
        for offset, label in enumerate(labels):
            row = start_row + offset
            required_mark = " *" if label in required_path_labels else ""
            ttk.Label(parent, text=f"{label}{required_mark}:").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=4,
            )
            entry_state = "readonly" if label == "Fixed Validation Dataset" else "normal"
            entry = ttk.Entry(parent, width=PATH_ENTRY_WIDTH, state=entry_state)
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=4)
            self.path_entries[label] = entry
            ttk.Button(
                parent,
                text="...",
                width=3,
                command=lambda target=label: self._browse(target),
            ).grid(row=row, column=2, padx=(8, 0), pady=4)
            self._grid_setting_hint(parent, label, row, column=3)

    def _grid_setting_hint(self, parent, label, row, column=2):
        message = SETTING_HINTS.get(label)
        if not message:
            return
        _info_button(parent, f"{label} Hint", message).grid(
            row=row,
            column=column,
            sticky="e",
            padx=(8, 0),
            pady=4,
        )

    def _add_form_entry_hints(self, form):
        form.columnconfigure(2, weight=0)
        for row, label in enumerate(form.entries):
            self._grid_setting_hint(form, label, row, column=2)

    def _add_packed_setting_hint(self, parent, label):
        message = SETTING_HINTS.get(label)
        if not message:
            return
        entry = getattr(parent, "entry", None)
        if entry is not None:
            entry.pack_forget()
        _info_button(parent, f"{label} Hint", message).pack(side=tk.RIGHT, padx=(8, 0))
        if entry is not None:
            entry.pack(side=tk.RIGHT)

    def _add_general_entry(
        self,
        parent,
        label,
        value,
        row,
        column,
    ):
        field = ttk.Frame(parent)
        field.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=6,
            pady=4,
        )
        field.columnconfigure(1, weight=1)

        ttk.Label(
            field,
            text=f"{label}{' *' if label in ('Model Name', 'Version') else ''}:",
            width=12,
            anchor="w",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 6),
        )

        entry = ttk.Entry(field)
        entry.insert(0, value)
        entry.grid(
            row=0,
            column=1,
            sticky="ew",
        )

        self.general_entries[label] = entry
        if label in {"Model Name", "Model Version"}:
            entry.bind("<FocusOut>", lambda _event: self._recover_latest_training_state())
            entry.bind("<Return>", lambda _event: self._recover_latest_training_state())


    def _add_general_combo(
        self,
        parent,
        label,
        options,
        default_value,
        row,
        column,
    ):
        field = ttk.Frame(parent)
        field.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=6,
            pady=4,
        )
        field.columnconfigure(1, weight=1)

        ttk.Label(
            field,
            text=f"{label}:",
            width=12,
            anchor="w",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 6),
        )

        combo = ttk.Combobox(
            field,
            values=options,
            state="readonly",
        )

        if default_value in options:
            combo.set(default_value)
        elif options:
            combo.current(0)

        combo.grid(
            row=0,
            column=1,
            sticky="ew",
        )

        self.general_combos[label] = combo

    def _load_saved_settings(self):
        saved = load_settings()
        settings = saved.get("training_tab", {})

        self._set_path_value(
            "Training Dataset",
            settings.get("dataset_root", ""),
        )
        self._set_path_value(
            "Fixed Validation Dataset",
            saved.get("checkpoint_registry", {}).get("fixed_validation_dataset", ""),
        )
        self._set_path_value(
            "Pretrained Checkpoint",
            settings.get("pretrained_model_path", ""),
        )
        self._set_path_value(
            "Phase 2 Dataset",
            settings.get("phase_2_dataset_root", ""),
        )
        self._set_path_value(
            "Output Folder",
            settings.get("artifacts_dir", ""),
        )
        self._set_path_value(
            "Grad-CAM Output Folder",
            settings.get("grad_cam_output", ""),
        )

        self._set_entry(
            self.general_entries["Model Name"],
            settings.get("model_name", "EfficientNetV2S"),
        )
        self._set_entry(
            self.general_entries["Model Version"],
            settings.get("model_version", "v1"),
        )
        self._set_entry(
            self.general_entries["Image Width"],
            settings.get("image_width", "384"),
        )
        self._set_entry(
            self.general_entries["Image Height"],
            settings.get("image_height", "384"),
        )
        self._set_combo(
            self.general_combos["Device"],
            settings.get("device", "GPU"),
        )
        self.status_bar.set_device(self.general_combos["Device"].get())
        self._set_combo(
            self.general_combos["Batch Size"],
            settings.get("batch_size", "16"),
        )
        self._set_entry(
            self.general_entries["Training Split (%)"],
            settings.get("split_percent", "80"),
        )

        self.run_phase1_var.set(
            bool(settings.get("run_phase_1", True))
        )
        self._set_entry(
            self.phase1_form.entries["Epochs"],
            settings.get("phase_1_epochs", "15"),
        )
        self._set_entry(
            self.phase1_form.entries["Learning Rate"],
            settings.get("head_learning_rate", "1e-3"),
        )
        self._set_combo(
            self.phase1_form.entries["Optimizer"],
            settings.get("head_optimizer", "Adam"),
        )
        self._set_entry(
            self.unfreeze_entry,
            settings.get("unfreeze_layers", "400"),
        )
        for form, prefix, defaults in [
            (self.stage1_form, "phase_1_stage1", {"Optimizer": "AdamW", "Learning Rate": "1e-4", "Epochs": "50"}),
            (self.stage2_form, "phase_1_stage2", {"Optimizer": "SGD", "Learning Rate": "1e-5", "Epochs": "50"}),
            (self.phase2_stage1_form, "phase_2_stage1", {"Optimizer": "AdamW", "Learning Rate": "5e-6", "Epochs": "30"}),
            (self.phase2_stage2_form, "phase_2_stage2", {"Optimizer": "None", "Learning Rate": "1e-6", "Epochs": "20"}),
        ]:
            self._set_combo(form.entries["Optimizer"], settings.get(f"{prefix}_optimizer", defaults["Optimizer"]))
            self._set_entry(form.entries["Learning Rate"], settings.get(f"{prefix}_lr", defaults["Learning Rate"]))
            self._set_entry(form.entries["Epochs"], settings.get(f"{prefix}_epochs", defaults["Epochs"]))

        self.run_phase2_var.set(
            bool(settings.get("run_phase_2", False))
        )
        self._set_entry(
            self.phase2_unfreeze_entry,
            settings.get("phase_2_unfreeze_parameters", "5"),
        )

        for label, key, default in [
            ("Focal Loss Gamma", "fine_tune_focal_gamma", "2.0"),
            ("Label Smoothing", "label_smoothing", "0.1"),
            ("Dropout Rate", "dropout_rate", "0.5"),
        ]:
            self._set_entry(self.loss_form.entries[label], settings.get(key, default))

        self.lr_scheduler_var.set(bool(settings.get("lr_scheduler_enabled", True)))
        self.export_grad_cam_var.set(bool(settings.get("export_grad_cam", False)))

        for label, key, default in [
            ("Crop Min Scale", "crop_min_scale", "0.85"),
            ("Crop Max Scale", "crop_max_scale", "1.0"),
            ("Horizontal Flip", "horizontal_flip_probability", "0.5"),
            ("Vertical Flip", "vertical_flip_probability", "0.5"),
            ("Brightness", "brightness_jitter", "0.1"),
            ("Contrast", "contrast_jitter", "0.1"),
            ("Saturation", "saturation_jitter", "0.1"),
        ]:
            self._set_slider(self.augmentation_sliders[label], settings.get(key, default))
        self._set_entry(
            self.augmentation_entries["Rotation Degrees"],
            settings.get("rotation_degrees", "180"),
        )

        for label, key, default in [
            ("LR Factor", "lr_scheduler_factor", "0.5"),
            ("LR Patience", "lr_scheduler_patience", "3"),
            ("Minimum LR", "min_learning_rate", "1e-7"),
            ("Early Stop Patience", "early_stopping_patience", "12"),
        ]:
            self._set_entry(self.training_control_form.entries[label], settings.get(key, default))

    def _save_settings(self):
        config = load_settings()
        previous_settings = config.get("training_tab", {})

        config["training_tab"] = {
            "dataset_root": self._get_path_value("Training Dataset"),
            "pretrained_model_path": self._get_path_value("Pretrained Checkpoint"),
            "phase_2_dataset_root": self._get_path_value("Phase 2 Dataset"),
            "save_model_path": previous_settings.get(
                "save_model_path",
                "",
            ),
            "artifacts_dir": self._get_path_value(
                "Output Folder"
            ),
            "grad_cam_output": self._get_path_value(
                "Grad-CAM Output Folder"
            ),
            "model_name": self.general_entries[
                "Model Name"
            ].get().strip(),
            "model_version": self.general_entries[
                "Model Version"
            ].get().strip(),
            "image_width": self.general_entries[
                "Image Width"
            ].get().strip(),
            "image_height": self.general_entries[
                "Image Height"
            ].get().strip(),
            "device": self.general_combos["Device"].get().strip(),
            "batch_size": self.general_combos[
                "Batch Size"
            ].get().strip(),
            "split_percent": self.general_entries[
                "Training Split (%)"
            ].get().strip(),
            "run_phase_1": self.run_phase1_var.get(),
            "run_phase_2": self.run_phase2_var.get(),
            "phase_1_epochs": self.phase1_form.entries[
                "Epochs"
            ].get().strip(),
            "head_learning_rate": self.phase1_form.entries[
                "Learning Rate"
            ].get().strip(),
            "head_optimizer": self.phase1_form.entries[
                "Optimizer"
            ].get().strip(),
            "dropout_rate": self.loss_form.entries[
                "Dropout Rate"
            ].get().strip(),
            "unfreeze_layers": self.unfreeze_entry.get().strip(),
            "phase_2_unfreeze_parameters": self.phase2_unfreeze_entry.get().strip(),
            "phase_1_stage1_optimizer": self.stage1_form.entries["Optimizer"].get().strip(),
            "phase_1_stage1_lr": self.stage1_form.entries["Learning Rate"].get().strip(),
            "phase_1_stage1_epochs": self.stage1_form.entries["Epochs"].get().strip(),
            "phase_1_stage2_optimizer": self.stage2_form.entries["Optimizer"].get().strip(),
            "phase_1_stage2_lr": self.stage2_form.entries["Learning Rate"].get().strip(),
            "phase_1_stage2_epochs": self.stage2_form.entries["Epochs"].get().strip(),
            "phase_2_stage1_optimizer": self.phase2_stage1_form.entries["Optimizer"].get().strip(),
            "phase_2_stage1_lr": self.phase2_stage1_form.entries["Learning Rate"].get().strip(),
            "phase_2_stage1_epochs": self.phase2_stage1_form.entries["Epochs"].get().strip(),
            "phase_2_stage2_optimizer": self.phase2_stage2_form.entries["Optimizer"].get().strip(),
            "phase_2_stage2_lr": self.phase2_stage2_form.entries["Learning Rate"].get().strip(),
            "phase_2_stage2_epochs": self.phase2_stage2_form.entries["Epochs"].get().strip(),
            "fine_tune_focal_gamma": self.loss_form.entries["Focal Loss Gamma"].get().strip(),
            "label_smoothing": self.loss_form.entries["Label Smoothing"].get().strip(),
            "use_class_weights": False,
            "weight_decay": "1e-4",
            "freeze_batchnorm_stats": True,
            "lr_scheduler_enabled": self.lr_scheduler_var.get(),
            "export_grad_cam": self.export_grad_cam_var.get(),
            "early_stopping_patience": self.training_control_form.entries["Early Stop Patience"].get().strip(),
            "lr_scheduler_factor": self.training_control_form.entries["LR Factor"].get().strip(),
            "lr_scheduler_patience": self.training_control_form.entries["LR Patience"].get().strip(),
            "min_learning_rate": self.training_control_form.entries["Minimum LR"].get().strip(),
            "crop_min_scale": f"{self.augmentation_sliders['Crop Min Scale'].get():.2f}",
            "crop_max_scale": f"{self.augmentation_sliders['Crop Max Scale'].get():.2f}",
            "rotation_degrees": self.augmentation_entries["Rotation Degrees"].get().strip(),
            "horizontal_flip_probability": f"{self.augmentation_sliders['Horizontal Flip'].get():.2f}",
            "vertical_flip_probability": f"{self.augmentation_sliders['Vertical Flip'].get():.2f}",
            "brightness_jitter": f"{self.augmentation_sliders['Brightness'].get():.2f}",
            "contrast_jitter": f"{self.augmentation_sliders['Contrast'].get():.2f}",
            "saturation_jitter": f"{self.augmentation_sliders['Saturation'].get():.2f}",
        }

        checkpoint_settings = config.setdefault("checkpoint_registry", {})
        checkpoint_settings["fixed_validation_dataset"] = self._get_path_value("Fixed Validation Dataset")

        synchronize_shared_settings(config, "training_tab")
        save_settings(config)

    def _recovery_settings(self):
        saved = load_settings().get("training_tab", {})
        return {
            "artifacts_dir": self._get_path_value("Output Folder") or saved.get("artifacts_dir", ""),
            "save_model_path": saved.get("save_model_path", ""),
            "model_name": self.general_entries["Model Name"].get().strip() or saved.get("model_name", "EfficientNetV2S"),
            "model_version": self.general_entries["Model Version"].get().strip() or saved.get("model_version", "v1"),
        }

    def _recover_latest_training_state(self):
        if not getattr(self, "logger", None):
            return
        try:
            state = recover_training_state(self._recovery_settings())
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self.logger.log(f"[Recovery Error] {message}")
            return
        self.recovered_state = state
        for line in state.get("log", []):
            self.logger.log(f"[Recovery] {line}")
        self._apply_recovered_state(state)

    def _apply_recovered_state(self, state):
        record = state.get("record")
        if record and state.get("checkpoint_valid"):
            self._latest_checkpoint_record = record
            self.status_bar.set_ckpt(state["checkpoint_path"])
        elif state.get("checkpoint_path"):
            self.status_bar.set_ckpt(state["checkpoint_path"])

        gate = state.get("gate") or {}
        promotion_gate = state.get("promotion_gate") or {}
        if gate.get("valid"):
            metrics = (record or {}).get("metrics", {})
            checkpoint_summary = (
                f"Recovered checkpoint: {Path(state['checkpoint_path']).name} | "
                f"Accuracy {float(metrics.get('accuracy', 0.0)):.2%}, "
                f"escape rate {float(metrics.get('escape_rate', 0.0)):.2%}"
            )
            if state.get("promotion_ready"):
                self.gate_state_var.set("Passed gate, ready to promote")
                self.gate_details_var.set(checkpoint_summary)
            else:
                failures = failed_rule_messages(promotion_gate)
                self.gate_state_var.set("Promotion blocked")
                self.gate_details_var.set(
                    "; ".join(failures[:6]) or checkpoint_summary
                )
        elif state.get("checkpoint_path"):
            self._latest_checkpoint_record = None
            self.gate_state_var.set("Checkpoint found — registration required")
            self.gate_details_var.set(f"Recovered checkpoint file: {Path(state['checkpoint_path']).name}")

        evaluation = state.get("evaluation") or {}
        if evaluation.get("status") == "complete":
            self.evaluation_status_var.set("Evaluation complete")
            self.evaluation_details_var.set(self._format_evaluation_details(evaluation))
        elif evaluation.get("status") == "invalid":
            self.evaluation_status_var.set("Evaluation invalid")
            self.evaluation_details_var.set(display_text(evaluation.get("error"), "Evaluation artifacts invalid."))
        elif evaluation.get("status") == "failed":
            self.evaluation_status_var.set("Evaluation failed")
            self.evaluation_details_var.set(display_text(evaluation.get("error"), "Evaluation failed."))
        else:
            self.evaluation_status_var.set("Evaluation not run")
            self.evaluation_details_var.set("Recovered checkpoint can be evaluated without retraining." if gate.get("valid") else "No valid evaluation state recovered.")

        thresholds = state.get("thresholds") or {}
        config = thresholds.get("config")
        if thresholds.get("status") == "saved" and config:
            self._display_threshold_config(config, thresholds.get("path"), saved=True)
            self.threshold_status_var.set(f"Saved thresholds recovered: {thresholds.get('path')}")
        elif thresholds.get("status") == "draft" and config:
            self._display_threshold_config(config, thresholds.get("path"), saved=False)
            self.threshold_status_var.set("Recommended thresholds recovered; save active values before promotion.")
        elif thresholds.get("status") == "invalid":
            self.threshold_status_var.set(f"Threshold file invalid: {display_text(thresholds.get('error'), 'Unknown validation error')}")
            self.threshold_unsaved_var.set("")
            self._set_threshold_editing_enabled(False)
        else:
            self.threshold_status_var.set("Thresholds disabled until valid evaluation predictions are available.")
            self.threshold_unsaved_var.set("")
            self._set_threshold_editing_enabled(False)

        multipliers = state.get("multipliers") or {}
        mconfig = multipliers.get("config")
        if multipliers.get("status") == "saved" and mconfig:
            self._display_multiplier_config(mconfig, multipliers.get("path"), multipliers.get("deployed_path"), saved=True)
            self.multiplier_status_var.set(f"Saved multipliers recovered: {multipliers.get('path')}")
        elif multipliers.get("status") == "draft" and mconfig:
            self._display_multiplier_config(mconfig, multipliers.get("path"), multipliers.get("deployed_path"), saved=False)
            self.multiplier_status_var.set("Recommended multipliers recovered; save them before promotion.")
        elif multipliers.get("status") == "invalid":
            self._show_multiplier_error(multipliers.get("error") or "Invalid multiplier configuration")
        else:
            self.multiplier_status_var.set("Multipliers disabled until valid evaluation predictions are available.")
            self._set_multiplier_editing_enabled(False)

        if state.get("promotion_ready"):
            self.training_progress_text.set("Recovered promotion-ready checkpoint")
        elif state.get("promotion_gate"):
            failures = failed_rule_messages(state["promotion_gate"])
            if failures:
                self.logger.log("[Gate] Promotion blocked: " + "; ".join(failures[:6]))

    def _set_entry(self, widget, value):
        widget.delete(0, tk.END)
        widget.insert(0, value)

    def _set_slider(self, widget, value):
        try:
            widget.var.set(float(value))
            widget._on_change(None)
        except (TypeError, ValueError):
            pass

    def _set_combo(self, widget, value):
        values = tuple(widget.cget("values"))
        if value in values:
            widget.set(value)

    def _set_path_value(self, label, value):
        entry = self.path_entries[label]
        original_state = str(entry.cget("state"))
        if original_state == "readonly":
            entry.configure(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, value)
        if original_state == "readonly":
            entry.configure(state="readonly")

    def _get_path_value(self, label):
        return self.path_entries[label].get().strip()

    def _get_saved_model_path(self):
        settings = load_settings().get("training_tab", {})
        return settings.get("save_model_path", "").strip()


    def _set_saved_model_path(self, path):
        config = load_settings()
        training_settings = config.setdefault("training_tab", {})
        training_settings["save_model_path"] = path
        save_settings(config)


    def _make_export_filename(self):
        model_name = self.general_entries[
            "Model Name"
        ].get().strip()
        version = self.general_entries[
            "Model Version"
        ].get().strip()

        model_name = model_name or "model"
        version = version or "v1"

        def clean_name(value):
            cleaned = "".join(
                character
                if character.isalnum() or character in ("-", "_")
                else "_"
                for character in value
            )
            return cleaned.strip("_") or "model"

        return f"{clean_name(model_name)}_{clean_name(version)}.pth"

    def _browse(self, target):
        if target == "Pretrained Checkpoint":
            path = filedialog.askopenfilename(
                title="Select Pretrained Checkpoint",
                filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")],
            )
        else:
            path = filedialog.askdirectory(
                title=f"Select {target}"
            )

        if not path:
            return

        if target == "Fixed Validation Dataset":
            current_path = self._get_path_value(target)
            if current_path and Path(current_path) != Path(path):
                confirmation = simpledialog.askstring(
                    "Change Fixed Validation Baseline",
                    "Changing the fixed validation dataset changes the promotion baseline. "
                    "Type CHANGE BASELINE to continue:",
                    parent=self,
                )
                if confirmation != "CHANGE BASELINE":
                    self.logger.log("[Validation Baseline] Change cancelled; fixed validation dataset was not changed.")
                    return

        self._set_path_value(target, path)
        self._save_settings()

        if target == "Training Dataset":
            self.status_bar.set_dataset(path)
        if target in {"Output Folder", "Fixed Validation Dataset"}:
            self._recover_latest_training_state()

        self.logger.log(f"[Path] {target} → {path}")

    def _build_training_settings(self):
        image_width = int(self.general_entries["Image Width"].get().strip() or "384")
        image_height = int(self.general_entries["Image Height"].get().strip() or "384")
        phase_1_epochs = int(self.phase1_form.entries["Epochs"].get().strip() or "15")
        dropout_rate = float(self.loss_form.entries["Dropout Rate"].get().strip() or "0.5")
        unfreeze_layers = int(self.unfreeze_entry.get().strip() or "400")
        phase2_unfreeze_parameters = int(self.phase2_unfreeze_entry.get().strip() or "5")

        stage1_epochs = int(self.stage1_form.entries["Epochs"].get().strip() or "50")
        stage1_lr = float(self.stage1_form.entries["Learning Rate"].get().strip() or "1e-4")
        stage1_opt = self.stage1_form.entries["Optimizer"].get().strip().lower()

        stage2_opt_raw = self.stage2_form.entries["Optimizer"].get().strip()
        phase_1_stages = [
            {
                "optimizer": stage1_opt,
                "learning_rate": stage1_lr,
                "epochs": stage1_epochs,
            }
        ]

        if stage2_opt_raw and stage2_opt_raw.lower() != "none":
            phase_1_stages.append(
                {
                    "optimizer": stage2_opt_raw.lower(),
                    "learning_rate": float(self.stage2_form.entries["Learning Rate"].get().strip() or "1e-5"),
                    "epochs": int(self.stage2_form.entries["Epochs"].get().strip() or "50"),
                }
            )

        phase2_stage1_epochs = int(self.phase2_stage1_form.entries["Epochs"].get().strip() or "30")
        phase2_stage1_lr = float(self.phase2_stage1_form.entries["Learning Rate"].get().strip() or "5e-6")
        phase2_stage1_opt = self.phase2_stage1_form.entries["Optimizer"].get().strip().lower()
        phase_2_stages = [
            {
                "optimizer": phase2_stage1_opt,
                "learning_rate": phase2_stage1_lr,
                "epochs": phase2_stage1_epochs,
            }
        ]

        phase2_stage2_opt_raw = self.phase2_stage2_form.entries["Optimizer"].get().strip()
        if phase2_stage2_opt_raw and phase2_stage2_opt_raw.lower() != "none":
            phase_2_stages.append(
                {
                    "optimizer": phase2_stage2_opt_raw.lower(),
                    "learning_rate": float(self.phase2_stage2_form.entries["Learning Rate"].get().strip() or "1e-6"),
                    "epochs": int(self.phase2_stage2_form.entries["Epochs"].get().strip() or "20"),
                }
            )

        planned_total_epochs = phase_1_epochs if self.run_phase1_var.get() else 0
        if self.run_phase1_var.get():
            planned_total_epochs += sum(stage["epochs"] for stage in phase_1_stages)
        if self.run_phase2_var.get():
            planned_total_epochs += sum(stage["epochs"] for stage in phase_2_stages)

        return {
            "dataset_root": self._get_path_value("Training Dataset"),
            "pretrained_model_path": self._get_path_value("Pretrained Checkpoint"),
            "phase_2_dataset_root": self._get_path_value("Phase 2 Dataset"),
            "save_model_path": "",
            "model_name": self.general_entries["Model Name"].get().strip(),
            "model_version": self.general_entries["Model Version"].get().strip(),
            "artifacts_dir": self._get_path_value(
                "Output Folder"
            ),
            "grad_cam_outside_dir": self._get_path_value(
                "Grad-CAM Output Folder"
            ),
            "device": self.general_combos["Device"].get().strip() or "GPU",
            "batch_size": int(
                self.general_combos["Batch Size"].get().strip()
                or "16"
            ),
            "image_size": (image_width, image_height),
            "split_percent": int(self.general_entries["Training Split (%)"].get().strip() or "80"),
            "run_phase_1": self.run_phase1_var.get(),
            "run_phase_2": self.run_phase2_var.get(),
            "phase_1_epochs": phase_1_epochs,
            "head_optimizer": self.phase1_form.entries["Optimizer"].get().strip().lower() or "adam",
            "head_learning_rate": float(self.phase1_form.entries["Learning Rate"].get().strip() or "1e-3"),
            "head_focal_gamma": 0.0,
            "dropout_rate": dropout_rate,
            "unfreeze_layers": unfreeze_layers,
            "phase_2_unfreeze_parameters": phase2_unfreeze_parameters,
            "fine_tune_focal_gamma": float(self.loss_form.entries["Focal Loss Gamma"].get().strip() or "2.0"),
            "label_smoothing": float(self.loss_form.entries["Label Smoothing"].get().strip() or "0.1"),
            "use_class_weights": False,
            "freeze_batchnorm_stats": True,
            "early_stopping_patience": int(self.training_control_form.entries["Early Stop Patience"].get().strip() or "12"),
            "lr_scheduler_enabled": self.lr_scheduler_var.get(),
            "export_grad_cam": self.export_grad_cam_var.get(),
            "lr_scheduler_factor": float(self.training_control_form.entries["LR Factor"].get().strip() or "0.5"),
            "lr_scheduler_patience": int(self.training_control_form.entries["LR Patience"].get().strip() or "3"),
            "min_learning_rate": float(self.training_control_form.entries["Minimum LR"].get().strip() or "1e-7"),
            "weight_decay": 1e-4,
            "crop_min_scale": float(self.augmentation_sliders["Crop Min Scale"].get()),
            "crop_max_scale": float(self.augmentation_sliders["Crop Max Scale"].get()),
            "rotation_degrees": float(self.augmentation_entries["Rotation Degrees"].get().strip() or "180"),
            "horizontal_flip_probability": float(self.augmentation_sliders["Horizontal Flip"].get()),
            "vertical_flip_probability": float(self.augmentation_sliders["Vertical Flip"].get()),
            "brightness_jitter": float(self.augmentation_sliders["Brightness"].get()),
            "contrast_jitter": float(self.augmentation_sliders["Contrast"].get()),
            "saturation_jitter": float(self.augmentation_sliders["Saturation"].get()),
            "phase_1_stages": phase_1_stages,
            "phase_2_stages": phase_2_stages,
            "planned_total_epochs": planned_total_epochs,
        }

    def _start_training(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self.logger.log("[Start] Training already running.")
            return

        if not self._get_path_value("Training Dataset"):
            self.logger.log("[Error] Training Dataset required.")
            return
        if self.run_phase2_var.get() and not self._get_path_value("Phase 2 Dataset"):
            self.logger.log("[Error] Phase 2 Dataset required when Phase 2 is enabled.")
            return

        self._save_settings()
        self.pb.stop()
        self.pb.configure(mode="indeterminate")
        self.training_progress_text.set("Preparing dataset and model...")
        self.training_progress.set(0.0)
        self.pb.start(12)
        self._reset_live_metrics()
        self.training_stop_event = threading.Event()
        settings = self._build_training_settings()
        model_name = settings.get("model_name", "").strip()
        model_version = settings.get("model_version", "").strip()
        if not model_name:
            self.pb.stop()
            self.pb.configure(mode="determinate")
            self.logger.log("[Error] Model Name is required.")
            return
        if not model_version:
            self.pb.stop()
            self.pb.configure(mode="determinate")
            self.logger.log("[Error] Version is required.")
            return
        self._set_training_actions_enabled(False, stop_enabled=True)
        self.status_bar.set_dataset(settings["dataset_root"])

        output_root = Path(settings.get("artifacts_dir") or Path(__file__).resolve().parents[2] / "artifacts")
        run_name = f"{model_name}_{model_version}"
        self.status_bar.set_ckpt(str(output_root / run_name / "checkpoints" / f"{run_name}.pth"))

        def worker():
            try:
                result = run_training_job(
                    settings,
                    self.logger,
                    stop_event=self.training_stop_event,
                    progress_callback=self.training_event_queue.put,
                )
                self.after(0, lambda: self._on_training_complete(result))
            except Exception as exc:
                self.after(0, lambda error=exc: self._on_training_failed(error))

        self.logger.log("[Start] Pipeline execution launched.")
        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        self.after(500, self._poll_training_events)

    def _on_training_complete(self, result):
        self._poll_training_events()
        self.pb.stop()
        self.pb.configure(mode="determinate")
        if result.get("stopped"):
            self.training_progress_text.set(
                f"Stopped at {result.get('completed_epochs', 0)}/{result.get('total_epochs', 0)} epochs"
            )
            self.logger.log("[Stop] Training stopped by user.")
            self._set_training_actions_enabled(True)
            return
        self.training_progress.set(100.0)
        self.training_stop_event = None
        self.training_progress_text.set("Validating checkpoint against production baseline...")
        self.status_bar.set_ckpt(result["final_model_path"])
        self.logger.log(f"[Done] Training finished. Model: {result['final_model_path']}")
        self.logger.log(f"[Done] Summary: {result['summary_path']}")
        self.logger.log(f"[Done] Artifacts: {result['artifacts_dir']}")
        self._validate_completed_checkpoint(result)

    def _on_training_failed(self, exc):
        self.pb.stop()
        self.pb.configure(mode="determinate")
        self.training_progress.set(0.0)
        self.training_progress_text.set("Training failed")
        self.logger.log(f"[Error] {exc}")
        self._set_training_actions_enabled(True)

    def _stop_training(self):
        if self._worker_thread and self._worker_thread.is_alive() and self.training_stop_event:
            self.training_stop_event.set()
            self.training_progress_text.set("Stopping training...")
            self.logger.log("[Stop] Stop requested. Waiting for backend checkpoint-safe stop.")
            return
        self.pb.stop()
        self.pb.configure(mode="determinate")
        self.training_progress_text.set("Stopped")

    def _clear_logs(self):
        self.logger.clear()

    def _validate_completed_checkpoint(self, training_result):
        checkpoint_path = self._resolve_best_checkpoint(training_result)
        self._run_validation_gate(
            checkpoint_path=checkpoint_path,
            artifacts_dir=training_result["artifacts_dir"],
            produced_by="training_run",
        )

    def _resolve_best_checkpoint(self, training_result):
        checkpoint_path = (
            training_result.get("best_model_path")
            or training_result.get("best_checkpoint_path")
            or training_result.get("final_model_path")
        )
        if not checkpoint_path:
            raise ValueError("Training completed but no checkpoint path was returned.")
        return checkpoint_path

    def _run_validation_gate(self, checkpoint_path, artifacts_dir, produced_by, image_size_override=None):
        settings = self._build_training_settings()
        saved = load_settings()
        fixed_validation_dataset = (
            saved.get("checkpoint_registry", {}).get("fixed_validation_dataset", "").strip()
        )
        if not fixed_validation_dataset:
            message = "Set a fixed validation dataset before training can be validated for promotion."
            self.training_progress_text.set("Validation blocked — fixed validation set required")
            self.gate_state_var.set("Validation unavailable — promotion blocked")
            self.gate_details_var.set(message)
            self.logger.log(f"[Gate Blocked] {message}")
            messagebox.showwarning("Fixed Validation Dataset Required", message, parent=self)
            self._set_training_actions_enabled(True)
            return

        settings.update({
            "dataset_root": fixed_validation_dataset,
            "model_checkpoint": checkpoint_path,
            "save_model_path": checkpoint_path,
            "artifacts_dir": artifacts_dir,
        })
        if image_size_override is not None:
            settings["image_size"] = tuple(image_size_override)
        self.training_progress_text.set("Validating checkpoint against production baseline...")
        self._post_training_busy = True
        self._set_training_actions_enabled(False, stop_enabled=False)

        def worker():
            logger = self._main_thread_logger()
            try:
                validation = run_validation_job(settings, logger)
                record = register_checkpoint(
                    checkpoint_path,
                    validation["metrics"],
                    produced_by=produced_by,
                    metadata={
                        "model_name": settings.get("model_name"),
                        "model_version": settings.get("model_version"),
                        "validation_report": validation.get("report_path"),
                        # Recorded so a later "Re-run Gate" on this
                        # same checkpoint can recall the resolution it was
                        # actually evaluated at, instead of guessing from
                        # whatever the GUI fields currently show.
                        "image_size": list(settings.get("image_size", ())),
                    },
                )
                record = self._run_automatic_evaluation_for_record(record, settings, validation, logger)
                self.after(0, lambda: self._on_gate_complete(record))
            except Exception as exc:
                self.after(0, lambda error=exc: self._on_gate_failed(error))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _main_thread_logger(self):
        tab = self

        class MainThreadLogger:
            def log(self, message):
                tab.after(0, lambda msg=str(message): tab.logger.log(msg))

        return MainThreadLogger()

    def _run_automatic_evaluation_for_record(self, record, settings, validation, logger=None):
        evaluation_settings = dict(settings)
        evaluation_settings.update(
            {
                "dataset_root": settings["dataset_root"],
                "test_dir": validation.get("dataset_dir") or settings["dataset_root"],
                "model_checkpoint": settings["model_checkpoint"],
                "save_model_path": settings["model_checkpoint"],
                "output_dir": settings.get("artifacts_dir"),
                "artifacts_dir": settings.get("artifacts_dir"),
                "skip_grad_cam": not bool(settings.get("export_grad_cam", False)),
                "append_run_folder": True,
            }
        )

        def progress(percent, text):
            self.after(
                0,
                lambda p=percent, t=text: self._set_automatic_evaluation_progress(p, t),
            )

        try:
            self.after(0, lambda: self.evaluation_status_var.set("Evaluation running"))
            self.after(0, lambda: self.training_progress_text.set("Evaluation 1/4 — Running validation inference"))
            result = run_test_job(evaluation_settings, logger or self._main_thread_logger(), progress_callback=progress)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self.after(0, lambda msg=message: self.evaluation_status_var.set("Evaluation failed"))
            self.after(0, lambda msg=message: self.evaluation_details_var.set(msg))
            self.after(0, lambda msg=message: self.logger.log(f"[Evaluation Error] Automatic evaluation failed: {msg}"))
            return update_checkpoint_metadata(
                record["id"],
                {
                    "automatic_evaluation": {
                        "status": "failed",
                        "error": message,
                    }
                },
            )

        details = self._format_evaluation_details(result)
        self.after(0, lambda text=details: self.evaluation_details_var.set(text))
        self.after(0, lambda path=result.get("report_path"): self.logger.log(f"[Evaluation] Automatic report: {path}"))
        threshold_metadata = self._generate_thresholds_after_evaluation(settings, result, logger or self._main_thread_logger())
        multiplier_metadata = self._generate_multipliers_after_evaluation(settings, result, logger or self._main_thread_logger())
        return update_checkpoint_metadata(
            record["id"],
            {
                "automatic_evaluation": {
                    "status": "complete",
                    "accuracy": result.get("accuracy"),
                    "macro_precision": result.get("macro_precision"),
                    "macro_recall": result.get("macro_recall"),
                    "macro_f1": result.get("macro_f1"),
                    "misclassified_count": result.get("misclassified_count"),
                    "summary_path": result.get("summary_path"),
                    "confusion_matrix_path": result.get("confusion_matrix_path"),
                    "report_path": result.get("report_path"),
                    "dataset_dir": result.get("dataset_dir"),
                    "sample_count": result.get("sample_count"),
                    "grad_cam_skipped": result.get("grad_cam_skipped"),
                    "prediction_cache_path": result.get("prediction_cache_path"),
                },
                "automatic_thresholds": threshold_metadata,
                "automatic_multipliers": multiplier_metadata,
                "multipliers_required": multiplier_metadata.get("status") == "complete",
            },
        )

    def _format_evaluation_details(self, result):
        return (
            f"Accuracy {float(result.get('accuracy', 0.0)):.2%} | "
            f"Macro precision {float(result.get('macro_precision', 0.0)):.2%} | "
            f"Macro recall {float(result.get('macro_recall', 0.0)):.2%} | "
            f"Macro F1 {float(result.get('macro_f1', 0.0)):.2%} | "
            f"Misclassified {int(result.get('misclassified_count', 0))}\n"
            f"Report: {result.get('report_path') or 'not generated'}\n"
            f"Confusion matrix: {result.get('confusion_matrix_path') or 'not generated'}"
        )

    def _generate_thresholds_after_evaluation(self, settings, evaluation_result, logger):
        prediction_cache_path = evaluation_result.get("prediction_cache_path")
        if not prediction_cache_path:
            message = "Evaluation did not produce a prediction cache."
            self.after(0, lambda msg=message: self._show_threshold_error(msg))
            return {"status": "failed", "error": message}
        threshold_settings = dict(settings)
        threshold_settings.update(
            {
                "dataset_root": evaluation_result.get("dataset_dir") or settings["dataset_root"],
                "model_checkpoint": settings["model_checkpoint"],
                "save_model_path": settings["model_checkpoint"],
                "artifacts_dir": settings.get("artifacts_dir"),
                "prediction_cache_path": prediction_cache_path,
                "threshold_strategy": "best_f1",
                "append_run_folder": True,
            }
        )
        try:
            self.after(0, lambda: self.threshold_status_var.set("Generating recommended thresholds…"))
            result = generate_threshold_recommendations(threshold_settings, logger)
            config = result["active_thresholds"]
            config_path = result["active_thresholds_path"]
            self.after(0, lambda cfg=config, path=config_path: self._display_threshold_config(cfg, path, saved=False))
            return {
                "status": "complete",
                "strategy": config.get("strategy"),
                "active_thresholds_path": config_path,
                "recommendations_path": result.get("results_json_path"),
                "prediction_cache_path": prediction_cache_path,
                "warnings": result.get("thresholds", {}).get("warnings", []),
            }
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self.after(0, lambda msg=message: self._show_threshold_error(msg))
            self.after(0, lambda msg=message: self.logger.log(f"[Threshold Error] {msg}"))
            return {"status": "failed", "error": message}

    def _generate_multipliers_after_evaluation(self, settings, evaluation_result, logger):
        cache_path = evaluation_result.get("prediction_cache_path")
        if not cache_path:
            message = "Evaluation did not produce a prediction cache."
            self.after(0, lambda msg=message: self._show_multiplier_error(msg))
            return {"status": "failed", "error": message}
        multiplier_settings = dict(settings)
        multiplier_settings.update({
            "dataset_root": evaluation_result.get("dataset_dir") or settings["dataset_root"],
            "model_checkpoint": settings["model_checkpoint"],
            "save_model_path": settings["model_checkpoint"],
            "artifacts_dir": settings.get("artifacts_dir"),
            "prediction_cache_path": cache_path,
        })
        if not hasattr(self, "multiplier_status_var"):
            return {"status": "not_started", "error": "Multiplier UI state is unavailable."}
        try:
            self.after(0, lambda: self.multiplier_status_var.set("Generating recommended defect multipliers…"))
            result = generate_multiplier_recommendations(multiplier_settings, logger)
            config = result["active_multipliers"]
            path = result["active_multipliers_path"]
            deployed = result["deployed_multipliers_path"]
            self.after(0, lambda cfg=config, p=path, d=deployed: self._display_multiplier_config(cfg, p, d, saved=False))
            return {"status":"complete", "strategy":config.get("strategy"), "active_multipliers_path":path, "recommendations_path":result.get("recommendations_path"), "deployed_multipliers_path":deployed, "prediction_cache_path":cache_path}
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self.after(0, lambda msg=message: self._show_multiplier_error(msg))
            self.after(0, lambda msg=message: self.logger.log(f"[Multiplier Error] {msg}"))
            return {"status":"failed", "error":message}

    def _show_multiplier_error(self, message):
        if not hasattr(self, "multiplier_status_var"):
            return
        self.multiplier_status_var.set(f"Multiplier recommendations failed: {message}")
        self.multiplier_unsaved_var.set("")
        self._set_multiplier_editing_enabled(False)
        self._recalculate_gate_display()

    def _display_multiplier_config(self, config, path=None, deployed_path=None, saved=False):
        self.multiplier_config = copy.deepcopy(config)
        self.multiplier_saved_config = copy.deepcopy(config) if saved else None
        self.multiplier_config_path = path or self.multiplier_config_path
        self.multiplier_deployed_path = deployed_path or self.multiplier_deployed_path
        self._rebuild_multiplier_rows()
        self._set_multiplier_editing_enabled(True)
        self._mark_multiplier_unsaved(not saved)

    def _rebuild_multiplier_rows(self):
        if not hasattr(self, "multiplier_rows_frame"): return
        for child in list(self.multiplier_rows_frame.winfo_children())[1:]: child.destroy()
        self.multiplier_rows = {}
        for name, values in (self.multiplier_config or {"classes":{}}).get("classes", {}).items():
            row=ttk.Frame(self.multiplier_rows_frame); row.pack(fill=tk.X,pady=1)
            self._configure_recommendation_row(row)
            rec, active = values.get("recommended"), values.get("active")
            var=tk.StringVar(value="" if active is None else f"{float(active):.4f}"); mode=tk.StringVar(value=str(values.get("mode","auto")).title())
            ttk.Label(row,text=name).grid(row=0,column=0,sticky="ew",padx=(0,4))
            ttk.Label(row,text="—" if rec is None else f"{float(rec):.4f}").grid(row=0,column=1,sticky="ew",padx=(0,4))
            entry=ttk.Entry(row,textvariable=var,width=10); entry.grid(row=0,column=2,sticky="ew",padx=(0,4))
            ttk.Label(row,textvariable=mode).grid(row=0,column=3,sticky="ew",padx=(0,4))
            ttk.Label(row,text=values.get("warning") or values.get("status") or "valid",wraplength=230,justify=tk.LEFT).grid(row=0,column=4,sticky="ew",padx=(0,4))
            button=ttk.Button(row,text="Restore",width=10,command=lambda n=name:self._restore_one_multiplier(n)); button.grid(row=0,column=5,sticky="ew")
            var.trace_add("write",lambda *_args,n=name:self._on_multiplier_value_changed(n))
            self.multiplier_rows[name]={"entry":entry,"active_var":var,"mode_var":mode,"restore_button":button}

    def _set_multiplier_editing_enabled(self, enabled):
        state=tk.NORMAL if enabled else tk.DISABLED
        for row in self.multiplier_rows.values(): row["entry"].configure(state=state); row["restore_button"].configure(state=state)
        if hasattr(self,"multiplier_restore_all_button"): self.multiplier_restore_all_button.configure(state=state)
        if hasattr(self,"multiplier_save_button"): self.multiplier_save_button.configure(state=state)

    def _on_multiplier_value_changed(self, class_name):
        if self._updating_multiplier_rows or not self.multiplier_config: return
        row=self.multiplier_rows.get(class_name)
        if not row: return
        self.multiplier_config["classes"][class_name]["active"]=row["active_var"].get(); self.multiplier_config["classes"][class_name]["mode"]="manual"; row["mode_var"].set("Manual")
        self._mark_multiplier_unsaved(True); self._recalculate_gate_display()

    def _restore_one_multiplier(self, class_name):
        try: self.multiplier_config=restore_recommended_multiplier(self.multiplier_config,class_name)
        except Exception as exc: messagebox.showerror("Restore Multiplier",str(exc),parent=self); return
        row=self.multiplier_rows[class_name]; self._updating_multiplier_rows=True; row["active_var"].set(f"{self.multiplier_config['classes'][class_name]['active']:.4f}"); self._updating_multiplier_rows=False; row["mode_var"].set("Auto")
        self._mark_multiplier_unsaved(True); self._recalculate_gate_display()

    def _restore_all_multipliers(self):
        try: self.multiplier_config=restore_recommended_multiplier(self.multiplier_config)
        except Exception as exc: messagebox.showerror("Restore Multipliers",str(exc),parent=self); return
        for name,row in self.multiplier_rows.items():
            self._updating_multiplier_rows=True; row["active_var"].set(f"{self.multiplier_config['classes'][name]['active']:.4f}"); self._updating_multiplier_rows=False; row["mode_var"].set("Auto")
        self._mark_multiplier_unsaved(True); self._recalculate_gate_display()

    def _save_active_multipliers(self):
        if not self.multiplier_config or not self.multiplier_config_path: messagebox.showinfo("Save Multipliers","Generate recommendations before saving multipliers.",parent=self); return
        try:
            config=copy.deepcopy(self.multiplier_config)
            for name,row in self.multiplier_rows.items():
                config=set_active_multiplier(config,name,row["active_var"].get()); config["classes"][name]["mode"]=self.multiplier_config["classes"][name].get("mode","manual")
            save_multiplier_config(config,self.multiplier_config_path,deployment_path=self.multiplier_deployed_path)
        except Exception as exc: messagebox.showerror("Save Multipliers",str(exc),parent=self); self.multiplier_status_var.set(f"Save failed: {exc}"); return
        self.multiplier_config=config; self.multiplier_saved_config=copy.deepcopy(config); self.multiplier_status_var.set(f"Active multipliers saved: {self.multiplier_config_path}"); self._mark_multiplier_unsaved(False)
        if self._latest_checkpoint_record:
            self._latest_checkpoint_record=update_checkpoint_metadata(self._latest_checkpoint_record["id"],{"active_multipliers":{"status":"saved","path":self.multiplier_config_path,"deployed_path":self.multiplier_deployed_path,"strategy":config.get("strategy"),"classes":config.get("classes",{})},"deployed_multipliers_path":self.multiplier_deployed_path})
        self._recover_latest_training_state()

    def _mark_multiplier_unsaved(self, unsaved):
        self.multiplier_unsaved_var.set("Unsaved multiplier changes" if unsaved else "")
        if unsaved and self.multiplier_config: self.multiplier_status_var.set("Review multipliers, then Save Active to activate.")

    def _multiplier_draft_has_unsaved_changes(self):
        return bool(self.multiplier_config and self.multiplier_saved_config != self.multiplier_config)

    def _show_threshold_error(self, message):
        self.threshold_status_var.set(f"Threshold recommendations failed: {message}")
        self.threshold_unsaved_var.set("")
        self._set_threshold_editing_enabled(False)

    def _display_threshold_config(self, config, path=None, saved=False):
        self._show_model_readiness()
        self.threshold_config = copy.deepcopy(config)
        if saved:
            self.threshold_saved_config = copy.deepcopy(config)
        self.threshold_config_path = path or self.threshold_config_path
        self._rebuild_threshold_rows()
        self._set_threshold_editing_enabled(True)
        self._mark_threshold_unsaved(not saved)

    def _rebuild_threshold_rows(self):
        if not hasattr(self, "threshold_rows_frame"):
            return
        for child in list(self.threshold_rows_frame.winfo_children())[1:]:
            child.destroy()
        self.threshold_rows = {}
        config = self.threshold_config or {"classes": {}}
        for class_name, values in config.get("classes", {}).items():
            row = ttk.Frame(self.threshold_rows_frame)
            row.pack(fill=tk.X, pady=1)
            self._configure_recommendation_row(row)
            recommended = values.get("recommended")
            active = values.get("active")
            active_var = tk.StringVar(value="" if active is None else f"{float(active):.4f}")
            mode_var = tk.StringVar(value=str(values.get("mode", "auto")).title())
            status_text = values.get("warning") or values.get("status") or "valid"
            ttk.Label(row, text=class_name).grid(row=0, column=0, sticky="ew", padx=(0, 4))
            ttk.Label(
                row,
                text="—" if recommended is None else f"{float(recommended):.4f}",
            ).grid(row=0, column=1, sticky="ew", padx=(0, 4))
            entry = ttk.Entry(row, textvariable=active_var, width=10)
            entry.grid(row=0, column=2, sticky="ew", padx=(0, 4))
            ttk.Label(row, textvariable=mode_var).grid(row=0, column=3, sticky="ew", padx=(0, 4))
            ttk.Label(
                row,
                text=status_text,
                wraplength=230,
                justify=tk.LEFT,
            ).grid(row=0, column=4, sticky="ew", padx=(0, 4))
            button = ttk.Button(
                row,
                text="Restore",
                width=10,
                command=lambda name=class_name: self._restore_one_threshold(name),
            )
            button.grid(row=0, column=5, sticky="ew")
            active_var.trace_add("write", lambda *_args, name=class_name: self._on_threshold_value_changed(name))
            self.threshold_rows[class_name] = {
                "entry": entry,
                "active_var": active_var,
                "mode_var": mode_var,
                "restore_button": button,
            }

    def _set_threshold_editing_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for row in getattr(self, "threshold_rows", {}).values():
            row["entry"].configure(state=state)
            row["restore_button"].configure(state=state)
        if hasattr(self, "threshold_restore_all_button"):
            self.threshold_restore_all_button.configure(state=state)
        if hasattr(self, "threshold_save_button"):
            self.threshold_save_button.configure(state=state)

    def _on_threshold_value_changed(self, class_name):
        if getattr(self, "_updating_threshold_rows", False):
            return
        if not self.threshold_config or class_name not in self.threshold_config.get("classes", {}):
            return
        row = self.threshold_rows.get(class_name)
        if not row:
            return
        self.threshold_config["classes"][class_name]["active"] = row["active_var"].get()
        self.threshold_config["classes"][class_name]["mode"] = "manual"
        row["mode_var"].set("Manual")
        self._mark_threshold_unsaved(True)
        self._recalculate_gate_display()

    def _restore_one_threshold(self, class_name):
        if not self.threshold_config:
            return
        try:
            self.threshold_config = restore_recommended(self.threshold_config, class_name)
        except Exception as exc:
            messagebox.showerror("Restore Threshold", str(exc), parent=self)
            return
        row = self.threshold_rows.get(class_name)
        if row:
            active = self.threshold_config["classes"][class_name]["active"]
            self._updating_threshold_rows = True
            row["active_var"].set(f"{float(active):.4f}")
            self._updating_threshold_rows = False
            row["mode_var"].set("Auto")
        self._mark_threshold_unsaved(True)
        self._recalculate_gate_display()

    def _restore_all_thresholds(self):
        if not self.threshold_config:
            return
        try:
            self.threshold_config = restore_recommended(self.threshold_config)
        except Exception as exc:
            messagebox.showerror("Restore Thresholds", str(exc), parent=self)
            return
        for class_name, row in self.threshold_rows.items():
            active = self.threshold_config["classes"][class_name]["active"]
            self._updating_threshold_rows = True
            row["active_var"].set(f"{float(active):.4f}")
            self._updating_threshold_rows = False
            row["mode_var"].set("Auto")
        self._mark_threshold_unsaved(True)
        self._recalculate_gate_display()

    def _save_active_thresholds(self):
        if not self.threshold_config or not self.threshold_config_path:
            messagebox.showinfo("Save Active", "Generate recommendations before saving thresholds.", parent=self)
            return
        try:
            config = copy.deepcopy(self.threshold_config)
            for class_name, row in self.threshold_rows.items():
                config = set_active_threshold(config, class_name, row["active_var"].get())
                config["classes"][class_name]["mode"] = self.threshold_config["classes"][class_name].get("mode", "manual")
            save_threshold_config(config, self.threshold_config_path)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            messagebox.showerror("Save Active", message, parent=self)
            self.threshold_status_var.set(f"Save failed: {message}")
            return
        self.threshold_config = config
        self.threshold_saved_config = copy.deepcopy(config)
        self.threshold_status_var.set(f"Active thresholds saved: {self.threshold_config_path}")
        self._mark_threshold_unsaved(False)
        if self._latest_checkpoint_record:
            try:
                self._latest_checkpoint_record = update_checkpoint_metadata(
                    self._latest_checkpoint_record["id"],
                    {
                        "active_thresholds": {
                            "status": "saved",
                            "path": self.threshold_config_path,
                            "strategy": config.get("strategy"),
                            "classes": config.get("classes", {}),
                        }
                    },
                )
                self._display_gate_summary(self._latest_checkpoint_record)
                self.gate_details_var.set(f"{self.gate_details_var.get()} | Active thresholds saved.")
            except Exception as exc:
                self.logger.log(f"[Threshold Warning] Saved thresholds, but registry metadata update failed: {exc}")
        self._recover_latest_training_state()

    def _mark_threshold_unsaved(self, unsaved):
        if unsaved:
            self.threshold_unsaved_var.set("Unsaved threshold changes")
            if self.threshold_config:
                self.threshold_status_var.set("Review thresholds, then Save Active to activate.")
        else:
            self.threshold_unsaved_var.set("")

    def _threshold_draft_has_unsaved_changes(self):
        if not self.threshold_config:
            return False
        return self.threshold_saved_config != self.threshold_config

    def _recalculate_gate_display(self):
        state = getattr(self, "recovered_state", None)
        if not state:
            return
        gate_result = evaluate_promotion_gate(
            state,
            requirements=load_registry().get("requirements") or {},
            model_name=self.general_entries["Model Name"].get().strip(),
            model_version=self.general_entries["Model Version"].get().strip(),
            unsaved_threshold_changes=self._threshold_draft_has_unsaved_changes(),
            unsaved_multiplier_changes=self._multiplier_draft_has_unsaved_changes(),
        )
        state["promotion_gate"] = gate_result
        state["promotion_ready"] = bool(gate_result.get("passed"))
        if gate_result.get("passed"):
            self.gate_state_var.set("Passed gate, ready to promote")
        else:
            failures = failed_rule_messages(gate_result)
            self.gate_state_var.set("Promotion blocked")
            self.gate_details_var.set("; ".join(failures[:6]) or "Promotion gate failed.")

    def _set_automatic_evaluation_progress(self, percent, text):
        if str(self.pb.cget("mode")) != "determinate":
            self.pb.stop()
            self.pb.configure(mode="determinate")
        self.training_progress.set(max(0.0, min(100.0, float(percent))))
        self.training_progress_text.set(str(text))

    def _set_training_actions_enabled(self, enabled, stop_enabled=None):
        state = tk.NORMAL if enabled else tk.DISABLED
        target_texts = {
            "Start",
            "Promote",
            "Re-run Gate",
            "Rollback",
            "Restore Recommended",
            "Save Active",
            "Restore",
        }

        def visit(widget):
            for child in widget.winfo_children():
                try:
                    if str(child.cget("text")) in target_texts:
                        child.configure(state=state)
                except tk.TclError:
                    pass
                visit(child)

        visit(self)
        if stop_enabled is not None:
            stop_state = tk.NORMAL if stop_enabled else tk.DISABLED

            def visit_stop(widget):
                for child in widget.winfo_children():
                    try:
                        if str(child.cget("text")) == "Stop":
                            child.configure(state=stop_state)
                    except tk.TclError:
                        pass
                    visit_stop(child)

            visit_stop(self)

    def _rerun_validation_gate(self):
        initial_dir, initial_file = self._checkpoint_browse_default()
        checkpoint_path = filedialog.askopenfilename(
            title="Select Checkpoint to Validate",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")],
            parent=self,
        )
        if not checkpoint_path:
            return

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        artifacts_dir = checkpoint.parent.parent if checkpoint.parent.name.lower() == "checkpoints" else checkpoint.parent

        image_size_override, recorded = self._recall_image_size_for_checkpoint(checkpoint)
        current_w = self.general_entries["Image Width"].get().strip() or "384"
        current_h = self.general_entries["Image Height"].get().strip() or "384"
        if recorded:
            self.logger.log(
                f"[Gate] Using recorded training resolution {image_size_override[0]}x{image_size_override[1]} "
                f"for this checkpoint (from registry metadata)."
            )
        else:
            # No record of what this checkpoint was actually trained at (e.g. it
            # predates this fix, or was never registered before). Guessing here
            # is exactly the silent-wrong-number failure this gate exists to
            # prevent, so require an explicit human confirmation instead.
            proceed = messagebox.askyesno(
                "Confirm Validation Resolution",
                "No recorded training resolution for this checkpoint.\n\n"
                f"Validation will run at the current Image Width/Height fields: "
                f"{current_w}x{current_h}.\n\n"
                "Confirm this matches the resolution this checkpoint was actually "
                "trained at. If unsure, click No and check before proceeding.",
                parent=self,
            )
            if not proceed:
                self.logger.log("[Gate] Cancelled — resolution not confirmed.")
                return
            image_size_override = (int(current_w), int(current_h))

        self.status_bar.set_ckpt(str(checkpoint))
        self.logger.log(f"[Gate] Re-running fixed validation for: {checkpoint}")
        self._run_validation_gate(
            checkpoint_path=str(checkpoint),
            artifacts_dir=str(artifacts_dir),
            produced_by="manual_validation",
            image_size_override=image_size_override,
        )

    def _recall_image_size_for_checkpoint(self, checkpoint_path):
        """Look up a previously-registered image_size for this exact checkpoint
        file. Returns (image_size_tuple_or_None, found_bool)."""
        registry = load_registry()
        target = str(Path(checkpoint_path).expanduser().resolve())
        for record in registry.get("checkpoints", []):
            if record.get("source_path") == target:
                size = record.get("metadata", {}).get("image_size")
                if size and len(size) == 2 and all(size):
                    return (int(size[0]), int(size[1])), True
        return None, False


    def _checkpoint_browse_default(self):
        settings = load_settings().get("training_tab", {})
        runtime_paths = build_runtime_paths(
            output_root=settings.get("artifacts_dir"),
            model_name=settings.get("model_name", "EfficientNetV2S"),
            model_version=settings.get("model_version", "v1"),
        )
        preferred_dir = Path(runtime_paths.checkpoints_dir)
        search_root = Path(settings.get("artifacts_dir") or Path(__file__).resolve().parents[2] / "artifacts")
        candidates = []
        for root in (preferred_dir, search_root):
            if root.is_dir():
                candidates.extend(path for pattern in ("*.pth", "*.pt") for path in root.rglob(pattern))
        candidates = list({path.resolve() for path in candidates if path.is_file()})
        if candidates:
            newest = max(candidates, key=lambda path: path.stat().st_mtime)
            return str(newest.parent), newest.name
        return str(preferred_dir if preferred_dir.exists() else search_root), ""

    def _display_gate_summary(self, record):
        """Update the gate labels from a checkpoint record. Display only —
        this must never change which checkpoint promotion will target."""
        gate = record.get("gate", {})
        metrics = record.get("metrics", {})
        baseline = gate.get("baseline", {})
        self.gate_state_var.set(gate.get("state", "Checkpoint gated"))
        self.gate_details_var.set(
            f"New: accuracy {float(metrics.get('accuracy', 0)):.2%}, "
            f"escape rate {float(metrics.get('escape_rate', 0)):.2%} | "
            f"Production baseline: accuracy {float(baseline.get('accuracy', 0)):.2%}, "
            f"escape rate {float(baseline.get('escape_rate', 0)):.2%}"
        )

    def _on_gate_complete(self, record):
        self._show_model_readiness()
        # This is the ONLY place a gate run (training or manual) claims the
        # promotion target for the current session.
        self._latest_checkpoint_record = record
        self._display_gate_summary(record)
        state = record.get("gate", {}).get("state", "Checkpoint gated")
        evaluation = record.get("metadata", {}).get("automatic_evaluation", {})
        if evaluation.get("status") == "complete":
            self.training_progress_text.set(f"{state} — automatic evaluation complete")
            self.evaluation_status_var.set("Evaluation complete")
            self.evaluation_details_var.set(self._format_evaluation_details(evaluation))
            self.logger.log(f"[Evaluation] Automatic evaluation complete: {evaluation.get('report_path')}")
        elif evaluation.get("status") == "failed":
            self.training_progress_text.set(f"{state} — automatic evaluation failed")
            self.evaluation_status_var.set("Evaluation failed")
            self.evaluation_details_var.set(display_text(evaluation.get("error"), "Automatic evaluation failed."))
            self.logger.log(f"[Evaluation Error] {evaluation.get('error')}")
        else:
            self.training_progress_text.set(state)
        self.logger.log(f"[Gate] {state}")
        self._post_training_busy = False
        self._set_training_actions_enabled(True)
        # Rebuild the complete promotion state after evaluation/threshold/
        # multiplier artifacts have been written.  Showing only the registry
        # accuracy gate here can incorrectly say "ready to promote" while the
        # evaluation is invalid or active tuning files are still unsaved.
        self._recover_latest_training_state()
        self._refresh_registry_ui()

    def _on_gate_failed(self, exc):
        self.training_progress_text.set("Validation gate failed to run")
        self.gate_state_var.set("Validation unavailable — promotion blocked")
        message = str(exc).strip() or exc.__class__.__name__
        self.gate_details_var.set(message)
        self.evaluation_status_var.set("Evaluation not run")
        self.evaluation_details_var.set("Validation gate failed before automatic evaluation.")
        self.logger.log(f"[Gate Error] {message}")
        self._post_training_busy = False
        self._set_training_actions_enabled(True)

    def _export_model(self):
        record = self._latest_checkpoint_record
        if record is None:
            # Cold start: no gate has run in this session. Fall back to the
            # registry, but only when the target is unambiguous. If several
            # checkpoints are gated-but-unpromoted, refuse to guess — the user
            # must pick one explicitly via Re-run Gate, so we never
            # promote a checkpoint they did not choose.
            pending = self._unpromoted_gated_checkpoints()
            if not pending:
                messagebox.showinfo(
                    "Promote to Production",
                    "Train and validate a checkpoint before promotion.",
                )
                return
            if len(pending) > 1:
                messagebox.showwarning(
                    "Select a Checkpoint to Promote",
                    "Multiple validated checkpoints are awaiting promotion. "
                    'Use "Re-run Gate" and select the specific '
                    "checkpoint you want, so the correct one is promoted.",
                    parent=self,
                )
                return
            record = pending[0]
            self._latest_checkpoint_record = record
        if not self._record_ready_for_promotion(record):
            failures = failed_rule_messages(getattr(self, "_last_promotion_gate", {}) or {})
            details = "\n".join(failures) if failures else "Promotion requires a valid gate, complete evaluation artifacts, and saved active thresholds."
            messagebox.showwarning(
                "Promotion Not Ready",
                details,
                parent=self,
            )
            self.logger.log("[Promotion] Blocked — " + "; ".join(failures or ["recovered state is not promotion-ready"]))
            return
        override = False
        if not record.get("gate", {}).get("passed"):
            confirmation = simpledialog.askstring(
                "Override Required",
                "This checkpoint failed the production baseline. Type OVERRIDE to promote it anyway:",
                parent=self,
            )
            if confirmation != "OVERRIDE":
                self.logger.log("[Promotion] Override cancelled; production was not changed.")
                return
            override = True
        try:
            settings = self._build_training_settings()
            package = build_promotion_package(
                self.recovered_state or recover_training_state(self._recovery_settings()),
                output_root=settings.get("artifacts_dir") or self._get_path_value("Output Folder") or Path(__file__).resolve().parents[2] / "artifacts",
                model_name=settings.get("model_name"),
                model_version=settings.get("model_version"),
                training_config=settings,
                requirements=load_registry().get("requirements") or {},
                unsaved_threshold_changes=self._threshold_draft_has_unsaved_changes(),
                unsaved_multiplier_changes=self._multiplier_draft_has_unsaved_changes(),
                override=override,
            )
            promoted = register_promoted_package(
                record["id"],
                package["package_dir"],
                package["model_path"],
                override=override,
            )
        except Exception as exc:
            messagebox.showerror("Promotion Failed", str(exc))
            self.logger.log(f"[Promotion Error] {exc}")
            return
        self.gate_state_var.set("Currently in production")
        self.training_progress_text.set("Currently in production")
        self.status_bar.set_ckpt(promoted["promoted_path"])
        self.logger.log(f"[Promotion] Production package updated: {promoted.get('promoted_package_path') or promoted['promoted_path']}")
        # Promotion consumed the target; require a fresh gate or explicit pick
        # before the next promotion so nothing lingers as an implicit target.
        self._latest_checkpoint_record = None
        self._refresh_registry_ui()

    def _record_ready_for_promotion(self, record):
        state = recover_training_state(self._recovery_settings())
        self.recovered_state = state
        target = state.get("record") or {}
        if target.get("id") != record.get("id"):
            self._last_promotion_gate = {
                "passed": False,
                "rules": [{"name": "checkpoint_identity", "passed": False, "reason": "Recovered checkpoint does not match selected checkpoint."}],
                "warnings": [],
            }
            return False
        self._last_promotion_gate = evaluate_promotion_gate(
            state,
            requirements=load_registry().get("requirements") or {},
            model_name=self.general_entries["Model Name"].get().strip(),
            model_version=self.general_entries["Model Version"].get().strip(),
            unsaved_threshold_changes=self._threshold_draft_has_unsaved_changes(),
            unsaved_multiplier_changes=self._multiplier_draft_has_unsaved_changes(),
        )
        if self._last_promotion_gate.get("passed"):
            return True
        overridable = OVERRIDABLE_PROMOTION_RULES
        failed = [
            rule.get("name")
            for rule in self._last_promotion_gate.get("rules", [])
            if not rule.get("passed")
        ]
        return bool(failed) and all(name in overridable for name in failed)

    def _unpromoted_gated_checkpoints(self, registry=None):
        registry = registry or load_registry()
        return [
            record
            for record in registry.get("checkpoints", [])
            if record.get("gate") is not None and not record.get("promoted_path")
        ]

    def _latest_unpromoted_gated_checkpoint(self, registry=None):
        candidates = self._unpromoted_gated_checkpoints(registry)
        if not candidates:
            return None
        return max(
            enumerate(candidates),
            key=lambda item: (item[1].get("created_at", ""), item[0]),
        )[1]

    def _refresh_registry_ui(self):
        if not self.rollback_tree:
            return
        for item in self.rollback_tree.get_children():
            self.rollback_tree.delete(item)
        registry = load_registry()
        # Display-only. Surface a pending checkpoint when no gate has run this
        # session, purely so the user can see something is waiting. Crucially,
        # this does NOT reassign self._latest_checkpoint_record — a UI refresh
        # must never change what Promote will act on.
        if self._latest_checkpoint_record is None:
            pending = self._latest_unpromoted_gated_checkpoint(registry)
            if pending:
                self._display_gate_summary(pending)
        current_id = registry.get("current_production_id")
        for record in promoted_checkpoints():
            metrics = record.get("metrics", {})
            state = "Currently in production" if record.get("id") == current_id else "Available for rollback"
            self.rollback_tree.insert(
                "",
                "end",
                iid=record["id"],
                values=(
                    Path(record.get("promoted_path", record.get("filename", ""))).name,
                    f"{float(metrics.get('accuracy', 0)):.2%}",
                    f"{float(metrics.get('escape_rate', 0)):.2%}",
                    f"{float(metrics.get('false_alarm_rate', 0)):.2%}",
                    display_text(record.get("promoted_at")),
                    state,
                ),
            )

    def _rollback_selected(self):
        selected = self.rollback_tree.selection() if self.rollback_tree else ()
        if not selected:
            messagebox.showinfo("Rollback", "Select a promoted checkpoint first.")
            return
        try:
            record = rollback_checkpoint(selected[0])
        except Exception as exc:
            messagebox.showerror("Rollback Failed", str(exc))
            self.logger.log(f"[Rollback Error] {exc}")
            return
        self.gate_state_var.set("Currently in production")
        self.gate_details_var.set(f"Rolled back to {Path(record['promoted_path']).name}.")
        self.status_bar.set_ckpt(record["promoted_path"])
        self.logger.log(f"[Rollback] Production now uses: {record['promoted_path']}")
        self._refresh_registry_ui()

    def _toggle_log_panel(self):
        if self.log_visible:
            self.display_panel.forget(self.console_panel)
            self.log_visible = False
            self.logger.log("[UI] Console Log hidden.")
        else:
            self.display_panel.add(self.console_panel, weight=CONSOLE_PANE_WEIGHT)
            self.log_visible = True
            self.logger.log("[UI] Console Log displayed.")
