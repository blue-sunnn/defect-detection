from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from backend.path_validation import require_path




def get_runtime_class_names() -> list[str]:
    """Read class names without importing PyTorch or the inference service."""
    default = ["Mousebite", "Open", "Pass", "Pinhole", "Protrusion", "Short", "Via"]
    try:
        config_module = importlib.import_module("backend.config_threshold_optimization")
        classes_path = Path(config_module.VAL_CLASSES_PATH)
        if not classes_path.exists():
            return default
        import numpy as np
        names = np.load(classes_path, allow_pickle=True).tolist()
        return [str(name) for name in names] or default
    except Exception:
        return default

@dataclass
class RuntimePaths:
    artifacts_dir: str
    checkpoints_dir: str
    phase1_final_model_path: str
    phase2_transfer_model_path: str
    phase2_transfer_confusion_matrix_path: str
    phase2_transfer_training_results_plot_path: str
    phase2_transfer_summary_txt_path: str
    phase2_transfer_misclassification_output_dir: str
    train_confusion_matrix_path: str
    validation_confusion_matrix_path: str
    test_confusion_matrix_path: str
    training_results_plot_path: str
    training_summary_txt_path: str
    validation_summary_txt_path: str
    test_summary_txt_path: str
    evaluation_predictions_path: str
    threshold_data_dir: str
    threshold_true_path: str
    threshold_prob_path: str
    threshold_classes_path: str
    threshold_histogram_path: str
    threshold_results_csv_path: str
    threshold_results_json_path: str
    threshold_active_json_path: str
    threshold_deployed_json_path: str
    multiplier_results_json_path: str
    multiplier_active_json_path: str
    multiplier_deployed_json_path: str
    train_misclassification_output_dir: str
    validation_misclassification_output_dir: str
    grad_cam_output_dir: str
    config_dir: str
    training_settings_path: str
    backend_config_snapshot_path: str


class LoggerWriter:
    _PROGRESS_PATTERNS = (
        re.compile(r"^\s*\d{1,3}%[|\s]"),
        re.compile(r"\d+/\d+.*(?:it/s|s/it)"),
        re.compile(r"(?:Training Batch|Grad-CAM|Generating Grad-CAM|Plotting Grad-CAM).*\d+%", re.IGNORECASE),
        re.compile(r"^.*\b(?:batch|image)s?\b.*\d+%.*(?:it/s|s/it).*$", re.IGNORECASE),
        re.compile(r"^[\s\u2588\u2589\u258a\u258b\u258c\u258d\u258e\u258f#=.-]+$"),
    )

    def __init__(self, logger):
        self.logger = logger
        self._buffer = ""

    @classmethod
    def _is_progress_line(cls, line):
        return any(pattern.search(line) for pattern in cls._PROGRESS_PATTERNS)

    def _emit(self, line):
        line = line.strip()
        if line and not self._is_progress_line(line):
            self.logger.log(line)

    def write(self, text):
        if not text:
            return 0
        # Terminal progress libraries redraw the same line with carriage returns.
        # Treat those fragments as disposable and keep only normal newline logs.
        normalized = text.replace("\r\n", "\n")
        if "\r" in normalized:
            fragments = normalized.split("\r")
            for fragment in fragments[:-1]:
                if "\n" in fragment:
                    for line in fragment.split("\n"):
                        self._emit(line)
            normalized = fragments[-1]
        self._buffer += normalized
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self):
        self._emit(self._buffer)
        self._buffer = ""


def _bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _backend_root() -> Path:
    return _bundle_root() / "backend"


def ensure_backend_paths():
    bundle_root = str(_bundle_root())
    backend_root = str(_backend_root())
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)


def import_backend_modules():
    ensure_backend_paths()
    module_names = [
        "backend.model_config",
        "backend.config_threshold_optimization",
        "backend.train_multiclass_defect_classifier",
        "backend.run_validation",
        "backend.run_test",
        "backend.run_threshold_pipeline",
        "backend.optimize_pass_threshold",
        "backend.utils_plotting",
        "backend.utils_training",
        "backend.model_factory",
    ]
    return {name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in module_names}


def resolve_dataset_root(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if (path / "train").is_dir() and (path / "val").is_dir():
        return str(path)
    if path.name.lower() == "val" and path.parent.is_dir() and (path.parent / "train").is_dir():
        return str(path.parent)
    return str(path)


def sanitize_path_part(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or fallback


def build_runtime_paths(
    output_root: str | None = None,
    save_model_path: str | None = None,
    grad_cam_root: str | None = None,
    model_name: str = "EfficientNetV2S",
    model_version: str = "v1",
    append_run_folder: bool = True,
) -> RuntimePaths:
    bundle_root = _bundle_root()
    safe_model_name = sanitize_path_part(model_name, "model")
    safe_model_version = sanitize_path_part(model_version, "v1")
    run_name = f"{safe_model_name}_{safe_model_version}"
    file_prefix = run_name.lower()

    base_artifacts_dir = Path(output_root).expanduser().resolve() if output_root else (bundle_root / "artifacts")
    artifacts_dir = base_artifacts_dir / run_name if append_run_folder else base_artifacts_dir
    checkpoints_dir = artifacts_dir / "checkpoints"
    phase1_final_model_path = (
        str(Path(save_model_path).expanduser().resolve())
        if save_model_path
        else str(checkpoints_dir / f"{run_name}.pth")
    )
    phase2_transfer_model_path = str(checkpoints_dir / f"{run_name}_phase2.pth")

    gcam_base = Path(grad_cam_root).expanduser().resolve() / run_name if grad_cam_root else artifacts_dir
    config_dir = artifacts_dir / "config"
    deployed_multipliers_path = artifacts_dir / "multiplier" / "defect_multipliers.json"

    return RuntimePaths(
        artifacts_dir=str(artifacts_dir),
        checkpoints_dir=str(checkpoints_dir),
        phase1_final_model_path=phase1_final_model_path,
        phase2_transfer_model_path=phase2_transfer_model_path,
        phase2_transfer_confusion_matrix_path=str(artifacts_dir / "phase2_transfer" / "plots" / "confusion_matrix.png"),
        phase2_transfer_training_results_plot_path=str(artifacts_dir / "phase2_transfer" / "plots" / "training_metrics_dashboard.png"),
        phase2_transfer_summary_txt_path=str(artifacts_dir / "phase2_transfer" / "summaries" / "phase2_transfer_summary.txt"),
        phase2_transfer_misclassification_output_dir=str(gcam_base / "phase2_transfer" / "grad_cam"),
        train_confusion_matrix_path=str(artifacts_dir / "phase1" / "plots" / "confusion_matrix.png"),
        validation_confusion_matrix_path=str(artifacts_dir / "validation" / "plots" / "confusion_matrix.png"),
        test_confusion_matrix_path=str(artifacts_dir / "test" / "plots" / "confusion_matrix.png"),
        evaluation_predictions_path=str(artifacts_dir / "evaluation" / "predictions.jsonl"),
        training_results_plot_path=str(artifacts_dir / "phase1" / "plots" / "training_metrics_dashboard.png"),
        training_summary_txt_path=str(artifacts_dir / "phase1" / "summaries" / "final_training_summary.txt"),
        validation_summary_txt_path=str(artifacts_dir / "validation" / "summaries" / "validation_summary.txt"),
        test_summary_txt_path=str(artifacts_dir / "test" / "summaries" / "test_summary.txt"),
        threshold_data_dir=str(artifacts_dir / "threshold" / "data"),
        threshold_true_path=str(artifacts_dir / "threshold" / "data" / "val_data_true.npy"),
        threshold_prob_path=str(artifacts_dir / "threshold" / "data" / "val_data_prob.npy"),
        threshold_classes_path=str(artifacts_dir / "threshold" / "data" / "val_data_classes.npy"),
        threshold_histogram_path=str(artifacts_dir / "threshold" / "plots" / "probability_histogram.png"),
        threshold_results_csv_path=str(artifacts_dir / "threshold" / "optimization" / "threshold_sweep.csv"),
        threshold_results_json_path=str(artifacts_dir / "threshold" / "optimization" / "recommended_thresholds.json"),
        threshold_active_json_path=str(artifacts_dir / "threshold" / "active_thresholds.json"),
        threshold_deployed_json_path=str(deployed_multipliers_path),
        multiplier_results_json_path=str(artifacts_dir / "multiplier" / "optimization" / "recommended_multipliers.json"),
        multiplier_active_json_path=str(artifacts_dir / "multiplier" / "active_multipliers.json"),
        multiplier_deployed_json_path=str(deployed_multipliers_path),
        train_misclassification_output_dir=str(gcam_base / "phase1" / "grad_cam"),
        validation_misclassification_output_dir=str(gcam_base / "validation" / "grad_cam"),
        grad_cam_output_dir=str(gcam_base / "test" / "grad_cam"),
        config_dir=str(config_dir),
        training_settings_path=str(config_dir / "training_settings.json"),
        backend_config_snapshot_path=str(config_dir / "backend_config_snapshot.json"),
    )


def apply_runtime_overrides(modules, dataset_root: str, runtime_paths: RuntimePaths, batch_size: int | None = None, image_size: tuple[int, int] | None = None):
    cfg = modules["model_config"]
    threshold_cfg = modules["config_threshold_optimization"]
    training_main = modules["train_multiclass_defect_classifier"]
    run_validation = modules["run_validation"]
    run_test = modules["run_test"]
    run_threshold_pipeline = modules["run_threshold_pipeline"]
    optimize_pass_threshold = modules["optimize_pass_threshold"]

    cfg.DATASET_DIR = dataset_root
    if batch_size is not None:
        cfg.BATCH_SIZE = batch_size
        run_test.BATCH_SIZE = batch_size
        run_threshold_pipeline.BATCH_SIZE = batch_size
    if image_size is not None:
        cfg.IMAGE_SIZE = image_size
        run_test.IMAGE_SIZE = image_size
        run_threshold_pipeline.IMAGE_SIZE = image_size

    cfg.ARTIFACTS_DIR = runtime_paths.artifacts_dir
    cfg.PHASE_1_FINAL_MODEL_PATH = runtime_paths.phase1_final_model_path
    cfg.PHASE_2_TRANSFER_MODEL_PATH = runtime_paths.phase2_transfer_model_path
    cfg.PHASE_2_INIT_CHECKPOINT_PATH = runtime_paths.phase1_final_model_path
    cfg.PHASE_2_TRANSFER_CONFUSION_MATRIX_PATH = runtime_paths.phase2_transfer_confusion_matrix_path
    cfg.PHASE_2_TRANSFER_TRAINING_RESULTS_PLOT_PATH = runtime_paths.phase2_transfer_training_results_plot_path
    cfg.PHASE_2_TRANSFER_SUMMARY_TXT_PATH = runtime_paths.phase2_transfer_summary_txt_path
    cfg.PHASE_2_TRANSFER_MISCLASSIFICATION_OUTPUT_DIR = runtime_paths.phase2_transfer_misclassification_output_dir
    cfg.TRAIN_CONFUSION_MATRIX_PATH = runtime_paths.train_confusion_matrix_path
    cfg.VALIDATION_CONFUSION_MATRIX_PATH = runtime_paths.validation_confusion_matrix_path
    cfg.TEST_CONFUSION_MATRIX_PATH = runtime_paths.test_confusion_matrix_path
    cfg.TRAINING_RESULTS_PLOT_PATH = runtime_paths.training_results_plot_path
    cfg.TRAINING_SUMMARY_TXT_PATH = runtime_paths.training_summary_txt_path
    cfg.VALIDATION_SUMMARY_TXT_PATH = runtime_paths.validation_summary_txt_path
    cfg.TEST_SUMMARY_TXT_PATH = runtime_paths.test_summary_txt_path
    cfg.TRAIN_MISCLASSIFICATION_OUTPUT_DIR = runtime_paths.train_misclassification_output_dir
    cfg.VALIDATION_MISCLASSIFICATION_OUTPUT_DIR = runtime_paths.validation_misclassification_output_dir
    cfg.GRAD_CAM_OUTPUT_DIR = runtime_paths.grad_cam_output_dir

    threshold_cfg.THRESHOLD_DATA_DIR = runtime_paths.threshold_data_dir
    threshold_cfg.VAL_TRUE_PATH = runtime_paths.threshold_true_path
    threshold_cfg.VAL_PROB_PATH = runtime_paths.threshold_prob_path
    threshold_cfg.VAL_CLASSES_PATH = runtime_paths.threshold_classes_path
    threshold_cfg.PROBABILITY_HISTOGRAM_PATH = runtime_paths.threshold_histogram_path
    threshold_cfg.DEFAULT_RESULTS_CSV_PATH = runtime_paths.threshold_results_csv_path
    threshold_cfg.DEPLOYED_MULTIPLIERS_JSON_PATH = runtime_paths.threshold_deployed_json_path

    training_main.DATASET_DIR = dataset_root
    training_main.PHASE_1_FINAL_MODEL_PATH = runtime_paths.phase1_final_model_path
    training_main.PHASE_2_TRANSFER_MODEL_PATH = runtime_paths.phase2_transfer_model_path
    training_main.PHASE_2_INIT_CHECKPOINT_PATH = runtime_paths.phase1_final_model_path
    training_main.PHASE_2_TRANSFER_CONFUSION_MATRIX_PATH = runtime_paths.phase2_transfer_confusion_matrix_path
    training_main.PHASE_2_TRANSFER_TRAINING_RESULTS_PLOT_PATH = runtime_paths.phase2_transfer_training_results_plot_path
    training_main.PHASE_2_TRANSFER_SUMMARY_TXT_PATH = runtime_paths.phase2_transfer_summary_txt_path
    training_main.PHASE_2_TRANSFER_MISCLASSIFICATION_OUTPUT_DIR = runtime_paths.phase2_transfer_misclassification_output_dir
    training_main.TRAIN_CONFUSION_MATRIX_PATH = runtime_paths.train_confusion_matrix_path
    training_main.TRAINING_RESULTS_PLOT_PATH = runtime_paths.training_results_plot_path
    training_main.TRAINING_SUMMARY_TXT_PATH = runtime_paths.training_summary_txt_path
    training_main.TRAIN_MISCLASSIFICATION_OUTPUT_DIR = runtime_paths.train_misclassification_output_dir
    training_main.PLOT_SETTINGS = training_main.build_plot_settings(
        model_name=training_main.MODEL_NAME,
        save_path=runtime_paths.train_confusion_matrix_path,
        colormap=training_main.COLORMAP,
        misclassification_output_dir=runtime_paths.train_misclassification_output_dir,
    )
    training_main.MISCLASSIFICATION_SETTINGS = training_main.PLOT_SETTINGS
    training_main.PHASE_2_TRANSFER_PLOT_SETTINGS = training_main.build_plot_settings(
        model_name=f"{training_main.MODEL_NAME} Phase 2 Transfer Learning",
        save_path=runtime_paths.phase2_transfer_confusion_matrix_path,
        colormap=training_main.COLORMAP,
        misclassification_output_dir=runtime_paths.phase2_transfer_misclassification_output_dir,
    )
    training_main.PHASE_2_TRANSFER_MISCLASSIFICATION_SETTINGS = training_main.PHASE_2_TRANSFER_PLOT_SETTINGS

    run_validation.VAL_TRUE_PATH = runtime_paths.threshold_true_path
    run_validation.VAL_PROB_PATH = runtime_paths.threshold_prob_path
    run_validation.VAL_CLASSES_PATH = runtime_paths.threshold_classes_path
    run_validation.DEPLOYED_MULTIPLIERS_JSON_PATH = runtime_paths.threshold_deployed_json_path

    run_test.PHASE_1_FINAL_MODEL_PATH = runtime_paths.phase1_final_model_path
    run_test.PHASE_2_TRANSFER_MODEL_PATH = runtime_paths.phase2_transfer_model_path
    run_test.TEST_SUMMARY_TXT_PATH = runtime_paths.test_summary_txt_path
    run_test.TEST_CONFUSION_MATRIX_PATH = runtime_paths.test_confusion_matrix_path
    run_test.GRAD_CAM_OUTPUT_DIR = runtime_paths.grad_cam_output_dir
    run_test.DEPLOYED_MULTIPLIERS_JSON_PATH = runtime_paths.threshold_deployed_json_path

    run_threshold_pipeline.DATASET_DIR = dataset_root
    run_threshold_pipeline.PHASE_1_FINAL_MODEL_PATH = runtime_paths.phase1_final_model_path
    run_threshold_pipeline.THRESHOLD_DATA_DIR = runtime_paths.threshold_data_dir
    run_threshold_pipeline.VAL_TRUE_PATH = runtime_paths.threshold_true_path
    run_threshold_pipeline.VAL_PROB_PATH = runtime_paths.threshold_prob_path
    run_threshold_pipeline.VAL_CLASSES_PATH = runtime_paths.threshold_classes_path
    run_threshold_pipeline.PROBABILITY_HISTOGRAM_PATH = runtime_paths.threshold_histogram_path
    run_threshold_pipeline.DEFAULT_RESULTS_CSV_PATH = runtime_paths.threshold_results_csv_path

    optimize_pass_threshold.VAL_TRUE_PATH = runtime_paths.threshold_true_path
    optimize_pass_threshold.VAL_PROB_PATH = runtime_paths.threshold_prob_path
    optimize_pass_threshold.VAL_CLASSES_PATH = runtime_paths.threshold_classes_path
    optimize_pass_threshold.DEFAULT_RESULTS_CSV_PATH = runtime_paths.threshold_results_csv_path
    optimize_pass_threshold.DEPLOYED_MULTIPLIERS_JSON_PATH = runtime_paths.threshold_deployed_json_path


@contextlib.contextmanager
def backend_logging(logger):
    writer = LoggerWriter(logger)
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        yield
    writer.flush()


@contextlib.contextmanager
def backend_cwd():
    original = Path.cwd()
    os.chdir(_backend_root())
    try:
        yield
    finally:
        os.chdir(original)


def write_json_file(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, default=str)


def build_backend_config_snapshot(cfg):
    names = [
        "MODEL_NAME",
        "DATASET_DIR",
        "PHASE_2_TRANSFER_DATASET_DIR",
        "IMAGE_SIZE",
        "BATCH_SIZE",
        "SPLIT_PERCENT",
        "HEAD_DROPOUT_RATE",
        "UNFREEZE_LAYERS",
        "HEAD_OPTIMIZER",
        "HEAD_LEARNING_RATE",
        "HEAD_FOCAL_GAMMA",
        "FINE_TUNE_FOCAL_GAMMA",
        "LABEL_SMOOTHING",
        "USE_CLASS_WEIGHTS",
        "FREEZE_BATCHNORM_STATS",
        "EARLY_STOPPING_PATIENCE",
        "LR_SCHEDULER_ENABLED",
        "LR_SCHEDULER_FACTOR",
        "LR_SCHEDULER_PATIENCE",
        "MIN_LEARNING_RATE",
        "WEIGHT_DECAY",
        "CROP_MIN_SCALE",
        "CROP_MAX_SCALE",
        "ROTATION_DEGREES",
        "HORIZONTAL_FLIP_PROBABILITY",
        "VERTICAL_FLIP_PROBABILITY",
        "BRIGHTNESS_JITTER",
        "CONTRAST_JITTER",
        "SATURATION_JITTER",
        "PHASE_1_FINE_TUNE_STAGES",
        "PHASE_2_TRANSFER_LEARNING_STAGES",
        "PHASE_1_FINAL_MODEL_PATH",
        "PHASE_2_TRANSFER_MODEL_PATH",
    ]
    return {name: getattr(cfg, name, None) for name in names}


def resolve_torch_device(device_value):
    import torch
    requested = str(device_value or "GPU").strip().lower()
    if requested in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was selected, but CUDA is not available. Select CPU or install a CUDA-enabled PyTorch build.")
        return torch.device("cuda")
    return torch.device("cpu")

def run_training_job(settings: dict, logger, stop_event=None, progress_callback=None):
    modules = import_backend_modules()
    dataset_root = resolve_dataset_root(settings["dataset_root"])
    selected_device = resolve_torch_device(settings.get("device", "GPU"))
    modules["utils_training"].DEVICE = selected_device
    modules["model_factory"].DEVICE = selected_device
    
    # Extract structural configs from your UI settings dictionary
    runtime_paths = build_runtime_paths(
        output_root=settings.get("artifacts_dir"),
        save_model_path=None,
        grad_cam_root=settings.get("grad_cam_outside_dir"),
        model_name=settings.get("model_name", "EfficientNetV2S"),
        model_version=settings.get("model_version", "v1"),
    )
    for directory in (
        runtime_paths.artifacts_dir,
        runtime_paths.checkpoints_dir,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)

    apply_runtime_overrides(
        modules,
        dataset_root=dataset_root,
        runtime_paths=runtime_paths,
        batch_size=settings.get("batch_size"),
        image_size=settings.get("image_size"),
    )

    cfg = modules["model_config"]
    cfg.MODEL_NAME = settings.get("model_name", cfg.MODEL_NAME)
    cfg.RUN_PHASE_1 = settings.get("run_phase_1", True)
    cfg.SPLIT_PERCENT = settings["split_percent"]
    cfg.PHASE_1_EPOCHS = settings["phase_1_epochs"]
    cfg.HEAD_DROPOUT_RATE = settings["dropout_rate"]
    cfg.UNFREEZE_LAYERS = settings["unfreeze_layers"]
    cfg.PHASE_1_FINE_TUNE_STAGES = settings["phase_1_stages"]
    cfg.HEAD_OPTIMIZER = settings["head_optimizer"]
    cfg.HEAD_LEARNING_RATE = settings["head_learning_rate"]
    cfg.HEAD_FOCAL_GAMMA = settings["head_focal_gamma"]
    cfg.FINE_TUNE_FOCAL_GAMMA = settings["fine_tune_focal_gamma"]
    cfg.LABEL_SMOOTHING = settings["label_smoothing"]
    cfg.USE_CLASS_WEIGHTS = settings["use_class_weights"]
    cfg.FREEZE_BATCHNORM_STATS = settings["freeze_batchnorm_stats"]
    cfg.EARLY_STOPPING_PATIENCE = settings["early_stopping_patience"]
    cfg.LR_SCHEDULER_ENABLED = settings["lr_scheduler_enabled"]
    cfg.LR_SCHEDULER_FACTOR = settings["lr_scheduler_factor"]
    cfg.LR_SCHEDULER_PATIENCE = settings["lr_scheduler_patience"]
    cfg.MIN_LEARNING_RATE = settings["min_learning_rate"]
    cfg.WEIGHT_DECAY = settings["weight_decay"]
    cfg.CROP_MIN_SCALE = settings["crop_min_scale"]
    cfg.CROP_MAX_SCALE = settings["crop_max_scale"]
    cfg.ROTATION_DEGREES = settings["rotation_degrees"]
    cfg.HORIZONTAL_FLIP_PROBABILITY = settings["horizontal_flip_probability"]
    cfg.VERTICAL_FLIP_PROBABILITY = settings["vertical_flip_probability"]
    cfg.BRIGHTNESS_JITTER = settings["brightness_jitter"]
    cfg.CONTRAST_JITTER = settings["contrast_jitter"]
    cfg.SATURATION_JITTER = settings["saturation_jitter"]
    cfg.RUN_PHASE_2_TRANSFER = settings.get("run_phase_2", False)
    cfg.PHASE_2_RUN = cfg.RUN_PHASE_2_TRANSFER
    phase_2_dataset_root = settings.get("phase_2_dataset_root", "")
    cfg.PHASE_2_TRANSFER_DATASET_DIR = resolve_dataset_root(phase_2_dataset_root) if phase_2_dataset_root else ""
    cfg.PHASE_2_DATASET_DIR = cfg.PHASE_2_TRANSFER_DATASET_DIR
    cfg.PHASE_2_TRANSFER_UNFREEZE_LAYERS = settings["phase_2_unfreeze_parameters"]
    cfg.PHASE_2_TRANSFER_LEARNING_STAGES = settings["phase_2_stages"]
    main_module = modules["train_multiclass_defect_classifier"]
    main_module.DEVICE = selected_device
    pretrained_model_path = str(settings.get("pretrained_model_path", "") or "").strip()
    main_module.PRETRAINED_MODEL_PATH = pretrained_model_path

    # Direct Phase 2 transfer learning must initialize from the checkpoint selected
    # in the Pretrained Checkpoint field. The runtime-generated Phase 1 path belongs to
    # the new model version and does not exist when Phase 1 is intentionally skipped.
    if cfg.RUN_PHASE_2_TRANSFER and not cfg.RUN_PHASE_1:
        if not pretrained_model_path:
            raise ValueError(
                "Pretrained Checkpoint is required when Phase 1 is disabled and "
                "Phase 2 transfer learning is enabled."
            )
        pretrained_checkpoint = Path(pretrained_model_path).expanduser()
        if not pretrained_checkpoint.is_file():
            raise FileNotFoundError(
                f"Selected Pretrained Checkpoint does not exist: {pretrained_checkpoint}"
            )
        cfg.PHASE_2_INIT_CHECKPOINT_PATH = str(pretrained_checkpoint)
        main_module.PHASE_2_INIT_CHECKPOINT_PATH = str(pretrained_checkpoint)
        logger.log(
            f"[Checkpoint] Phase 2 initialization checkpoint: {pretrained_checkpoint}"
        )
    else:
        cfg.PHASE_2_INIT_CHECKPOINT_PATH = runtime_paths.phase1_final_model_path
        main_module.PHASE_2_INIT_CHECKPOINT_PATH = runtime_paths.phase1_final_model_path

    main_module.EXPORT_MISCLASSIFICATION_IMAGES = bool(settings.get("export_grad_cam", False))
    logger.log(f"[Runtime] Training device: {selected_device.type.upper()}")
    main_module.MODEL_NAME = cfg.MODEL_NAME
    main_module.RUN_PHASE_1 = cfg.RUN_PHASE_1
    main_module.SPLIT_PERCENT = cfg.SPLIT_PERCENT
    main_module.PHASE_1_EPOCHS = cfg.PHASE_1_EPOCHS
    main_module.HEAD_DROPOUT_RATE = cfg.HEAD_DROPOUT_RATE
    main_module.UNFREEZE_LAYERS = cfg.UNFREEZE_LAYERS
    main_module.PHASE_1_FINE_TUNE_STAGES = cfg.PHASE_1_FINE_TUNE_STAGES
    main_module.RUN_PHASE_2_TRANSFER = cfg.RUN_PHASE_2_TRANSFER
    main_module.PHASE_2_TRANSFER_DATASET_DIR = cfg.PHASE_2_TRANSFER_DATASET_DIR
    main_module.PHASE_2_TRANSFER_UNFREEZE_LAYERS = cfg.PHASE_2_TRANSFER_UNFREEZE_LAYERS
    main_module.PHASE_2_TRANSFER_LEARNING_STAGES = cfg.PHASE_2_TRANSFER_LEARNING_STAGES
    for name in (
        "SPLIT_PERCENT",
        "HEAD_LEARNING_RATE",
        "HEAD_OPTIMIZER",
        "HEAD_FOCAL_GAMMA",
        "FINE_TUNE_FOCAL_GAMMA",
        "LABEL_SMOOTHING",
        "USE_CLASS_WEIGHTS",
        "FREEZE_BATCHNORM_STATS",
        "EARLY_STOPPING_PATIENCE",
        "LR_SCHEDULER_ENABLED",
        "LR_SCHEDULER_FACTOR",
        "LR_SCHEDULER_PATIENCE",
        "MIN_LEARNING_RATE",
        "WEIGHT_DECAY",
        "CROP_MIN_SCALE",
        "CROP_MAX_SCALE",
        "ROTATION_DEGREES",
        "HORIZONTAL_FLIP_PROBABILITY",
        "VERTICAL_FLIP_PROBABILITY",
        "BRIGHTNESS_JITTER",
        "CONTRAST_JITTER",
        "SATURATION_JITTER",
    ):
        setattr(main_module, name, getattr(cfg, name))
    main_module.PLOT_SETTINGS = main_module.build_plot_settings(
        model_name=main_module.MODEL_NAME,
        save_path=runtime_paths.train_confusion_matrix_path,
        colormap=main_module.COLORMAP,
        misclassification_output_dir=runtime_paths.train_misclassification_output_dir,
    )
    main_module.MISCLASSIFICATION_SETTINGS = main_module.PLOT_SETTINGS
    main_module.PHASE_2_TRANSFER_PLOT_SETTINGS = main_module.build_plot_settings(
        model_name=f"{main_module.MODEL_NAME} Phase 2 Transfer Learning",
        save_path=runtime_paths.phase2_transfer_confusion_matrix_path,
        colormap=main_module.COLORMAP,
        misclassification_output_dir=runtime_paths.phase2_transfer_misclassification_output_dir,
    )
    main_module.PHASE_2_TRANSFER_MISCLASSIFICATION_SETTINGS = main_module.PHASE_2_TRANSFER_PLOT_SETTINGS

    completed_epochs = 0
    total_epochs = settings.get("planned_total_epochs", 0) or 1

    def handle_training_event(event):
        nonlocal completed_epochs
        if event.get("type") == "epoch":
            completed_epochs += 1
            event["completed_epochs"] = completed_epochs
            event["total_epochs"] = total_epochs
            event["progress"] = min(100.0, (completed_epochs / total_epochs) * 100.0)
        if progress_callback:
            progress_callback(event)

    try:
        with backend_cwd(), backend_logging(logger):
            main_module.main(event_callback=handle_training_event, stop_event=stop_event)
    except Exception as exc:
        if exc.__class__.__name__ == "TrainingStopped":
            return {
                "stopped": True,
                "final_model_path": runtime_paths.phase1_final_model_path,
                "summary_path": runtime_paths.training_summary_txt_path,
                "artifacts_dir": runtime_paths.artifacts_dir,
                "completed_epochs": completed_epochs,
                "total_epochs": total_epochs,
            }
        raise

    final_model_path = (
        runtime_paths.phase2_transfer_model_path
        if settings.get("run_phase_2", False)
        else runtime_paths.phase1_final_model_path
    )
    summary_path = (
        runtime_paths.phase2_transfer_summary_txt_path
        if settings.get("run_phase_2", False)
        else runtime_paths.training_summary_txt_path
    )

    return {
        "final_model_path": final_model_path,
        "summary_path": summary_path,
        "artifacts_dir": runtime_paths.artifacts_dir,
    }


def _should_append_run_folder(settings: dict, checkpoint_path: str) -> bool:
    model_name = settings.get("model_name", "EfficientNetV2S")
    model_version = settings.get("model_version", "v1")
    run_name = (
        f"{sanitize_path_part(model_name, 'model')}_"
        f"{sanitize_path_part(model_version, 'v1')}"
    )
    output_root = settings.get("artifacts_dir")
    if output_root:
        output_path = Path(output_root).expanduser()
        if output_path.name == run_name:
            return False

    checkpoint = Path(checkpoint_path).expanduser()
    if output_root and checkpoint.parent.name == "checkpoints" and checkpoint.parent.parent.name == run_name:
        try:
            if Path(output_root).expanduser().resolve() == checkpoint.parent.parent.resolve():
                return False
        except OSError:
            pass

    return bool(settings.get("append_run_folder", True))


def _build_threshold_runtime_paths(settings: dict, checkpoint_path: str) -> RuntimePaths:
    return build_runtime_paths(
        output_root=settings.get("artifacts_dir"),
        save_model_path=checkpoint_path,
        model_name=settings.get("model_name", "EfficientNetV2S"),
        model_version=settings.get("model_version", "v1"),
        append_run_folder=_should_append_run_folder(settings, checkpoint_path),
    )


def run_threshold_sweep(settings: dict, logger):
    """Optimize per-class thresholds from cached evaluation predictions."""
    from backend.prediction_cache import PredictionResultCache
    from backend.threshold_service import optimize_thresholds

    # The prediction cache is tied to the exact directory evaluated (for example,
    # <dataset>/val). Do not normalize a val directory back to its parent here,
    # otherwise cache identity validation compares different roots and incorrectly
    # reports that the dataset changed.
    dataset_root = require_path(
        settings["dataset_root"],
        label="Threshold dataset directory",
        must_exist=True,
    )
    checkpoint_path = require_path(settings.get("model_checkpoint"), label="Model checkpoint", must_exist=True)
    runtime_paths = _build_threshold_runtime_paths(settings, checkpoint_path)
    cache_path = Path(settings.get("prediction_cache_path") or runtime_paths.evaluation_predictions_path)
    prediction_cache = PredictionResultCache.read_jsonl(cache_path)
    prediction_cache.validate_identity(
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        class_names=prediction_cache.metadata.class_names,
    )
    result = optimize_thresholds(
        prediction_cache,
        prediction_cache.metadata.class_names,
        strategy=settings.get("threshold_strategy", "best_f1"),
    )
    output_path = Path(runtime_paths.threshold_results_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(output_path)
    logger.log(f"[Threshold] Loaded cached predictions: {cache_path}")
    logger.log(f"[Threshold] Recommended thresholds saved: {output_path}")
    for warning in result.warnings:
        logger.log(f"[Warning] {warning}")
    return {
        "thresholds": result.to_dict(),
        "metrics": _threshold_result_as_legacy_metrics(result.to_dict()),
        "results_json_path": str(output_path),
        "prediction_cache_path": str(cache_path),
        "deployed_json_path": runtime_paths.threshold_deployed_json_path,
    }


def generate_threshold_recommendations(settings: dict, logger):
    from backend.threshold_persistence import build_threshold_config, load_threshold_config

    result = run_threshold_sweep(settings, logger)
    checkpoint_path = require_path(settings.get("model_checkpoint"), label="Model checkpoint", must_exist=True)
    runtime_paths = _build_threshold_runtime_paths(settings, checkpoint_path)
    active_path = runtime_paths.threshold_active_json_path
    saved_config = load_threshold_config(active_path)
    config = build_threshold_config(
        recommendations=result["thresholds"],
        checkpoint=checkpoint_path,
        saved_config=saved_config,
    )
    logger.log(f"[Threshold] Recommended thresholds ready for review: {active_path}")
    result["active_thresholds"] = config
    result["active_thresholds_path"] = active_path
    return result



def generate_multiplier_recommendations(settings: dict, logger):
    """Optimize defect multipliers from the same cached validation predictions."""
    from backend.multiplier_persistence import build_multiplier_config, load_multiplier_config
    from backend.multiplier_service import optimize_multipliers
    from backend.prediction_cache import PredictionResultCache

    dataset_root = require_path(settings["dataset_root"], label="Multiplier dataset directory", must_exist=True)
    checkpoint_path = require_path(settings.get("model_checkpoint"), label="Model checkpoint", must_exist=True)
    runtime_paths = _build_threshold_runtime_paths(settings, checkpoint_path)
    cache_path = Path(settings.get("prediction_cache_path") or runtime_paths.evaluation_predictions_path)
    cache = PredictionResultCache.read_jsonl(cache_path)
    cache.validate_identity(checkpoint_path=checkpoint_path, dataset_root=dataset_root, class_names=cache.metadata.class_names)
    options = dict(settings.get("multiplier_optimizer_options") or {})
    result = optimize_multipliers(cache, cache.metadata.class_names, **options)
    recommendations_path = Path(runtime_paths.multiplier_results_json_path)
    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    temp = recommendations_path.with_suffix(recommendations_path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(recommendations_path)
    active_path = runtime_paths.multiplier_active_json_path
    saved = load_multiplier_config(active_path)
    config = build_multiplier_config(recommendations=result, checkpoint=checkpoint_path, saved_config=saved)
    logger.log(f"[Multiplier] Loaded cached predictions: {cache_path}")
    logger.log(f"[Multiplier] Recommended multipliers ready for review: {recommendations_path}")
    return {
        "multipliers": result,
        "active_multipliers": config,
        "active_multipliers_path": active_path,
        "recommendations_path": str(recommendations_path),
        "deployed_multipliers_path": runtime_paths.multiplier_deployed_json_path,
        "prediction_cache_path": str(cache_path),
    }

def _threshold_result_as_legacy_metrics(result: dict) -> dict:
    classes = result.get("classes", {})
    return {
        "thresholds": [
            class_result.get("recommended")
            for class_result in classes.values()
        ],
        "threshold_classes": list(classes.keys()),
        "warnings": list(result.get("warnings", [])),
        "f1_score": max(
            [
                float(class_result.get("metric_value"))
                for class_result in classes.values()
                if class_result.get("metric_value") is not None
            ]
            or [0.0]
        ),
    }


def run_validation_job(settings: dict, logger):
    modules = import_backend_modules()
    dataset_root = resolve_dataset_root(settings["dataset_root"])
    checkpoint_path = require_path(
        settings.get("model_checkpoint") or settings.get("save_model_path"),
        label="Model checkpoint",
        must_exist=True,
    )

    # Extract structural configs from your UI settings dictionary
    runtime_paths = build_runtime_paths(
        output_root=settings.get("artifacts_dir") or settings.get("output_dir"),
        save_model_path=checkpoint_path,
        grad_cam_root=settings.get("grad_cam_outside_dir"),
        model_name=settings.get("model_name", "EfficientNetV2S"),
        model_version=settings.get("model_version", "v1"),
        append_run_folder=bool(settings.get("append_run_folder", False)),
    )
    apply_runtime_overrides(
        modules,
        dataset_root=dataset_root,
        runtime_paths=runtime_paths,
        batch_size=settings.get("batch_size"),
        image_size=settings.get("image_size"),
    )

    run_validation_module = modules["run_validation"]
    run_validation_module.DEPLOYED_MULTIPLIERS_JSON_PATH = runtime_paths.multiplier_deployed_json_path
    run_validation_module.EXPORT_MISCLASSIFICATION_IMAGES = bool(settings.get("export_grad_cam", False))

    with backend_cwd(), backend_logging(logger):
        metrics = modules["run_validation"].run_validation(model_path=checkpoint_path)

    return {
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "escape_rate": float(metrics["escape_rate"] if metrics["escape_rate"] is not None else 1.0),
            "false_alarm_rate": float(metrics["false_alarm_rate"] if metrics.get("false_alarm_rate") is not None else 0.0),
        },
        "summary_path": runtime_paths.validation_summary_txt_path,
        "confusion_matrix_path": runtime_paths.validation_confusion_matrix_path,
        "grad_cam_dir": runtime_paths.validation_misclassification_output_dir,
        "report_path": str(Path(runtime_paths.validation_summary_txt_path).with_name(Path(runtime_paths.validation_summary_txt_path).stem + "_report.json")),
        "dataset_dir": str(Path(dataset_root) / "val") if (Path(dataset_root) / "val").is_dir() else dataset_root,
    }


def run_test_job(settings: dict, logger, progress_callback=None):
    modules = import_backend_modules()
    dataset_root = resolve_dataset_root(settings["dataset_root"])
    checkpoint_path = require_path(
        settings.get("model_checkpoint") or settings.get("save_model_path"),
        label="Model checkpoint",
        must_exist=True,
    )
    test_dir = require_path(
        settings.get("test_dir"),
        label="Test dataset directory",
        must_exist=True,
    )

    # Extract structural configs from your UI settings dictionary
    runtime_paths = build_runtime_paths(
        output_root=settings.get("artifacts_dir") or settings.get("output_dir"),
        save_model_path=checkpoint_path,
        grad_cam_root=settings.get("grad_cam_outside_dir"),
        append_run_folder=False,
    )
    apply_runtime_overrides(
        modules,
        dataset_root=dataset_root,
        runtime_paths=runtime_paths,
        batch_size=settings.get("batch_size"),
        image_size=settings.get("image_size"),
    )
    selected_device = resolve_torch_device(settings.get("device", "GPU"))
    modules["utils_training"].DEVICE = selected_device
    modules["model_factory"].DEVICE = selected_device
    modules["run_test"].DEVICE = selected_device
    modules["run_test"].MODEL_NAME = settings.get("model_name", modules["run_test"].MODEL_NAME)
    modules["run_test"].HEAD_DROPOUT_RATE = settings.get("dropout_rate", modules["run_test"].HEAD_DROPOUT_RATE)
    modules["run_test"].PREDICTION_CACHE_PATH = runtime_paths.evaluation_predictions_path
    logger.log(f"[Runtime] Evaluation device: {selected_device.type.upper()}")

    with backend_cwd(), backend_logging(logger):
        run_details = modules["run_test"].run_test(
            test_dir=test_dir,
            model_path=checkpoint_path,
            progress_callback=progress_callback,
            skip_grad_cam=bool(settings.get("skip_grad_cam", True)),
        )

    return {
        "summary_path": runtime_paths.test_summary_txt_path,
        "confusion_matrix_path": runtime_paths.test_confusion_matrix_path,
        "grad_cam_dir": runtime_paths.grad_cam_output_dir,
        "report_path": str(Path(runtime_paths.test_summary_txt_path).with_name(Path(runtime_paths.test_summary_txt_path).stem + "_report.json")),
        "dataset_dir": test_dir,
        "grad_cam_skipped": bool(run_details.get("grad_cam_skipped", False)),
        "sample_count": int(run_details.get("sample_count", 0)),
        "accuracy": float(run_details.get("accuracy", 0.0)),
        "macro_precision": float(run_details.get("macro_precision", 0.0)),
        "macro_recall": float(run_details.get("macro_recall", 0.0)),
        "macro_f1": float(run_details.get("macro_f1", 0.0)),
        "misclassified_count": int(run_details.get("misclassified_count", 0)),
        "metrics": run_details.get("metrics", {}),
        "prediction_cache_path": run_details.get("prediction_cache_path") or runtime_paths.evaluation_predictions_path,
    }


def resource_path(relative_path):
    """Get an absolute path to a project resource."""
    base_path = Path(__file__).resolve().parents[1]
    return os.path.join(base_path, relative_path)
