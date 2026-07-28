from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.path_validation import require_path
from backend.prediction_cache import build_prediction_cache


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class EvaluationConfig:
    test_dir: str
    model_path: str
    model_name: str
    image_size: tuple[int, int]
    batch_size: int
    device: Any
    head_dropout_rate: float
    summary_output_path: str
    confusion_matrix_path: str
    grad_cam_output_dir: str
    deployed_multipliers_json_path: str
    prediction_cache_path: str | None = None
    colormap: str = "Blues"


@dataclass(frozen=True)
class EvaluationResult:
    sample_count: int
    class_names: list[str]
    grad_cam_skipped: bool
    metrics: dict[str, Any]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    misclassified_count: int
    prediction_cache_path: str | None


def _default_dependencies() -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from backend.config_threshold_optimization import load_multipliers_from_json
    from backend.model_factory import build_model
    from backend.utils_grad_cam import export_test_set_grad_cam_examples
    from backend.utils_misclassification import collect_predictions
    from backend.utils_plotting import build_plot_settings, plot_confusion_matrix
    from backend.utils_training import evaluate_model, load_checkpoint_or_exit

    return {
        "DataLoader": DataLoader,
        "datasets": datasets,
        "transforms": transforms,
        "load_multipliers_from_json": load_multipliers_from_json,
        "build_model": build_model,
        "load_checkpoint_or_exit": load_checkpoint_or_exit,
        "collect_predictions": collect_predictions,
        "evaluate_model": evaluate_model,
        "build_plot_settings": build_plot_settings,
        "plot_confusion_matrix": plot_confusion_matrix,
        "export_test_set_grad_cam_examples": export_test_set_grad_cam_examples,
    }


def _report(progress_callback: ProgressCallback | None, percent: float, text: str) -> None:
    print(text)
    if progress_callback:
        progress_callback(float(percent), str(text))


def _evaluation_settings(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "optimizer": "test-evaluation",
        "learning_rate": 0.0,
        "epochs": 0,
        "unfreeze_layers": 0,
        "batch_size": config.batch_size,
        "dropout_rate": config.head_dropout_rate,
    }


def _mean_metric(metrics: dict[str, Any], key: str) -> float:
    rows = metrics.get("per_class") or []
    if not rows:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def _misclassified_count(predictions: dict[str, Any]) -> int:
    true_labels = predictions.get("true_labels", [])
    pred_labels = predictions.get("pred_labels", [])
    return int(sum(1 for true, pred in zip(true_labels, pred_labels) if true != pred))


def run_evaluation_pipeline(
    config: EvaluationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    skip_grad_cam: bool = True,
    dependencies: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Run model evaluation once and reuse predictions for all downstream outputs."""
    deps = dependencies or _default_dependencies()

    test_dir = require_path(config.test_dir, label="Test dataset directory", must_exist=True)
    model_path = require_path(config.model_path, label="Model checkpoint", must_exist=True)

    _report(progress_callback, 2, "[Evaluation 1/4] Loading test dataset...")
    normalize = deps["transforms"].Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    test_transform = deps["transforms"].Compose(
        [
            deps["transforms"].Resize(config.image_size),
            deps["transforms"].ToTensor(),
            normalize,
        ]
    )
    test_dataset = deps["datasets"].ImageFolder(test_dir, transform=test_transform)
    test_loader = deps["DataLoader"](
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.device.type == "cuda"),
    )
    class_names = list(test_dataset.classes)
    _report(
        progress_callback,
        5,
        f"[Dataset] Found {len(test_dataset)} image(s) in {len(class_names)} class(es).",
    )

    _report(progress_callback, 7, f"[Evaluation 1/4] Loading {config.model_name} onto {config.device}...")
    model = deps["build_model"](len(class_names), config.head_dropout_rate)
    model = deps["load_checkpoint_or_exit"](
        model,
        model_path,
        config.model_name,
        checkpoint_label="final Phase 1 checkpoint",
        device=config.device,
    )

    active_multipliers = deps["load_multipliers_from_json"](
        config.deployed_multipliers_json_path,
        class_names,
    )
    _report(progress_callback, 10, f"[Evaluation 1/4] Using deployed defect multipliers: {active_multipliers}")

    _report(progress_callback, 12, "Evaluation 1/4 — Running validation inference")

    def batch_progress(batch_index: int, total_batches: int) -> None:
        percent = 12 + (batch_index / max(total_batches, 1)) * 58
        _report(progress_callback, percent, f"Evaluation 1/4 — Running validation inference ({batch_index}/{total_batches})")

    predictions = deps["collect_predictions"](
        model,
        test_loader,
        class_names,
        file_paths=[path for path, _ in getattr(test_dataset, "samples", [])],
        defect_multipliers=active_multipliers,
        progress_callback=batch_progress,
    )
    prediction_cache = build_prediction_cache(
        predictions=predictions,
        class_names=class_names,
        checkpoint_path=model_path,
        dataset_root=test_dir,
        model_name=config.model_name,
        image_size=config.image_size,
        preprocessing={
            "resize": list(config.image_size),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "defect_multipliers": active_multipliers,
        },
    )
    if config.prediction_cache_path:
        prediction_cache.write_jsonl(config.prediction_cache_path)
    predictions = prediction_cache.to_legacy_predictions(class_names)

    settings = _evaluation_settings(config)
    _report(progress_callback, 72, "Evaluation 2/4 — Calculating metrics")
    metrics = deps["evaluate_model"](
        model,
        test_loader,
        class_names,
        settings,
        model_name=f"{config.model_name} (Test Set)",
        defect_multipliers=active_multipliers,
        summary_output_path=config.summary_output_path,
        predictions=predictions,
    )

    _report(progress_callback, 80, "Evaluation 3/4 — Building confusion matrix")
    test_plot_settings = deps["build_plot_settings"](
        model_name=f"{config.model_name} (Test Set)",
        save_path=config.confusion_matrix_path,
        colormap=config.colormap,
        misclassification_output_dir=config.grad_cam_output_dir,
    )
    deps["plot_confusion_matrix"](
        model,
        test_loader,
        class_names,
        test_plot_settings,
        defect_multipliers=active_multipliers,
        predictions=predictions,
    )

    if skip_grad_cam:
        _report(progress_callback, 100, "Evaluation 4/4 — Generating report")
    else:
        _report(progress_callback, 88, "Evaluation 4/4 — Generating report")
        deps["export_test_set_grad_cam_examples"](
            model,
            test_loader.dataset,
            class_names,
            predictions,
            output_dir=config.grad_cam_output_dir,
            target_filenames=[],
        )
        _report(progress_callback, 100, "Evaluation 4/4 — Generating report")

    metrics = metrics or {}

    return EvaluationResult(
        sample_count=len(test_dataset),
        class_names=class_names,
        grad_cam_skipped=bool(skip_grad_cam),
        metrics=metrics,
        accuracy=float(metrics.get("accuracy", 0.0)),
        macro_precision=_mean_metric(metrics, "precision"),
        macro_recall=_mean_metric(metrics, "recall"),
        macro_f1=_mean_metric(metrics, "f1"),
        misclassified_count=_misclassified_count(predictions),
        prediction_cache_path=config.prediction_cache_path,
    )
