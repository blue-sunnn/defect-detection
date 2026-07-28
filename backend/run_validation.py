import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config_threshold_optimization import (
    DEPLOYED_MULTIPLIERS_JSON_PATH,
    load_multipliers_from_json,
)
from backend.path_validation import require_path
from backend.utils_training import DEVICE, evaluate_model, load_checkpoint_or_exit, load_datasets

EXPORT_MISCLASSIFICATION_IMAGES = False


def run_validation(model_type="efficientnetv2s", model_path=None):
    if model_type != "efficientnetv2s":
        raise ValueError(f"Unsupported model_type: {model_type}")

    from backend.model_config import (
        BATCH_SIZE,
        COLORMAP,
        DATASET_DIR,
        HEAD_DROPOUT_RATE,
        IMAGE_SIZE,
        MODEL_NAME,
        VALIDATION_CONFUSION_MATRIX_PATH,
        VALIDATION_MISCLASSIFICATION_OUTPUT_DIR,
        VALIDATION_SUMMARY_TXT_PATH,
    )
    from backend.model_factory import build_model
    from backend.utils_misclassification import export_false_positive_false_negative_examples
    from backend.utils_plotting import build_plot_settings, plot_confusion_matrix

    model_path = require_path(model_path, label="Model checkpoint", must_exist=True)
    dataset_dir = require_path(DATASET_DIR, label="Validation dataset root", must_exist=True)

    validation_plot_settings = build_plot_settings(
        model_name=MODEL_NAME,
        save_path=VALIDATION_CONFUSION_MATRIX_PATH,
        colormap=COLORMAP,
        misclassification_output_dir=VALIDATION_MISCLASSIFICATION_OUTPUT_DIR,
    )

    print(f"Rerunning validation for {MODEL_NAME} on device: {DEVICE}")

    _, val_loader, class_names, val_file_paths = load_datasets(
        dataset_dir,
        IMAGE_SIZE,
        BATCH_SIZE,
    )
    active_multipliers = load_multipliers_from_json(DEPLOYED_MULTIPLIERS_JSON_PATH, class_names)
    print(f"Using deployed defect multipliers: {active_multipliers}")

    model = build_model(len(class_names), HEAD_DROPOUT_RATE)
    model = load_checkpoint_or_exit(
        model,
        model_path,
        MODEL_NAME,
        checkpoint_label="final Phase 1 checkpoint",
        device=DEVICE,
    )

    settings = {
        "optimizer": "validation-only",
        "learning_rate": 0.0,
        "epochs": 0,
        "unfreeze_layers": 0,
        "batch_size": BATCH_SIZE,
        "dropout_rate": HEAD_DROPOUT_RATE,
    }

    metrics = evaluate_model(
        model,
        val_loader,
        class_names,
        settings,
        model_name=MODEL_NAME,
        defect_multipliers=active_multipliers,
        summary_output_path=VALIDATION_SUMMARY_TXT_PATH,
    )
    plot_confusion_matrix(
        model,
        val_loader,
        class_names,
        validation_plot_settings,
        defect_multipliers=active_multipliers,
    )
    if EXPORT_MISCLASSIFICATION_IMAGES:
        export_false_positive_false_negative_examples(
            model,
            val_loader,
            class_names,
            val_file_paths,
            validation_plot_settings,
            defect_multipliers=active_multipliers,
        )
    return metrics


if __name__ == "__main__":
    run_validation(model_type="efficientnetv2s", model_path=None)
