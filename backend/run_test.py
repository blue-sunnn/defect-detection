import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config_threshold_optimization import DEPLOYED_MULTIPLIERS_JSON_PATH
from backend.evaluation_service import EvaluationConfig, run_evaluation_pipeline
from backend.model_config import (
    BATCH_SIZE,
    GRAD_CAM_OUTPUT_DIR,
    HEAD_DROPOUT_RATE,
    IMAGE_SIZE,
    MODEL_NAME,
    TEST_CONFUSION_MATRIX_PATH,
    TEST_SUMMARY_TXT_PATH,
)

DEVICE = None
PREDICTION_CACHE_PATH = None


def _resolve_device():
    global DEVICE
    if DEVICE is None:
        from backend.utils_training import DEVICE as default_device

        DEVICE = default_device
    return DEVICE


def run_test(test_dir=None, model_path=None, progress_callback=None, skip_grad_cam=True):
    result = run_evaluation_pipeline(
        EvaluationConfig(
            test_dir=test_dir,
            model_path=model_path,
            model_name=MODEL_NAME,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            device=_resolve_device(),
            head_dropout_rate=HEAD_DROPOUT_RATE,
            summary_output_path=TEST_SUMMARY_TXT_PATH,
            confusion_matrix_path=TEST_CONFUSION_MATRIX_PATH,
            grad_cam_output_dir=GRAD_CAM_OUTPUT_DIR,
            deployed_multipliers_json_path=DEPLOYED_MULTIPLIERS_JSON_PATH,
            prediction_cache_path=PREDICTION_CACHE_PATH,
        ),
        progress_callback=progress_callback,
        skip_grad_cam=skip_grad_cam,
    )
    return {
        "sample_count": result.sample_count,
        "class_names": result.class_names,
        "grad_cam_skipped": result.grad_cam_skipped,
        "metrics": result.metrics,
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "misclassified_count": result.misclassified_count,
        "prediction_cache_path": result.prediction_cache_path,
    }


if __name__ == "__main__":
    run_test()
