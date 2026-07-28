import os
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config_threshold_optimization import (
    DEFAULT_RESULTS_CSV_PATH,
    PROBABILITY_HISTOGRAM_PATH,
    THRESHOLD_DATA_DIR,
    VAL_CLASSES_PATH,
    VAL_PROB_PATH,
    VAL_TRUE_PATH,
)
from backend.model_config import (
    BATCH_SIZE,
    DATASET_DIR,
    HEAD_DROPOUT_RATE,
    IMAGE_SIZE,
)
from backend.model_factory import build_model
from backend.path_validation import require_path
from backend.optimize_pass_threshold import run_optimization_sweep
from backend.utils_plotting import plot_pass_probability_histogram
from backend.utils_training import DEVICE, load_datasets


def run_pipeline(
    model_type="efficientnetv2s",
    model_path=None,
    results_csv_name=DEFAULT_RESULTS_CSV_PATH,
):
    model_path = require_path(model_path, label="Model checkpoint", must_exist=True)
    dataset_dir = require_path(DATASET_DIR, label="Threshold dataset root", must_exist=True)

    print("=== Step 1/3: Export validation predictions ===")
    train_loader, validation_loader, class_names, _ = load_datasets(
        dataset_dir,
        IMAGE_SIZE,
        BATCH_SIZE,
    )
    del train_loader

    num_classes = len(class_names)
    model = build_model(num_classes, HEAD_DROPOUT_RATE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find checkpoint '{model_path}'. Train the model first.")

    print(f"Loading EfficientNetV2S checkpoint onto {DEVICE} from: {model_path}")
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_prob = []

    print("Extracting validation probabilities...")
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            y_prob.append(probabilities.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    true_labels = np.concatenate(y_true, axis=0)
    prob_matrix = np.concatenate(y_prob, axis=0)

    os.makedirs(THRESHOLD_DATA_DIR, exist_ok=True)
    export_result = {
        "true_path": VAL_TRUE_PATH,
        "prob_path": VAL_PROB_PATH,
        "classes_path": VAL_CLASSES_PATH,
    }

    np.save(export_result["true_path"], true_labels)
    np.save(export_result["prob_path"], prob_matrix)
    np.save(export_result["classes_path"], np.array(class_names, dtype=object))

    print(f"Export complete: labels shape={true_labels.shape}, prob shape={prob_matrix.shape}")

    print("\n=== Step 2/3: Plot Pass probability histogram ===")
    plot_pass_probability_histogram(
        true_path=export_result["true_path"],
        prob_path=export_result["prob_path"],
        classes_path=export_result["classes_path"],
        output_path=PROBABILITY_HISTOGRAM_PATH,
    )

    print("\n=== Step 3/3: Optimize per-defect multipliers ===")
    best_result = run_optimization_sweep(
        true_path=export_result["true_path"],
        prob_path=export_result["prob_path"],
        classes_path=export_result["classes_path"],
        results_csv_path=results_csv_name,
    )

    print("\nPipeline complete.")
    print(f"Threshold data directory: {THRESHOLD_DATA_DIR}")
    if results_csv_name:
        print(f"Sweep CSV: {results_csv_name}")
        
    return best_result


if __name__ == "__main__":
    run_pipeline()
