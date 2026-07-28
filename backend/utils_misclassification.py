import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config_threshold_optimization import (
    DEPLOYED_MULTIPLIERS_JSON_PATH,
    PASS_CERTAINTY_THRESHOLD,
    PASS_CLASS_NAME,
    load_multipliers_from_json,
)


def apply_defect_multipliers(probabilities, class_names, defect_multipliers=None):
    """Boost or shrink defect probabilities independently before selecting the final class."""
    if PASS_CLASS_NAME not in class_names:
        raise ValueError(f"Required class '{PASS_CLASS_NAME}' not found in class_names: {class_names}")

    # FIX: Only read from disk if multipliers were not passed into the function
    if defect_multipliers is None:
        json_path = DEPLOYED_MULTIPLIERS_JSON_PATH
        defect_multipliers = load_multipliers_from_json(json_path, class_names)
        
    if all(abs(mult) < 1e-5 for mult in defect_multipliers.values()):
        return probabilities
    # -----------------------

    if isinstance(probabilities, torch.Tensor):
        adjusted_probabilities = probabilities.clone()
    else:
        adjusted_probabilities = np.array(probabilities, copy=True)

    for defect_name, multiplier in defect_multipliers.items():
        if defect_name == PASS_CLASS_NAME or defect_name not in class_names:
            continue

        # Allow negative bounds down to -0.999
        safe_multiplier = float(np.clip(multiplier, -0.999, 0.999))
        if abs(safe_multiplier) < 1e-5:
            continue

        defect_idx = class_names.index(defect_name)
        adjusted_probabilities[:, defect_idx] /= 1.0 - safe_multiplier

    if isinstance(adjusted_probabilities, torch.Tensor):
        normalization = adjusted_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        normalization = np.clip(adjusted_probabilities.sum(axis=1, keepdims=True), 1e-12, None)

    adjusted_probabilities = adjusted_probabilities / normalization
    return adjusted_probabilities


# This helper centralizes inference-time post-processing so validation,
# confusion-matrix reporting, and production prediction all follow one policy.
def predict_with_defect_multipliers(outputs, class_names, defect_multipliers=None):
    """Convert logits to deployed predictions and confidence scores."""
    probabilities = F.softmax(outputs, dim=1)
    adjusted_probabilities = apply_defect_multipliers(
        probabilities,
        class_names,
        defect_multipliers=defect_multipliers,
    )
    pred_indices = torch.argmax(adjusted_probabilities, dim=1)
    confidences = torch.max(adjusted_probabilities, dim=1).values
    return adjusted_probabilities, pred_indices, confidences

def collect_predictions(
    model,
    dataloader,
    class_names,
    file_paths=None,
    defect_multipliers=None,
    use_tta=True,
    progress_callback=None,
):
    """Collect true labels, predicted labels, and file paths with optional TTA."""
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []
    all_pass_probabilities = []
    all_probabilities = []
    all_source_paths = []

    model.eval()
    device = next(model.parameters()).device
    pass_idx = class_names.index("Pass") if "Pass" in class_names else -1

    total_batches = max(len(dataloader), 1)
    with torch.no_grad():
        seen_count = 0
        for batch_index, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(device)
            all_true_labels.extend(labels.cpu().numpy())
            batch_size = int(labels.size(0)) if hasattr(labels, "size") else len(labels)
            if file_paths:
                all_source_paths.extend(str(path) for path in file_paths[seen_count:seen_count + batch_size])
            else:
                all_source_paths.extend(f"sample-{seen_count + index:08d}" for index in range(batch_size))
            seen_count += batch_size

            if use_tta:
                # --- Keep your existing TTA code here for run_test.py ---

                # --- A. Generate Flipped Variants ---
                images_hflip = torch.flip(images, dims=[3])  # Flip width (columns)
                images_vflip = torch.flip(images, dims=[2])  # Flip height (rows)

                # --- B. Run Forward Passes ---
                outputs_orig = model(images)
                outputs_hflip = model(images_hflip)
                outputs_vflip = model(images_vflip)

                # --- C. Convert Logits to Probabilities ---
                probs_orig = F.softmax(outputs_orig, dim=1)
                probs_hflip = F.softmax(outputs_hflip, dim=1)
                probs_vflip = F.softmax(outputs_vflip, dim=1)

                # -------------------------------------------------------------
                # FIX: Spatially align probabilities across flips.
                # Since these are global classification probabilities, if the 
                # model uses spatial position context, the aggregation needs to 
                # be handled strictly at an un-flipped, unified base image level.
                # -------------------------------------------------------------
                stacked_probs = torch.stack([probs_orig, probs_hflip, probs_vflip], dim=0)
                mean_probs = stacked_probs.mean(dim=0)  # Shape: [Batch, Classes]

                # --- E. Cleaned Pessimistic Pass Logic ---
                if pass_idx != -1:
                    # Compare the true un-flipped base pass confidence directly against 
                    # the maximum doubt generated across the evaluation variants
                    min_pass_probs, _ = torch.min(stacked_probs[:, :, pass_idx], dim=0)
                    mean_probs[:, pass_idx] = min_pass_probs
                    
                    # Re-normalize probabilities cleanly
                    mean_probs = mean_probs / mean_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

            else:
                # --- FAST PATH: Single forward pass for training validation ---
                outputs = model(images)
                mean_probs = F.softmax(outputs, dim=1)

            # Apply Defect Multipliers
            adjusted_probs = apply_defect_multipliers(mean_probs, class_names, defect_multipliers=defect_multipliers)

            final_pred_indices = []
            final_confidences = []
            
            for i in range(adjusted_probs.size(0)):
                # 1. Isolate the absolute strongest candidate defect index
                defect_probs = adjusted_probs[i].clone()
                if pass_idx != -1:
                    defect_probs[pass_idx] = -1.0
                max_defect_idx = torch.argmax(defect_probs).item()
                
                # 2. Extract the clean Pass probability
                raw_pass_prob = adjusted_probs[i, pass_idx].item() if pass_idx != -1 else 0.0
                
                # 3. Apply the absolute gatekeeper rule
                if pass_idx != -1 and raw_pass_prob >= PASS_CERTAINTY_THRESHOLD:
                    # The board is only allowed to pass if it clears the high confidence floor
                    final_pred_indices.append(pass_idx)
                    final_confidences.append(raw_pass_prob)
                else:
                    # Force a secure reject to entirely eliminate escapes
                    final_pred_indices.append(max_defect_idx)
                    final_confidences.append(adjusted_probs[i, max_defect_idx].item())

            all_pred_labels.extend(final_pred_indices)
            all_confidences.extend(final_confidences)
            all_pass_probabilities.extend(
                adjusted_probs[:, pass_idx].detach().cpu().numpy()
                if pass_idx != -1
                else np.zeros(adjusted_probs.size(0), dtype=np.float32)
            )
            all_probabilities.extend(adjusted_probs.detach().cpu().numpy())
            if progress_callback:
                progress_callback(batch_index, total_batches)

    return {
        "true_labels": np.array(all_true_labels),
        "pred_labels": np.array(all_pred_labels),
        "confidences": np.array(all_confidences),
        "pass_probabilities": np.array(all_pass_probabilities),
        "probabilities": np.array(all_probabilities),
        "source_paths": np.array(all_source_paths, dtype=object),
    }


def split_misclassified_records(records, pass_class_name=PASS_CLASS_NAME):
    """Split wrong predictions into generic misclassified, escapes, and false alarms."""
    misclassified_records = []
    escape_records = []
    false_alarm_records = []

    for record in records:
        if record["true_idx"] == record["pred_idx"]:
            continue
        misclassified_records.append(record)
        if record["actual_class"] != pass_class_name and record["predicted_class"] == pass_class_name:
            escape_records.append(record)
        elif record["actual_class"] == pass_class_name and record["predicted_class"] != pass_class_name:
            false_alarm_records.append(record)

    other_misclassified_records = [
        record
        for record in misclassified_records
        if record not in escape_records and record not in false_alarm_records
    ]

    return misclassified_records, escape_records, false_alarm_records, other_misclassified_records


def export_false_positive_false_negative_examples(
    model,
    dataloader,
    class_names,
    file_paths,
    settings,
    defect_multipliers=None,
):
    """Export per-image original and Grad-CAM views for wrong predictions."""
    from backend.utils_grad_cam import build_prediction_records, export_grad_cam_records, find_last_conv_layer

    predictions = collect_predictions(
        model,
        dataloader,
        class_names,
        file_paths=file_paths,
        defect_multipliers=defect_multipliers,
    )

    if "Pass" not in class_names:
        print("\n[Warning]: 'Pass' class not found in class_names. Skipping FP/FN export.")
        return

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "samples"):
        print("Dataset metadata unavailable; skipping misclassification image export.")
        return

    records = build_prediction_records(dataset, predictions, class_names)
    misclassified_records, escape_records, false_alarm_records, other_misclassified_records = split_misclassified_records(records)
    total_misclassified = len(misclassified_records)
    if total_misclassified == 0:
        print("No misclassified samples found; skipping misclassification image export.")
        return

    target_layer = find_last_conv_layer(model)

    if other_misclassified_records:
        print(
            f"Exporting {len(other_misclassified_records)} wrong prediction samples to "
            f"{settings['misclassified_examples_dir']}..."
        )
        export_grad_cam_records(
            model,
            dataset,
            other_misclassified_records,
            settings["misclassified_examples_dir"],
            target_layer,
        )
        print(
            f"Saved {len(other_misclassified_records)} wrong prediction originals to "
            f"{settings['misclassified_original_dir']}"
        )
        print(
            f"Saved {len(other_misclassified_records)} wrong prediction Grad-CAM overlays to "
            f"{settings['misclassified_gradcam_dir']}"
        )
    else:
        print("No wrong prediction samples found; skipping generic misclassified export.")

    if escape_records:
        print(
            f"Exporting {len(escape_records)} escape samples to "
            f"{settings['escape_examples_dir']}..."
        )
        export_grad_cam_records(
            model,
            dataset,
            escape_records,
            settings["escape_examples_dir"],
            target_layer,
        )
        print(
            f"Saved {len(escape_records)} escape originals to "
            f"{settings['escape_original_dir']}"
        )
        print(
            f"Saved {len(escape_records)} escape Grad-CAM overlays to "
            f"{settings['escape_gradcam_dir']}"
        )
    else:
        print("No escape samples found; skipping escape export.")

    if false_alarm_records:
        # --- NEW LIMIT ADDED HERE ---
        limited_false_alarms = false_alarm_records[:300]
        
        print(
            f"Exporting {len(limited_false_alarms)} (out of {len(false_alarm_records)}) false alarm samples to "
            f"{settings['false_alarm_examples_dir']}..."
        )
        export_grad_cam_records(
            model,
            dataset,
            limited_false_alarms, # Use the limited list here
            settings["false_alarm_examples_dir"],
            target_layer,
        )
        print(
            f"Saved {len(limited_false_alarms)} false alarm originals to "
            f"{settings['false_alarm_original_dir']}"
        )
        print(
            f"Saved {len(limited_false_alarms)} false alarm Grad-CAM overlays to "
            f"{settings['false_alarm_gradcam_dir']}"
        )
    else:
        print("No false alarm samples found; skipping false alarm export.")
