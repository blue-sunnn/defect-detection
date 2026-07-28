import csv
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config_threshold_optimization import PASS_CLASS_NAME


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def sanitize_filename(value):
    """Convert labels and filenames into filesystem-safe fragments."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "sample"


def denormalize_image(image_tensor):
    """Convert a normalized tensor back into an RGB image array for display."""
    image_tensor = image_tensor.detach().cpu()
    image_tensor = image_tensor * IMAGENET_STD + IMAGENET_MEAN
    image_tensor = image_tensor.clamp(0, 1)
    return image_tensor.permute(1, 2, 0).numpy()


def find_last_conv_layer(model):
    """Return the deepest Conv2d module for Grad-CAM hooks."""
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("Could not find a Conv2d layer for Grad-CAM.")


def generate_grad_cam(model, image_tensor, target_class_idx, target_layer, use_tta=True):
    """Generate an aggregated Grad-CAM heatmap using Test-Time Augmentation (TTA)."""
    
    def get_single_cam(img_tensor):
        activations = {}
        def forward_hook(module, input, output):
            activations["value"] = output
            output.retain_grad()

        forward_handle = target_layer.register_forward_hook(forward_hook)

        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                img_tensor.requires_grad_(True)
                outputs = model(img_tensor)
                score = outputs[:, target_class_idx].sum()
                score.backward()

            activation = activations["value"]
            gradient = activation.grad

            if gradient is None:
                raise RuntimeError("Gradient is None. The computational graph failed to build.")

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
            
            return cam.detach().squeeze().cpu().numpy()
        finally:
            forward_handle.remove()

    was_training = model.training
    model.eval()

    cams = []
    
    # 1. Base Image
    cams.append(get_single_cam(image_tensor))

    if use_tta:
        # 2. Horizontal Flip
        img_hflip = torch.flip(image_tensor, dims=[3])
        cam_hflip = get_single_cam(img_hflip)
        cams.append(np.flip(cam_hflip, axis=1)) # Un-flip the heatmap

        # 3. Vertical Flip
        img_vflip = torch.flip(image_tensor, dims=[2])
        cam_vflip = get_single_cam(img_vflip)
        cams.append(np.flip(cam_vflip, axis=0)) # Un-flip the heatmap

    # Aggregate the heatmaps to find consensus
    cam_avg = np.mean(cams, axis=0)

    # Clean Min-Max Normalization with a Noise Gate
    cam_max = float(cam_avg.max())
    if cam_max > 1e-7:
        # NOISE GATE: Suppress any background activation below 20% of the peak intensity
        threshold = 0.20 * cam_max
        cam_avg = np.where(cam_avg < threshold, 0, cam_avg)
        
        # Normalize (Since ReLU sets min to 0, we just divide by max)
        cam_avg = cam_avg / cam_max
    else:
        cam_avg = np.zeros_like(cam_avg)

    if was_training:
        model.train()

    return cam_avg


def overlay_heatmap_on_image(base_image, heatmap, alpha=0.4):
    """Blend a Grad-CAM heatmap over the original image."""
    colorized_heatmap = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = ((1 - alpha) * base_image) + (alpha * colorized_heatmap)
    return np.clip(overlay, 0, 1)


def save_prediction_figure(output_path, image, metadata, panel_title):
    """Save one annotated image with prediction metadata."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.set_title(panel_title)
    ax.axis("off")

    fig.suptitle(
        (
            f"{metadata['sample_name']}\n"
            f"Actual: {metadata['actual_class']} | "
            f"Pred: {metadata['predicted_class']} ({metadata['confidence']:.1%})"
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_prediction_records(dataset, predictions, class_names):
    """Pair dataset sample metadata with deployed predictions."""
    records = []
    pass_probabilities = predictions.get("pass_probabilities")
    if pass_probabilities is None:
        pass_probabilities = np.full(len(predictions["true_labels"]), np.nan)

    for index, ((file_path, _), true_idx, pred_idx, confidence, pass_probability) in enumerate(
        zip(
            dataset.samples,
            predictions["true_labels"],
            predictions["pred_labels"],
            predictions["confidences"],
            pass_probabilities,
        )
    ):
        records.append(
            {
                "index": index,
                "file_path": file_path,
                "sample_name": os.path.basename(file_path),
                "true_idx": int(true_idx),
                "pred_idx": int(pred_idx),
                "actual_class": class_names[int(true_idx)],
                "predicted_class": class_names[int(pred_idx)],
                "confidence": float(confidence),
                "pass_probability": float(pass_probability),
            }
        )
    return records


def calculate_grad_cam_metrics(heatmap, active_threshold=0.0):
    """Measure activation spread on a normalized, noise-gated Grad-CAM heatmap."""
    active_mask = heatmap > active_threshold
    total_pixels = int(active_mask.size)
    active_pixels = int(active_mask.sum())
    active_ratio = active_pixels / total_pixels if total_pixels else 0.0

    labeled_mask, component_count = ndimage.label(active_mask)
    if component_count:
        component_sizes = np.bincount(labeled_mask.ravel())[1:]
        largest_component_pixels = int(component_sizes.max())
    else:
        largest_component_pixels = 0

    largest_component_ratio = largest_component_pixels / total_pixels if total_pixels else 0.0
    activation_sum = float(heatmap.sum())
    if activation_sum > 1e-12:
        top_pixel_count = max(1, int(total_pixels * 0.01))
        top_values = np.partition(heatmap.reshape(-1), -top_pixel_count)[-top_pixel_count:]
        peak_concentration = float(top_values.sum() / activation_sum)
    else:
        peak_concentration = 0.0

    return {
        "active_ratio": active_ratio,
        "largest_component_ratio": largest_component_ratio,
        "component_count": component_count,
        "peak_concentration": peak_concentration,
    }


def infer_lot_name(file_path):
    """Best-effort lot name from ImageFolder path: lot/class/image."""
    path = Path(file_path)
    if len(path.parents) >= 2:
        return path.parents[1].name
    return "unknown"


def should_simulate_grad_cam_override(metrics, active_ratio_threshold=0.15):
    """Candidate-only gate: diffuse and fragmented CAM becomes simulated Pass."""
    return (
        metrics["active_ratio"] > active_ratio_threshold
        and (
            metrics["largest_component_ratio"] < active_ratio_threshold * 0.67
            or metrics["component_count"] >= 3
        )
    )


def save_grad_cam_audit_plots(rows, output_dir):
    """Plot CAM metric distributions without changing predictions."""
    if not rows:
        return []

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    saved_paths = []

    def values_for(group_name, metric_name):
        return [row[metric_name] for row in rows if row["audit_group"] == group_name]

    groups = ["true_defect", "false_alarm"]
    group_labels = ["True Defect", "False Alarm"]
    metrics = [
        ("active_ratio", "Active Ratio"),
        ("largest_component_ratio", "Largest Component Ratio"),
        ("component_count", "Component Count"),
        ("peak_concentration", "Peak Concentration"),
    ]

    for metric_name, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [values_for(group, metric_name) for group in groups]
        data = [values for values in data if values]
        labels = [label for group, label in zip(groups, group_labels) if values_for(group, metric_name)]
        if not data:
            plt.close(fig)
            continue
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"Grad-CAM Audit - {metric_label}")
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle="--", alpha=0.4)
        output_path = os.path.join(plots_dir, f"{metric_name}_true_defect_vs_false_alarm.png")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    class_names = sorted({row["predicted_class"] for row in rows})
    active_ratio_by_class = [
        [row["active_ratio"] for row in rows if row["predicted_class"] == class_name]
        for class_name in class_names
    ]
    if any(active_ratio_by_class):
        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.4), 5))
        ax.boxplot(active_ratio_by_class, labels=class_names, showmeans=True)
        ax.set_title("Grad-CAM Audit - Active Ratio by Predicted Class")
        ax.set_ylabel("Active Ratio")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        output_path = os.path.join(plots_dir, "active_ratio_by_predicted_class.png")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    lot_names = sorted({row["lot_name"] for row in rows})
    if len(lot_names) > 1:
        active_ratio_by_lot = [
            [row["active_ratio"] for row in rows if row["lot_name"] == lot_name]
            for lot_name in lot_names
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(lot_names) * 1.5), 5))
        ax.boxplot(active_ratio_by_lot, labels=lot_names, showmeans=True)
        ax.set_title("Grad-CAM Audit - Active Ratio by Lot")
        ax.set_ylabel("Active Ratio")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        output_path = os.path.join(plots_dir, "active_ratio_by_lot.png")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def export_grad_cam_audit(
    model,
    dataset,
    class_names,
    predictions,
    output_dir,
    active_ratio_threshold=0.15,
):
    """Compute numeric Grad-CAM audit metrics for predicted defects without overriding."""
    records = build_prediction_records(dataset, predictions, class_names)
    defect_records = [
        record for record in records if record["predicted_class"] != PASS_CLASS_NAME
    ]
    audit_dir = os.path.join(output_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    csv_path = os.path.join(audit_dir, "grad_cam_audit.csv")

    if not defect_records:
        print("No predicted defects found; writing empty Grad-CAM audit CSV.")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "index",
                "image_path",
                "lot_name",
                "true_class",
                "predicted_class",
                "confidence",
                "pass_probability",
                "cam_active_ratio",
                "largest_component_ratio",
                "component_count",
                "peak_concentration",
                "audit_group",
                "candidate_override_pass",
                "candidate_false_alarm_removed",
                "candidate_new_escape",
            ])
            writer.writeheader()
        return csv_path, []

    target_layer = find_last_conv_layer(model)
    device = next(model.parameters()).device
    rows = []
    progress_bar_width = 20

    for index, record in enumerate(defect_records, start=1):
        image_tensor, _ = dataset[record["index"]]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        heatmap = generate_grad_cam(
            model,
            image_tensor,
            target_class_idx=record["pred_idx"],
            target_layer=target_layer,
            use_tta=True,
        )
        metrics = calculate_grad_cam_metrics(heatmap)
        candidate_override = should_simulate_grad_cam_override(
            metrics,
            active_ratio_threshold=active_ratio_threshold,
        )
        is_false_alarm = record["actual_class"] == PASS_CLASS_NAME
        is_true_defect = record["actual_class"] != PASS_CLASS_NAME
        row = {
            "index": record["index"],
            "image_path": record["file_path"],
            "lot_name": infer_lot_name(record["file_path"]),
            "true_class": record["actual_class"],
            "predicted_class": record["predicted_class"],
            "confidence": record["confidence"],
            "pass_probability": record["pass_probability"],
            "cam_active_ratio": metrics["active_ratio"],
            "active_ratio": metrics["active_ratio"],
            "largest_component_ratio": metrics["largest_component_ratio"],
            "component_count": metrics["component_count"],
            "peak_concentration": metrics["peak_concentration"],
            "audit_group": "false_alarm" if is_false_alarm else "true_defect",
            "candidate_override_pass": candidate_override,
            "candidate_false_alarm_removed": bool(candidate_override and is_false_alarm),
            "candidate_new_escape": bool(candidate_override and is_true_defect),
        }
        rows.append(row)

        if index % 10 == 0 or index == len(defect_records):
            filled_width = int(progress_bar_width * index / len(defect_records))
            progress_bar = "#" * filled_width + "-" * (progress_bar_width - filled_width)
            print(f"\rGrad-CAM audit [{progress_bar}] {index}/{len(defect_records)}", end="", flush=True)

    print()

    csv_fieldnames = [
        "index",
        "image_path",
        "lot_name",
        "true_class",
        "predicted_class",
        "confidence",
        "pass_probability",
        "cam_active_ratio",
        "largest_component_ratio",
        "component_count",
        "peak_concentration",
        "audit_group",
        "candidate_override_pass",
        "candidate_false_alarm_removed",
        "candidate_new_escape",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fieldnames})

    plot_paths = save_grad_cam_audit_plots(rows, audit_dir)
    false_alarm_removed = sum(row["candidate_false_alarm_removed"] for row in rows)
    new_escapes = sum(row["candidate_new_escape"] for row in rows)
    print(
        "Grad-CAM audit saved: "
        f"{csv_path} | simulated false alarms removed={false_alarm_removed} | "
        f"simulated new escapes={new_escapes}"
    )
    return csv_path, plot_paths


def export_grad_cam_records(model, dataset, records, output_dir, target_layer):
    """Save original and Grad-CAM versions per selected prediction record."""
    original_output_dir = os.path.join(output_dir, "original")
    gradcam_output_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(original_output_dir, exist_ok=True)
    os.makedirs(gradcam_output_dir, exist_ok=True)

    total_records = len(records)
    progress_bar_width = 20

    for index, record in enumerate(records, start=1):
        image_tensor, _ = dataset[record["index"]]
        image_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)

        # --- FIX 1: The Escape Intercept ---
        # If the model predicted Pass (an Escape), we cannot target Pass.
        # We must target the actual Defect class to see what it missed.
        if record["predicted_class"] == PASS_CLASS_NAME and record["actual_class"] != PASS_CLASS_NAME:
            target_idx = record["true_idx"]
            record_note = f" (Targeting missed {record['actual_class']} features)"
        else:
            target_idx = record["pred_idx"]
            record_note = ""

        # --- FIX 2 & 3: Call the new TTA-enabled generator ---
        heatmap = generate_grad_cam(
            model,
            image_tensor,
            target_class_idx=target_idx,
            target_layer=target_layer,
            use_tta=True  # Ensure TTA matches your inference loop
        )
        
        original_image = denormalize_image(image_tensor.squeeze(0))
        overlay_image = overlay_heatmap_on_image(original_image, heatmap)

        file_stem = sanitize_filename(os.path.splitext(record["sample_name"])[0])
        pred_label = sanitize_filename(record["predicted_class"])
        output_path = os.path.join(
            gradcam_output_dir,
            f"{record['index']:04d}_{file_stem}_pred_{pred_label}.png",
        )
        original_output_path = os.path.join(
            original_output_dir,
            f"{record['index']:04d}_{file_stem}_pred_{pred_label}.png",
        )
        save_prediction_figure(original_output_path, original_image, record, "Original")
        save_prediction_figure(output_path, overlay_image, record, "Grad-CAM")

        if index % 10 == 0 or index == total_records:
            filled_width = int(progress_bar_width * index / total_records)
            progress_bar = "#" * filled_width + "-" * (progress_bar_width - filled_width)
            print(f"\r[{progress_bar}] {index}/{total_records}", end="", flush=True)

    print()


def export_test_set_grad_cam_examples(
    model,
    dataset,
    class_names,
    predictions,
    output_dir,
    target_filenames=None,  # <-- CHANGED: Replaced random parameters
):
    """Export Grad-CAM overlays for misclassified samples and explicitly targeted file names."""
    from backend.utils_misclassification import split_misclassified_records

    records = build_prediction_records(dataset, predictions, class_names)
    _, escape_records, false_alarm_records, other_misclassified_records = split_misclassified_records(
        records,
        pass_class_name=PASS_CLASS_NAME,
    )

    target_layer = find_last_conv_layer(model)
    misclassified_output_dir = os.path.join(output_dir, "misclassified")
    escape_output_dir = os.path.join(output_dir, "escape")
    false_alarm_output_dir = os.path.join(output_dir, "false_alarm")
    targeted_output_dir = os.path.join(output_dir, "targeted_samples")

    if other_misclassified_records:
        export_grad_cam_records(
            model,
            dataset,
            other_misclassified_records,
            misclassified_output_dir,
            target_layer,
        )
        print(
            f"Saved {len(other_misclassified_records)} Grad-CAM overlays for wrong prediction samples to "
            f"{misclassified_output_dir}"
        )
    else:
        print("No wrong prediction samples found; skipping generic misclassified Grad-CAM export.")

    if escape_records:
        export_grad_cam_records(
            model,
            dataset,
            escape_records,
            escape_output_dir,
            target_layer,
        )
        print(f"Saved {len(escape_records)} Grad-CAM overlays for escape samples to {escape_output_dir}")
    else:
        print("No escape samples found; skipping escape Grad-CAM export.")

    if false_alarm_records:
        export_grad_cam_records(
            model,
            dataset,
            false_alarm_records,
            false_alarm_output_dir,
            target_layer,
        )
        print(f"Saved {len(false_alarm_records)} Grad-CAM overlays for false alarm samples to {false_alarm_output_dir}")
    else:
        print("No false alarm samples found; skipping false alarm Grad-CAM export.")

    if not records:
        print("Dataset is empty; skipping targeted Grad-CAM export.")
        return

    # 2. NEW: Filter records based on your custom filename list
    if target_filenames:
        selected_records = []
        for record in records:
            # Check if any string in your target list matches the current file name
            if any(target.lower() in record["sample_name"].lower() for target in target_filenames):
                selected_records.append(record)

        if selected_records:
            export_grad_cam_records(
                model,
                dataset,
                selected_records,
                targeted_output_dir,
                target_layer,
            )
            print(f"Saved {len(selected_records)} Grad-CAM overlays for targeted samples to {targeted_output_dir}")
        else:
            print(f"None of the provided target filenames matched images in the active dataset split.")
