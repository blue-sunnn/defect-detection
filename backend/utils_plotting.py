import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils_misclassification import collect_predictions
from backend.utils_training import calculate_classification_metrics


DEFAULT_PLOT_CONFIG = {
    "fig_width": 10,
    "fig_height": 7,
    "save_dpi": 300,
    "font_family": "Arial",
    "base_font_size": 11,
    "title_size": 16,
    "label_size": 12,
    "tick_size": 10,
    "annot_size": 11,
    "colorbar_tick_size": 10,
    "show_grid": True,
    "max_examples_per_plot": 12,
}


def apply_plot_style():
    """Apply shared Matplotlib defaults for training visualizations."""
    plt.rcParams["font.family"] = DEFAULT_PLOT_CONFIG["font_family"]
    plt.rcParams["font.size"] = DEFAULT_PLOT_CONFIG["base_font_size"]


def build_plot_settings(
    model_name,
    save_path,
    colormap,
    misclassification_output_dir="plot",
    overrides=None,
):
    """Build the shared plotting settings dict with model-specific paths."""
    misclassified_examples_dir = os.path.join(misclassification_output_dir, "misclassified")
    escape_examples_dir = os.path.join(misclassification_output_dir, "escape")
    false_alarm_examples_dir = os.path.join(misclassification_output_dir, "false_alarm")
    settings = {
        "model_name": model_name,
        "fig_width": DEFAULT_PLOT_CONFIG["fig_width"],
        "fig_height": DEFAULT_PLOT_CONFIG["fig_height"],
        "save_path": save_path,
        "save_dpi": DEFAULT_PLOT_CONFIG["save_dpi"],
        "label_size": DEFAULT_PLOT_CONFIG["label_size"],
        "tick_size": DEFAULT_PLOT_CONFIG["tick_size"],
        "title_size": DEFAULT_PLOT_CONFIG["title_size"],
        "annot_size": DEFAULT_PLOT_CONFIG["annot_size"],
        "colorbar_tick_size": DEFAULT_PLOT_CONFIG["colorbar_tick_size"],
        "colormap": colormap,
        "show_grid": DEFAULT_PLOT_CONFIG["show_grid"],
        "misclassification_output_dir": misclassification_output_dir,
        "misclassified_examples_dir": misclassified_examples_dir,
        "misclassified_original_dir": os.path.join(misclassified_examples_dir, "original"),
        "misclassified_gradcam_dir": os.path.join(misclassified_examples_dir, "gradcam"),
        "escape_examples_dir": escape_examples_dir,
        "escape_original_dir": os.path.join(escape_examples_dir, "original"),
        "escape_gradcam_dir": os.path.join(escape_examples_dir, "gradcam"),
        "false_alarm_examples_dir": false_alarm_examples_dir,
        "false_alarm_original_dir": os.path.join(false_alarm_examples_dir, "original"),
        "false_alarm_gradcam_dir": os.path.join(false_alarm_examples_dir, "gradcam"),
        "max_examples_per_plot": DEFAULT_PLOT_CONFIG["max_examples_per_plot"],
    }

    if overrides:
        settings.update(overrides)

    return settings


def plot_confusion_matrix(
    model,
    dataset,
    class_names,
    settings,
    evaluation_settings=None,
    optimizer_epoch_usage=None,
    defect_multipliers=None,
    predictions=None,
):
    """Plot and save a normalized confusion matrix for the given dataset."""
    print(f"\nGenerating confusion matrix for {settings['model_name']}...")

    if predictions is None:
        predictions = collect_predictions(
            model,
            dataset,
            class_names,
            defect_multipliers=defect_multipliers,
        )
    metrics = calculate_classification_metrics(
        predictions["true_labels"],
        predictions["pred_labels"],
        class_names,
    )

    if metrics["escape_rate"] is None and "Pass" not in class_names:
        print("\n[Warning]: 'Pass' class not found in class_names. Skipping Escape Rate calculation.")

    cm = confusion_matrix(
        predictions["true_labels"],
        predictions["pred_labels"],
        labels=list(range(len(class_names))),
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(settings["fig_width"], settings["fig_height"]))
    heatmap = ax.imshow(cm, cmap=settings["colormap"], vmin=0, vmax=1)
    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.ax.tick_params(labelsize=settings["colorbar_tick_size"])

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=settings["tick_size"])
    ax.set_yticklabels(class_names, fontsize=settings["tick_size"])

    if settings["show_grid"]:
        ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=settings["annot_size"],
                fontweight="bold",
                color="white" if cm[i, j] > threshold else "black",
            )

    ax.set_title(
        f"Confusion Matrix - {settings['model_name']}",
        fontsize=settings["title_size"],
        pad=15,
    )
    ax.set_ylabel("Actual Class", fontsize=settings["label_size"])
    ax.set_xlabel("Predicted Class", fontsize=settings["label_size"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    output_dir = os.path.dirname(settings["save_path"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(settings["save_path"], dpi=settings["save_dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_examples(rows, output_path, title, settings):
    """Plot a gallery of all misclassified images."""
    if not rows:
        print(f"No samples found for {title.lower()}.")
        return

    sample_rows = rows
    cols = min(5, len(sample_rows))
    rows_count = math.ceil(len(sample_rows) / cols)

    fig, axes = plt.subplots(rows_count, cols, figsize=(4 * cols, 4 * rows_count))
    axes = np.atleast_1d(axes).flatten()

    for ax, row in zip(axes, sample_rows):
        image = plt.imread(row["file_path"])
        ax.imshow(image)
        ax.set_title(
            f"Actual: {row['actual_class']}\nPred: {row['predicted_class']} ({row['confidence']:.1%})",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(sample_rows) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=settings["title_size"])
    fig.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=settings["save_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {title.lower()} gallery to {output_path}")

def plot_training_results(
    train_losses,
    val_losses,
    val_accs,
    escape_rates=None,
    false_alarm_rates=None,
    output_path="plot/results.png",
):
    """Plot training metrics across epochs in a 2-panel dashboard."""
    if not train_losses:
        raise ValueError("plot_training_results requires at least one epoch of training data.")

    epochs = np.arange(1, len(train_losses) + 1)
    
    # Create exactly 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Left Graph: Training & Validation Loss ---
    ax1.plot(epochs, train_losses, color='#1f77b4', linewidth=1.5, label='Train Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, color='#ff7f0e', linewidth=1.5, label='Val Loss')
        
    ax1.set_title("Training & Validation Loss", fontsize=13, pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper right", fontsize=10)

    # --- Right Graph: Performance Metrics ---
    if val_accs:
        ax2.plot(epochs, val_accs, color='#2ca02c', linewidth=1.5, label='Accuracy')
    if escape_rates is not None and len(escape_rates) > 0:
        ax2.plot(epochs, escape_rates, color='#d62728', linewidth=1.5, label='Escape Rate')
    if false_alarm_rates is not None and len(false_alarm_rates) > 0:
        ax2.plot(epochs, false_alarm_rates, color='#9467bd', linewidth=1.5, label='False Alarm (Over-Reject)')
        
    ax2.set_title("Model Performance Metrics", fontsize=13, pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Rate / Probability", fontsize=11)
    
    # Set y-axis to strictly show 0% to 100% distribution 
    ax2.set_ylim(0, 1.05) 
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="center right", fontsize=10)

    plt.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training dashboard successfully saved to {output_path}")


def plot_pass_probability_histogram(
    true_path="val_data_true.npy",
    prob_path="val_data_prob.npy",
    classes_path="val_data_classes.npy",
    output_path="plot/probability_histogram.png",
):
    """Plot the model confidence distribution for the Pass class."""
    if not os.path.exists(true_path) or not os.path.exists(prob_path):
        raise FileNotFoundError("Missing .npy files. Please run the export step first.")

    y_true = np.load(true_path)
    y_prob = np.load(prob_path)
    class_names = np.load(classes_path, allow_pickle=True).tolist()

    if "Pass" not in class_names:
        raise ValueError("The class 'Pass' was not found in the class names array.")

    pass_idx = class_names.index("Pass")
    pass_class_probabilities = y_prob[:, pass_idx]

    actual_pass_mask = y_true == pass_idx
    probs_of_actual_pass = pass_class_probabilities[actual_pass_mask]

    actual_defect_mask = y_true != pass_idx
    probs_of_actual_defects = pass_class_probabilities[actual_defect_mask]

    plt.figure(figsize=(10, 6))
    plt.hist(
        probs_of_actual_defects,
        bins=50,
        alpha=0.6,
        color="#d97706",
        label="Actual Defects",
        density=False,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.hist(
        probs_of_actual_pass,
        bins=50,
        alpha=0.6,
        color="#0b3d91",
        label="Actual Pass",
        density=False,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.title("Model Confidence Distribution for the 'Pass' Class", fontsize=14, pad=15)
    plt.xlabel("Probability Assigned to 'Pass' (0.0 to 1.0)", fontsize=12)
    plt.ylabel("Number of Validation Samples (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.axvline(x=0.3, color="gray", linestyle="--", alpha=0.7)
    plt.axvline(x=0.7, color="gray", linestyle="--", alpha=0.7)
    plt.text(
        0.5,
        0.92,
        "Uncertainty Zone",
        transform=plt.gca().transAxes,
        horizontalalignment="center",
        color="gray",
        fontsize=10,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper center")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully generated probability histogram at: {output_path}")
