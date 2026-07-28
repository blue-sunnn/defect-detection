import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils_misclassification import collect_predictions, predict_with_defect_multipliers


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma  
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss_unweighted = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        pt = torch.exp(-ce_loss_unweighted)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_unweighted

        if self.weight is not None:
            weights = self.weight[targets]
            focal_loss = focal_loss * weights

        if self.reduction == 'mean':
            return focal_loss.sum() / weights.sum() if self.weight is not None else focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calculate_classification_metrics(true_labels, pred_labels, class_names):
    """Calculate evaluation metrics from predicted and true labels."""
    labels = list(range(len(class_names)))
    cm_raw = confusion_matrix(true_labels, pred_labels, labels=labels, normalize=None)

    total_samples = np.sum(cm_raw)
    correct_predictions = np.trace(cm_raw)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, zero_division=0
    )
    per_class = []
    for index, class_name in enumerate(class_names):
        per_class.append({
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "correct": int(cm_raw[index, index]),
        })

    top_confusions = []
    for true_index, true_name in enumerate(class_names):
        for pred_index, pred_name in enumerate(class_names):
            if true_index == pred_index:
                continue
            count = int(cm_raw[true_index, pred_index])
            if count:
                top_confusions.append({
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": count,
                })
    top_confusions.sort(key=lambda item: item["count"], reverse=True)

    metrics = {
        "cm_raw": cm_raw,
        "class_names": list(class_names),
        "per_class": per_class,
        "top_confusions": top_confusions,
        "accuracy": accuracy,
        "escape_rate": None,
        "false_alarm_rate": None, 
        "overall_false_alarm_rate": None,
        "total_actual_defects": 0,
        "total_escaped_defects": 0,
        "total_actual_pass": 0,    
        "total_false_alarms": 0,
        "total_samples": total_samples,
    }

    if "Pass" not in class_names:
        return metrics

    pass_idx = class_names.index("Pass")
    total_actual_defects = 0
    total_escaped_defects = 0

    for i, name in enumerate(class_names):
        if name == "Pass":
            continue
        total_actual_defects += np.sum(cm_raw[i, :])
        total_escaped_defects += cm_raw[i, pass_idx]

    escape_rate = total_escaped_defects / total_actual_defects if total_actual_defects > 0 else 0
    
    total_actual_pass = np.sum(cm_raw[pass_idx, :])
    total_false_alarms = total_actual_pass - cm_raw[pass_idx, pass_idx]
    
    false_alarm_rate = total_false_alarms / total_actual_pass if total_actual_pass > 0 else 0
    overall_false_alarm_rate = total_false_alarms / total_samples if total_samples > 0 else 0

    metrics.update(
        {
            "escape_rate": escape_rate,
            "false_alarm_rate": false_alarm_rate, 
            "overall_false_alarm_rate": overall_false_alarm_rate,
            "total_actual_defects": total_actual_defects,
            "total_escaped_defects": total_escaped_defects,
            "total_actual_pass": total_actual_pass, 
            "total_false_alarms": total_false_alarms, 
        }
    )
    return metrics


def format_evaluation_summary(
    settings,
    accuracy,
    escape_rate,
    false_alarm_rate,
    overall_false_alarm_rate,
    total_escaped_defects,
    total_actual_defects,
    total_false_alarms,
    total_actual_pass,
    total_samples,
    optimizer_epoch_usage=None,
):
    """Build the evaluation summary text for terminal output or file export."""
    lines = [
        "",
        "=" * 45,
        "  Active Settings",
        f"  Learning Rate:         {settings['learning_rate']}",
        f"  Unfreeze Parameters:   {settings['unfreeze_layers']}",
        f"  Batch Size:            {settings['batch_size']}",
        f"  Dropout:               {settings['dropout_rate']}",
        f"  Optimizer:             {settings['optimizer'].upper()}",
        f"  Epochs:                {settings['epochs']}",
        "-" * 45,
        f"  Overall Model Accuracy: {accuracy * 100:.2f}%",
    ]

    if escape_rate is not None and false_alarm_rate is not None:
        quality_score = ((accuracy + (1 - escape_rate)) / 2) * 100
        lines.extend(
            [
                f"  Total Defects Escaped:  {total_escaped_defects} / {total_actual_defects}",
                f"  Model Escape Rate:      {escape_rate * 100:.2f}%",
                "-" * 45,
                f"  Total False Alarms / Total Pass:  {total_false_alarms} / {total_actual_pass}",
                f"  False Alarm Rate (Pass Data):     {false_alarm_rate * 100:.2f}%",
                f"  Total False Alarms / Total Input: {total_false_alarms} / {total_samples}",
                f"  False Alarm Rate (All Data):      {overall_false_alarm_rate * 100:.2f}%",
                "-" * 45,
                f"  Model Quality Score:    {quality_score:.2f}%",
            ]
        )

    if optimizer_epoch_usage:
        lines.append("  Optimizer Epoch Usage:")
        for optimizer_name, epoch_count in optimizer_epoch_usage.items():
            lines.append(f"    {optimizer_name.upper()}: {epoch_count}")

    lines.extend(["=" * 45, ""])
    return "\n".join(lines)


def print_evaluation_summary(
    settings, 
    accuracy, 
    escape_rate, 
    false_alarm_rate, 
    overall_false_alarm_rate,
    total_escaped_defects, 
    total_actual_defects,
    total_false_alarms,
    total_actual_pass,
    total_samples,
    optimizer_epoch_usage=None,
    output_path=None,
):
    """Print the active settings together with the evaluation metrics."""
    summary_text = format_evaluation_summary(
        settings,
        accuracy,
        escape_rate,
        false_alarm_rate,
        overall_false_alarm_rate,
        total_escaped_defects,
        total_actual_defects,
        total_false_alarms,
        total_actual_pass,
        total_samples,
        optimizer_epoch_usage=optimizer_epoch_usage,
    )
    print(summary_text)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary_text)

    return summary_text


def evaluate_model(
    model,
    dataloader,
    class_names,
    settings,
    model_name,
    optimizer_epoch_usage=None,
    defect_multipliers=None,
    print_summary=True,
    summary_output_path=None,
    predictions=None,
):
    """Print evaluation metrics for the current training settings."""
    if print_summary:
        print(f"\nEvaluating {model_name} with the current settings...")

    if defect_multipliers is None:
        defect_multipliers = {name: 0.0 for name in class_names if name != "Pass"}
        
    if predictions is None:
        predictions = collect_predictions(
            model,
            dataloader,
            class_names,
            defect_multipliers=defect_multipliers,
            use_tta=False,
        )
    
    metrics = calculate_classification_metrics(
        predictions["true_labels"],
        predictions["pred_labels"],
        class_names,
    )

    if metrics["escape_rate"] is None and "Pass" not in class_names:
        print("\n[Warning]: 'Pass' class not found in class_names. Skipping Escape Rate calculation.")

    summary_text = format_evaluation_summary(
        settings,
        metrics["accuracy"],
        metrics["escape_rate"],
        metrics["false_alarm_rate"],
        metrics["overall_false_alarm_rate"],
        metrics["total_escaped_defects"],
        metrics["total_actual_defects"],
        metrics["total_false_alarms"],
        metrics["total_actual_pass"],
        metrics["total_samples"],
        optimizer_epoch_usage=optimizer_epoch_usage,
    )

    if print_summary or summary_output_path:
        print_evaluation_summary(
            settings,
            metrics["accuracy"],
            metrics["escape_rate"],
            metrics["false_alarm_rate"],
            metrics["overall_false_alarm_rate"],
            metrics["total_escaped_defects"],
            metrics["total_actual_defects"],
            metrics["total_false_alarms"],
            metrics["total_actual_pass"],
            metrics["total_samples"],
            optimizer_epoch_usage=optimizer_epoch_usage,
            output_path=summary_output_path,
        )

    metrics["summary_text"] = summary_text

    if summary_output_path:
        report_path = os.path.splitext(summary_output_path)[0] + "_report.json"
        report_payload = {
            "model_name": model_name,
            "class_names": metrics["class_names"],
            "confusion_matrix": metrics["cm_raw"].astype(int).tolist(),
            "per_class": metrics["per_class"],
            "top_confusions": metrics["top_confusions"],
            "accuracy": float(metrics["accuracy"]),
            "escape_rate": None if metrics["escape_rate"] is None else float(metrics["escape_rate"]),
            "false_alarm_rate": None if metrics["false_alarm_rate"] is None else float(metrics["false_alarm_rate"]),
            "overall_false_alarm_rate": None if metrics["overall_false_alarm_rate"] is None else float(metrics["overall_false_alarm_rate"]),
            "total_actual_defects": int(metrics["total_actual_defects"]),
            "total_escaped_defects": int(metrics["total_escaped_defects"]),
            "total_actual_pass": int(metrics["total_actual_pass"]),
            "total_false_alarms": int(metrics["total_false_alarms"]),
            "total_samples": int(metrics["total_samples"]),
            "settings": settings,
        }
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report_payload, report_file, indent=2)
        metrics["report_path"] = report_path

    return metrics


def load_datasets(
    dataset_dir,
    image_size,
    batch_size,
    augmentation_settings=None,
    split_percent=80,
    random_seed=42,
):
    """Load train/val datasets, splitting raw class folders in memory when needed."""
    print("Loading datasets...")
    if not 1 <= int(split_percent) <= 99:
        raise ValueError("split_percent must be between 1 and 99.")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # ---------------------------------------------------------
    # FIX 1: Tame the Color Jitter (Broadened Color Jitter Backfire)
    # Dialed back from 0.3 to 0.1 to preserve pristine board baseline
    # ---------------------------------------------------------
    augmentation_settings = augmentation_settings or {}
    crop_min_scale = float(augmentation_settings.get("crop_min_scale", 0.85))
    crop_max_scale = float(augmentation_settings.get("crop_max_scale", 1.0))
    crop_min_aspect_ratio = float(augmentation_settings.get("crop_min_aspect_ratio", 0.9))
    crop_max_aspect_ratio = float(augmentation_settings.get("crop_max_aspect_ratio", 1.1))
    rotation_degrees = float(augmentation_settings.get("rotation_degrees", 180))
    horizontal_flip_probability = float(augmentation_settings.get("horizontal_flip_probability", 0.5))
    vertical_flip_probability = float(augmentation_settings.get("vertical_flip_probability", 0.5))
    brightness_jitter = float(augmentation_settings.get("brightness_jitter", 0.1))
    contrast_jitter = float(augmentation_settings.get("contrast_jitter", 0.1))
    saturation_jitter = float(augmentation_settings.get("saturation_jitter", 0.1))

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(
            image_size,
            scale=(crop_min_scale, crop_max_scale),
            ratio=(crop_min_aspect_ratio, crop_max_aspect_ratio),
        ),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
        transforms.RandomVerticalFlip(p=vertical_flip_probability),
        transforms.ColorJitter(
            brightness=brightness_jitter,
            contrast=contrast_jitter,
            saturation=saturation_jitter,
        ),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,  
    ])
    
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    has_train = os.path.isdir(train_dir)
    has_val = os.path.isdir(val_dir)
    if has_train != has_val:
        missing = "val" if has_train else "train"
        raise ValueError(f"Dataset root contains only one predefined split. Missing '{missing}' folder.")

    if has_train and has_val:
        print("[Dataset] Found train/ and val/ folders; Split % is ignored.")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        if train_dataset.classes != val_dataset.classes:
            raise ValueError(
                f"Train/val class order mismatch. Train: {train_dataset.classes}, Val: {val_dataset.classes}."
            )
        class_names = val_dataset.classes
        val_file_paths = [path for path, _ in val_dataset.samples]
    else:
        print(f"[Dataset] No train/val folders found; creating a stratified {split_percent}/{100-int(split_percent)} split.")
        train_source_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
        val_source_dataset = datasets.ImageFolder(dataset_dir, transform=val_transform)
        indices = np.arange(len(train_source_dataset.samples))
        targets = np.array(train_source_dataset.targets)

        train_indices, val_indices = train_test_split(
            indices,
            train_size=float(split_percent) / 100.0,
            stratify=targets,
            random_state=random_seed,
            shuffle=True,
        )
        train_dataset = Subset(train_source_dataset, train_indices.tolist())
        val_dataset = Subset(val_source_dataset, val_indices.tolist())
        train_dataset.classes = train_source_dataset.classes
        val_dataset.classes = val_source_dataset.classes
        class_names = val_source_dataset.classes
        val_file_paths = [val_source_dataset.samples[index][0] for index in val_indices]

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=(DEVICE.type == "cuda") 
    )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=(DEVICE.type == "cuda")
    )
    
    return train_loader, val_loader, class_names, val_file_paths


def build_class_weights_tensor(class_weight_dict, num_classes, device=DEVICE):
    """Convert a class-weight mapping into a tensor ordered by class index."""
    weight_list = [class_weight_dict[i] for i in range(num_classes)]
    return torch.tensor(weight_list, dtype=torch.float, device=device)


def build_weighted_criterion(
    class_weight_dict,
    num_classes,
    gamma=1.0,
    label_smoothing=0.1,
    use_class_weights=False,
    device=DEVICE,
):
    """Create a weighted Focal Loss for the current dataset."""
    class_weights = None
    if use_class_weights:
        class_weights = build_class_weights_tensor(class_weight_dict, num_classes, device=device)
    
    return FocalLoss(
        weight=class_weights,
        gamma=gamma,  
        label_smoothing=label_smoothing
    )


def calculate_class_weights(train_loader):
    """Calculate balanced class weights from the training dataset."""
    dataset = train_loader.dataset
    if isinstance(dataset, Subset):
        train_labels = np.array([dataset.dataset.targets[index] for index in dataset.indices])
    else:
        train_labels = np.array(dataset.targets)
    
    unique_classes = np.unique(train_labels)
    calculated_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )

    class_weight_dict = dict(zip(unique_classes, calculated_weights))
    num_classes = len(unique_classes)
    return class_weight_dict, num_classes


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device=DEVICE,
    freeze_batchnorm_stats=True,
    check_stop=None,
):
    """Execute one training epoch while keeping BatchNorm statistics frozen."""
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty. At least one training batch is required.")

    model.train()

    if freeze_batchnorm_stats:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training Batch", leave=False, disable=True)

    for images, labels in progress_bar:
        if check_stop is not None:
            check_stop()
        inputs, targets = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        if check_stop is not None:
            check_stop()

    return total_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, class_names, device=DEVICE, check_stop=None):
    """Calculate validation loss, accuracy, and escape rate in a single pass."""
    if len(val_loader) == 0:
        raise ValueError("val_loader is empty. At least one validation batch is required.")

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []

    zero_multipliers = {name: 0.0 for name in class_names if name != "Pass"}

    with torch.no_grad():
        for images, labels in val_loader:
            if check_stop is not None:
                check_stop()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predictions, _ = predict_with_defect_multipliers(
                outputs, 
                class_names, 
                defect_multipliers=zero_multipliers
            )

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(predictions.cpu().tolist())
            if check_stop is not None:
                check_stop()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0
    metrics = calculate_classification_metrics(true_labels, pred_labels, class_names)
    return avg_loss, accuracy, metrics["escape_rate"], metrics["false_alarm_rate"]


def load_checkpoint_or_exit(
    model,
    checkpoint_path,
    model_name,
    checkpoint_label="checkpoint",
    device=DEVICE,
):
    """Load a checkpoint or exit with a clear error if it does not exist."""
    if not os.path.exists(checkpoint_path):
        print(
            f"Error: Could not find '{checkpoint_path}'. "
            f"Required {model_name} {checkpoint_label} is missing."
        )
        raise SystemExit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in '{checkpoint_path}'.")
    cleaned_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    incompatible = model.load_state_dict(cleaned_state, strict=False)
    if incompatible.missing_keys:
        print(f"[Checkpoint] Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {len(incompatible.unexpected_keys)}")
    model.to(device)
    print(f"Successfully loaded {model_name} {checkpoint_label}.")
    return model


def freeze_all_and_unfreeze_last(model, num_params_to_unfreeze):
    """Freeze all parameters, then unfreeze the last N parameters."""
    all_params = list(model.parameters())

    for param in all_params:
        param.requires_grad = False

    if num_params_to_unfreeze <= 0:
        return

    for param in all_params[-num_params_to_unfreeze:]:
        param.requires_grad = True


def get_optimizer(
    model,
    optimizer_name,
    learning_rate,
    weight_decay=1e-4,
    sgd_momentum=0.9,
    rmsprop_alpha=0.9,
    head_lr_multiplier=1.0,
    deep_lr_multiplier=0.1,
    shallow_lr_multiplier=0.005,
):
    """Instantiate the optimizer using Layer-Wise Learning Rate Decay (LLRD)."""
    feature_blocks = list(model.features.children())
    num_blocks = len(feature_blocks)
    split_point = num_blocks // 2
    
    shallow_params = []
    deep_params = []
    head_params = list(model.classifier.parameters())

    for i, block in enumerate(feature_blocks):
        if i < split_point:
            shallow_params.extend([p for p in block.parameters() if p.requires_grad])
        else:
            deep_params.extend([p for p in block.parameters() if p.requires_grad])

    param_groups = [
        {"params": head_params, "lr": learning_rate * head_lr_multiplier},
        {"params": deep_params, "lr": learning_rate * deep_lr_multiplier},
        {"params": shallow_params, "lr": learning_rate * shallow_lr_multiplier},
    ]

    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    if optimizer_name == "adam":
        optimizer = optim.Adam(param_groups)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(param_groups, momentum=sgd_momentum)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(param_groups, alpha=rmsprop_alpha)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer
