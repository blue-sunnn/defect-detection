import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.model_config import (
    BATCH_SIZE,
    COLORMAP,
    BRIGHTNESS_JITTER,
    DATASET_DIR,
    CONTRAST_JITTER,
    CROP_MAX_ASPECT_RATIO,
    CROP_MAX_SCALE,
    CROP_MIN_ASPECT_RATIO,
    CROP_MIN_SCALE,
    DEEP_BACKBONE_LR_MULTIPLIER,
    EARLY_STOPPING_PATIENCE,
    FINE_TUNE_FOCAL_GAMMA,
    FREEZE_BATCHNORM_STATS,
    HEAD_FOCAL_GAMMA,
    HEAD_LEARNING_RATE,
    HEAD_OPTIMIZER,
    PHASE_2_TRANSFER_DATASET_DIR,
    PHASE_2_INIT_CHECKPOINT_PATH,
    TRAIN_CONFUSION_MATRIX_PATH,
    HEAD_DROPOUT_RATE,
    HEAD_LR_MULTIPLIER,
    HORIZONTAL_FLIP_PROBABILITY,
    IMAGE_SIZE,
    LABEL_SMOOTHING,
    LR_SCHEDULER_ENABLED,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MIN_LEARNING_RATE,
    MODEL_NAME,
    PHASE_1_EPOCHS,
    PHASE_1_FINAL_MODEL_PATH,
    PHASE_1_FINE_TUNE_STAGES,
    PHASE_2_TRANSFER_CONFUSION_MATRIX_PATH,
    PHASE_2_TRANSFER_LEARNING_STAGES,
    PHASE_2_TRANSFER_MISCLASSIFICATION_OUTPUT_DIR,
    RUN_PHASE_1,
    RUN_PHASE_2_TRANSFER,
    PHASE_2_TRANSFER_MODEL_PATH,
    PHASE_2_TRANSFER_SUMMARY_TXT_PATH,
    PHASE_2_TRANSFER_TRAINING_RESULTS_PLOT_PATH,
    PHASE_2_TRANSFER_UNFREEZE_LAYERS,
    RMSPROP_ALPHA,
    ROTATION_DEGREES,
    SPLIT_PERCENT,
    TRAIN_MISCLASSIFICATION_OUTPUT_DIR,
    TRAINING_RESULTS_PLOT_PATH,
    TRAINING_SUMMARY_TXT_PATH,
    SATURATION_JITTER,
    SGD_MOMENTUM,
    SHALLOW_BACKBONE_LR_MULTIPLIER,
    UNFREEZE_LAYERS,
    USE_CLASS_WEIGHTS,
    VERTICAL_FLIP_PROBABILITY,
    WEIGHT_DECAY,
)
from backend.utils_misclassification import export_false_positive_false_negative_examples
from backend.model_factory import build_model
from backend.utils_plotting import (
    apply_plot_style,
    build_plot_settings,
    plot_confusion_matrix,
    plot_training_results,
)

from backend.utils_training import (
    DEVICE,
    build_weighted_criterion,
    calculate_class_weights,
    evaluate_model,
    freeze_all_and_unfreeze_last,
    get_optimizer,
    load_checkpoint_or_exit,
    load_datasets,
    train_one_epoch,
    validate_one_epoch,
)

apply_plot_style()

PLOT_SETTINGS = build_plot_settings(
    model_name=MODEL_NAME,
    save_path=TRAIN_CONFUSION_MATRIX_PATH,
    colormap=COLORMAP,
    misclassification_output_dir=TRAIN_MISCLASSIFICATION_OUTPUT_DIR,
)

MISCLASSIFICATION_SETTINGS = PLOT_SETTINGS

PHASE_2_TRANSFER_PLOT_SETTINGS = build_plot_settings(
    model_name=f"{MODEL_NAME} Phase 2 Transfer Learning",
    save_path=PHASE_2_TRANSFER_CONFUSION_MATRIX_PATH,
    colormap=COLORMAP,
    misclassification_output_dir=PHASE_2_TRANSFER_MISCLASSIFICATION_OUTPUT_DIR,
)

PHASE_2_TRANSFER_MISCLASSIFICATION_SETTINGS = PHASE_2_TRANSFER_PLOT_SETTINGS
TRAINING_EVENT_CALLBACK = None
TRAINING_STOP_EVENT = None
PRETRAINED_MODEL_PATH = ""
EXPORT_MISCLASSIFICATION_IMAGES = False


class TrainingStopped(Exception):
    """Raised when the UI requests a cooperative training stop."""


def should_stop_training():
    return TRAINING_STOP_EVENT is not None and TRAINING_STOP_EVENT.is_set()


def check_training_stop():
    if should_stop_training():
        raise TrainingStopped("Training stopped by user.")


def emit_training_event(event):
    if TRAINING_EVENT_CALLBACK is not None:
        TRAINING_EVENT_CALLBACK(event)


def get_learning_rates(optimizer):
    """Return the current learning rate for each optimizer parameter group."""
    return [param_group["lr"] for param_group in optimizer.param_groups]


def build_empty_history():
    """Create history container shared across Phase 1 substeps."""
    return {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "escape_rate": [],
        "false_alarm_rate": [],
    }


def build_augmentation_settings():
    return {
        "crop_min_scale": CROP_MIN_SCALE,
        "crop_max_scale": CROP_MAX_SCALE,
        "crop_min_aspect_ratio": CROP_MIN_ASPECT_RATIO,
        "crop_max_aspect_ratio": CROP_MAX_ASPECT_RATIO,
        "rotation_degrees": ROTATION_DEGREES,
        "horizontal_flip_probability": HORIZONTAL_FLIP_PROBABILITY,
        "vertical_flip_probability": VERTICAL_FLIP_PROBABILITY,
        "brightness_jitter": BRIGHTNESS_JITTER,
        "contrast_jitter": CONTRAST_JITTER,
        "saturation_jitter": SATURATION_JITTER,
    }


def make_optimizer(model, optimizer_name, learning_rate):
    return get_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        sgd_momentum=SGD_MOMENTUM,
        rmsprop_alpha=RMSPROP_ALPHA,
        head_lr_multiplier=HEAD_LR_MULTIPLIER,
        deep_lr_multiplier=DEEP_BACKBONE_LR_MULTIPLIER,
        shallow_lr_multiplier=SHALLOW_BACKBONE_LR_MULTIPLIER,
    )


def assert_same_class_names(expected_class_names, actual_class_names, dataset_label):
    """Fail fast if transfer-learning dataset class order differs from Phase 1."""
    if expected_class_names != actual_class_names:
        raise ValueError(
            f"{dataset_label} class order mismatch. "
            f"Expected {expected_class_names}, found {actual_class_names}."
        )


def run_phase_1_head_tuning(train_loader, val_loader, class_names, class_weight_dict, num_classes):
    """Run Phase 1 head-tuning substep."""
    print(f"\n=== RUNNING {MODEL_NAME} PHASE 1: HEAD TUNING SUBSTEP ===")

    model = build_model(num_classes, HEAD_DROPOUT_RATE)
    if PRETRAINED_MODEL_PATH:
        print(f"[Checkpoint] Loading pretrained model: {PRETRAINED_MODEL_PATH}")
        model = load_checkpoint_or_exit(
            model, PRETRAINED_MODEL_PATH, MODEL_NAME,
            checkpoint_label="pretrained model", device=DEVICE,
        )

    optimizer = make_optimizer(model, optimizer_name=HEAD_OPTIMIZER, learning_rate=HEAD_LEARNING_RATE)
    criterion = build_weighted_criterion(
        class_weight_dict,
        num_classes,
        gamma=HEAD_FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        use_class_weights=USE_CLASS_WEIGHTS,
        device=DEVICE,
    )

    history = build_empty_history()

    for epoch in range(PHASE_1_EPOCHS):
        check_training_stop()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            freeze_batchnorm_stats=FREEZE_BATCHNORM_STATS,
            check_stop=check_training_stop,
        )
        val_loss, val_acc, escape_rate, false_alarm_rate = validate_one_epoch(
            model,
            val_loader,
            criterion,
            class_names,
            check_stop=check_training_stop,
        )
        check_training_stop()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["escape_rate"].append(escape_rate)
        history["false_alarm_rate"].append(false_alarm_rate)

        print(
            f"Epoch {epoch+1}/{PHASE_1_EPOCHS} "
            f"(Phase 1 head tuning) - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc * 100:.2f}% - "
            f"Escape: {escape_rate * 100:.2f}% - "
            f"False Alarms: {false_alarm_rate * 100:.2f}%"
        )
        emit_training_event(
            {
                "type": "epoch",
                "phase": "phase1_head",
                "stage": HEAD_OPTIMIZER,
                "epoch": epoch + 1,
                "stage_epochs": PHASE_1_EPOCHS,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "escape_rate": escape_rate,
                "false_alarm_rate": false_alarm_rate,
                "learning_rates": get_learning_rates(optimizer),
            }
        )

    return model, history


def load_phase_1_model(num_classes):
    """Load saved final Phase 1 checkpoint."""
    print(f"\n=== SKIPPING PHASE 1 TRAINING: LOADING FINAL {MODEL_NAME} CHECKPOINT ===")

    model = build_model(num_classes, HEAD_DROPOUT_RATE)
    return load_checkpoint_or_exit(
        model,
        PHASE_1_FINAL_MODEL_PATH,
        MODEL_NAME,
        checkpoint_label="final Phase 1 checkpoint",
        device=DEVICE,
    )


def load_model_from_checkpoint(num_classes, checkpoint_path, checkpoint_label):
    """Build model and load weights from chosen checkpoint path."""
    model = build_model(num_classes, HEAD_DROPOUT_RATE)
    return load_checkpoint_or_exit(
        model,
        checkpoint_path,
        MODEL_NAME,
        checkpoint_label=checkpoint_label,
        device=DEVICE,
    )


def run_staged_fine_tuning(
    model,
    train_loader,
    val_loader,
    class_names,
    class_weight_dict,
    num_classes,
    stage_configs,
    unfreeze_layers,
    checkpoint_path,
    run_label,
    stage_label,
    checkpoint_label,
):
    """Run staged fine-tuning pipeline shared by Phase 1 and Phase 2."""
    print(f"\n=== RUNNING {MODEL_NAME} {run_label} ===")

    freeze_all_and_unfreeze_last(model, unfreeze_layers)
    criterion = build_weighted_criterion(
        class_weight_dict,
        num_classes,
        gamma=FINE_TUNE_FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        use_class_weights=USE_CLASS_WEIGHTS,
        device=DEVICE,
    )

    history = build_empty_history()
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = deepcopy(model.state_dict())
    best_stage_settings = None
    optimizer_epoch_usage = {}
    total_epoch = 0
    early_stopping_patience = EARLY_STOPPING_PATIENCE

    if not stage_configs:
        print(f"\n[Warning]: No stages configured for {run_label}. Skipping staged fine-tuning.")
        fallback_settings = {
            "optimizer": "n/a",
            "learning_rate": 0.0,
            "epochs": 0,
            "unfreeze_layers": unfreeze_layers,
            "batch_size": BATCH_SIZE,
            "dropout_rate": HEAD_DROPOUT_RATE,
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_model_state, checkpoint_path)
        return model, fallback_settings, optimizer_epoch_usage, history

    for stage in stage_configs:
        check_training_stop()
        # Reset patience tracking at the start of each unique optimization stage
        epochs_without_improvement = 0
        stage_epochs_completed = 0

        stage_settings = {
            "optimizer": stage["optimizer"],
            "learning_rate": stage["learning_rate"],
            "epochs": stage["epochs"],
            "unfreeze_layers": unfreeze_layers,
            "batch_size": BATCH_SIZE,
            "dropout_rate": HEAD_DROPOUT_RATE,
        }
        print(
            f"\nRunning {stage_label} with {stage_settings['optimizer'].upper()} "
            f"at learning rate {stage_settings['learning_rate']} for {stage_settings['epochs']} epochs..."
        )

        optimizer = make_optimizer(
            model,
            optimizer_name=stage["optimizer"],
            learning_rate=stage["learning_rate"]
        )

        lr_scheduler = None
        if LR_SCHEDULER_ENABLED:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
            )

        for epoch in range(stage["epochs"]):
            check_training_stop()
            total_epoch += 1
            stage_epochs_completed += 1
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                freeze_batchnorm_stats=FREEZE_BATCHNORM_STATS,
                check_stop=check_training_stop,
            )
            val_loss, val_acc, escape_rate, false_alarm_rate = validate_one_epoch(
                model,
                val_loader,
                criterion,
                class_names,
                check_stop=check_training_stop,
            )
            check_training_stop()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["escape_rate"].append(escape_rate)
            history["false_alarm_rate"].append(false_alarm_rate)

            print(
                f"Epoch {epoch+1}/{stage['epochs']} "
                f"({run_label} total {total_epoch}) - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc * 100:.2f}% - "
                f"Escape Rate: {escape_rate * 100:.2f}% - "
                f"False Alarms: {false_alarm_rate * 100:.2f}%"
            )
            emit_training_event(
                {
                    "type": "epoch",
                    "phase": run_label,
                    "stage": stage["optimizer"],
                    "epoch": epoch + 1,
                    "stage_epochs": stage["epochs"],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "escape_rate": escape_rate,
                    "false_alarm_rate": false_alarm_rate,
                    "learning_rates": get_learning_rates(optimizer),
                }
            )

            previous_lrs = get_learning_rates(optimizer)
            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            current_lrs = get_learning_rates(optimizer)

            if current_lrs != previous_lrs:
                for group_index, (old_lr, new_lr) in enumerate(zip(previous_lrs, current_lrs), start=1):
                    if old_lr != new_lr:
                        print(
                            f"[LR SCHEDULER] Reduced learning rate for param group {group_index} "
                            f"from {old_lr:.2e} to {new_lr:.2e} at total epoch {total_epoch}."
                        )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = total_epoch
                best_model_state = deepcopy(model.state_dict())
                best_stage_settings = dict(stage_settings)
                best_stage_settings["epochs"] = total_epoch
                epochs_without_improvement = 0
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_model_state, checkpoint_path)
                print(
                    f"New best validation loss at Total Epoch {total_epoch}: "
                    f"{best_val_loss:.4f}. {checkpoint_label} updated."
                )
            else:
                epochs_without_improvement += 1
                print(f"No val_loss improvement for {epochs_without_improvement} epoch(s). Patience: {early_stopping_patience}.")

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"\n[STAGE EARLY STOP] Triggered at {stage['optimizer'].upper()} epoch {epoch+1}. "
                    f"Advancing to next {stage_label}..."
                )
                # Load the best overall weights up to this point so the next optimizer builds on top of them
                model.load_state_dict(best_model_state)
                break

        optimizer_epoch_usage[stage["optimizer"]] = (
            optimizer_epoch_usage.get(stage["optimizer"], 0) + stage_epochs_completed
        )

    model.load_state_dict(best_model_state)
    print(
        f"\n=== {run_label} Complete ===\n"
        f"Restored absolute best global weights from Total Epoch {best_epoch} "
        f"with absolute minimum val_loss {best_val_loss:.4f}."
    )

    print(f"Saving {MODEL_NAME} {checkpoint_label.lower()} to {checkpoint_path}...")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    return model, best_stage_settings, optimizer_epoch_usage, history


def run_phase_1_fine_tuning_substep(
    model,
    train_loader,
    val_loader,
    class_names,
    class_weight_dict,
    num_classes,
):
    """Run Phase 1 fine-tuning substep across sequential optimizer stages."""
    return run_staged_fine_tuning(
        model,
        train_loader,
        val_loader,
        class_names,
        class_weight_dict,
        num_classes,
        stage_configs=PHASE_1_FINE_TUNE_STAGES,
        unfreeze_layers=UNFREEZE_LAYERS,
        checkpoint_path=PHASE_1_FINAL_MODEL_PATH,
        run_label="PHASE 1: FINE-TUNING SUBSTEP",
        stage_label="Phase 1 fine-tuning stage",
        checkpoint_label="Final Phase 1 checkpoint",
    )


def merge_histories(*histories):
    """Merge sequential training histories into one plot-ready history."""
    merged_history = build_empty_history()
    for history in histories:
        for key in merged_history:
            merged_history[key].extend(history.get(key, []))
    return merged_history


def build_phase_1_evaluation_settings():
    """Build fallback settings for evaluation when training metadata is unavailable."""
    return {
        "optimizer": "phase1-checkpoint",
        "learning_rate": 0.0,
        "epochs": 0,
        "unfreeze_layers": UNFREEZE_LAYERS,
        "batch_size": BATCH_SIZE,
        "dropout_rate": HEAD_DROPOUT_RATE,
    }


def run_phase_1(train_loader, val_loader, class_names, class_weight_dict, num_classes):
    """Run complete Phase 1 pipeline: head tuning followed by fine-tuning."""
    print(f"\n=== RUNNING {MODEL_NAME} PHASE 1: FULL TRAINING PIPELINE ===")
    model, head_history = run_phase_1_head_tuning(
        train_loader,
        val_loader,
        class_names,
        class_weight_dict,
        num_classes,
    )
    model, best_stage_settings, optimizer_epoch_usage, fine_tune_history = run_phase_1_fine_tuning_substep(
        model,
        train_loader,
        val_loader,
        class_names,
        class_weight_dict,
        num_classes,
    )
    combined_history = merge_histories(head_history, fine_tune_history)
    return model, best_stage_settings, optimizer_epoch_usage, combined_history


def run_phase_2_transfer_learning(expected_class_names):
    """Run Phase 2 transfer learning on same-class dataset."""
    if not PHASE_2_TRANSFER_DATASET_DIR:
        raise ValueError("PHASE_2_TRANSFER_DATASET_DIR is empty. Set transfer-learning dataset path first.")
    check_training_stop()

    print(f"\n=== RUNNING {MODEL_NAME} PHASE 2: TRANSFER LEARNING ===")
    transfer_train_loader, transfer_val_loader, transfer_class_names, transfer_val_file_paths = load_datasets(
        PHASE_2_TRANSFER_DATASET_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        augmentation_settings=build_augmentation_settings(),
        split_percent=SPLIT_PERCENT,
    )
    assert_same_class_names(expected_class_names, transfer_class_names, "Phase 2 transfer-learning dataset")

    class_weight_dict, num_classes = calculate_class_weights(transfer_train_loader)
    model = load_model_from_checkpoint(
        num_classes,
        PHASE_2_INIT_CHECKPOINT_PATH,
        checkpoint_label="Phase 2 init checkpoint",
    )
    model, best_stage_settings, optimizer_epoch_usage, history = run_staged_fine_tuning(
        model,
        transfer_train_loader,
        transfer_val_loader,
        transfer_class_names,
        class_weight_dict,
        num_classes,
        stage_configs=PHASE_2_TRANSFER_LEARNING_STAGES,
        unfreeze_layers=PHASE_2_TRANSFER_UNFREEZE_LAYERS,
        checkpoint_path=PHASE_2_TRANSFER_MODEL_PATH,
        run_label="PHASE 2: TRANSFER LEARNING",
        stage_label="Phase 2 transfer-learning stage",
        checkpoint_label="Phase 2 transfer-learning checkpoint",
    )

    evaluate_model(
        model,
        transfer_val_loader,
        transfer_class_names,
        best_stage_settings,
        model_name=f"{MODEL_NAME} (Phase 2 Transfer)",
        optimizer_epoch_usage=optimizer_epoch_usage,
        print_summary=True,
        summary_output_path=PHASE_2_TRANSFER_SUMMARY_TXT_PATH,
    )
    plot_training_results(
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        val_accs=history["val_acc"],
        escape_rates=history["escape_rate"],
        false_alarm_rates=history["false_alarm_rate"],
        output_path=PHASE_2_TRANSFER_TRAINING_RESULTS_PLOT_PATH,
    )
    plot_confusion_matrix(
        model,
        transfer_val_loader,
        transfer_class_names,
        PHASE_2_TRANSFER_PLOT_SETTINGS,
        evaluation_settings=None,
        optimizer_epoch_usage=optimizer_epoch_usage,
    )
    if EXPORT_MISCLASSIFICATION_IMAGES:
        export_false_positive_false_negative_examples(
            model,
            transfer_val_loader,
            transfer_class_names,
            transfer_val_file_paths,
            PHASE_2_TRANSFER_MISCLASSIFICATION_SETTINGS,
        )


def main(event_callback=None, stop_event=None):
    """Run Phase 1 training or load final Phase 1 checkpoint, then evaluate."""
    global TRAINING_EVENT_CALLBACK, TRAINING_STOP_EVENT
    TRAINING_EVENT_CALLBACK = event_callback
    TRAINING_STOP_EVENT = stop_event
    check_training_stop()
    # Note: train_ds/val_ds are now PyTorch DataLoaders
    train_loader, val_loader, class_names, val_file_paths = load_datasets(
        DATASET_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        augmentation_settings=build_augmentation_settings(),
        split_percent=SPLIT_PERCENT,
    )
    class_weight_dict, num_classes = calculate_class_weights(train_loader)

    # When Phase 1 is skipped and Phase 2 is enabled, go directly to transfer
    # learning. PHASE_2_INIT_CHECKPOINT_PATH is supplied by the UI bridge from
    # the selected Pretrained Model, so no new-version Phase 1 checkpoint should
    # be loaded first.
    if RUN_PHASE_2_TRANSFER and not RUN_PHASE_1:
        check_training_stop()
        if not PHASE_2_INIT_CHECKPOINT_PATH:
            raise ValueError(
                "Phase 2 initialization checkpoint is empty. Select a Pretrained Model."
            )
        print(
            f"[Checkpoint] Direct Phase 2 initialization: "
            f"{PHASE_2_INIT_CHECKPOINT_PATH}"
        )
        run_phase_2_transfer_learning(class_names)
        return

    if RUN_PHASE_1:
        model, best_stage_settings, optimizer_epoch_usage, combined_history = run_phase_1(
            train_loader,
            val_loader,
            class_names,
            class_weight_dict,
            num_classes,
        )
    else:
        check_training_stop()
        model = load_phase_1_model(num_classes)
        best_stage_settings = build_phase_1_evaluation_settings()
        optimizer_epoch_usage = {}
        combined_history = None

    if RUN_PHASE_2_TRANSFER:
        check_training_stop()
        run_phase_2_transfer_learning(class_names)
        return

    evaluate_model(
        model,
        val_loader,
        class_names,
        best_stage_settings,
        model_name=MODEL_NAME,
        optimizer_epoch_usage=optimizer_epoch_usage,
        print_summary=True,
        summary_output_path=TRAINING_SUMMARY_TXT_PATH,
    )
    if combined_history is not None:
        plot_training_results(
            train_losses=combined_history["train_loss"],
            val_losses=combined_history["val_loss"],
            val_accs=combined_history["val_acc"],
            escape_rates=combined_history["escape_rate"],
            false_alarm_rates=combined_history["false_alarm_rate"],
            output_path=TRAINING_RESULTS_PLOT_PATH,
        )
    plot_confusion_matrix(
        model,
        val_loader,
        class_names,
        PLOT_SETTINGS,
        evaluation_settings=None,
        optimizer_epoch_usage=optimizer_epoch_usage,
    )
    if EXPORT_MISCLASSIFICATION_IMAGES:
        export_false_positive_false_negative_examples(
            model,
            val_loader,
            class_names,
            val_file_paths,
            MISCLASSIFICATION_SETTINGS,
        )


if __name__ == "__main__":
    main()
