import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap


MODEL_NAME = "EfficientNetV2S"
MODEL_SLUG = "efficientnetv2s"

# ========================================
# 1. Define Configuration Variables
# ========================================

DATASET_DIR = None
PHASE_2_TRANSFER_DATASET_DIR = None
IMAGE_SIZE = (384, 384)
BATCH_SIZE = 16
SPLIT_PERCENT = 80
HEAD_DROPOUT_RATE = 0.5
UNFREEZE_LAYERS = 400
HEAD_OPTIMIZER = "adam"
HEAD_LEARNING_RATE = 1e-3
HEAD_FOCAL_GAMMA = 0.0
FINE_TUNE_FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = False
FREEZE_BATCHNORM_STATS = True
EARLY_STOPPING_PATIENCE = 12
LR_SCHEDULER_ENABLED = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-4
SGD_MOMENTUM = 0.9
RMSPROP_ALPHA = 0.9
HEAD_LR_MULTIPLIER = 1.0
DEEP_BACKBONE_LR_MULTIPLIER = 0.1
SHALLOW_BACKBONE_LR_MULTIPLIER = 0.005
CROP_MIN_SCALE = 0.85
CROP_MAX_SCALE = 1.0
CROP_MIN_ASPECT_RATIO = 0.9
CROP_MAX_ASPECT_RATIO = 1.1
ROTATION_DEGREES = 180
HORIZONTAL_FLIP_PROBABILITY = 0.5
VERTICAL_FLIP_PROBABILITY = 0.5
BRIGHTNESS_JITTER = 0.1
CONTRAST_JITTER = 0.1
SATURATION_JITTER = 0.1
PHASE_2_TRANSFER_UNFREEZE_LAYERS = 5
PHASE_1_EPOCHS = 15
PHASE_1_FINE_TUNE_STAGES = [
    # {"optimizer": "adam", "learning_rate": 1e-6, "epochs": 5},
    {"optimizer": "adamw", "learning_rate": 1e-4, "epochs": 50},
    {"optimizer": "sgd", "learning_rate": 5e-5, "epochs": 60},
    # {"optimizer": "rmsprop", "learning_rate": 1e-6, "epochs": 5},
]

# Reserved for future transfer learning implementation.
PHASE_2_TRANSFER_LEARNING_STAGES = [
    {"optimizer": "adamw", "learning_rate": 5e-6, "epochs": 30},
]

# --- CONTROLLER FLAGS ---
RUN_PHASE_1 = False
RUN_PHASE_2_TRANSFER = True
PHASE_2_RUN = RUN_PHASE_2_TRANSFER
PHASE_2_DATASET_DIR = PHASE_2_TRANSFER_DATASET_DIR

ARTIFACTS_DIR = "artifacts"

TRAIN_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, "train")
TRAIN_CHECKPOINTS_DIR = os.path.join(TRAIN_ARTIFACTS_DIR, "checkpoints")
TRAIN_PLOTS_DIR = os.path.join(TRAIN_ARTIFACTS_DIR, "plots")
TRAIN_SUMMARIES_DIR = os.path.join(TRAIN_ARTIFACTS_DIR, "summaries")
TRAIN_GRAD_CAM_DIR = os.path.join(TRAIN_ARTIFACTS_DIR, "grad_cam")

VALIDATION_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, "validation")
VALIDATION_PLOTS_DIR = os.path.join(VALIDATION_ARTIFACTS_DIR, "plots")
VALIDATION_SUMMARIES_DIR = os.path.join(VALIDATION_ARTIFACTS_DIR, "summaries")
VALIDATION_GRAD_CAM_DIR = os.path.join(VALIDATION_ARTIFACTS_DIR, "grad_cam")

TEST_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, "test")
TEST_PLOTS_DIR = os.path.join(TEST_ARTIFACTS_DIR, "plots")
TEST_SUMMARIES_DIR = os.path.join(TEST_ARTIFACTS_DIR, "summaries")
TEST_GRAD_CAM_DIR = os.path.join(TEST_ARTIFACTS_DIR, "grad_cam")

PHASE_2_TRANSFER_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, "phase2_transfer")
PHASE_2_TRANSFER_PLOTS_DIR = os.path.join(PHASE_2_TRANSFER_ARTIFACTS_DIR, "plots")
PHASE_2_TRANSFER_SUMMARIES_DIR = os.path.join(PHASE_2_TRANSFER_ARTIFACTS_DIR, "summaries")
PHASE_2_TRANSFER_GRAD_CAM_DIR = os.path.join(PHASE_2_TRANSFER_ARTIFACTS_DIR, "grad_cam")

PHASE_1_FINAL_MODEL_PATH = os.path.join(
    TRAIN_CHECKPOINTS_DIR,
    f"{MODEL_SLUG}_phase1_final_checkpoint.pth",
)
PHASE_2_TRANSFER_MODEL_PATH = os.path.join(
    TRAIN_CHECKPOINTS_DIR,
    f"{MODEL_SLUG}_phase2_transfer_checkpoint.pth",
)
PHASE_2_INIT_CHECKPOINT_PATH = PHASE_1_FINAL_MODEL_PATH

# --- PLOTTING SETTINGS ---
TRAIN_CONFUSION_MATRIX_PATH = os.path.join(TRAIN_PLOTS_DIR, "confusion_matrix.png")
VALIDATION_CONFUSION_MATRIX_PATH = os.path.join(VALIDATION_PLOTS_DIR, "confusion_matrix.png")
TEST_CONFUSION_MATRIX_PATH = os.path.join(TEST_PLOTS_DIR, "confusion_matrix.png")
TRAINING_RESULTS_PLOT_PATH = os.path.join(TRAIN_PLOTS_DIR, "training_metrics_dashboard.png")
PHASE_2_TRANSFER_CONFUSION_MATRIX_PATH = os.path.join(PHASE_2_TRANSFER_PLOTS_DIR, "confusion_matrix.png")
PHASE_2_TRANSFER_TRAINING_RESULTS_PLOT_PATH = os.path.join(
    PHASE_2_TRANSFER_PLOTS_DIR,
    "training_metrics_dashboard.png",
)
COLORMAP = LinearSegmentedColormap.from_list(
    "darkblue",
    ["#f4f8ff", "#0b3d91"],
)
TRAIN_MISCLASSIFICATION_OUTPUT_DIR = TRAIN_GRAD_CAM_DIR
PHASE_2_TRANSFER_MISCLASSIFICATION_OUTPUT_DIR = PHASE_2_TRANSFER_GRAD_CAM_DIR
VALIDATION_MISCLASSIFICATION_OUTPUT_DIR = VALIDATION_GRAD_CAM_DIR
TRAINING_SUMMARY_TXT_PATH = os.path.join(TRAIN_SUMMARIES_DIR, "final_training_summary.txt")
PHASE_2_TRANSFER_SUMMARY_TXT_PATH = os.path.join(
    PHASE_2_TRANSFER_SUMMARIES_DIR,
    "phase2_transfer_summary.txt",
)
VALIDATION_SUMMARY_TXT_PATH = os.path.join(VALIDATION_SUMMARIES_DIR, "validation_summary.txt")
TEST_SUMMARY_TXT_PATH = os.path.join(TEST_SUMMARIES_DIR, "test_summary.txt")
GRAD_CAM_OUTPUT_DIR = TEST_GRAD_CAM_DIR
