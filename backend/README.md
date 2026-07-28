# Backend

This folder contains the model, training, evaluation, threshold, and inference code used by the desktop app.

Key files:

- `model_factory.py` builds the EfficientNet-based classifier.
- `model_config.py` stores shared model and artifact path settings.
- `utils_training.py` contains reusable training and evaluation helpers.
- `train_multiclass_defect_classifier.py` runs the main training workflow.
- `run_validation.py`, `run_test.py`, and `evaluation_service.py` handle validation and test evaluation.
- `inference_service.py` loads promoted model packages and predicts image results.
- `threshold_service.py`, `multiplier_service.py`, and related persistence files manage threshold and multiplier recommendations.
- `classification_costs.json` stores default defect-class cost settings.

The desktop UI usually calls this code through `app/backend_bridge.py`.
