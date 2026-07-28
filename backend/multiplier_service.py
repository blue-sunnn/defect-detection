from __future__ import annotations

from typing import Any
import numpy as np

from backend.config_threshold_optimization import PASS_CLASS_NAME, load_classification_costs_from_json
from backend.optimize_pass_threshold import run_optimization_from_arrays
from backend.prediction_cache import PredictionResultCache


def optimize_multipliers(prediction_cache: PredictionResultCache, class_names: list[str] | None = None, **optimizer_options: Any) -> dict[str, Any]:
    names = list(class_names or prediction_cache.metadata.class_names)
    legacy = prediction_cache.to_legacy_predictions(names)
    y_true = np.asarray(legacy["true_labels"], dtype=np.int64)
    y_prob = np.asarray(legacy["probabilities"], dtype=np.float64)
    costs_path = optimizer_options.pop("costs_json_path", "classification_costs.json")
    costs = load_classification_costs_from_json(costs_path, names)
    metrics = run_optimization_from_arrays(y_true, y_prob, names, classification_costs_map=costs, save_deployed=False, **optimizer_options)
    defects = [name for name in names if name != PASS_CLASS_NAME]
    return {
        "schema_version": 1,
        "strategy": "differential_evolution_kfold",
        "classes": {name: {"recommended": float(value), "status": "valid"} for name, value in zip(defects, metrics["multipliers"])},
        "metrics": metrics,
    }
