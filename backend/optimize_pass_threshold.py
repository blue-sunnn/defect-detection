import csv
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config_threshold_optimization import (
    DEFAULT_MULTIPLIER_BOUNDS,
    DEFAULT_RESULTS_CSV_PATH,
    DEPLOYED_MULTIPLIERS_JSON_PATH,
    PASS_CLASS_NAME,
    VAL_CLASSES_PATH,
    VAL_PROB_PATH,
    VAL_TRUE_PATH,
    save_multipliers_to_json,
    load_multipliers_from_json,
    load_classification_costs_from_json
)

# --- Define Global Variables ---
active_multipliers = None
classification_costs = None

def load_artifacts(true_path, prob_path, classes_path, json_path, costs_json_path):
    """Loads artifacts and assigns them to global configuration variables."""
    global active_multipliers, classification_costs  # Tell Python to modify the global variables
    
    y_true = np.load(true_path)
    y_prob = np.load(prob_path)
    class_names = np.load(classes_path, allow_pickle=True).tolist()
    
    # Load only once into global scope
    active_multipliers = load_multipliers_from_json(json_path, class_names)
    classification_costs = load_classification_costs_from_json(costs_json_path, class_names)

    return y_true, y_prob, class_names

def validate_inputs(y_true, y_prob, class_names, classification_costs_map=None):
    if y_prob.ndim != 2 or y_true.ndim != 1:
        raise ValueError("Invalid array dimensions for y_true or y_prob.")
    if len(y_true) != len(y_prob):
        raise ValueError("Mismatched sample counts.")
    if PASS_CLASS_NAME not in class_names:
        raise ValueError(f"Required class '{PASS_CLASS_NAME}' not found.")
    
    costs = classification_costs_map if classification_costs_map is not None else classification_costs
    if costs is None:
        raise ValueError("Classification costs are not loaded.")
    missing_cost_rows = [class_name for class_name in class_names if class_name not in costs]
    if missing_cost_rows:
        raise ValueError(
            f"Missing CLASSIFICATION_COSTS rows for classes: {missing_cost_rows}"
        )
    missing_cost_columns = []
    for true_name in class_names:
        for pred_name in class_names:
            if pred_name not in costs[true_name]:
                missing_cost_columns.append((true_name, pred_name))
    if missing_cost_columns:
        raise ValueError(
            f"Missing CLASSIFICATION_COSTS entries for class pairs: {missing_cost_columns}"
        )

def build_cost_matrix(class_names, classification_costs_map=None):
    global classification_costs
    costs = classification_costs_map
    if costs is None:
        if classification_costs is None:
            classification_costs = load_classification_costs_from_json("classification_costs.json", class_names)
        costs = classification_costs
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for true_idx, true_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            matrix[true_idx, pred_idx] = costs[true_name][pred_name]
    return matrix

def evaluate_5d_threshold(multipliers, y_true, y_prob, class_names, cost_matrix):
    """Evaluates the confusion matrix and metrics for a given 5D multiplier vector."""
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for multiplier evaluation. Install scikit-learn or skip defect multipliers.") from exc
    try:
        from backend.utils_misclassification import apply_defect_multipliers
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for multiplier evaluation. Install torch or skip defect multipliers.") from exc

    pass_idx = class_names.index(PASS_CLASS_NAME)
    defect_indices = [idx for idx, name in enumerate(class_names) if name != PASS_CLASS_NAME]

    defect_multipliers = {
        class_names[defect_idx]: multipliers[i]
        for i, defect_idx in enumerate(defect_indices)
    }
    adjusted_scores = apply_defect_multipliers(y_prob, class_names, defect_multipliers=defect_multipliers)

    y_pred = np.argmax(adjusted_scores, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    total_cost = int(np.sum(cm * cost_matrix))
    escapes = int(sum(cm[idx, pass_idx] for idx in defect_indices))
    false_alarms = int(np.sum(cm[pass_idx, :]) - cm[pass_idx, pass_idx])
    total_samples = len(y_true)
    overall_far_pct = (false_alarms / total_samples) * 100

    total_true_defect = int(sum(np.sum(cm[idx, :]) for idx in defect_indices))
    escape_rate = escapes / total_true_defect if total_true_defect > 0 else 0.0

    total_true_pass = int(np.sum(cm[pass_idx, :]))
    false_alarm_rate = false_alarms / total_true_pass if total_true_pass > 0 else 0.0

    # Binary deployment view: every non-Pass class is treated as a defect.
    true_positive = total_true_defect - escapes
    false_negative = escapes
    false_positive = false_alarms
    true_negative = int(cm[pass_idx, pass_idx])
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0

    return {
        "multipliers": np.round(multipliers, 4).tolist(),
        "total_cost": total_cost,
        "escapes": escapes,
        "false_alarms": false_alarms,
        "overall_far_pct": overall_far_pct,
        "escape_rate": escape_rate,
        "false_alarm_rate": false_alarm_rate,
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
        "false_negative": int(false_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }

def objective_function(
    multipliers,
    y_true,
    y_prob,
    class_names,
    cost_matrix,
    max_escapes=0,
    target_far_pct=10.0,
    escape_penalty=1_000_000_000,
    false_alarm_penalty=5_000_000,
    far_penalty_exponent=3,
    continuous_far_pressure=50_000,
):
    metrics = evaluate_5d_threshold(multipliers, y_true, y_prob, class_names, cost_matrix)

    cost = metrics["total_cost"]
    escapes = metrics["escapes"]
    overall_far_pct = metrics["overall_far_pct"]

    penalty = 0.0
    excess_escapes = max(0, escapes - int(max_escapes))
    if excess_escapes:
        penalty += float(escape_penalty) * excess_escapes

    if overall_far_pct > float(target_far_pct):
        far_excess = overall_far_pct - float(target_far_pct)
        penalty += (far_excess ** int(far_penalty_exponent)) * float(false_alarm_penalty)

    penalty += overall_far_pct * float(continuous_far_pressure)
    return cost + penalty

def run_optimization_from_arrays(
    y_true,
    y_prob,
    class_names,
    *,
    classification_costs_map,
    results_csv_path=None,
    max_escapes=0,
    target_far_pct=10.0,
    escape_penalty=1_000_000_000,
    false_alarm_penalty=5_000_000,
    k_folds=5,
    max_iterations=100,
    population_size=25,
    strategy="best1bin",
    random_seed=42,
    far_penalty_exponent=3,
    continuous_far_pressure=50_000,
    save_deployed=False,
    deployed_json_path=DEPLOYED_MULTIPLIERS_JSON_PATH,
):
    """Run the existing cross-validated differential-evolution optimizer on cached arrays."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError("SciPy is required for multiplier optimization. Install scipy or skip defect multipliers.") from exc
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for multiplier optimization. Install scikit-learn or skip defect multipliers.") from exc

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    class_names = list(class_names)
    validate_inputs(y_true, y_prob, class_names, classification_costs_map)
    cost_matrix = build_cost_matrix(class_names, classification_costs_map)
    defect_class_names = [name for name in class_names if name != PASS_CLASS_NAME]
    bounds = [DEFAULT_MULTIPLIER_BOUNDS for _ in defect_class_names]
    k_folds = int(k_folds)
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2.")
    counts = np.bincount(y_true, minlength=len(class_names))
    positive_counts = counts[counts > 0]
    if not len(positive_counts) or k_folds > int(positive_counts.min()):
        raise ValueError("k_folds cannot exceed the sample count of the smallest class.")
    valid_strategies = {"best1bin", "best1exp", "rand1bin", "rand1exp", "rand2bin", "rand2exp", "best2bin", "best2exp", "currenttobest1bin", "currenttobest1exp", "randtobest1bin", "randtobest1exp"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unsupported differential evolution strategy: {strategy}")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=int(random_seed))
    fold_best_multipliers = []
    def cv_objective(multipliers, train_indices):
        return objective_function(multipliers, y_true[train_indices], y_prob[train_indices], class_names, cost_matrix, max_escapes=max_escapes, target_far_pct=target_far_pct, escape_penalty=escape_penalty, false_alarm_penalty=false_alarm_penalty, far_penalty_exponent=far_penalty_exponent, continuous_far_pressure=continuous_far_pressure)
    for fold, (train_idx, _val_idx) in enumerate(skf.split(y_prob, y_true)):
        result = differential_evolution(func=cv_objective, bounds=bounds, args=(train_idx,), strategy=strategy, maxiter=int(max_iterations), popsize=int(population_size), seed=int(random_seed)+fold, disp=False)
        fold_best_multipliers.append(result.x)
    optimized_multipliers = np.mean(fold_best_multipliers, axis=0)
    best_metrics = evaluate_5d_threshold(optimized_multipliers, y_true, y_prob, class_names, cost_matrix)
    if results_csv_path:
        os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)
        with open(results_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([*[f"{name.lower()}_mult" for name in defect_class_names], "total_cost", "escapes", "false_alarms", "overall_far_pct", "escape_rate", "false_alarm_rate", "escape_rate_pct", "false_alarm_rate_pct"])
            writer.writerow(best_metrics["multipliers"] + [best_metrics["total_cost"], best_metrics["escapes"], best_metrics["false_alarms"], best_metrics["overall_far_pct"], best_metrics["escape_rate"], best_metrics["false_alarm_rate"], best_metrics["escape_rate"]*100, best_metrics["false_alarm_rate"]*100])
    if save_deployed:
        save_multipliers_to_json(dict(zip(defect_class_names, best_metrics["multipliers"])), json_path=deployed_json_path)
    return best_metrics


def run_optimization_sweep(
    true_path=VAL_TRUE_PATH,
    prob_path=VAL_PROB_PATH,
    classes_path=VAL_CLASSES_PATH,
    results_csv_path=None,
    max_escapes=0,
    target_far_pct=10.0,
    escape_penalty=1_000_000_000,
    false_alarm_penalty=5_000_000,
    k_folds=5,
    max_iterations=100,
    population_size=25,
    strategy="best1bin",
    random_seed=42,
    far_penalty_exponent=3,
    continuous_far_pressure=50_000,
):
    """Run the legacy file-based entry point and save its deployed JSON for compatibility."""
    y_true, y_prob, class_names = load_artifacts(true_path, prob_path, classes_path, json_path=DEPLOYED_MULTIPLIERS_JSON_PATH, costs_json_path="classification_costs.json")
    return run_optimization_from_arrays(
        y_true, y_prob, class_names,
        classification_costs_map=classification_costs,
        results_csv_path=results_csv_path, max_escapes=max_escapes, target_far_pct=target_far_pct,
        escape_penalty=escape_penalty, false_alarm_penalty=false_alarm_penalty, k_folds=k_folds,
        max_iterations=max_iterations, population_size=population_size, strategy=strategy, random_seed=random_seed,
        far_penalty_exponent=far_penalty_exponent, continuous_far_pressure=continuous_far_pressure,
        save_deployed=True, deployed_json_path=DEPLOYED_MULTIPLIERS_JSON_PATH,
    )

if __name__ == "__main__":
    # Call the optimization sweep once (it handles loading internally now)
    run_optimization_sweep(
        true_path=VAL_TRUE_PATH,
        prob_path=VAL_PROB_PATH,
        classes_path=VAL_CLASSES_PATH,
        results_csv_path=DEFAULT_RESULTS_CSV_PATH,
    )
    
    # You can safely read the global variable anytime after run_optimization_sweep runs
    print(f"Current deployed defect multipliers: {active_multipliers}")
