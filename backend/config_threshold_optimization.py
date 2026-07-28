import json
import os

from backend.model_config import ARTIFACTS_DIR

PASS_CLASS_NAME = "Pass"
PASS_CERTAINTY_THRESHOLD = 0.75

DEFAULT_MULTIPLIER_BOUNDS = (0.00, 0.30)

THRESHOLD_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, "threshold")
THRESHOLD_DATA_DIR = os.path.join(THRESHOLD_ARTIFACTS_DIR, "data")
THRESHOLD_PLOTS_DIR = os.path.join(THRESHOLD_ARTIFACTS_DIR, "plots")
THRESHOLD_OPTIMIZATION_DIR = os.path.join(THRESHOLD_ARTIFACTS_DIR, "optimization")
THRESHOLD_DEPLOYED_DIR = os.path.join(THRESHOLD_ARTIFACTS_DIR, "deployed")

VAL_TRUE_PATH = os.path.join(THRESHOLD_DATA_DIR, "val_data_true.npy")
VAL_PROB_PATH = os.path.join(THRESHOLD_DATA_DIR, "val_data_prob.npy")
VAL_CLASSES_PATH = os.path.join(THRESHOLD_DATA_DIR, "val_data_classes.npy")
PROBABILITY_HISTOGRAM_PATH = os.path.join(THRESHOLD_PLOTS_DIR, "probability_histogram.png")
DEFAULT_RESULTS_CSV_PATH = os.path.join(THRESHOLD_OPTIMIZATION_DIR, "threshold_sweep.csv")
DEPLOYED_MULTIPLIERS_JSON_PATH = os.path.join(
    THRESHOLD_DEPLOYED_DIR,
    "deployed_multipliers.json",
)


def load_multipliers_from_json(json_source, class_names, default_value=0.0, verbose=False): # <-- Changed to verbose=False by default
    data = {}
    is_file_path = json_source.endswith('.json') or ('{' not in json_source)
    
    if is_file_path:
        if os.path.exists(json_source):
            try:
                with open(json_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if verbose: # <-- Add conditional check
                    print(f"[JSON] Successfully loaded multipliers from file: {json_source}")
            except Exception as e:
                print(f"[JSON Error] Failed to read file {json_source}: {e}.")
        else:
            if verbose: # <-- Add conditional check
                print(f"[JSON Info] '{json_source}' not found. Initializing defaults...")
            
            default_template = {name: default_value for name in class_names if name != PASS_CLASS_NAME}
            try:
                os.makedirs(os.path.dirname(json_source) or '.', exist_ok=True)
                with open(json_source, 'w', encoding='utf-8') as f:
                    json.dump(default_template, f, indent=4, ensure_ascii=False)
                if verbose: # <-- Add conditional check
                    print(f"[JSON] Created new baseline multipliers file at: {json_source}")
                data = default_template
            except Exception as e:
                print(f"[JSON Error] Could not create baseline file: {e}")
                data = default_template
    else:
        try:
            data = json.loads(json_source)
            if verbose: # <-- Add conditional check
                print("[JSON] Successfully parsed multipliers from JSON string.")
        except json.JSONDecodeError:
            data = {}

    # (Keep rest of the index parsing logic unchanged)
    multipliers = {}
    for name in class_names:
        if name == PASS_CLASS_NAME:
            continue
        if name in data:
            multipliers[name] = float(data[name])
        else:
            multipliers[name] = default_value
            
    return multipliers


def save_multipliers_to_json(multipliers, json_path="defect_multipliers.json"):
    """
    Saves a dictionary of defect multipliers out to a formatted JSON file.
    
    Args:
        multipliers (dict): Dictionary containing defect_name: value pairs.
        json_path (str): File path where the output JSON should be written.
    """
    try:
        os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(multipliers, f, indent=4, ensure_ascii=False)
        print(f"[JSON] Defect multipliers successfully saved to: {json_path}")
    except Exception as e:
        print(f"[JSON Error] Failed to write multipliers to {json_path}: {e}")

def load_classification_costs_from_json(json_source, class_names):
    """
    Loads classification cost profiles from a JSON file or JSON string and 
    maps them into a full nested dictionary matching the active class names.
    """
    # 1. Parse JSON file or fallback raw string
    data = {}
    if os.path.exists(json_source):
        try:
            with open(json_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[JSON Error] Failed to read cost file {json_source}: {e}. Using strict defaults.")
    else:
        try:
            data = json.loads(json_source)
        except json.JSONDecodeError:
            data = {}

    # 2. Extract configuration rules with fallbacks
    default_cost = data.get("DEFAULT_MISCLASSIFICATION_COST", 2)
    false_alarm_cost = data.get("DEFAULT_FALSE_ALARM_COST", 50)
    escape_cost = data.get("DEFAULT_ESCAPE_COST", 250)
    overrides = data.get("SPECIFIC_OVERRIDE_COSTS", {})

    # 3. Build the runtime mapping dynamically
    costs = {}
    for true_class in class_names:
        costs[true_class] = {}
        for pred_class in class_names:
            if true_class == pred_class:
                costs[true_class][pred_class] = 0
                continue

            # Case A: ESCAPE (Defect predicted as Pass)
            if true_class != PASS_CLASS_NAME and pred_class == PASS_CLASS_NAME:
                current_cost = escape_cost
            # Case B: FALSE ALARM (Pass predicted as a Defect)
            elif true_class == PASS_CLASS_NAME and pred_class != PASS_CLASS_NAME:
                current_cost = false_alarm_cost
            # Case C: Cross-defect nuisance
            else:
                current_cost = default_cost

            # Apply any specific pair overrides from JSON (e.g., "Short->Protrusion")
            override_key = f"{true_class}->{pred_class}"
            if override_key in overrides:
                current_cost = overrides[override_key]

            costs[true_class][pred_class] = current_cost

    return costs
