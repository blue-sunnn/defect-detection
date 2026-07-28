import json
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "config.json"
BUNDLED_CONFIG_FILE = CONFIG_FILE

DEFAULT_CONFIG = {
    "shared_settings": {
        "model_name": "EfficientNetV2S",
        "model_version": "v1",
        "training_checkpoint": "",
        "dataset_root": "",
        "output_root": "",
        "output_folder": "",
        "device": "GPU",
        "batch_size": "16",
        "class_mapping": ""
    },
    "live_tab": {
        "promoted_package": "",
        "class_mapping": "",
        "input_source": "",
        "save_result_folder": "",
        "input_mode": "Watcher Mode",
        "polling_interval_ms": "2000",
        "device": "CPU",
        "result_format": "JSON"
    },
    "dataset_tab": {
        "json_log_source": "",
        "ng_folder_source": "",
        "ok_folder_source": "",
        "destination_root": "",
        "promoted_package": "",
        "review_filter_mode": "All"
    },
    "training_tab": {
        "model_name": "EfficientNetV2S",
        "model_version": "v1",
        "dataset_root": "",
        "phase_2_dataset_root": "",
        "split_output_path": "",
        "save_model_path": "",
        "artifacts_dir": "",
        "image_width": "384",
        "image_height": "384",
        "batch_size": "16",
        "split_percent": "80",
        "run_phase_1": True,
        "run_phase_2": False,
        "phase_1_epochs": "15",
        "dropout_rate": "0.5",
        "unfreeze_layers": "400",
        "phase_2_unfreeze_parameters": "5"
    },
    "checkpoint_registry": {
        "fixed_validation_dataset": ""
    }
}


def _deep_merge(defaults, loaded):
    merged = json.loads(json.dumps(defaults))
    for key, value in (loaded or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged



SHARED_FIELD_MAP = {
    # Live keeps promoted-package state separate. Only device is
    # shared because it is one application-wide runtime preference.
    "live_tab": {"device": "device"},
    # Dataset publishes the built dataset root. Its promoted package remains
    # task-local and is validated independently from the Live tab selection.
    "dataset_tab": {"destination_root": "dataset_root"},
    "training_tab": {
        "dataset_root": "dataset_root",
        "artifacts_dir": "output_root",
        "save_model_path": "training_checkpoint",
        "model_name": "model_name",
        "model_version": "model_version",
        "device": "device",
        "batch_size": "batch_size",
    },
}


LEGACY_SHARED_ALIASES = {
    "output_folder": "output_root",
    "model_checkpoint": "training_checkpoint",
}


PATH_FIELDS = {
    "training_tab": {
        "files": ("pretrained_model_path", "save_model_path"),
        "dirs": ("dataset_root", "phase_2_dataset_root"),
    },
    "dataset_tab": {
        "files": (),
        "dirs": ("promoted_package",),
    },
    "live_tab": {
        "files": ("class_mapping",),
        "dirs": ("promoted_package",),
    },
    "checkpoint_registry": {"dirs": ("fixed_validation_dataset",)},
    "shared_settings": {
        "files": ("training_checkpoint", "class_mapping"),
        "dirs": ("dataset_root",),
    },
}


def synchronize_shared_settings(config_dict, source_tab):
    """Copy same-purpose settings from one tab to every tab.

    The source tab remains authoritative, including when a user intentionally
    clears a field. This function only synchronizes fields that actually exist
    on that tab.
    """
    shared = config_dict.setdefault("shared_settings", {})
    _normalize_shared_aliases(shared)
    source = config_dict.setdefault(source_tab, {})
    source_map = SHARED_FIELD_MAP.get(source_tab, {})

    for source_key, shared_key in source_map.items():
        if source_key in source:
            shared[shared_key] = source[source_key]
    if "output_root" in shared:
        shared["output_folder"] = shared["output_root"]

    for tab_name, field_map in SHARED_FIELD_MAP.items():
        tab_settings = config_dict.setdefault(tab_name, {})
        for tab_key, shared_key in field_map.items():
            if shared_key in shared:
                tab_settings[tab_key] = shared[shared_key]
    return config_dict


def _initialize_shared_settings(config_dict):
    """Populate shared values from existing configs without discarding data."""
    shared = config_dict.setdefault("shared_settings", {})
    _normalize_shared_aliases(shared)
    candidates = {
        "model_name": [("training_tab", "model_name")],
        "model_version": [("training_tab", "model_version")],
        "training_checkpoint": [("training_tab", "save_model_path"), ("evaluation_tab", "model_checkpoint"), ("threshold_tab", "model_checkpoint")],
        "dataset_root": [("evaluation_tab", "dataset_dir"), ("evaluation_tab", "test_dataset_dir"), ("threshold_tab", "dataset_root"), ("training_tab", "dataset_root")],
        "output_root": [("evaluation_tab", "output_folder"), ("threshold_tab", "artifacts_dir"), ("training_tab", "artifacts_dir")],
        "device": [("evaluation_tab", "device"), ("training_tab", "device"), ("live_tab", "device")],
        "batch_size": [("evaluation_tab", "batch_size"), ("threshold_tab", "batch_size"), ("training_tab", "batch_size")],
        "class_mapping": [("live_tab", "class_mapping")],
    }
    for shared_key, locations in candidates.items():
        current = shared.get(shared_key)
        if current not in (None, ""):
            continue
        for tab_name, tab_key in locations:
            value = config_dict.get(tab_name, {}).get(tab_key)
            if value not in (None, ""):
                shared[shared_key] = value
                break
    if shared.get("output_root") not in (None, ""):
        shared["output_folder"] = shared["output_root"]
    for tab_name, field_map in SHARED_FIELD_MAP.items():
        tab_settings = config_dict.setdefault(tab_name, {})
        for tab_key, shared_key in field_map.items():
            if shared.get(shared_key) not in (None, ""):
                tab_settings[tab_key] = shared[shared_key]
    return config_dict


def _normalize_shared_aliases(shared):
    for old_key, new_key in LEGACY_SHARED_ALIASES.items():
        if shared.get(new_key) in (None, "") and shared.get(old_key) not in (None, ""):
            shared[new_key] = shared[old_key]
    if shared.get("output_root") not in (None, ""):
        shared["output_folder"] = shared["output_root"]
    return shared


def validate_recovered_paths(config_dict):
    """Clear stale persisted input paths and report what was rejected.

    Output folders are intentionally excluded because users may select a new
    target folder before it exists. Backend jobs still receive explicit paths.
    """
    invalid = []
    for section, groups in PATH_FIELDS.items():
        values = config_dict.setdefault(section, {})
        for key in groups.get("files", ()):
            raw = values.get(key)
            if raw in (None, ""):
                continue
            if not Path(raw).expanduser().is_file():
                invalid.append({"section": section, "key": key, "path": raw, "reason": "missing_file"})
                values[key] = ""
        for key in groups.get("dirs", ()):
            raw = values.get(key)
            if raw in (None, ""):
                continue
            if not Path(raw).expanduser().is_dir():
                invalid.append({"section": section, "key": key, "path": raw, "reason": "missing_dir"})
                values[key] = ""
    config_dict["_invalid_paths"] = invalid
    return config_dict


def normalize_settings(config_dict):
    merged = _deep_merge(DEFAULT_CONFIG, config_dict or {})
    migrated = _initialize_shared_settings(merged)
    return validate_recovered_paths(migrated)

def load_settings():
    """Load settings from the local defect_detection/config.json file."""
    if not CONFIG_FILE.exists():
        if BUNDLED_CONFIG_FILE.exists():
            try:
                with open(BUNDLED_CONFIG_FILE, "r", encoding="utf-8") as handle:
                    initial = normalize_settings(json.load(handle))
                save_settings(initial)
                return initial
            except Exception:
                pass
        initial = normalize_settings(DEFAULT_CONFIG)
        save_settings(initial)
        return initial

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
            return normalize_settings(json.load(handle))
    except Exception:
        return normalize_settings(DEFAULT_CONFIG)


def save_settings(config_dict):
    """Save local bundle settings beside app/backend folders."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
            json.dump(config_dict, handle, indent=4)
    except Exception as exc:
        print(f"Error saving configuration: {exc}")
