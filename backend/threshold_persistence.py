from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def build_threshold_config(
    *,
    recommendations: dict[str, Any],
    checkpoint: str,
    saved_config: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    saved_classes = (saved_config or {}).get("classes", {})
    classes = {}
    for class_name, recommendation in recommendations.get("classes", {}).items():
        recommended = recommendation.get("recommended")
        saved = saved_classes.get(class_name, {})
        saved_mode = str(saved.get("mode", "")).lower()
        saved_active = saved.get("active")
        if saved_mode == "manual" and _is_valid_threshold(saved_active):
            active = float(saved_active)
            mode = "manual"
        else:
            active = None if recommended is None else float(recommended)
            mode = "auto"
        classes[class_name] = {
            "recommended": None if recommended is None else float(recommended),
            "active": active,
            "mode": mode,
            "status": recommendation.get("status"),
            "warning": recommendation.get("warning"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": recommendations.get("strategy", "best_f1"),
        "checkpoint": str(checkpoint),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "classes": classes,
    }


def load_threshold_config(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    config_path = Path(path)
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Threshold configuration is invalid: {config_path}") from exc
    if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported threshold configuration schema: {data.get('schema_version')}")
    classes = data.get("classes")
    if not isinstance(classes, dict):
        raise ValueError(f"Threshold configuration missing classes: {config_path}")
    return data


def save_threshold_config(config: dict[str, Any], path: str | os.PathLike[str]) -> None:
    validate_threshold_config(config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(output_path)


def validate_threshold_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported threshold configuration schema: {config.get('schema_version')}")
    for class_name, values in config.get("classes", {}).items():
        active = values.get("active")
        recommended = values.get("recommended")
        if active is not None and not _is_valid_threshold(active):
            raise ValueError(f"Active threshold for {class_name} must be between 0 and 1.")
        if recommended is not None and not _is_valid_threshold(recommended):
            raise ValueError(f"Recommended threshold for {class_name} must be between 0 and 1.")
        mode = values.get("mode")
        if mode not in {"auto", "manual"}:
            raise ValueError(f"Threshold mode for {class_name} must be auto or manual.")


def set_active_threshold(config: dict[str, Any], class_name: str, value: Any) -> dict[str, Any]:
    threshold = parse_threshold(value, class_name=class_name)
    updated = deepcopy(config)
    if class_name not in updated.get("classes", {}):
        raise ValueError(f"Unknown threshold class: {class_name}")
    updated["classes"][class_name]["active"] = threshold
    updated["classes"][class_name]["mode"] = "manual"
    validate_threshold_config(updated)
    return updated


def restore_recommended(config: dict[str, Any], class_name: str | None = None) -> dict[str, Any]:
    updated = deepcopy(config)
    target_names = [class_name] if class_name else list(updated.get("classes", {}).keys())
    for target_name in target_names:
        if target_name not in updated.get("classes", {}):
            raise ValueError(f"Unknown threshold class: {target_name}")
        recommended = updated["classes"][target_name].get("recommended")
        if recommended is None:
            raise ValueError(f"No recommendation available for {target_name}.")
        updated["classes"][target_name]["active"] = float(recommended)
        updated["classes"][target_name]["mode"] = "auto"
    validate_threshold_config(updated)
    return updated


def parse_threshold(value: Any, *, class_name: str = "threshold") -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Active threshold for {class_name} must be a number.") from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Active threshold for {class_name} must be between 0 and 1.")
    return threshold


def _is_valid_threshold(value: Any) -> bool:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= threshold <= 1.0
