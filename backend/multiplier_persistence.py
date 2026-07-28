from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MIN_MULTIPLIER = -0.999
MAX_MULTIPLIER = 0.999


def build_multiplier_config(*, recommendations: dict[str, Any], checkpoint: str, saved_config: dict[str, Any] | None = None, generated_at: str | None = None) -> dict[str, Any]:
    saved_classes = (saved_config or {}).get("classes", {})
    classes: dict[str, Any] = {}
    for class_name, recommendation in recommendations.get("classes", {}).items():
        recommended = recommendation.get("recommended")
        saved = saved_classes.get(class_name, {})
        if str(saved.get("mode", "")).lower() == "manual" and _is_valid(saved.get("active")):
            active, mode = float(saved["active"]), "manual"
        else:
            active = None if recommended is None else float(recommended)
            mode = "auto"
        classes[class_name] = {
            "recommended": None if recommended is None else float(recommended),
            "active": active,
            "mode": mode,
            "status": recommendation.get("status", "valid"),
            "warning": recommendation.get("warning"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": recommendations.get("strategy", "differential_evolution_kfold"),
        "checkpoint": str(checkpoint),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "classes": classes,
    }


def load_multiplier_config(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Multiplier configuration is invalid: {p}") from exc
    validate_multiplier_config(data)
    return data


def save_multiplier_config(config: dict[str, Any], path: str | os.PathLike[str], *, deployment_path: str | os.PathLike[str] | None = None) -> None:
    validate_multiplier_config(config)
    _atomic_json(Path(path), config)
    if deployment_path:
        flat = {name: float(values["active"]) for name, values in config["classes"].items()}
        _atomic_json(Path(deployment_path), flat)


def validate_multiplier_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported multiplier configuration schema: {config.get('schema_version')}")
    classes = config.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("Multiplier configuration missing classes.")
    for name, values in classes.items():
        for key in ("recommended", "active"):
            value = values.get(key)
            if value is not None and not _is_valid(value):
                raise ValueError(f"{key.title()} multiplier for {name} must be between {MIN_MULTIPLIER} and {MAX_MULTIPLIER}.")
        if values.get("mode") not in {"auto", "manual"}:
            raise ValueError(f"Multiplier mode for {name} must be auto or manual.")


def set_active_multiplier(config: dict[str, Any], class_name: str, value: Any) -> dict[str, Any]:
    updated = deepcopy(config)
    if class_name not in updated.get("classes", {}):
        raise ValueError(f"Unknown multiplier class: {class_name}")
    updated["classes"][class_name]["active"] = parse_multiplier(value, class_name=class_name)
    updated["classes"][class_name]["mode"] = "manual"
    validate_multiplier_config(updated)
    return updated


def restore_recommended_multiplier(config: dict[str, Any], class_name: str | None = None) -> dict[str, Any]:
    updated = deepcopy(config)
    names = [class_name] if class_name else list(updated.get("classes", {}))
    for name in names:
        if name not in updated.get("classes", {}):
            raise ValueError(f"Unknown multiplier class: {name}")
        value = updated["classes"][name].get("recommended")
        if value is None:
            raise ValueError(f"No multiplier recommendation available for {name}.")
        updated["classes"][name]["active"] = float(value)
        updated["classes"][name]["mode"] = "auto"
    validate_multiplier_config(updated)
    return updated


def parse_multiplier(value: Any, *, class_name: str = "multiplier") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Active multiplier for {class_name} must be a number.") from exc
    if not MIN_MULTIPLIER <= number <= MAX_MULTIPLIER:
        raise ValueError(f"Active multiplier for {class_name} must be between {MIN_MULTIPLIER} and {MAX_MULTIPLIER}.")
    return number


def _is_valid(value: Any) -> bool:
    try:
        return MIN_MULTIPLIER <= float(value) <= MAX_MULTIPLIER
    except (TypeError, ValueError):
        return False


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)
