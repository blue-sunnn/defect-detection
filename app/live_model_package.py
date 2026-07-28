from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.promotion_package import validate_promotion_package


@dataclass(frozen=True)
class LivePackage:
    package_dir: str
    checkpoint_path: str
    labels_path: str
    thresholds_path: str
    multiplier_path: str | None
    metadata_path: str
    class_names: list[str]
    active_thresholds: dict[str, float]
    model_name: str
    model_version: str
    architecture: str
    image_size: tuple[int, int]
    summary: str


def resolve_live_package(package_dir: str | Path) -> LivePackage:
    root = Path(package_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Promoted package folder not found: {root}")
    validate_promotion_package(root)

    model_path = _require_file(root / "model.pt", "model checkpoint")
    labels_path = _require_file(root / "labels.json", "labels")
    thresholds_path = _require_file(root / "thresholds.json", "active thresholds")
    metadata_path = _require_file(root / "model_metadata.json", "model metadata")
    multiplier_path = root / "defect_multipliers.json"
    multipliers = _read_json(multiplier_path) if multiplier_path.is_file() else None

    labels = _read_json(labels_path)
    thresholds = _read_json(thresholds_path)
    metadata = _read_json(metadata_path)
    class_names = _labels_class_order(labels)
    active_thresholds = _active_thresholds(thresholds, class_names)
    _validate_metadata(metadata, class_names)
    if multipliers is not None:
        _validate_multipliers(multipliers, class_names)

    model_name = str(metadata.get("model_name") or root.parent.name)
    model_version = str(metadata.get("model_version") or root.name)
    architecture = str(metadata.get("architecture") or model_name)
    image_size = _image_size(metadata)
    summary = (
        f"Model: {model_name}\n"
        f"Version: {model_version}\n"
        f"Checkpoint: {model_path.name}\n"
        f"Threshold source: {thresholds_path.name}\n"
        f"Multiplier source: {multiplier_path.name if multipliers is not None else 'Not provided'}\n"
        "Package status: Valid"
    )
    return LivePackage(
        package_dir=str(root),
        checkpoint_path=str(model_path),
        labels_path=str(labels_path),
        thresholds_path=str(thresholds_path),
        multiplier_path=str(multiplier_path) if multipliers is not None else None,
        metadata_path=str(metadata_path),
        class_names=class_names,
        active_thresholds=active_thresholds,
        model_name=model_name,
        model_version=model_version,
        architecture=architecture,
        image_size=image_size,
        summary=summary,
    )


def custom_model_defaults(checkpoint_path: str | Path, class_names: list[str] | None = None) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    names = class_names or ["Mousebite", "Open", "Pass", "Pinhole", "Protrusion", "Short", "Via"]
    return {
        "checkpoint_path": str(checkpoint),
        "class_names": names,
        "active_thresholds": {},
        "defect_multipliers": {name: 0.0 for name in names if name != "Pass"},
        "model_name": checkpoint.stem,
        "model_version": checkpoint.stem,
        "architecture": "custom",
        "image_size": (384, 384),
        "summary": (
            f"Model: {checkpoint.stem}\n"
            f"Version: {checkpoint.stem}\n"
            f"Checkpoint: {checkpoint.name}\n"
            "Threshold source: Safe defaults\n"
            "Multiplier source: Not provided\n"
            "Package status: Custom standalone model"
        ),
    }


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Promoted package missing {label}: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in promoted package file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Promoted package JSON root must be object: {path}")
    return data


def _labels_class_order(labels: dict[str, Any]) -> list[str]:
    class_names = labels.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("labels.json must contain non-empty class_names list.")
    mapping = labels.get("class_to_index") or {}
    for index, name in enumerate(class_names):
        if str(mapping.get(name)) != str(index):
            raise ValueError(f"Label mapping class order mismatch for {name}.")
    return [str(name) for name in class_names]


def _active_thresholds(thresholds: dict[str, Any], class_names: list[str]) -> dict[str, float]:
    classes = thresholds.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("thresholds.json must contain classes object.")
    result = {}
    known = set(class_names)
    for class_name, values in classes.items():
        if class_name not in known:
            raise ValueError(f"Threshold class not present in labels: {class_name}")
        active = values.get("active")
        if active is None:
            raise ValueError(f"Active threshold missing for {class_name}.")
        active_float = float(active)
        if not 0.0 <= active_float <= 1.0:
            raise ValueError(f"Active threshold for {class_name} must be between 0 and 1.")
        result[class_name] = active_float
    return result


def _validate_metadata(metadata: dict[str, Any], class_names: list[str]) -> None:
    for key in ("model_name", "model_version", "architecture", "image_size"):
        if key not in metadata or metadata.get(key) in (None, ""):
            raise ValueError(f"model_metadata.json missing required field: {key}")
    metadata_classes = metadata.get("class_names")
    if metadata_classes is not None and list(metadata_classes) != class_names:
        raise ValueError("Model metadata class order does not match labels.json.")
    _image_size(metadata)


def _image_size(metadata: dict[str, Any]) -> tuple[int, int]:
    size = metadata.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("model_metadata.json image_size must be [width, height].")
    return int(size[0]), int(size[1])


def _validate_multipliers(multipliers: dict[str, Any], class_names: list[str]) -> None:
    known = set(class_names)
    for class_name, value in multipliers.items():
        if class_name not in known:
            raise ValueError(f"Multiplier class not present in labels: {class_name}")
        float(value)
