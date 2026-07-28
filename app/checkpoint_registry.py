from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config_manager import load_settings, save_settings

MIN_ACCURACY = 0.90
MAX_ESCAPE_RATE = 0.01
DEFAULT_BASELINE_KEY = "__default__"


def _bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def registry_dir() -> Path:
    return _bundle_root() / "artifacts" / "checkpoints"


def registry_path() -> Path:
    return registry_dir() / "registry.json"


def production_pointer_path() -> Path:
    return registry_dir() / "current_production.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _default_registry() -> dict:
    return {
        "schema_version": 2,
        "requirements": {
            "minimum_accuracy": MIN_ACCURACY,
            "maximum_escape_rate": MAX_ESCAPE_RATE,
        },
        "baseline": None,
        "baselines": {},
        "current_production_id": None,
        "current_production_ids": {},
        "checkpoints": [],
    }


def load_registry() -> dict:
    path = registry_path()
    if not path.exists():
        return _default_registry()
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _default_registry()
    registry = _default_registry()
    registry.update(loaded if isinstance(loaded, dict) else {})
    registry.setdefault("checkpoints", [])
    registry.setdefault("baselines", {})
    registry.setdefault("current_production_ids", {})
    if not registry["baselines"] and registry.get("baseline"):
        baseline_model = _baseline_model_name(registry, registry["baseline"])
        baseline_key = _baseline_key(baseline_model)
        registry["baselines"][baseline_key] = dict(registry["baseline"])
        if registry.get("current_production_id"):
            registry["current_production_ids"].setdefault(baseline_key, registry["current_production_id"])
    return registry


def _baseline_key(model_name: str | None) -> str:
    model_name = str(model_name or "").strip()
    return model_name or DEFAULT_BASELINE_KEY


def _record_model_name(record: dict | None) -> str | None:
    metadata = (record or {}).get("metadata") or {}
    return metadata.get("model_name")


def _baseline_model_name(registry: dict, baseline: dict | None) -> str | None:
    checkpoint_id = (baseline or {}).get("checkpoint_id")
    if not checkpoint_id:
        return None
    record = next((item for item in registry.get("checkpoints", []) if item.get("id") == checkpoint_id), None)
    return _record_model_name(record)


def save_registry(registry: dict) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
    temp.replace(path)


def baseline_metrics(registry: dict | None = None, model_name: str | None = None) -> dict:
    registry = registry or load_registry()
    baselines = registry.get("baselines") or {}
    baseline = baselines.get(_baseline_key(model_name))
    if baseline is None and not model_name:
        baseline = registry.get("baseline")
    baseline = baseline or {}
    metrics = baseline.get("metrics") or {}
    requirements = registry.get("requirements") or {}
    return {
        "accuracy": float(metrics.get("accuracy", requirements.get("minimum_accuracy", MIN_ACCURACY))),
        "escape_rate": float(metrics.get("escape_rate", requirements.get("maximum_escape_rate", MAX_ESCAPE_RATE))),
    }


def gate_result(metrics: dict, registry: dict | None = None, model_name: str | None = None) -> dict:
    registry = registry or load_registry()
    baseline = baseline_metrics(registry, model_name=model_name)
    requirements = registry.get("requirements") or {}
    accuracy_floor = float(requirements.get("minimum_accuracy", MIN_ACCURACY))
    accuracy = float(metrics["accuracy"])
    escape_rate = float(metrics["escape_rate"])
    # Accuracy is judged against a fixed floor, not the ratcheting production
    # baseline: small accuracy swings (subclass confusion, not missed defects)
    # are expected month to month and shouldn't force an override on their
    # own. Escape rate is the safety-critical metric and must not regress
    # past whatever is currently in production.
    accuracy_ok = accuracy >= accuracy_floor
    escape_ok = escape_rate <= baseline["escape_rate"]
    passed = accuracy_ok and escape_ok
    if passed:
        state = "Passed gate, ready to promote"
    elif not escape_ok and not accuracy_ok:
        state = f"Failed gate — below {accuracy_floor:.0%} accuracy floor and escape rate regressed, needs override"
    elif not escape_ok:
        state = "Failed gate — escape rate regressed vs. production, needs override"
    else:
        state = f"Failed gate — below {accuracy_floor:.0%} accuracy floor, needs override"
    return {
        "passed": passed,
        "state": state,
        "baseline": baseline,
        "accuracy_delta": accuracy - baseline["accuracy"],
        "escape_rate_delta": escape_rate - baseline["escape_rate"],
    }


def register_checkpoint(checkpoint_path: str, metrics: dict, produced_by: str, metadata: dict | None = None) -> dict:
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    registry = load_registry()
    existing = next((item for item in registry["checkpoints"] if item.get("source_path") == str(source)), None)
    record = existing or {"id": uuid4().hex, "created_at": _now()}
    metadata = metadata or {}
    model_name = metadata.get("model_name")
    baseline_key = _baseline_key(model_name)
    registry.setdefault("baselines", {})
    if baseline_key not in registry["baselines"]:
        registry["baselines"][baseline_key] = {
            "checkpoint_id": record["id"],
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "escape_rate": float(metrics["escape_rate"]),
            },
            "created_at": _now(),
            "source_path": str(source),
            "provisional": True,
        }
    record.update({
        "filename": source.name,
        "source_path": str(source),
        "produced_by": produced_by,
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "escape_rate": float(metrics["escape_rate"]),
            "false_alarm_rate": float(metrics.get("false_alarm_rate", 0.0)),
        },
        "gate": gate_result(metrics, registry, model_name=model_name),
        "metadata": metadata,
    })
    if existing is None:
        registry["checkpoints"].append(record)
    save_registry(registry)
    return record


def update_checkpoint_metadata(checkpoint_id: str, metadata: dict) -> dict:
    registry = load_registry()
    record = next((item for item in registry["checkpoints"] if item.get("id") == checkpoint_id), None)
    if record is None:
        raise ValueError("Checkpoint is not registered.")
    current = dict(record.get("metadata") or {})
    current.update(metadata or {})
    record["metadata"] = current
    record["updated_at"] = _now()
    save_registry(registry)
    return record


def _versioned_destination(record: dict) -> Path:
    source = Path(record["source_path"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in source.stem)
    return registry_dir() / "versions" / f"{safe_stem}_{stamp}_{record['id'][:8]}{source.suffix or '.pth'}"


def _write_pointer(record: dict) -> None:
    pointer = {
        "checkpoint_id": record["id"],
        "checkpoint_path": record["promoted_path"],
        "package_path": record.get("promoted_package_path"),
        "updated_at": _now(),
    }
    path = production_pointer_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(pointer, handle, indent=2)
    temp.replace(path)


def _sync_config_checkpoint(checkpoint_path: str) -> None:
    config = load_settings()
    config.setdefault("shared_settings", {})["training_checkpoint"] = checkpoint_path
    config.setdefault("training_tab", {})["save_model_path"] = checkpoint_path
    save_settings(config)


def register_promoted_package(checkpoint_id: str, package_dir: str, model_path: str, override: bool = False) -> dict:
    registry = load_registry()
    record = next((item for item in registry["checkpoints"] if item.get("id") == checkpoint_id), None)
    if record is None:
        raise ValueError("Checkpoint is not registered.")
    if not record.get("gate", {}).get("passed") and not override:
        raise PermissionError("This checkpoint failed the baseline gate.")
    package = Path(package_dir).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not package.is_dir():
        raise FileNotFoundError(f"Promoted package is missing: {package}")
    if not model.is_file():
        raise FileNotFoundError(f"Promoted package model is missing: {model}")

    promoted_at = _now()
    record["promoted_package_path"] = str(package)
    record["promoted_path"] = str(model)
    record["promoted_at"] = promoted_at
    record["promotion_override"] = bool(override)
    registry["current_production_id"] = record["id"]
    model_name = _record_model_name(record)
    baseline = {
        "checkpoint_id": record["id"],
        "metrics": dict(record["metrics"]),
        "promoted_at": promoted_at,
        "package_path": str(package),
    }
    registry.setdefault("baselines", {})[_baseline_key(model_name)] = baseline
    registry.setdefault("current_production_ids", {})[_baseline_key(model_name)] = record["id"]
    registry["baseline"] = baseline
    _write_pointer(record)
    _sync_config_checkpoint(record["promoted_path"])
    save_registry(registry)
    return record


def promote_checkpoint(checkpoint_id: str, override: bool = False) -> dict:
    registry = load_registry()
    record = next((item for item in registry["checkpoints"] if item.get("id") == checkpoint_id), None)
    if record is None:
        raise ValueError("Checkpoint is not registered.")
    if not record.get("gate", {}).get("passed") and not override:
        raise PermissionError("This checkpoint failed the baseline gate.")

    source = Path(record["source_path"])
    destination = Path(record.get("promoted_path") or _versioned_destination(record))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

    promoted_at = _now()
    record["promoted_path"] = str(destination.resolve())
    record["promoted_at"] = promoted_at
    record["promotion_override"] = bool(override)
    registry["current_production_id"] = record["id"]
    model_name = _record_model_name(record)
    baseline = {
        "checkpoint_id": record["id"],
        "metrics": dict(record["metrics"]),
        "promoted_at": promoted_at,
    }
    registry.setdefault("baselines", {})[_baseline_key(model_name)] = baseline
    registry.setdefault("current_production_ids", {})[_baseline_key(model_name)] = record["id"]
    registry["baseline"] = baseline
    _write_pointer(record)
    _sync_config_checkpoint(record["promoted_path"])
    save_registry(registry)
    return record


def rollback_checkpoint(checkpoint_id: str) -> dict:
    registry = load_registry()
    record = next((item for item in registry["checkpoints"] if item.get("id") == checkpoint_id), None)
    if record is None or not record.get("promoted_path"):
        raise ValueError("Only a previously promoted checkpoint can be restored.")
    if not Path(record["promoted_path"]).is_file():
        raise FileNotFoundError(f"Promoted checkpoint is missing: {record['promoted_path']}")
    registry["current_production_id"] = record["id"]
    model_name = _record_model_name(record)
    baseline = {
        "checkpoint_id": record["id"],
        "metrics": dict(record.get("metrics") or {}),
        "promoted_at": record.get("promoted_at"),
        "package_path": record.get("promoted_package_path"),
    }
    registry.setdefault("baselines", {})[_baseline_key(model_name)] = baseline
    registry.setdefault("current_production_ids", {})[_baseline_key(model_name)] = record["id"]
    registry["baseline"] = baseline
    record["last_rollback_at"] = _now()
    _write_pointer(record)
    _sync_config_checkpoint(record["promoted_path"])
    save_registry(registry)
    return record


def resolve_production_checkpoint(fallback: str | None = None) -> str | None:
    try:
        with production_pointer_path().open("r", encoding="utf-8") as handle:
            path = json.load(handle).get("checkpoint_path")
        if path and Path(path).is_file():
            return str(Path(path).resolve())
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return fallback


def promoted_checkpoints() -> list[dict]:
    registry = load_registry()
    rows = [item for item in registry["checkpoints"] if item.get("promoted_path")]
    return sorted(rows, key=lambda item: item.get("promoted_at", ""), reverse=True)
