from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable

from app.backend_bridge import sanitize_path_part
from app.promotion_gate import (
    OVERRIDABLE_PROMOTION_RULES,
    evaluate_promotion_gate,
    failed_rule_messages,
)
from backend.prediction_cache import PredictionResultCache, checkpoint_identity


CopyFn = Callable[[str | Path, str | Path], Any]


def build_promotion_package(
    state: dict[str, Any],
    *,
    output_root: str | Path,
    model_name: str,
    model_version: str,
    training_config: dict[str, Any] | None = None,
    requirements: dict[str, Any] | None = None,
    unsaved_threshold_changes: bool = False,
    unsaved_multiplier_changes: bool = False,
    override: bool = False,
    copy_fn: CopyFn = shutil.copy2,
) -> dict[str, Any]:
    gate = evaluate_promotion_gate(
        state,
        requirements=requirements,
        model_name=model_name,
        model_version=model_version,
        unsaved_threshold_changes=unsaved_threshold_changes,
        unsaved_multiplier_changes=unsaved_multiplier_changes,
    )
    failed_rules = [
        rule.get("name")
        for rule in gate.get("rules", [])
        if not rule.get("passed")
    ]
    override_applied = bool(
        override
        and failed_rules
        and all(name in OVERRIDABLE_PROMOTION_RULES for name in failed_rules)
    )
    gate["override"] = {
        "requested": bool(override),
        "applied": override_applied,
        "overridden_rules": failed_rules if override_applied else [],
    }
    if not gate.get("passed") and not override_applied:
        raise ValueError("Promotion gate failed: " + "; ".join(failed_rule_messages(gate)))

    package_dir = (
        Path(output_root)
        / "promoted"
        / sanitize_path_part(model_name, "model")
        / sanitize_path_part(model_version, "v1")
    )
    if package_dir.exists():
        raise FileExistsError(f"Promoted package already exists: {package_dir}")
    temp_dir = package_dir.parent / f".{package_dir.name}.tmp-{uuid4().hex}"
    if temp_dir.exists():
        raise FileExistsError(f"Temporary promotion package path already exists: {temp_dir}")

    record = state["record"]
    metadata = record.get("metadata") or {}
    evaluation = state["evaluation"]
    thresholds = state["thresholds"]
    multipliers = state.get("multipliers") or {}
    required_sources = {
        "model.pt": state.get("checkpoint_path"),
        "thresholds.json": thresholds.get("path"),
        "metrics.json": evaluation.get("report_path"),
        "gate_result.json": None,
        "model_metadata.json": None,
        "training_config.json": None,
        "labels.json": None,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required_sources.items()
        if path is not None and not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing promotion file(s): " + ", ".join(missing))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "checkpoint_id": record.get("id"),
        "files": {},
    }
    try:
        temp_dir.mkdir(parents=True)
        _copy_file(required_sources["model.pt"], temp_dir / "model.pt", manifest, copy_fn)
        _copy_file(required_sources["thresholds.json"], temp_dir / "thresholds.json", manifest, copy_fn)
        if multipliers.get("deployed_path") and Path(multipliers.get("deployed_path")).is_file():
            _copy_file(multipliers.get("deployed_path"), temp_dir / "defect_multipliers.json", manifest, copy_fn)
        _copy_file(required_sources["metrics.json"], temp_dir / "metrics.json", manifest, copy_fn)
        labels = _labels_from_state(state)
        _write_json(temp_dir / "labels.json", labels, manifest)
        _write_json(temp_dir / "gate_result.json", gate, manifest)
        _write_json(temp_dir / "training_config.json", training_config or metadata.get("training_config") or {}, manifest)
        _write_json(
            temp_dir / "model_metadata.json",
            {
                "model_name": model_name,
                "model_version": model_version,
                "checkpoint_id": record.get("id"),
                "promotion_override": override_applied,
                "overridden_gate_rules": failed_rules if override_applied else [],
                "source_checkpoint": state.get("checkpoint_path"),
                "source_checkpoint_identity": checkpoint_identity(state["checkpoint_path"]),
                "architecture": metadata.get("model_name") or model_name,
                "image_size": metadata.get("image_size"),
                "evaluation": {
                    "prediction_cache_path": evaluation.get("prediction_cache_path"),
                    "confusion_matrix_path": evaluation.get("confusion_matrix_path"),
                    "report_path": evaluation.get("report_path"),
                },
                "thresholds": {
                    "active_thresholds_path": thresholds.get("path"),
                    "mode_count": _mode_count((thresholds.get("config") or {}).get("classes", {})),
                },
                "multipliers": {
                    "active_multipliers_path": multipliers.get("path"),
                    "deployed_multipliers_path": multipliers.get("deployed_path"),
                    "mode_count": _mode_count((multipliers.get("config") or {}).get("classes", {})),
                },
            },
            manifest,
        )
        _write_json(temp_dir / "manifest.json", manifest, manifest=None)
        temp_dir.replace(package_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    validate_promotion_package(package_dir)
    return {
        "package_dir": str(package_dir),
        "model_path": str(package_dir / "model.pt"),
        "manifest_path": str(package_dir / "manifest.json"),
        "gate_result": gate,
    }


def validate_promotion_package(package_dir: str | Path) -> dict[str, Any]:
    package_path = Path(package_dir)
    manifest_path = package_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Promotion manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, info in manifest.get("files", {}).items():
        path = package_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"Promotion package file missing: {relative}")
        digest = _sha256(path)
        if digest != info.get("sha256"):
            raise ValueError(f"Promotion package hash mismatch: {relative}")
    return manifest


def _copy_file(source: str | Path, destination: Path, manifest: dict[str, Any], copy_fn: CopyFn) -> None:
    if not source:
        raise FileNotFoundError(f"Missing source for {destination.name}")
    copy_fn(source, destination)
    _record_file(destination, manifest)


def _write_json(path: Path, data: dict[str, Any] | list[Any], manifest: dict[str, Any] | None) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    if manifest is not None:
        _record_file(path, manifest)


def _record_file(path: Path, manifest: dict[str, Any]) -> None:
    manifest["files"][path.name] = {
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _labels_from_state(state: dict[str, Any]) -> dict[str, Any]:
    cache_path = state.get("evaluation", {}).get("prediction_cache_path")
    if cache_path and Path(cache_path).is_file():
        cache = PredictionResultCache.read_jsonl(cache_path)
        return {
            "class_names": cache.metadata.class_names,
            "class_to_index": {name: index for index, name in enumerate(cache.metadata.class_names)},
        }
    classes = list((state.get("thresholds", {}).get("config") or {}).get("classes", {}).keys())
    return {
        "class_names": classes,
        "class_to_index": {name: index for index, name in enumerate(classes)},
    }


def _mode_count(classes: dict[str, Any]) -> dict[str, int]:
    counts = {"auto": 0, "manual": 0}
    for values in classes.values():
        mode = values.get("mode")
        if mode in counts:
            counts[mode] += 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
