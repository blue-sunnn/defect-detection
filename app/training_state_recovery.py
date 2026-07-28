from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.backend_bridge import build_runtime_paths
from app.checkpoint_registry import load_registry
from app.promotion_gate import evaluate_promotion_gate
from backend.prediction_cache import PredictionResultCache
from backend.threshold_persistence import build_threshold_config, load_threshold_config
from backend.multiplier_persistence import build_multiplier_config, load_multiplier_config


def recover_training_state(settings: dict[str, Any], registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    runtime_paths = build_runtime_paths(
        output_root=settings.get("artifacts_dir"),
        save_model_path=settings.get("save_model_path") or None,
        model_name=settings.get("model_name", "EfficientNetV2S"),
        model_version=settings.get("model_version", "v1"),
        append_run_folder=True,
    )
    log: list[str] = []
    candidates = _matching_records(
        registry.get("checkpoints", []),
        model_name=settings.get("model_name", "EfficientNetV2S"),
        model_version=settings.get("model_version", "v1"),
        artifacts_dir=runtime_paths.artifacts_dir,
    )
    record = candidates[0] if candidates else None
    checkpoint_path = _record_checkpoint(record)
    if record and checkpoint_path:
        log.append(f"Recovered registry checkpoint: {Path(checkpoint_path).name}")
    elif record:
        log.append("Latest matching registry checkpoint missing source file.")
    else:
        checkpoint_path = _latest_checkpoint_file(runtime_paths.checkpoints_dir)
        if checkpoint_path:
            log.append(f"Recovered checkpoint file without registry record: {Path(checkpoint_path).name}")
        else:
            log.append("No persisted checkpoint found for selected model/version.")

    checkpoint_valid = bool(checkpoint_path and Path(checkpoint_path).is_file())
    gate = _recover_gate(record, checkpoint_valid, log)
    evaluation = _recover_evaluation(record, checkpoint_path, log)
    thresholds = _recover_thresholds(
        record=record,
        runtime_paths=runtime_paths,
        checkpoint_path=checkpoint_path,
        evaluation=evaluation,
        log=log,
    )
    multipliers = _recover_multipliers(record=record, runtime_paths=runtime_paths, checkpoint_path=checkpoint_path, evaluation=evaluation, log=log)
    state = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_valid": checkpoint_valid,
        "record": record,
        "gate": gate,
        "evaluation": evaluation,
        "thresholds": thresholds,
        "multipliers": multipliers,
        "runtime_paths": runtime_paths,
    }
    gate_result = evaluate_promotion_gate(
        state,
        requirements=registry.get("requirements") or {},
        model_name=settings.get("model_name", "EfficientNetV2S"),
        model_version=settings.get("model_version", "v1"),
    )
    promotion_ready = bool(gate_result.get("passed"))
    if promotion_ready:
        log.append("Promotion-ready state recovered.")
    elif checkpoint_valid and gate.get("valid") and evaluation.get("status") != "complete":
        log.append("Checkpoint can be evaluated after restart.")

    state.update({
        "promotion_ready": promotion_ready,
        "promotion_gate": gate_result,
        "log": log,
    })
    return state


def _matching_records(records: list[dict[str, Any]], *, model_name: str, model_version: str, artifacts_dir: str) -> list[dict[str, Any]]:
    root = str(Path(artifacts_dir).expanduser().resolve())

    def matches(record: dict[str, Any]) -> bool:
        metadata = record.get("metadata") or {}
        if metadata.get("model_name") == model_name and metadata.get("model_version") == model_version:
            return True
        source = record.get("source_path")
        return bool(source and str(Path(source).expanduser().resolve()).startswith(root))

    matched = [record for record in records if matches(record)]
    return sorted(
        matched,
        key=lambda record: (record.get("updated_at") or record.get("created_at") or "", record.get("id") or ""),
        reverse=True,
    )


def _record_checkpoint(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    source = record.get("source_path")
    if source and Path(source).is_file():
        return str(Path(source).expanduser().resolve())
    return source


def _latest_checkpoint_file(checkpoints_dir: str) -> str | None:
    root = Path(checkpoints_dir)
    if not root.is_dir():
        return None
    candidates = [path for pattern in ("*.pth", "*.pt") for path in root.rglob(pattern) if path.is_file()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime).resolve())


def _recover_gate(record: dict[str, Any] | None, checkpoint_valid: bool, log: list[str]) -> dict[str, Any]:
    if not record:
        return {"valid": False, "passed": False, "state": "Checkpoint not registered."}
    gate = record.get("gate")
    metrics = record.get("metrics")
    if not checkpoint_valid:
        log.append("Gate blocked: checkpoint file missing.")
        return {"valid": False, "passed": False, "state": "Checkpoint file missing."}
    if not isinstance(gate, dict) or not isinstance(metrics, dict):
        log.append("Gate not available: registry record lacks gate metrics.")
        return {"valid": False, "passed": False, "state": "Gate result missing."}
    log.append(f"Gate recovered: {gate.get('state', 'Checkpoint gated')}")
    recovered = dict(gate)
    recovered["valid"] = True
    return recovered


def _recover_evaluation(record: dict[str, Any] | None, checkpoint_path: str | None, log: list[str]) -> dict[str, Any]:
    metadata = (record or {}).get("metadata") or {}
    evaluation = metadata.get("automatic_evaluation") or {}
    if not evaluation:
        return {"status": "not_started", "error": None}
    if evaluation.get("status") == "failed":
        log.append(f"Evaluation recovered as failed: {evaluation.get('error')}")
        return dict(evaluation)
    if evaluation.get("status") != "complete":
        return {"status": evaluation.get("status", "not_started"), "error": None}

    required_paths = {
        "prediction_cache_path": evaluation.get("prediction_cache_path"),
        "confusion_matrix_path": evaluation.get("confusion_matrix_path"),
        "report_path": evaluation.get("report_path"),
    }
    missing = [name for name, path in required_paths.items() if not path or not Path(path).is_file()]
    if missing:
        message = "Evaluation artifacts missing: " + ", ".join(missing)
        log.append(message)
        invalid = dict(evaluation)
        invalid["status"] = "invalid"
        invalid["error"] = message
        return invalid
    try:
        _validate_json_if_json_path(required_paths["report_path"])
        cache = PredictionResultCache.read_jsonl(required_paths["prediction_cache_path"])
        dataset_dir = evaluation.get("dataset_dir")
        if checkpoint_path and dataset_dir:
            cache.validate_identity(
                checkpoint_path=checkpoint_path,
                dataset_root=dataset_dir,
                class_names=cache.metadata.class_names,
            )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        log.append(f"Evaluation invalid: {message}")
        invalid = dict(evaluation)
        invalid["status"] = "invalid"
        invalid["error"] = message
        return invalid
    log.append("Evaluation artifacts validated.")
    return dict(evaluation)


def _recover_thresholds(
    *,
    record: dict[str, Any] | None,
    runtime_paths,
    checkpoint_path: str | None,
    evaluation: dict[str, Any],
    log: list[str],
) -> dict[str, Any]:
    metadata = (record or {}).get("metadata") or {}
    active_meta = metadata.get("active_thresholds") or {}
    active_path = active_meta.get("path") or runtime_paths.threshold_active_json_path
    try:
        active_config = load_threshold_config(active_path)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        log.append(f"Threshold file invalid: {message}")
        return {"status": "invalid", "path": active_path, "error": message, "config": None}
    if active_config:
        log.append("Active thresholds validated.")
        return {"status": "saved", "path": active_path, "error": None, "config": active_config}

    recommendations_path = (metadata.get("automatic_thresholds") or {}).get("recommendations_path") or runtime_paths.threshold_results_json_path
    try:
        recommendations = _load_recommendations(recommendations_path)
    except Exception as exc:
        if evaluation.get("status") == "complete":
            message = str(exc).strip() or exc.__class__.__name__
            log.append(f"Threshold recommendations unavailable: {message}")
            return {"status": "not_saved", "path": active_path, "error": message, "config": None}
        return {"status": "not_started", "path": active_path, "error": None, "config": None}
    config = build_threshold_config(
        recommendations=recommendations,
        checkpoint=checkpoint_path or "",
        saved_config=None,
    )
    log.append("Recommended thresholds recovered; active thresholds not saved.")
    return {"status": "draft", "path": active_path, "error": None, "config": config}



def _recover_multipliers(*, record, runtime_paths, checkpoint_path, evaluation, log):
    metadata = (record or {}).get("metadata") or {}
    active_meta = metadata.get("active_multipliers") or {}
    active_path = active_meta.get("path") or runtime_paths.multiplier_active_json_path
    deployed_path = active_meta.get("deployed_path") or metadata.get("deployed_multipliers_path") or runtime_paths.multiplier_deployed_json_path
    try:
        config = load_multiplier_config(active_path)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        log.append(f"Multiplier file invalid: {message}")
        return {"status":"invalid","path":active_path,"deployed_path":deployed_path,"error":message,"config":None}
    if config:
        if not Path(deployed_path).is_file():
            return {"status":"invalid","path":active_path,"deployed_path":deployed_path,"error":"Active multiplier deployment file is missing.","config":config}
        log.append("Active multipliers validated.")
        return {"status":"saved","path":active_path,"deployed_path":deployed_path,"error":None,"config":config}
    rec_path = (metadata.get("automatic_multipliers") or {}).get("recommendations_path") or runtime_paths.multiplier_results_json_path
    try:
        data = json.loads(Path(rec_path).read_text(encoding="utf-8"))
        if not isinstance(data.get("classes"), dict): raise ValueError(f"Multiplier recommendations invalid: {rec_path}")
    except Exception as exc:
        if evaluation.get("status") == "complete":
            return {"status":"not_saved","path":active_path,"deployed_path":deployed_path,"error":str(exc),"config":None}
        return {"status":"not_started","path":active_path,"deployed_path":deployed_path,"error":None,"config":None}
    config = build_multiplier_config(recommendations=data, checkpoint=checkpoint_path or "", saved_config=None)
    log.append("Recommended multipliers recovered; active multipliers not saved.")
    return {"status":"draft","path":active_path,"deployed_path":deployed_path,"error":None,"config":config}

def _load_recommendations(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Threshold recommendations not found: {path}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("classes"), dict):
        raise ValueError(f"Threshold recommendations invalid: {path}")
    return data


def _validate_json_if_json_path(path: str | None) -> None:
    if path and str(path).lower().endswith(".json"):
        json.loads(Path(path).read_text(encoding="utf-8"))
