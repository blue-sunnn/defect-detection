from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_EVALUATION_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "misclassified_count",
)

# Only baseline-performance rules may be bypassed by an explicit production
# override. Artifact integrity, saved configuration, model identity, and
# evaluation-completeness rules must always pass.
OVERRIDABLE_PROMOTION_RULES = frozenset({
    "registration_gate",
    "minimum_accuracy",
    "maximum_escape_rate",
})


def evaluate_promotion_gate(
    state: dict[str, Any],
    *,
    requirements: dict[str, Any] | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    thresholds_required: bool = True,
    unsaved_threshold_changes: bool = False,
    multipliers_required: bool | None = None,
    unsaved_multiplier_changes: bool = False,
) -> dict[str, Any]:
    """Promotion gate rules.

    Existing accuracy-floor behavior remains: checkpoint registration still
    decides `gate.passed` from minimum accuracy plus ratcheting escape-rate
    baseline. These rules add persisted-artifact checks required for promotion.
    Defect multipliers are deployment tuning inputs. They are required when
    automatic multiplier recommendations were generated for a trained model;
    otherwise missing multiplier files are warning-only.
    """
    requirements = requirements or {}
    rules: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    record = state.get("record") or {}
    metadata = record.get("metadata") or {}
    evaluation = state.get("evaluation") or {}
    thresholds = state.get("thresholds") or {}
    multipliers = state.get("multipliers") or {}

    _rule(rules, "checkpoint_valid", bool(state.get("checkpoint_valid")), path=state.get("checkpoint_path"))
    _rule(rules, "checkpoint_registered", bool(record), checkpoint_id=record.get("id"))
    _rule(
        rules,
        "model_identity",
        (not model_name or metadata.get("model_name") == model_name)
        and (not model_version or metadata.get("model_version") == model_version),
        actual={"model_name": metadata.get("model_name"), "model_version": metadata.get("model_version")},
        required={"model_name": model_name, "model_version": model_version},
    )
    gate = state.get("gate") or {}
    _rule(rules, "registration_gate", bool(gate.get("valid") and gate.get("passed")), state=gate.get("state"))

    metrics = record.get("metrics") or {}
    accuracy_required = float(requirements.get("minimum_accuracy", 0.0))
    escape_required = requirements.get("maximum_escape_rate")
    _metric_rule(rules, "minimum_accuracy", metrics, "accuracy", min_value=accuracy_required)
    if escape_required is not None:
        _metric_rule(rules, "maximum_escape_rate", metrics, "escape_rate", max_value=float(escape_required))

    _rule(rules, "evaluation_complete", evaluation.get("status") == "complete", status=evaluation.get("status"), reason=evaluation.get("error"))
    for metric in REQUIRED_EVALUATION_METRICS:
        _rule(rules, f"evaluation_metric_{metric}", metric in evaluation and evaluation.get(metric) is not None, actual=evaluation.get(metric))

    for path_key in ("prediction_cache_path", "confusion_matrix_path", "report_path"):
        path = evaluation.get(path_key)
        _rule(rules, f"evaluation_file_{path_key}", bool(path and Path(path).is_file()), path=path)

    _per_class_rules(rules, evaluation, requirements.get("per_class") or {})

    if thresholds_required:
        _rule(rules, "thresholds_saved", thresholds.get("status") == "saved", status=thresholds.get("status"), reason=thresholds.get("error"))
        _rule(rules, "threshold_file_valid", bool(thresholds.get("config")), path=thresholds.get("path"))
        _rule(rules, "thresholds_no_unsaved_changes", not unsaved_threshold_changes, reason="Active thresholds contain unsaved changes." if unsaved_threshold_changes else None)
    else:
        _rule(rules, "thresholds_optional", True)

    if multipliers_required is None:
        multipliers_required = bool(metadata.get("multipliers_required") or metadata.get("automatic_multipliers")) and not bool(metadata.get("external_model", False))
    if multipliers_required:
        _rule(rules, "multiplier_recommendations", multipliers.get("status") in {"draft", "saved"}, status=multipliers.get("status"), reason=multipliers.get("error"))
        _rule(rules, "multipliers_saved", multipliers.get("status") == "saved", status=multipliers.get("status"), reason=multipliers.get("error"))
        _rule(rules, "multiplier_file_valid", bool(multipliers.get("config")) and _path_present(multipliers.get("deployed_path")), path=multipliers.get("deployed_path"))
        _rule(rules, "multipliers_no_unsaved_changes", not unsaved_multiplier_changes, reason="Active multipliers contain unsaved changes." if unsaved_multiplier_changes else None)
    else:
        _rule(rules, "multipliers_optional_external_model", True)
        if not _path_present(metadata.get("deployed_multipliers_path")):
            warnings.append({"name":"optional_multipliers","passed":True,"reason":"No deployed multiplier file recorded; multipliers are optional for this model."})

    passed = all(rule.get("passed") for rule in rules)
    return {"passed": passed, "rules": rules, "warnings": warnings}


def failed_rule_messages(result: dict[str, Any]) -> list[str]:
    messages = []
    for rule in result.get("rules", []):
        if rule.get("passed"):
            continue
        reason = (
            rule.get("reason")
            or rule.get("status")
            or rule.get("state")
            or rule.get("actual")
            or "failed"
        )
        messages.append(f"{rule.get('name')}: {reason}")
    return messages


def _rule(rules: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    row = {"name": name, "passed": bool(passed)}
    row.update({key: value for key, value in details.items() if value is not None})
    rules.append(row)


def _metric_rule(
    rules: list[dict[str, Any]],
    name: str,
    metrics: dict[str, Any],
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    actual = metrics.get(key)
    if actual is None:
        _rule(rules, name, False, reason=f"Missing metric: {key}")
        return
    actual_float = float(actual)
    if min_value is not None:
        _rule(rules, name, actual_float >= min_value, actual=actual_float, required=min_value)
    else:
        _rule(rules, name, actual_float <= float(max_value), actual=actual_float, required=max_value)


def _per_class_rules(rules: list[dict[str, Any]], evaluation: dict[str, Any], per_class_requirements: dict[str, Any]) -> None:
    if not per_class_requirements:
        return
    rows = evaluation.get("metrics", {}).get("per_class") or evaluation.get("per_class") or []
    by_name = {row.get("class") or row.get("class_name") or row.get("name"): row for row in rows if isinstance(row, dict)}
    for class_name, requirements in per_class_requirements.items():
        row = by_name.get(class_name)
        if row is None:
            _rule(rules, f"per_class_{class_name}", False, reason=f"Missing per-class result for {class_name}")
            continue
        for metric, minimum in (requirements or {}).items():
            actual = row.get(metric)
            _rule(
                rules,
                f"per_class_{class_name}_{metric}",
                actual is not None and float(actual) >= float(minimum),
                actual=actual,
                required=minimum,
            )


def _path_present(path: str | None) -> bool:
    return bool(path and Path(path).is_file())
