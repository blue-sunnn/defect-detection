from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from backend.prediction_cache import PredictionResultCache


PASS_CLASS_NAME = "Pass"
DEFAULT_STRATEGY = "best_f1"


@dataclass(frozen=True)
class ClassThresholdResult:
    recommended: float | None
    metric_value: float | None
    sample_count: int
    positive_count: int
    negative_count: int
    status: str
    warning: str | None
    diagnostics: dict[str, float | int | str | None]


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    strategy: str
    class_names: list[str]
    classes: dict[str, ClassThresholdResult]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "class_names": list(self.class_names),
            "classes": {
                class_name: asdict(result)
                for class_name, result in self.classes.items()
            },
            "warnings": list(self.warnings),
        }


def optimize_thresholds(
    prediction_cache: PredictionResultCache,
    class_names: Iterable[str],
    *,
    strategy: str = DEFAULT_STRATEGY,
    include_pass: bool = False,
) -> ThresholdOptimizationResult:
    """Calculate per-class score thresholds from cached predictions."""
    expected_classes = list(class_names)
    prediction_cache.validate_class_order(expected_classes)
    if strategy != DEFAULT_STRATEGY:
        raise ValueError(f"Unsupported threshold optimization strategy: {strategy}")

    legacy = prediction_cache.to_legacy_predictions(expected_classes)
    probabilities = np.asarray(legacy["probabilities"], dtype=np.float64)
    if probabilities.size == 0:
        probabilities = np.zeros((0, len(expected_classes)), dtype=np.float64)
    true_labels = np.asarray(legacy["true_labels"], dtype=np.int64)
    _validate_probabilities(probabilities, expected_classes)
    if probabilities.shape[0] != true_labels.shape[0]:
        raise ValueError("Prediction cache sample count mismatch between labels and probabilities.")

    target_classes = [
        class_name
        for class_name in expected_classes
        if include_pass or class_name != PASS_CLASS_NAME
    ]
    results: dict[str, ClassThresholdResult] = {}
    warnings: list[str] = []
    for class_name in target_classes:
        class_index = expected_classes.index(class_name)
        result = _best_f1_for_class(
            class_name=class_name,
            class_index=class_index,
            scores=probabilities[:, class_index],
            positives=(true_labels == class_index),
        )
        results[class_name] = result
        if result.warning:
            warnings.append(f"{class_name}: {result.warning}")

    return ThresholdOptimizationResult(
        strategy=strategy,
        class_names=expected_classes,
        classes=results,
        warnings=warnings,
    )


def _validate_probabilities(probabilities: np.ndarray, class_names: list[str]) -> None:
    if probabilities.ndim != 2:
        raise ValueError("Prediction scores must be a 2D probability matrix.")
    if probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Prediction score class count {probabilities.shape[1]} does not match class list {len(class_names)}."
        )
    if probabilities.size and not np.isfinite(probabilities).all():
        raise ValueError("Prediction scores contain NaN or infinite values.")
    if probabilities.size and ((probabilities < 0.0).any() or (probabilities > 1.0).any()):
        raise ValueError("Prediction scores must be probabilities in the range [0, 1].")


def _best_f1_for_class(
    *,
    class_name: str,
    class_index: int,
    scores: np.ndarray,
    positives: np.ndarray,
) -> ClassThresholdResult:
    sample_count = int(scores.shape[0])
    positive_count = int(np.sum(positives))
    negative_count = int(sample_count - positive_count)
    base_diagnostics = {
        "class_index": class_index,
        "evaluated_thresholds": 0,
        "best_precision": None,
        "best_recall": None,
    }
    if sample_count == 0 or positive_count == 0 and negative_count == 0:
        return _invalid_result(sample_count, positive_count, negative_count, "absent", "Class absent from validation set.", base_diagnostics)
    if positive_count == 0:
        return _invalid_result(sample_count, positive_count, negative_count, "no_positive_samples", "No positive validation samples for this class.", base_diagnostics)
    if negative_count == 0:
        return _invalid_result(sample_count, positive_count, negative_count, "no_negative_samples", "No negative validation samples for this class.", base_diagnostics)

    candidates = sorted({0.0, 1.0, *[float(value) for value in scores.tolist()]})
    best_threshold = None
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    for threshold in candidates:
        predicted_positive = scores >= threshold
        true_positive = int(np.sum(predicted_positive & positives))
        false_positive = int(np.sum(predicted_positive & ~positives))
        false_negative = int(np.sum(~predicted_positive & positives))
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best_f1 or (f1 == best_f1 and (best_threshold is None or threshold > best_threshold)):
            best_threshold = threshold
            best_f1 = f1
            best_precision = precision
            best_recall = recall

    return ClassThresholdResult(
        recommended=round(float(best_threshold), 6),
        metric_value=round(float(best_f1), 6),
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        status="valid",
        warning=None,
        diagnostics={
            "class_index": class_index,
            "evaluated_thresholds": len(candidates),
            "best_precision": round(float(best_precision), 6),
            "best_recall": round(float(best_recall), 6),
        },
    )


def _invalid_result(
    sample_count: int,
    positive_count: int,
    negative_count: int,
    status: str,
    warning: str,
    diagnostics: dict[str, float | int | str | None],
) -> ClassThresholdResult:
    return ClassThresholdResult(
        recommended=None,
        metric_value=None,
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        status=status,
        warning=warning,
        diagnostics=diagnostics,
    )
