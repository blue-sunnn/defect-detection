from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictionCacheMetadata:
    schema_version: int
    checkpoint_identity: dict[str, Any]
    dataset_identity: dict[str, Any]
    class_names: list[str]
    model_name: str
    image_size: list[int]
    preprocessing: dict[str, Any]


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    source_path: str | None
    true_index: int
    true_class: str
    predicted_index: int | None
    predicted_class: str | None
    probabilities: list[float]
    confidence: float | None
    error: str | None = None


@dataclass(frozen=True)
class PredictionResultCache:
    metadata: PredictionCacheMetadata
    records: list[PredictionRecord]

    def validate_class_order(self, class_names: Iterable[str]) -> None:
        expected = list(class_names)
        if self.metadata.class_names != expected:
            raise ValueError(
                "Prediction cache class order mismatch. "
                f"Cache={self.metadata.class_names} Expected={expected}"
            )

    def validate_identity(
        self,
        *,
        checkpoint_path: str,
        dataset_root: str,
        class_names: Iterable[str],
    ) -> None:
        self.validate_class_order(class_names)
        checkpoint = checkpoint_identity(checkpoint_path)
        dataset = dataset_identity(dataset_root)
        if self.metadata.checkpoint_identity.get("signature") != checkpoint.get("signature"):
            raise ValueError("Prediction cache is stale: checkpoint identity changed.")
        if self.metadata.dataset_identity.get("signature") != dataset.get("signature"):
            raise ValueError("Prediction cache is stale: dataset identity changed.")

    def to_legacy_predictions(self, class_names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        if class_names is not None:
            self.validate_class_order(class_names)
        probabilities = np.array([record.probabilities for record in self.records], dtype=np.float32)
        true_labels = np.array([record.true_index for record in self.records], dtype=np.int64)
        pred_labels = np.array(
            [
                -1 if record.predicted_index is None else int(record.predicted_index)
                for record in self.records
            ],
            dtype=np.int64,
        )
        confidences = np.array(
            [
                0.0 if record.confidence is None else float(record.confidence)
                for record in self.records
            ],
            dtype=np.float32,
        )
        pass_idx = self.metadata.class_names.index("Pass") if "Pass" in self.metadata.class_names else -1
        pass_probabilities = (
            probabilities[:, pass_idx] if pass_idx != -1 and probabilities.size else np.zeros(len(self.records), dtype=np.float32)
        )
        return {
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "confidences": confidences,
            "pass_probabilities": pass_probabilities,
            "probabilities": probabilities,
            "source_paths": np.array([record.source_path or record.sample_id for record in self.records], dtype=object),
        }

    def write_jsonl(self, path: str | os.PathLike[str]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "metadata", **asdict(self.metadata)}, sort_keys=True) + "\n")
            for record in self.records:
                handle.write(json.dumps({"type": "record", **asdict(record)}, sort_keys=True) + "\n")
        temp_path.replace(output_path)

    @classmethod
    def read_jsonl(cls, path: str | os.PathLike[str]) -> "PredictionResultCache":
        cache_path = Path(path)
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            raise FileNotFoundError(f"Prediction cache not found: {cache_path}") from exc
        if not lines:
            raise ValueError(f"Prediction cache is empty or partial: {cache_path}")

        try:
            first = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Prediction cache metadata is corrupt: {cache_path}") from exc
        if first.get("type") != "metadata":
            raise ValueError(f"Prediction cache missing metadata header: {cache_path}")
        try:
            metadata = PredictionCacheMetadata(
                schema_version=int(first["schema_version"]),
                checkpoint_identity=dict(first["checkpoint_identity"]),
                dataset_identity=dict(first["dataset_identity"]),
                class_names=list(first["class_names"]),
                model_name=str(first["model_name"]),
                image_size=list(first["image_size"]),
                preprocessing=dict(first["preprocessing"]),
            )
        except KeyError as exc:
            raise ValueError(f"Prediction cache metadata is incomplete: {cache_path}") from exc
        if metadata.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported prediction cache schema: {metadata.schema_version}")

        records = []
        for line_number, line in enumerate(lines[1:], start=2):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Prediction cache record is corrupt at line {line_number}: {cache_path}") from exc
            if row.get("type") != "record":
                raise ValueError(f"Prediction cache has invalid row type at line {line_number}: {cache_path}")
            probabilities = [float(value) for value in row["probabilities"]]
            if len(probabilities) != len(metadata.class_names):
                raise ValueError(f"Prediction cache probability length mismatch at line {line_number}: {cache_path}")
            records.append(
                PredictionRecord(
                    sample_id=str(row["sample_id"]),
                    source_path=row.get("source_path"),
                    true_index=int(row["true_index"]),
                    true_class=str(row["true_class"]),
                    predicted_index=None if row.get("predicted_index") is None else int(row["predicted_index"]),
                    predicted_class=row.get("predicted_class"),
                    probabilities=probabilities,
                    confidence=None if row.get("confidence") is None else float(row["confidence"]),
                    error=row.get("error"),
                )
            )
        return cls(metadata=metadata, records=records)


def checkpoint_identity(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a stable identity for a checkpoint's actual model bytes.

    The previous implementation included the absolute path and modification time
    in the signature.  A harmless copy, registry update, antivirus touch, or
    packaging step could therefore invalidate a freshly-created prediction
    cache even when the checkpoint bytes had not changed.

    Hashing the file contents makes cache validation follow the model itself.
    The path and mtime are still recorded for diagnostics, but they no longer
    decide whether predictions belong to the checkpoint.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    content_sha256 = digest.hexdigest()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "content_sha256": content_sha256,
        "signature": content_sha256,
    }


def dataset_identity(dataset_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    digest = hashlib.sha256()
    digest.update(str(root).encode("utf-8", errors="surrogatepass"))
    file_count = 0
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda item: str(item).lower(),
    ):
        stat = path.stat()
        relative = path.relative_to(root)
        digest.update(str(relative).encode("utf-8", errors="surrogatepass"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        file_count += 1
    return {
        "path": str(root),
        "image_file_count": file_count,
        "signature": digest.hexdigest(),
    }


def build_prediction_cache(
    *,
    predictions: dict[str, Any],
    class_names: list[str],
    checkpoint_path: str,
    dataset_root: str,
    model_name: str,
    image_size: tuple[int, int],
    preprocessing: dict[str, Any] | None = None,
) -> PredictionResultCache:
    probabilities = predictions.get("probabilities")
    if probabilities is None:
        raise ValueError("Prediction cache requires full probability matrix.")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Prediction probability matrix shape {probabilities.shape} does not match {len(class_names)} classes."
        )
    true_labels = np.asarray(predictions["true_labels"], dtype=np.int64)
    pred_labels = np.asarray(predictions["pred_labels"], dtype=np.int64)
    confidences = np.asarray(predictions.get("confidences", np.zeros(len(true_labels))), dtype=np.float32)
    source_paths = list(predictions.get("source_paths", []))
    if not source_paths:
        source_paths = [f"sample-{index:08d}" for index in range(len(true_labels))]
    if not (len(true_labels) == len(pred_labels) == len(confidences) == len(source_paths) == probabilities.shape[0]):
        raise ValueError("Prediction cache inputs have mismatched sample counts.")

    records = []
    for index, (true_idx, pred_idx, confidence, source_path, probs) in enumerate(
        zip(true_labels, pred_labels, confidences, source_paths, probabilities)
    ):
        true_int = int(true_idx)
        pred_int = int(pred_idx)
        source_text = None if source_path is None else str(source_path)
        records.append(
            PredictionRecord(
                sample_id=source_text or f"sample-{index:08d}",
                source_path=source_text,
                true_index=true_int,
                true_class=class_names[true_int],
                predicted_index=pred_int,
                predicted_class=class_names[pred_int] if 0 <= pred_int < len(class_names) else None,
                probabilities=[float(value) for value in probs.tolist()],
                confidence=float(confidence),
            )
        )

    return PredictionResultCache(
        metadata=PredictionCacheMetadata(
            schema_version=SCHEMA_VERSION,
            checkpoint_identity=checkpoint_identity(checkpoint_path),
            dataset_identity=dataset_identity(dataset_root),
            class_names=list(class_names),
            model_name=model_name,
            image_size=[int(image_size[0]), int(image_size[1])],
            preprocessing=preprocessing or {},
        ),
        records=records,
    )
