from __future__ import annotations

import hashlib
import json
import os
import traceback
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


MODEL_NAME = "EfficientNetV2S"
IMAGE_SIZE = (384, 384)
HEAD_DROPOUT_RATE = 0.5
PASS_CLASS_NAME = "Pass"
PASS_CERTAINTY_THRESHOLD = 0.75

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional default for external callers that cannot pass package_path directly.
# Set DEFECT_MODEL_PACKAGE_PATH to a promoted package folder on the deployment machine.
MODEL_PACKAGE_PATH = os.environ.get("DEFECT_MODEL_PACKAGE_PATH", "")

ERROR_DEFECT_NAME = "Error"


@dataclass
class PredictionResult:
    image_path: str
    predicted_class: str
    confidence: float
    scores: dict[str, float]


@dataclass
class InferenceSession:
    model: torch.nn.Module
    class_names: list[str]
    device: torch.device
    defect_multipliers: dict[str, float]
    checkpoint_path: str
    active_thresholds: dict[str, float] | None = None
    image_size: tuple[int, int] = IMAGE_SIZE


@dataclass(frozen=True)
class LiveModelSettings:
    checkpoint_path: str
    class_names: list[str]
    defect_multipliers: dict[str, float]
    active_thresholds: dict[str, float]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class LivePackage:
    checkpoint_path: str
    labels_path: str
    thresholds_path: str
    multiplier_path: str | None
    metadata_path: str
    class_names: list[str]
    active_thresholds: dict[str, float]
    image_size: tuple[int, int]


def require_path(value: str | PathLike[str] | None, *, label: str, must_exist: bool = False) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required. Select a path in the GUI or pass it explicitly.")

    path = Path(str(value).strip()).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def load_multipliers_from_json(json_path, class_names, default_value=0.0, verbose=False):
    # Per-defect-class score boosts loaded from the JSON path supplied by the caller.
    # If the file is missing, initialize neutral (0.0) multipliers for each defect class.
    path = Path(json_path)
    default_template = {name: default_value for name in class_names if name != PASS_CLASS_NAME}

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if verbose:
                print(f"[JSON] Successfully loaded multipliers from file: {path}")
        except Exception as error:
            print(f"[JSON Error] Failed to read file {path}: {error}.")
            data = default_template
    else:
        if verbose:
            print(f"[JSON Info] '{path}' not found. Initializing defaults...")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(default_template, file, indent=4, ensure_ascii=False)
        except Exception as error:
            print(f"[JSON Error] Could not create baseline file: {error}")
        data = default_template

    return {
        name: float(data.get(name, default_value))
        for name in class_names
        if name != PASS_CLASS_NAME
    }


def apply_defect_multipliers(probabilities, class_names, defect_multipliers):
    if PASS_CLASS_NAME not in class_names:
        raise ValueError(f"Required class '{PASS_CLASS_NAME}' not found in class_names: {class_names}")

    if all(abs(mult) < 1e-5 for mult in defect_multipliers.values()):
        return probabilities

    if isinstance(probabilities, torch.Tensor):
        adjusted_probabilities = probabilities.clone()
    else:
        adjusted_probabilities = np.array(probabilities, copy=True)

    # Dividing by (1 - multiplier) scales a class's probability up (positive multiplier) or
    # down (negative) before renormalizing, letting deployment-time calibration correct for
    # classes the model over/under-predicts without retraining.
    for defect_name, multiplier in defect_multipliers.items():
        if defect_name == PASS_CLASS_NAME or defect_name not in class_names:
            continue

        safe_multiplier = float(np.clip(multiplier, -0.999, 0.999))  # keep the divisor away from 0
        if abs(safe_multiplier) < 1e-5:
            continue

        defect_idx = class_names.index(defect_name)
        adjusted_probabilities[:, defect_idx] /= 1.0 - safe_multiplier

    # Renormalize so scores sum back to 1 after the per-class scaling above.
    if isinstance(adjusted_probabilities, torch.Tensor):
        normalization = adjusted_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        normalization = np.clip(adjusted_probabilities.sum(axis=1, keepdims=True), 1e-12, None)

    return adjusted_probabilities / normalization


def build_model(num_classes, dropout_rate):
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # Freeze the pretrained backbone; only the replacement classifier head below gets trained.
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, num_classes),
    )

    return model.to(DEFAULT_DEVICE)


def load_checkpoint_or_exit(
    model,
    checkpoint_path,
    model_name,
    checkpoint_label="checkpoint",
    device=DEFAULT_DEVICE,
):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Could not find '{checkpoint_path}'. "
            f"Required {model_name} {checkpoint_label} is missing."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Some training scripts save the raw state_dict, others wrap it inside a checkpoint dict
    # alongside optimizer/epoch info; unwrap whichever key holds the actual weights.
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in '{checkpoint_path}'.")

    # Strip the "module." prefix DataParallel/DistributedDataParallel adds to every key.
    cleaned_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    incompatible = model.load_state_dict(cleaned_state, strict=False)
    if incompatible.missing_keys:
        print(f"[Checkpoint] Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {len(incompatible.unexpected_keys)}")
    model.to(device)
    # print(f"Successfully loaded {model_name} {checkpoint_label}.")
    return model


def resolve_device(device_override: str | None = None) -> torch.device:
    if not device_override:
        return DEFAULT_DEVICE

    normalized = device_override.strip().lower()
    if normalized == "gpu":
        normalized = "cuda"

    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    return torch.device(normalized)


def build_inference_transform(image_size: tuple[int, int] | None = None):
    image_size = image_size or IMAGE_SIZE
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_inference_model(
    checkpoint_path: str | None = None,
    class_names: list[str] | None = None,
    device_override: str | None = None,
    defect_multipliers: dict[str, float] | None = None,
    active_thresholds: dict[str, float] | None = None,
    image_size: tuple[int, int] | None = None,
) -> InferenceSession:
    if not class_names:
        raise ValueError(
            "Class names are required when loading a checkpoint directly. "
            "For package-based inference, call load_live_inference_model(package_path)."
        )
    class_names = [str(name) for name in class_names]
    checkpoint_path = require_path(
        checkpoint_path,
        label="Model checkpoint",
        must_exist=True,
    )
    device = resolve_device(device_override)

    model = build_model(len(class_names), HEAD_DROPOUT_RATE)
    model = load_checkpoint_or_exit(
        model,
        checkpoint_path,
        MODEL_NAME,
        checkpoint_label="final Phase 1 checkpoint",
        device=device,
    )
    model.to(device)
    model.eval()

    if defect_multipliers is None:
        defect_multipliers = {
            name: 0.0 for name in class_names if name != PASS_CLASS_NAME
        }

    return InferenceSession(
        model=model,
        class_names=class_names,
        device=device,
        defect_multipliers=defect_multipliers,
        checkpoint_path=checkpoint_path,
        active_thresholds=active_thresholds or {},
        image_size=image_size or IMAGE_SIZE,
    )


def load_image_tensor(image_path: str | Path, transform=None) -> torch.Tensor:
    transform = transform or build_inference_transform()
    with Image.open(image_path) as image:
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        return transform(rgb_image).unsqueeze(0)


def _predict_batch_with_tta(
    model: torch.nn.Module,
    batch: torch.Tensor,
    class_names: list[str],
    defect_multipliers: dict[str, float],
    pass_certainty_threshold: float = PASS_CERTAINTY_THRESHOLD,
):
    pass_idx = class_names.index(PASS_CLASS_NAME) if PASS_CLASS_NAME in class_names else -1

    with torch.no_grad():
        # Test-time augmentation: run the same image through the model 3 ways (original,
        # horizontal flip, vertical flip) and combine the predictions for a steadier result.
        outputs_orig = model(batch)
        outputs_hflip = model(torch.flip(batch, dims=[3]))
        outputs_vflip = model(torch.flip(batch, dims=[2]))

        probs_orig = F.softmax(outputs_orig, dim=1)
        probs_hflip = F.softmax(outputs_hflip, dim=1)
        probs_vflip = F.softmax(outputs_vflip, dim=1)

        stacked_probs = torch.stack([probs_orig, probs_hflip, probs_vflip], dim=0)
        mean_probs = stacked_probs.mean(dim=0)

        if pass_idx != -1:
            # For "Pass" specifically use the minimum (most pessimistic) of the 3 views instead
            # of the mean: a board should only be called good if every augmented view agrees,
            # since this is a defect inspector and false negatives (missed defects) are costly.
            min_pass_probs, _ = torch.min(stacked_probs[:, :, pass_idx], dim=0)
            mean_probs[:, pass_idx] = min_pass_probs
            mean_probs = mean_probs / mean_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        adjusted_probs = apply_defect_multipliers(
            mean_probs,
            class_names,
            defect_multipliers=defect_multipliers,
        )

        # Best defect class among the non-Pass classes, used as the fallback prediction below.
        defect_probs = adjusted_probs.clone()
        if pass_idx != -1:
            defect_probs[:, pass_idx] = -1.0

        max_defect_indices = torch.argmax(defect_probs, dim=1)
        final_pred_indices = []
        final_confidences = []

        for row_index in range(adjusted_probs.size(0)):
            # Only call a board "Pass" if the model clears a high confidence bar; otherwise
            # report the most likely defect even if its own probability is fairly low, since
            # an uncertain defect call is safer than a false pass.
            raw_pass_prob = adjusted_probs[row_index, pass_idx].item() if pass_idx != -1 else 0.0
            if pass_idx != -1 and raw_pass_prob >= pass_certainty_threshold:
                final_pred_indices.append(pass_idx)
                final_confidences.append(raw_pass_prob)
            else:
                chosen_defect = max_defect_indices[row_index].item()
                final_pred_indices.append(chosen_defect)
                final_confidences.append(adjusted_probs[row_index, chosen_defect].item())

        return (
            adjusted_probs.cpu(),
            torch.tensor(final_pred_indices, dtype=torch.long),
            torch.tensor(final_confidences, dtype=torch.float32),
        )


def predict_image(
    session: InferenceSession,
    image_path: str | Path,
    class_names: list[str] | None = None,
    defect_multipliers: dict[str, float] | None = None,
) -> PredictionResult:
    class_names = class_names or session.class_names
    defect_multipliers = defect_multipliers or session.defect_multipliers

    batch = load_image_tensor(image_path, transform=build_inference_transform(session.image_size)).to(session.device)
    adjusted_probs, pred_indices, confidences = _predict_batch_with_tta(
        session.model,
        batch,
        class_names,
        defect_multipliers,
    )

    scores = {
        class_name: float(score)
        for class_name, score in zip(class_names, adjusted_probs[0].tolist())
    }
    pred_index = int(pred_indices[0].item())
    predicted_class = class_names[pred_index]
    confidence = float(confidences[0].item())
    active_thresholds = session.active_thresholds or {}
    if (
        predicted_class != PASS_CLASS_NAME
        and predicted_class in active_thresholds
        and confidence < float(active_thresholds[predicted_class])
    ):
        pass_idx = class_names.index(PASS_CLASS_NAME) if PASS_CLASS_NAME in class_names else pred_index
        predicted_class = class_names[pass_idx]
        confidence = float(scores.get(predicted_class, confidence))

    return PredictionResult(
        image_path=str(image_path),
        predicted_class=predicted_class,
        confidence=confidence,
        scores=scores,
    )


def _configured_live_package_path() -> str:
    return str(MODEL_PACKAGE_PATH).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in promoted package file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Promoted package JSON root must be object: {path}")
    return data


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Promoted package missing {label}: {path}")
    return path


def validate_promotion_package(package_dir: str | Path) -> dict:
    package_path = Path(package_dir)
    manifest_path = package_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Promotion manifest missing: {manifest_path}")

    manifest = _read_json(manifest_path)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"Promotion manifest files must be an object: {manifest_path}")

    for relative, info in files.items():
        if not isinstance(info, dict):
            raise ValueError(f"Promotion manifest entry must be an object: {relative}")
        path = package_path / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Promotion package file missing: {relative}")
        digest = _sha256(path)
        if digest != info.get("sha256"):
            raise ValueError(f"Promotion package hash mismatch: {relative}")
    return manifest


def _labels_class_order(labels: dict) -> list[str]:
    class_names = labels.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("labels.json must contain non-empty class_names list.")

    mapping = labels.get("class_to_index") or {}
    for index, name in enumerate(class_names):
        if str(mapping.get(name)) != str(index):
            raise ValueError(f"Label mapping class order mismatch for {name}.")
    return [str(name) for name in class_names]


def _active_thresholds(thresholds: dict, class_names: list[str]) -> dict[str, float]:
    classes = thresholds.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("thresholds.json must contain classes object.")

    result = {}
    known = set(class_names)
    for class_name, values in classes.items():
        if class_name not in known:
            raise ValueError(f"Threshold class not present in labels: {class_name}")
        if not isinstance(values, dict):
            raise ValueError(f"Threshold entry must be an object for {class_name}.")
        active = values.get("active")
        if active is None:
            raise ValueError(f"Active threshold missing for {class_name}.")
        active_float = float(active)
        if not 0.0 <= active_float <= 1.0:
            raise ValueError(f"Active threshold for {class_name} must be between 0 and 1.")
        result[str(class_name)] = active_float
    return result


def _image_size(metadata: dict) -> tuple[int, int]:
    size = metadata.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("model_metadata.json image_size must be [width, height].")
    return int(size[0]), int(size[1])


def _validate_metadata(metadata: dict, class_names: list[str]) -> None:
    for key in ("model_name", "model_version", "architecture", "image_size"):
        if key not in metadata or metadata.get(key) in (None, ""):
            raise ValueError(f"model_metadata.json missing required field: {key}")
    metadata_classes = metadata.get("class_names")
    if metadata_classes is not None and list(metadata_classes) != class_names:
        raise ValueError("Model metadata class order does not match labels.json.")
    _image_size(metadata)


def _validate_multipliers(multipliers: dict, class_names: list[str]) -> None:
    known = set(class_names)
    for class_name, value in multipliers.items():
        if class_name not in known:
            raise ValueError(f"Multiplier class not present in labels: {class_name}")
        float(value)


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

    labels = _read_json(labels_path)
    thresholds = _read_json(thresholds_path)
    metadata = _read_json(metadata_path)
    multipliers = _read_json(multiplier_path) if multiplier_path.is_file() else None

    class_names = _labels_class_order(labels)
    active_thresholds = _active_thresholds(thresholds, class_names)
    _validate_metadata(metadata, class_names)
    if multipliers is not None:
        _validate_multipliers(multipliers, class_names)

    return LivePackage(
        checkpoint_path=str(model_path),
        labels_path=str(labels_path),
        thresholds_path=str(thresholds_path),
        multiplier_path=str(multiplier_path) if multipliers is not None else None,
        metadata_path=str(metadata_path),
        class_names=class_names,
        active_thresholds=active_thresholds,
        image_size=_image_size(metadata),
    )


def _read_live_multipliers(multiplier_path: str | None, class_names: list[str]) -> dict[str, float]:
    defect_classes = [name for name in class_names if name != PASS_CLASS_NAME]
    if not multiplier_path:
        return {name: 0.0 for name in defect_classes}

    path = Path(multiplier_path)
    if not path.is_file():
        raise FileNotFoundError(f"Promoted package multiplier file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Multiplier JSON root must be an object.")

    allowed = {str(name).strip().casefold() for name in class_names}
    unexpected = [key for key in data if str(key).strip().casefold() not in allowed]
    if unexpected:
        raise ValueError(f"Multiplier class not present in labels: {unexpected[0]}")

    lookup = {str(key).strip().casefold(): value for key, value in data.items()}
    multipliers = {}
    for class_name in defect_classes:
        try:
            value = float(lookup.get(str(class_name).strip().casefold(), 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid multiplier for '{class_name}'.") from exc
        multipliers[class_name] = float(np.clip(value, -0.999, 0.999))
    return multipliers


def resolve_live_model_settings(package_path: str | None = None) -> LiveModelSettings:
    package_path = str(package_path or _configured_live_package_path()).strip()
    if not package_path:
        raise ValueError(
            "Promoted Package required. Pass package_path or set DEFECT_MODEL_PACKAGE_PATH."
        )

    package = resolve_live_package(package_path)
    return LiveModelSettings(
        checkpoint_path=package.checkpoint_path,
        class_names=package.class_names,
        defect_multipliers=_read_live_multipliers(package.multiplier_path, package.class_names),
        active_thresholds=package.active_thresholds,
        image_size=package.image_size,
    )


def load_live_inference_model(
    package_path: str | None = None,
    device_override: str | None = None,
) -> InferenceSession:
    settings = resolve_live_model_settings(package_path)
    return load_inference_model(
        checkpoint_path=settings.checkpoint_path,
        class_names=settings.class_names,
        device_override=device_override,
        defect_multipliers=settings.defect_multipliers,
        active_thresholds=settings.active_thresholds,
        image_size=settings.image_size,
    )


def _image_name(image_path):
    return "" if not image_path else str(image_path)


def _error_defect_name(error=None):
    if error is None:
        return ERROR_DEFECT_NAME
    return f"{ERROR_DEFECT_NAME}: {type(error).__name__}: {error}"


def _error_result(image_path, error=None):
    return {
        "preparedImagePath": _image_name(image_path),
        "DefectiveScore": 0,
        "DefectiveName": _error_defect_name(error),
    }


def _iter_image_paths(request):
    # `request` is untrusted JSON from the calling GUI, so each level is defensively type-checked;
    # a malformed group/image entry yields "" rather than raising, so predict() can still report
    # an error result for that slot instead of aborting the whole batch.
    for group in request:
        if not isinstance(group, dict):
            yield ""
            continue

        images = group.get("Images", [])
        if not isinstance(images, list):
            yield ""
            continue

        for image in images:
            if not isinstance(image, dict):
                yield ""
                continue
            yield image.get("preparedImagePath", "").strip()


def predict(request, package_path: str | None = None):
    """Run inference using only a promoted-package folder.

    Pass ``package_path`` explicitly, or set ``DEFECT_MODEL_PACKAGE_PATH`` for callers that
    cannot supply function arguments. The checkpoint, labels, thresholds, metadata, and
    optional defect multipliers are all resolved from that package.

    Always returns a {"Results": [...]} dict; failures are reported as Error results.
    """
    results = []
    image_paths = list(_iter_image_paths(request))

    try:
        session = load_live_inference_model(package_path=package_path)
    except Exception as error:
        # Model failed to load (missing checkpoint, bad device, etc.) — every image in this
        # batch is unscoreable, so report the same error for each one instead of crashing.
        print(f"[MODEL INIT ERROR] {type(error).__name__}: {error}")
        traceback.print_exc()
        return {
            "Results": [
                _error_result(image_path, error)
                for image_path in image_paths
            ]
        }

    for image_path in image_paths:
        try:
            if not image_path:
                raise ValueError("Each image entry must contain preparedImagePath.")

            result = predict_image(
                session,
                image_path,
            )
            results.append({
                "preparedImagePath": _image_name(image_path),
                "DefectiveScore": round(result.confidence * 100, 1),
                "DefectiveName": result.predicted_class,
            })
        except Exception as error:
            # Isolate per-image failures (e.g. a corrupt/unreadable file) so one bad image
            # doesn't stop the rest of the batch from being scored.
            print(f"[IMAGE ERROR] {image_path}: {type(error).__name__}: {error}")
            traceback.print_exc()
            results.append(_error_result(image_path, error))

    return {"Results": results}
