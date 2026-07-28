from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from backend.model_config import HEAD_DROPOUT_RATE, IMAGE_SIZE, MODEL_NAME
from backend.config_threshold_optimization import (
    DEPLOYED_MULTIPLIERS_JSON_PATH,
    PASS_CERTAINTY_THRESHOLD,
    VAL_CLASSES_PATH,
    load_multipliers_from_json,
)
from backend.model_factory import build_model
from backend.path_validation import require_path
from backend.utils_misclassification import apply_defect_multipliers
from backend.utils_training import DEVICE as DEFAULT_DEVICE, load_checkpoint_or_exit


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


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


def get_runtime_class_names() -> list[str]:
    classes_path = Path(VAL_CLASSES_PATH)
    if classes_path.exists():
        return np.load(classes_path, allow_pickle=True).tolist()

    return ["Mousebite", "Open", "Pass", "Pinhole", "Protrusion", "Short", "Via"]


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
    class_names = class_names or get_runtime_class_names()
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
        defect_multipliers = load_multipliers_from_json(DEPLOYED_MULTIPLIERS_JSON_PATH, class_names)

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
    pass_idx = class_names.index("Pass") if "Pass" in class_names else -1

    with torch.no_grad():
        outputs_orig = model(batch)
        outputs_hflip = model(torch.flip(batch, dims=[3]))
        outputs_vflip = model(torch.flip(batch, dims=[2]))

        probs_orig = F.softmax(outputs_orig, dim=1)
        probs_hflip = F.softmax(outputs_hflip, dim=1)
        probs_vflip = F.softmax(outputs_vflip, dim=1)

        stacked_probs = torch.stack([probs_orig, probs_hflip, probs_vflip], dim=0)
        mean_probs = stacked_probs.mean(dim=0)

        if pass_idx != -1:
            min_pass_probs, _ = torch.min(stacked_probs[:, :, pass_idx], dim=0)
            mean_probs[:, pass_idx] = min_pass_probs
            mean_probs = mean_probs / mean_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        adjusted_probs = apply_defect_multipliers(
            mean_probs,
            class_names,
            defect_multipliers=defect_multipliers,
        )

        defect_probs = adjusted_probs.clone()
        if pass_idx != -1:
            defect_probs[:, pass_idx] = -1.0

        max_defect_indices = torch.argmax(defect_probs, dim=1)
        final_pred_indices = []
        final_confidences = []

        for row_index in range(adjusted_probs.size(0)):
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
    if predicted_class != "Pass" and predicted_class in active_thresholds and confidence < float(active_thresholds[predicted_class]):
        pass_idx = class_names.index("Pass") if "Pass" in class_names else pred_index
        predicted_class = class_names[pass_idx]
        confidence = float(scores.get(predicted_class, confidence))

    return PredictionResult(
        image_path=str(image_path),
        predicted_class=predicted_class,
        confidence=confidence,
        scores=scores,
    )


def iter_image_paths(input_dir: str | Path) -> Iterable[Path]:
    """Yield supported images recursively, preserving arbitrary folder depth."""
    root = Path(input_dir)
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
