from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.inference_service import load_inference_model, predict_image


def build_parser():
    parser = argparse.ArgumentParser(description="Predict new images with trained EfficientNetV2-S checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained checkpoint (.pth). Required.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing unlabeled images to predict.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Single image path to predict.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override. Example: cpu or cuda.",
    )
    return parser


def _print_result(result):
    image_name = Path(result.image_path).name
    print(f"{image_name}: {result.predicted_class} ({result.confidence:.2%})")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Provide at least one of --image or --input-dir.")
    if not args.checkpoint or not args.checkpoint.strip():
        parser.error("--checkpoint is required.")

    session = load_inference_model(
        checkpoint_path=args.checkpoint,
        device_override=args.device,
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            parser.error(f"Image not found: {image_path}")
        result = predict_image(session, args.image)
        _print_result(result)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            parser.error(f"Input directory not found: {input_dir}")

        for path in sorted(input_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
                print(f"[Skip] Unsupported file type: {path.name}")
                continue
            try:
                result = predict_image(session, path)
            except Exception as exc:
                print(f"[Warning] Failed on {path.name}: {exc}")
                continue
            _print_result(result)


if __name__ == "__main__":
    main()
