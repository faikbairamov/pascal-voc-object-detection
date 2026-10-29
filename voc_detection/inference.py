"""Shared offline inference for the command line and local web demo."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms.functional import to_tensor

from .dataset import VOC_CLASSES
from .model import create_model
from .project import PROJECT_ROOT

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models/best.pt"


def load_detector(checkpoint_path=DEFAULT_CHECKPOINT):
    """Load a trusted local model checkpoint; never fetch pretrained weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = checkpoint.get("classes", list(VOC_CLASSES))
    if list(classes) != list(VOC_CLASSES):
        raise ValueError("Checkpoint class order does not match Pascal VOC.")
    model = create_model(num_classes=21, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    settings = checkpoint.get("inference_config", {})
    model.transform.min_size = (settings.get("min_size", 320),)
    model.transform.max_size = settings.get("max_size", 640)
    proposals = settings.get("rpn_proposals", 150)
    model.rpn._pre_nms_top_n["testing"] = proposals
    model.rpn._post_nms_top_n["testing"] = proposals
    return model.eval(), checkpoint


def predict_image(model, image, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    started = time.perf_counter()
    tensor = to_tensor(image)
    with torch.inference_mode():
        output = model([tensor])[0]
    elapsed = time.perf_counter() - started
    detections = [
        {"label": VOC_CLASSES[int(label) - 1], "class_id": int(label),
         "score": float(score), "box": [float(v) for v in box]}
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"])
        if float(score) >= threshold
    ]
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        box = detection["box"]
        color = (45, 225, 170)
        draw.rectangle(box, outline=color, width=3)
        text = f"{detection['label']} {detection['score']:.0%}"
        x, y = box[0], max(0, box[1] - 16)
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill=color)
        draw.text((x, y), text, fill=(8, 25, 30))
    return annotated, {
        "width": image.width, "height": image.height, "threshold": threshold,
        "inference_seconds": elapsed, "detections": detections,
        "box_format": "xyxy, zero-based pixel coordinates, exclusive end",
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Detect the 20 VOC categories in a new image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.png"))
    args = parser.parse_args()
    model, _ = load_detector(args.checkpoint)
    with Image.open(args.image) as image:
        annotated, result = predict_image(model, image, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
