"""Export portable inference weights without optimizer state or local paths."""
import argparse
import json
from pathlib import Path

import torch

from .dataset import VOC_CLASSES
from .inference import DEFAULT_CHECKPOINT, load_detector, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--min-size", type=int, default=320)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--proposals", type=int, default=150)
    args = parser.parse_args()
    if args.min_size < 32 or args.max_size < args.min_size or args.proposals < 1:
        parser.error("Use positive proposal count and valid image sizes.")
    model, checkpoint = load_detector(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "classes": list(VOC_CLASSES), "epoch": checkpoint["epoch"],
        "mlflow_run_id": checkpoint["mlflow_run_id"],
        "source_sha256": sha256(args.source),
        "config": checkpoint["config"],
        "inference_config": {"min_size": args.min_size, "max_size": args.max_size,
                             "rpn_proposals": args.proposals},
    }
    torch.save({**metadata, "model_state_dict": model.state_dict()}, args.output)
    metadata.update({"sha256": sha256(args.output),
                     "bytes": args.output.stat().st_size,
                     "source_bytes": args.source.stat().st_size})
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
