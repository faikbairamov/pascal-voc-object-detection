"""Recompute all-category metrics on the saved dev split or official val."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import VOCDetectionDataset, detection_collate_fn
from .evaluation import evaluate_model
from .inference import DEFAULT_CHECKPOINT, load_detector, sha256
from .project import PROJECT_ROOT, load_config


def configure_proposals(model, proposals):
    if proposals < 1:
        raise ValueError("Proposal count must be positive.")
    model.rpn._pre_nms_top_n["testing"] = proposals
    model.rpn._post_nms_top_n["testing"] = proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["dev", "val"], default="dev")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--proposals", type=int)
    parser.add_argument("--min-size", type=int)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    dataset_dir = args.dataset or PROJECT_ROOT / config["dataset_dir"]
    split_file = (PROJECT_ROOT / "splits/dev.txt" if args.split == "dev"
                  else dataset_dir / "ImageSets/Main/val.txt")
    dataset = VOCDetectionDataset(dataset_dir, split_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=detection_collate_fn)
    model, checkpoint = load_detector(args.checkpoint)
    if args.proposals is not None:
        configure_proposals(model, args.proposals)
    if args.min_size is not None:
        model.transform.min_size = (args.min_size,)
    if args.max_size is not None:
        model.transform.max_size = args.max_size
    if model.transform.min_size[0] < 32 or model.transform.max_size < model.transform.min_size[0]:
        parser.error("Image sizes must be at least 32, with max-size >= min-size.")
    started = time.perf_counter()
    result = evaluate_model(model, dataset, loader, torch.device("cpu"))
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split, "images": len(dataset), "map50": result["map50"],
        "evaluation_seconds": elapsed, "checkpoint_sha256": sha256(args.checkpoint),
        "source_run": checkpoint["mlflow_run_id"], "epoch": checkpoint["epoch"],
        "rpn_proposals": model.rpn._post_nms_top_n["testing"],
        "min_size": model.transform.min_size[0], "max_size": model.transform.max_size,
        "batch_size": 2, "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(), "torch": torch.__version__,
        "split_sha256": sha256(split_file),
        "metric": "VOC all-points interpolated AP, IoU > 0.5; difficult matches ignored",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    result["table"].to_csv(args.output / "per_category.csv")
    torch.save(result["records"], args.output / "predictions.pt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
