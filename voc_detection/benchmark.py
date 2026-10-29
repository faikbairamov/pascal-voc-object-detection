"""Paired, interleaved CPU latency comparison on fixed development images."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import configure_proposals
from .inference import DEFAULT_CHECKPOINT, load_detector, sha256
from .project import load_config, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--images", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/optimization/latency.json"))
    args = parser.parse_args()
    dataset, _ = make_loader(load_config(), "dev")
    if not 1 <= args.images <= len(dataset) or args.repeats < 1:
        parser.error("Choose a positive image count within the dev split, and positive repeats.")
    indices = np.linspace(0, len(dataset) - 1, args.images, dtype=int)
    # Decode and convert before timing: measure detector forward only, batch size one.
    images = [dataset[int(index)][0] for index in indices]
    model, _ = load_detector(args.checkpoint)
    timings = {150: [], 75: []}
    with torch.inference_mode():
        for proposals in timings:
            configure_proposals(model, proposals)
            for image in images[:5]:
                model([image])
        for repeat in range(args.repeats):
            for index, image in enumerate(images):
                order = [150, 75] if (repeat + index) % 2 == 0 else [75, 150]
                for proposals in order:
                    configure_proposals(model, proposals)
                    started = time.perf_counter()
                    model([image])
                    timings[proposals].append(time.perf_counter() - started)
            print(f"Latency repeat {repeat + 1}/{args.repeats} complete", flush=True)
    result = {
        "images": args.images, "repeats": args.repeats, "batch_size": 1,
        "warmup_images_per_setting": min(5, args.images),
        "timing_scope": "CPU detector forward only; decoded tensors; paired interleaved settings",
        "platform": platform.platform(), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "checkpoint_sha256": sha256(args.checkpoint),
        "image_ids": [dataset.image_ids[int(index)] for index in indices],
        "settings": {
            str(proposals): {"mean_ms": float(np.mean(values) * 1000),
                             "median_ms": float(np.median(values) * 1000),
                             "p95_ms": float(np.percentile(values, 95) * 1000),
                             "samples_ms": [v * 1000 for v in values]}
            for proposals, values in timings.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: {m: v for m, v in s.items() if m != "samples_ms"}
           for k, s in result["settings"].items()})


if __name__ == "__main__":
    main()
