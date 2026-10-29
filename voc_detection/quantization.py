"""Dev-only experiment: dynamic INT8 linear heads, leaving convolutions FP32.

Uses the legacy torch.ao API available in the pinned environment. This API is
deprecated upstream; the experiment is isolated from the supported FP32 demo.
"""
import argparse
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .evaluation import evaluate_model
from .inference import DEFAULT_CHECKPOINT, load_detector, sha256
from .project import load_config, make_loader


def serialized_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("reports/optimization/int8"))
    parser.add_argument("--images", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    data, loader = make_loader(load_config(), "dev")
    if not 1 <= args.images <= len(data) or args.repeats < 1:
        parser.error("Choose positive image and repeat counts within the dev split.")
    model, _ = load_detector(args.checkpoint)
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        raise RuntimeError("This experiment requires the qnnpack CPU backend.")
    torch.backends.quantized.engine = "qnnpack"
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    args.output.mkdir(parents=True, exist_ok=True)
    result = evaluate_model(quantized, data, loader, torch.device("cpu"))
    result["table"].to_csv(args.output / "per_category.csv")
    indices = np.linspace(0, len(data) - 1, args.images, dtype=int)
    images = [data[int(index)][0] for index in indices]
    models = {"fp32": model, "int8_linear": quantized}
    timings = {name: [] for name in models}
    with torch.inference_mode():
        for candidate in models.values():
            for image in images[:5]:
                candidate([image])
        for repeat in range(args.repeats):
            for index, image in enumerate(images):
                names = list(models)
                if (index + repeat) % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter()
                    models[name]([image])
                    timings[name].append((time.perf_counter() - start) * 1000)
    summary = {
        "split": "dev", "evaluation_images": len(data), "map50": result["map50"],
        "method": "dynamic qint8 for nn.Linear only; convolutions and other operators remain FP32",
        "backend": "qnnpack", "torch": torch.__version__, "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(), "checkpoint_sha256": sha256(args.checkpoint),
        "latency_protocol": "paired interleaved, decoded tensors, batch size 1, 5 warmups per setting",
        "latency_images": args.images, "repeats": args.repeats,
        "image_ids": [data.image_ids[int(index)] for index in indices],
        "api_status": "torch.ao.quantization is deprecated; isolated experiment, not default serving path",
        "settings": {name: {"state_dict_bytes": serialized_bytes(candidate),
                             "mean_ms": float(np.mean(timings[name])),
                             "median_ms": float(np.median(timings[name])),
                             "p95_ms": float(np.percentile(timings[name], 95)),
                             "samples_ms": timings[name]}
                     for name, candidate in models.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "settings"}, indent=2))
    print({name: {k:v for k,v in setting.items() if k != "samples_ms"}
           for name,setting in summary["settings"].items()})


if __name__ == "__main__":
    main()
