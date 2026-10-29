"""Validate split isolation, image decoding, and all train/val XML boxes."""
import argparse
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image

from .dataset import VOC_CLASSES
from .inference import sha256
from .project import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/data"))
    args = parser.parse_args()
    root = PROJECT_ROOT / load_config()["dataset_dir"]
    manifests = {"train": PROJECT_ROOT / "splits/train.txt", "dev": PROJECT_ROOT / "splits/dev.txt",
                 "val": root / "ImageSets/Main/val.txt"}
    splits = {name: path.read_text().splitlines() for name, path in manifests.items()}
    errors = []
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate image IDs in {name}.")
    for a, b in [("train", "dev"), ("train", "val"), ("dev", "val")]:
        if set(splits[a]) & set(splits[b]):
            errors.append(f"Split overlap: {a}/{b}.")
    official_train = set((root / "ImageSets/Main/train.txt").read_text().splitlines())
    if set(splits["train"]) | set(splits["dev"]) != official_train:
        errors.append("Fitting/development splits do not partition official train.")
    rows = []
    per_class = Counter()
    for split, ids in splits.items():
        for image_id in ids:
            try:
                with Image.open(root / "JPEGImages" / f"{image_id}.jpg") as image:
                    image.load()
                    width, height = image.size
                annotation = ET.parse(root / "Annotations" / f"{image_id}.xml").getroot()
                if (int(annotation.findtext("size/width")), int(annotation.findtext("size/height"))) != (width, height):
                    errors.append(f"Image/XML dimensions disagree: {image_id}.")
                objects = annotation.findall("object")
                difficult_count = 0
                small_count = 0
                for obj in objects:
                    label = obj.findtext("name")
                    if label not in VOC_CLASSES:
                        errors.append(f"Unknown label in {image_id}: {label}.")
                    box = obj.find("bndbox")
                    x1, y1, x2, y2 = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                    if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
                        errors.append(f"Invalid box in {image_id}: {[x1,y1,x2,y2]}.")
                    difficult = int(obj.findtext("difficult", "0"))
                    difficult_count += difficult
                    small_count += int((x2 - x1 + 1) * (y2 - y1 + 1) < 32 * 32)
                    per_class[(split, label, bool(difficult))] += 1
                rows.append({"split": split, "image_id": image_id, "width": width, "height": height,
                             "objects": len(objects), "difficult": difficult_count, "small_objects": small_count})
            except (OSError, ET.ParseError, TypeError, ValueError) as exc:
                errors.append(f"{image_id}: {exc}")
        print(f"Audited {split}: {len(ids)} images", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "images.csv", index=False)
    pd.DataFrame([{"split": s, "category": c, "difficult": d, "objects": n}
                  for (s,c,d), n in sorted(per_class.items())]).to_csv(args.output / "objects.csv", index=False)
    summary = {"images": {s: len(ids) for s, ids in splits.items()},
               "split_sha256": {s: sha256(p) for s, p in manifests.items()},
               "checks": ["unique IDs", "no split ID overlap", "official train partition",
                          "full image decode", "XML dimensions", "known labels", "box bounds"],
               "errors": errors}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
