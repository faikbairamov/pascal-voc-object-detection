"""Generate traceable experiment tables, plots, and structured dev error counts."""
import json
import sqlite3
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.ops import box_iou

from .dataset import VOC_CLASSES
from .evaluation import summarize_predictions
from .project import PROJECT_ROOT, load_config, make_loader
from .visualization import plot_category_examples


def error_counts(records, threshold=0.5):
    counts = {name: Counter() for name in VOC_CLASSES}
    for record in records:
        truth, prediction = record["target"], record["prediction"]
        matched = torch.zeros(len(truth["labels"]), dtype=torch.bool)
        for index in prediction["scores"].argsort(descending=True).tolist():
            if prediction["scores"][index] < threshold:
                continue
            label = int(prediction["labels"][index])
            counter = counts[VOC_CLASSES[label - 1]]
            overlaps = box_iou(prediction["boxes"][index:index+1], truth["boxes"])[0]
            same = torch.where(truth["labels"] == label)[0]
            if len(same):
                best = int(same[overlaps[same].argmax()])
                if overlaps[best] > .5:
                    if truth["difficult"][best]:
                        counter["ignored_difficult"] += 1
                    elif matched[best]:
                        counter["duplicate"] += 1
                    else:
                        counter["true_positive"] += 1
                        matched[best] = True
                    continue
            best_iou = float(overlaps.max()) if len(overlaps) else 0
            counter["class_confusion" if best_iou > .5 else
                    "localization" if best_iou > .1 else "background"] += 1
        for index, label in enumerate(truth["labels"].tolist()):
            if truth["difficult"][index]:
                continue
            counter = counts[VOC_CLASSES[label-1]]
            counter["objects"] += 1
            box = truth["boxes"][index]
            small = (box[2]-box[0]) * (box[3]-box[1]) < 32 * 32
            if small:
                counter["small_objects"] += 1
            if not matched[index]:
                counter["missed"] += 1
                if small:
                    counter["small_missed"] += 1
    columns = ["objects", "true_positive", "missed", "duplicate", "class_confusion",
               "localization", "background", "ignored_difficult", "small_objects", "small_missed"]
    return pd.DataFrame([{ "category": c, **{k: n[k] for k in columns}}
                         for c,n in counts.items()]).set_index("category")


def export_history():
    database = PROJECT_ROOT / "mlflow_local/tracking.db"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    baseline = "d43c960633a7477e8e8d719de3a27338"
    runs = [(baseline, 1, "Baseline epoch 1", "training", .001),
            (baseline, 2, "Baseline epoch 2", "training", .001),
            (baseline, 3, "Epoch 3 control; reset shuffle", "training", .001),
            ("1e1ca85634974b3289a6dac12ea3df12", 3,
             "Epoch 3 lower LR; reset shuffle", "training", .0001),
            ("6bd00a50ee294e1aa5502ae1d16b146c", 2,
             "Epoch 2 weights; 640 resolution", "evaluation_only", .001)]
    rows = []
    for run, epoch, name, kind, lr in runs:
        metrics = dict(connection.execute("select key,value from metrics where run_uuid=? and step=?",
                                          (run, epoch)).fetchall())
        rows.append({"experiment": name, "kind": kind, "run_id": run, "epoch": epoch,
                     "learning_rate": lr, "dev_map50": metrics["dev_map50"],
                     "train_loss": metrics.get("train_loss"), "epoch_seconds": metrics.get("epoch_seconds"),
                     "min_size": 640 if kind == "evaluation_only" else 320,
                     "training_device": "cpu", "horizontal_flip_prob": 0.0,
                     "training_min_sizes": "[320]", "evaluation_rpn_proposals": 150})
    gpu_runs = connection.execute("select run_uuid,name from runs where (name like 'mps-%' or name like 'colab-%') and status='FINISHED'").fetchall()
    for run, name in gpu_runs:
        params = dict(connection.execute("select key,value from params where run_uuid=?", (run,)).fetchall())
        epochs = connection.execute("select step,value from metrics where run_uuid=? and key='dev_map50' order by step", (run,)).fetchall()
        for epoch, score in epochs:
            metrics = dict(connection.execute("select key,value from metrics where run_uuid=? and step=?", (run,epoch)).fetchall())
            rows.append({"experiment": name, "kind": "training", "run_id": run,
                         "epoch": epoch, "learning_rate": float(params["learning_rate"]),
                         "dev_map50": score, "train_loss": metrics.get("train_loss"),
                         "epoch_seconds": metrics.get("epoch_seconds"),
                         "min_size": int(params.get("evaluation_min_size",320)),
                         "training_device": params.get("device","mps"), "horizontal_flip_prob": float(params.get("horizontal_flip_prob",0)),
                         "training_min_sizes": params.get("training_min_sizes", "[320]"),
                         "evaluation_rpn_proposals": int(params.get("evaluation_rpn_proposals",150))})
    connection.close()
    pd.DataFrame(rows).to_csv(PROJECT_ROOT / "reports/experiments.csv", index=False)


def main():
    reports = PROJECT_ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    export_history()
    records = torch.load(reports / "evaluation/dev/predictions.pt", weights_only=True)
    result = summarize_predictions(records)
    errors = error_counts(records)
    errors.to_csv(reports / "evaluation/dev/error_counts_at_050.csv")
    totals = errors.sum().astype(int).to_dict()
    totals["confidence_threshold"] = .5
    totals["small_object_definition"] = "original image box area < 32*32 pixels"
    (reports / "evaluation/dev/error_summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    table = result["table"].sort_values("ap50")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(table.index, table["ap50"] * 100, color="#168b77")
    ax.axvline(result["map50"] * 100, color="#bd6635", linestyle="--", label=f"Mean {result['map50']:.2%}")
    ax.set(xlabel="AP@0.5 (%)", xlim=(0, 100), title="Development performance · 1,144 images")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "dev_per_category.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, category in zip(axes.flat, ["person", "cat", "bottle", "pottedplant"]):
        curve = result["categories"][category]
        ax.plot(curve["recall"], curve["precision"], color="#168b77")
        ax.set(xlim=(0,1), ylim=(0,1.02), xlabel="Recall", ylabel="Precision",
               title=f"{category} · AP {curve['ap']:.2%}")
    fig.tight_layout(); fig.savefig(figures / "dev_precision_recall.png", dpi=160); plt.close(fig)
    dataset, _ = make_loader(load_config(), "dev")
    for category in ("bottle", "pottedplant"):
        fig = plot_category_examples(records, dataset, category=category, threshold=.5, limit=3)
        fig.savefig(figures / f"{category}_examples.png", dpi=120); plt.close(fig)
    data = pd.read_csv(reports / "data/objects.csv")
    train = data[(data["split"] == "train") & ~data["difficult"]].set_index("category")["objects"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(train.index, train.values, color="#437b9c")
    ax.set(xlabel="Non-difficult annotated objects", title="Fitting split class distribution · 4,573 images")
    fig.tight_layout(); fig.savefig(figures / "training_class_distribution.png", dpi=160); plt.close(fig)
    print("Generated experiment table, structured error counts, and five figures.")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
