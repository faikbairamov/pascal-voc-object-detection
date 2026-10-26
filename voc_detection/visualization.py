"""Visual error inspection using already-collected predictions."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .dataset import VOC_CLASSES


def plot_category_examples(records, dataset, category="bottle", threshold=0.20, limit=3):
    """Plot the first matching examples; threshold affects display only, not AP."""
    class_id = VOC_CLASSES.index(category) + 1
    selected = [
        (index, record) for index, record in enumerate(records)
        if ((record["target"]["labels"] == class_id)
            & ~record["target"]["difficult"]).any().item()
    ][:limit]
    if not selected:
        raise ValueError(f"No non-difficult examples found for {category}.")
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 5 * len(selected)), squeeze=False)
    for row, (index, record) in enumerate(selected):
        if dataset.image_ids[index] != record["image_id"]:
            raise ValueError("Prediction records and dataset image order do not match.")
        image, _ = dataset[index]
        photo = image.permute(1, 2, 0)
        height, width = photo.shape[:2]
        for ax in axes[row]:
            ax.imshow(photo, extent=(0, width, height, 0))
            ax.axis("off")
        target = record["target"]
        for box, label, difficult in zip(target["boxes"], target["labels"], target["difficult"]):
            if label.item() != class_id:
                continue
            x1, y1, x2, y2 = box.tolist()
            axes[row, 0].add_patch(Rectangle(
                (x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime", linewidth=2,
                linestyle="--" if difficult.item() else "-",
            ))
        prediction = record["prediction"]
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            if label.item() != class_id or score.item() < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            axes[row, 1].add_patch(Rectangle(
                (x1, y1), x2-x1, y2-y1, fill=False, edgecolor="orange", linewidth=2,
            ))
            axes[row, 1].text(x1, y1, f"{category}: {score.item():.2f}",
                              bbox={"facecolor": "orange", "alpha": .8})
        axes[row, 0].set_title(f"{record['image_id']} — {category} ground truth")
        axes[row, 1].set_title(f"{category} predictions ≥ {threshold:.2f}")
    fig.tight_layout()
    return fig
