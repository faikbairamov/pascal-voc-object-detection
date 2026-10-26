"""Aggregate VOC metrics without mixing them with experiment tracking."""
import pandas as pd

from .dataset import VOC_CLASSES
from .engine import collect_predictions
from .metrics import evaluate_category


def summarize_predictions(records):
    """Score all 20 categories, preserving curves and image-level predictions."""
    categories = {
        category: evaluate_category(records, class_id)
        for class_id, category in enumerate(VOC_CLASSES, start=1)
    }
    table = pd.DataFrame([
        {"category": category, "ap50": result["ap"], "objects": result["objects"]}
        for category, result in categories.items()
    ]).set_index("category")
    return {"map50": float(table["ap50"].mean()), "table": table,
            "categories": categories, "records": records}


def evaluate_model(model, dataset, loader, device):
    """Generate current predictions, then calculate development AP."""
    return summarize_predictions(collect_predictions(model, dataset, loader, device))
