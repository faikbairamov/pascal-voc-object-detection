"""VOC all-points AP and detection matching on zero-based, exclusive-end boxes."""
import numpy as np
from torchvision.ops import box_iou

def calculate_voc_ap(recall, precision):
    extended_recall = np.concatenate(([0.0], recall, [1.0]))

    extended_precision = np.concatenate(([0.0], precision, [0.0]))

    for index in range(len(extended_precision) - 2, -1, -1):
        extended_precision[index] = max(
            extended_precision[index],
            extended_precision[index + 1],
        )

    change_indices = np.where(
        extended_recall[1:] != extended_recall[:-1]
    )[0]

    ap = np.sum(
        (
            extended_recall[change_indices + 1]
            - extended_recall[change_indices]
        )
        * extended_precision[change_indices + 1]
    )

    return float(ap)

def evaluate_category(records, class_id, iou_threshold=0.5):
    ground_truth = {}

    detections = []

    positive_count = 0

    for record in records:
        image_id = record["image_id"]

        target = record["target"]

        mask = target["labels"] == class_id

        boxes = target["boxes"][mask]

        difficult = target["difficult"][mask]

        ground_truth[image_id] = {
            "boxes": boxes,
            "difficult": difficult,
            "matched": np.zeros(len(boxes), dtype=bool),
        }

        positive_count += int((~difficult).sum().item())

        prediction = record["prediction"]

        prediction_mask = prediction["labels"] == class_id

        for score, box in zip(
            prediction["scores"][prediction_mask],
            prediction["boxes"][prediction_mask],
        ):
            detections.append((score.item(), image_id, box))

    assert positive_count > 0, f"No non-difficult objects for class {class_id}."

    detections.sort(key=lambda detection: detection[0], reverse=True)

    true_positives = np.zeros(len(detections))

    false_positives = np.zeros(len(detections))

    for index, (_, image_id, box) in enumerate(detections):
        truth = ground_truth[image_id]

        if len(truth["boxes"]) == 0:
            false_positives[index] = 1

            continue

        overlaps = box_iou(box.unsqueeze(0), truth["boxes"])[0]

        best_index = int(overlaps.argmax().item())

        best_iou = overlaps[best_index].item()

        if best_iou > iou_threshold:
            if truth["difficult"][best_index].item():
                continue

            if not truth["matched"][best_index]:
                true_positives[index] = 1

                truth["matched"][best_index] = True

            else:
                false_positives[index] = 1

        else:
            false_positives[index] = 1

    cumulative_tp = np.cumsum(true_positives)

    cumulative_fp = np.cumsum(false_positives)

    recall = cumulative_tp / positive_count

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        np.finfo(float).eps,
    )

    return {
        "ap": calculate_voc_ap(recall, precision),
        "recall": recall,
        "precision": precision,
        "objects": positive_count,
    }
