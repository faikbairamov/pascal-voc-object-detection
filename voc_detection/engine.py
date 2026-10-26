"""Reusable training and inference; notebooks own experiment orchestration."""
import time

import mlflow
import torch
from torch.utils.data import SequentialSampler

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    total_weighted_loss = 0.0

    images_seen = 0

    started_at = time.perf_counter()

    for batch_number, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]

        model_targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            for target in targets
        ]

        optimizer.zero_grad()

        losses = model(images, model_targets)

        loss = sum(losses.values())

        assert torch.isfinite(loss), "Non-finite training loss."

        loss.backward()

        optimizer.step()

        loss_value = loss.detach().item()

        batch_size = len(images)

        total_weighted_loss += loss_value * batch_size

        images_seen += batch_size

        global_step = (epoch - 1) * len(loader) + batch_number

        if batch_number == 1 or batch_number % 100 == 0 or batch_number == len(loader):
            mlflow.log_metric("train_batch_loss", loss_value, step=global_step)

            print(
                f"Epoch {epoch} | Batch {batch_number}/{len(loader)} "
                f"| Loss: {loss_value:.4f}"
            )

    average_loss = total_weighted_loss / images_seen

    elapsed_seconds = time.perf_counter() - started_at

    return {
        "train_loss": average_loss,
        "epoch_seconds": elapsed_seconds,
    }


def collect_predictions(model, dataset, loader, device):
    """Collect fresh CPU predictions in manifest order without changing weights.

    A sequential loader is required because image IDs are assigned by order.
    Restore the model's previous training/evaluation mode on exit.
    """
    if loader.dataset is not dataset or not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Use this dataset's unshuffled evaluation loader.")
    records = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch_number, (images, targets) in enumerate(loader, start=1):
                images = [image.to(device) for image in images]
                predictions = model(images)
                for prediction, target in zip(predictions, targets):
                    records.append({
                        "image_id": dataset.image_ids[len(records)],
                        "prediction": {key: value.detach().cpu() for key, value in prediction.items()},
                        "target": {key: value.detach().cpu() for key, value in target.items()},
                    })
                if batch_number % 100 == 0 or batch_number == len(loader):
                    print(f"Evaluated {len(records)}/{len(dataset)} images")
    finally:
        model.train(was_training)
    if len(records) != len(dataset):
        raise ValueError("Evaluation image count mismatch.")
    return records
