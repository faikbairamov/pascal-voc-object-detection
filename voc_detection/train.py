"""Explicit training/resume entry point with MLflow tracking and checkpoints."""
import argparse
from pathlib import Path

import mlflow
import torch

from .engine import train_one_epoch
from .evaluation import evaluate_model
from .model import create_model
from .project import PROJECT_ROOT, load_config, make_loader, seed_everything
from .tracking import configure_tracking, log_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2, help="Additional epochs to train")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--horizontal-flip", type=float)
    parser.add_argument("--name", default="cli-training")
    parser.add_argument("--train-min-sizes", type=int, nargs="+")
    parser.add_argument("--train-max-size", type=int)
    parser.add_argument("--restart-shuffle", action="store_true",
                        help="Restart seed-42 order as in the historical epoch-3 comparison")
    args = parser.parse_args()
    if args.epochs < 1 or (args.lr is not None and args.lr <= 0):
        parser.error("Epochs and learning rate must be positive.")
    if args.horizontal_flip is not None and not 0 <= args.horizontal_flip <= 1:
        parser.error("Horizontal-flip probability must be between zero and one.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("The MPS GPU is unavailable in this process.")
    if args.train_min_sizes and min(args.train_min_sizes) < 32:
        parser.error("Training minimum sides must be at least 32 pixels.")
    checkpoint = (torch.load(args.resume, map_location="cpu", weights_only=True)
                  if args.resume else None)
    config = {**load_config(), **(checkpoint["config"] if checkpoint else {})}
    config["device"] = args.device
    if args.horizontal_flip is not None:
        config["horizontal_flip_prob"] = args.horizontal_flip
    if args.train_min_sizes:
        config["training_min_sizes"] = args.train_min_sizes
    if args.train_max_size:
        config["training_max_size"] = args.train_max_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.dataset:
        config["dataset_dir"] = str(args.dataset.resolve())
    seed_everything(config["seed"])
    training_data, training_loader = make_loader(config, "train", shuffle=True)
    dev_data, dev_loader = make_loader(config, "dev", shuffle=False)
    # Training must preserve the original COCO-pretrained frozen BN and backbone policy.
    # First use downloads TorchVision weights; subsequent runs use its cache.
    device = torch.device(args.device)
    model = create_model(config["num_classes"], pretrained=True).to(device)
    model.transform.min_size = tuple(config.get("training_min_sizes", [320]))
    model.transform.max_size = config.get("training_max_size", 640)
    if model.transform.max_size < max(model.transform.min_size):
        parser.error("Training max side must be at least the largest minimum side.")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                               lr=config["learning_rate"], momentum=config["momentum"],
                               weight_decay=config["weight_decay"])
    first_epoch = 1
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" not in checkpoint:
            parser.error("Resume from a training checkpoint, not inference-only weights.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        first_epoch = checkpoint["epoch"] + 1
        if not args.restart_shuffle and "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
            training_loader.generator.set_state(checkpoint["loader_rng_state"])
            if args.device == "mps" and "mps_rng_state" in checkpoint:
                torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    configure_tracking()
    directory = PROJECT_ROOT / "mlflow_local/checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=args.name) as run:
        mlflow.log_params({**config, "additional_epochs": args.epochs,
                           "starting_epoch": first_epoch - 1,
                           "restart_shuffle": args.restart_shuffle,
                           "evaluation_device": "cpu", "evaluation_min_size": 320,
                           "evaluation_max_size": 640})
        if args.resume:
            mlflow.set_tag("source_checkpoint", args.resume.name)
        for split in ("train", "dev"):
            mlflow.log_artifact(str(PROJECT_ROOT / "splits" / f"{split}.txt"), artifact_path="splits")
        for epoch in range(first_epoch, first_epoch + args.epochs):
            result = train_one_epoch(model, training_loader, optimizer, device, epoch)
            mlflow.log_metrics(result, step=epoch)
            path = directory / f"{run.info.run_id}_epoch_{epoch}.pt"
            state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "config": config,
                        "mlflow_run_id": run.info.run_id, "torch_rng_state": torch.get_rng_state(),
                        "loader_rng_state": training_loader.generator.get_state()}
            if args.device == "mps":
                state["mps_rng_state"] = torch.mps.get_rng_state()
            torch.save(state, path)
            mlflow.log_artifact(str(path), artifact_path="checkpoints")
            # Select on CPU inference, matching the supported deployment and earlier scores.
            if args.device == "cpu":
                evaluation_model = model
            else:
                with torch.random.fork_rng(devices=[]):
                    evaluation_model = create_model(config["num_classes"], pretrained=False)
                evaluation_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
            training_sizes = (model.transform.min_size, model.transform.max_size)
            evaluation_model.transform.min_size = (320,)
            evaluation_model.transform.max_size = 640
            try:
                evaluation = evaluate_model(evaluation_model, dev_data, dev_loader, torch.device("cpu"))
            finally:
                model.transform.min_size, model.transform.max_size = training_sizes
            if args.device != "cpu":
                del evaluation_model
            log_evaluation(evaluation, epoch)
            print(f"Epoch {epoch}: dev mAP@0.5 {evaluation['map50']:.2%}; {path}", flush=True)


if __name__ == "__main__":
    main()
