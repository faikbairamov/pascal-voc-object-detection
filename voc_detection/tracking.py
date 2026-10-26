"""Local MLflow setup and epoch-specific evaluation artifacts."""
import mlflow

from .project import PROJECT_ROOT


def configure_tracking():
    """Reuse the existing SQLite database and experiment without moving artifacts."""
    directory = PROJECT_ROOT / "mlflow_local"
    directory.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (directory / "tracking.db").as_posix())
    name = "pascal-voc-2012-detection"
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name, artifact_location=(directory / "artifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def log_evaluation(result, epoch):
    """Attach evaluation to the caller's active run at the correct epoch."""
    if mlflow.active_run() is None:
        raise RuntimeError("Open the checkpoint's MLflow run before logging evaluation.")
    mlflow.log_metric("dev_map50", result["map50"], step=epoch)
    for category, score in result["categories"].items():
        mlflow.log_metric(f"dev_ap50_{category}", score["ap"], step=epoch)
    mlflow.set_tag("evaluation_method", "VOC all-points interpolated AP; IoU > 0.5")
    mlflow.set_tag("evaluation_difficult_policy", "Ignore matches to difficult objects")
    mlflow.log_text(result["table"].to_csv(),
                    artifact_file=f"evaluation/dev_per_category_ap50_epoch_{epoch}.csv")
