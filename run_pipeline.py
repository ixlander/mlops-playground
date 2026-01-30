import click
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


@click.command()
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=str,
    help="Path to the training data CSV file.",
)
def main(data_path: str):
    """
    Run the training pipeline.
    
    Args:
        data_path: Path to the training data
    """
    tracking_uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}")
    
    run = train_pipeline(data_path=data_path)
    
    print("\nTraining Pipeline Completed!")
    print(f"Pipeline run name: {run.name}")
    print(f"\nView results in MLflow UI:")
    print(f"mlflow ui --backend-store-uri '{tracking_uri}'")


if __name__ == "__main__":
    main()