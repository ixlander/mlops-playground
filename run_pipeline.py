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
    
    r2, mse, rmse = train_pipeline(data_path=data_path)
    
    print("Training Pipeline Completed")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()