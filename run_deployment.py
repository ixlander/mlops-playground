import click
from rich import print

from pipelines.deployment_pipeline import continuous_deployment_pipeline
from pipelines.inference_pipeline import inference_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    show_default=True,
    help="Execution mode: deploy, predict, or both.",
)
@click.option(
    "--data-path",
    "-d",
    required=True,
    type=str,
    help="Path to CSV used for deploy/predict.",
)
@click.option(
    "--min-r2",
    default=0.2,
    show_default=True,
    type=float,
    help="Minimum R2 required to deploy the model.",
)
def main(config: str, data_path: str, min_r2: float):
    """
    Run deployment and/or inference pipeline.
    
    Args:
        config: Execution mode
        data_path: Path to the data
        min_r2: Minimum R2 score for deployment
    """
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    if deploy:
        print("\n[bold blue]Running Continuous Deployment Pipeline...[/bold blue]")
        continuous_deployment_pipeline(
            data_path=data_path,
            min_r2=min_r2,
            workers=3,
            timeout=60,
        )
        print("[bold green]✓ Deployment pipeline completed[/bold green]")

    if predict:
        print("\n[bold blue]Running Inference Pipeline...[/bold blue]")
        preds_df = inference_pipeline(data_path=data_path)
        print("\n[bold green]Predictions:[/bold green]")
        print(preds_df.head())

    print(
        "\n[bold yellow]MLflow UI:[/bold yellow]\n"
        f"[italic green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]"
    )


if __name__ == "__main__":
    main()