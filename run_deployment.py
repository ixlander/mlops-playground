import click
from rich import print

from pipelines.deployment_pipeline import continuous_deployment_pipeline
from pipelines.inference_pipeline import inference_pipeline  # если у тебя есть файл inference_pipeline.py
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
)
@click.option(
    "--data-path",
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
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    if deploy:
        continuous_deployment_pipeline(
            data_path=data_path,
            min_r2=min_r2,
            workers=3,
            timeout=60,
        )

    if predict:
        preds_df = inference_pipeline(data_path=data_path)
        print(preds_df.head())

    print(
        "MLflow UI:\n"
        f"[italic green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]"
    )


if __name__ == "__main__":
    main()
