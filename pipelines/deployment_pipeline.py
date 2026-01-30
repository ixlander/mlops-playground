from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from pydantic import BaseModel

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.config import ModelNameConfig


class DeploymentTriggerConfig(BaseModel):
    min_r2: float = 0.2


@step
def deployment_trigger(r2: float, config: DeploymentTriggerConfig) -> bool:
    return r2 >= config.min_r2


@pipeline(enable_cache=True)
def continuous_deployment_pipeline(
    data_path: str,
    min_r2: float = 0.2,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)

    model = train_model(X_train, y_train, ModelNameConfig())
    r2, mse, rmse = evaluate_model(model, X_test, y_test)

    should_deploy = deployment_trigger(r2, DeploymentTriggerConfig(min_r2=min_r2))

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=should_deploy,
        model_name="model",
        workers=workers,
        timeout=timeout,
    )
