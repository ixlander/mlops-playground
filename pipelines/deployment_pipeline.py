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
    """Configuration for deployment decision"""
    min_r2: float = 0.2


@step
def deployment_trigger(r2: float, config: DeploymentTriggerConfig) -> bool:
    """
    Decides whether to deploy the model based on R2 score.
    
    Args:
        r2: R2 score of the model
        config: Deployment trigger configuration
        
    Returns:
        Boolean indicating whether to deploy
    """
    return r2 >= config.min_r2


@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    data_path: str,
    min_r2: float = 0.2,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """
    Continuous deployment pipeline that trains, evaluates, and conditionally deploys a model.
    
    Args:
        data_path: Path to the training data
        min_r2: Minimum R2 score required for deployment
        workers: Number of workers for the deployment service
        timeout: Timeout for service start/stop
    """
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