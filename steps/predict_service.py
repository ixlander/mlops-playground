from __future__ import annotations

from typing import Optional
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step
def predict_from_deployed_model(
    X: pd.DataFrame,
    pipeline_name: str = "continuous_deployment_pipeline",
    pipeline_step_name: str = "deploy_model",
    model_name: str = "model",
    timeout: int = 120,
) -> pd.Series:
    """Fetch the deployed MLflow service and run inference."""
    deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )

    if not services:
        raise RuntimeError(
            "No deployed MLflow service found. Run deployment first."
        )

    service = services[0]
    service = service 

    if hasattr(service, "start") and not service.is_running:
        service.start(timeout=timeout)

    if hasattr(service, "predict"):
        preds = service.predict(X) 
        return pd.Series(preds)
    
    raise RuntimeError("MLflow service does not expose a predict() method in this setup.")