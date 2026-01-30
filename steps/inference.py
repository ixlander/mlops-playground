from typing import Optional

import pandas as pd
import requests
from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step
def predict_from_deployed_model(
    data: pd.DataFrame,
    pipeline_name: str = "continuous_deployment_pipeline",
    pipeline_step_name: str = "mlflow_model_deployer_step",
    model_name: str = "model",
    timeout: int = 60,
) -> pd.DataFrame:
    """Load the active MLflow deployed service and run predictions."""

    deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )

    if not services:
        raise RuntimeError(
            "No deployed MLflow service found.\n"
            "Run deployment first:\n"
            "py run_deployment.py --config deploy --data-path <path>"
        )

    service = services[0]
    if not isinstance(service, MLFlowDeploymentService):
        raise TypeError("Found service is not an MLFlowDeploymentService.")

    if not service.is_running:
        service.start(timeout=timeout)

    url = service.prediction_url.rstrip("/") + "/invocations"
    payload = {
        "dataframe_split": {
            "columns": list(data.columns),
            "data": data.values.tolist(),
        }
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()

    preds = resp.json()
    if isinstance(preds, dict) and "predictions" in preds:
        preds = preds["predictions"]

    out = data.copy()
    out["prediction"] = preds
    return out
