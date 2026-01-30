import logging
import pandas as pd 
import mlflow

from zenml import step 
from zenml.client import Client
from sklearn.base import RegressorMixin

from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, R2, RMSE

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "R2 Score"],
    Annotated[float, "Mean Squared Error"],
    Annotated[float, "Root Mean Squared Error"]
]:
    """
    Evaluates the model on the ingested data. 

    Args:
        df (pd.DataFrame): The ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE", mse)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2", r2)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)  
        mlflow.log_metric("RMSE", rmse)
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e

    return r2, mse, rmse