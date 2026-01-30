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
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse"],
    Annotated[float, "rmse"]
]:
    """
    Evaluates the model on the test data. 

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        r2_score: R2 Score
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    try:
        prediction = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)  
        mlflow.log_metric("rmse", rmse)
        
        return r2, mse, rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e