import logging 
import pandas as pd 
from zenml import step 

import mlflow
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """ 
    Train a machine learning model using the provided DataFrame.
    
    Args:
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        
    Returns:
        Trained machine learning model
    """
    model = None
    if config.model_name == "LinearRegression":
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))