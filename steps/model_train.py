import logging 
import pandas as pd 
from zenml import step 

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step 
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
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))