from zenml import pipeline  # Fixed: was from zenml.pipelines import pipeline
from typing import Tuple
from typing_extensions import Annotated

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


@pipeline(enable_cache=False)
def train_pipeline(data_path: str) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse"],
    Annotated[float, "rmse"]
]:
    """
    Training pipeline that ingests data, cleans it, trains a model, and evaluates it.
    
    Args:
        data_path: Path to the training data CSV file
        
    Returns:
        r2_score: R2 Score of the model
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    config = ModelNameConfig()
    model = train_model(X_train, y_train, config)
    r2, mse, rmse = evaluate_model(model, X_test, y_test)
    return r2, mse, rmse 