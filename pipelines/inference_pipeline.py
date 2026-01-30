from zenml import pipeline
import pandas as pd
from typing_extensions import Annotated

from steps.ingest_data import ingest_df
from steps.preprocess_inference import preprocess_inference_df
from steps.inference import predict_from_deployed_model


@pipeline(enable_cache=False)
def inference_pipeline(data_path: str) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Inference pipeline that loads data, preprocesses it, and makes predictions.
    
    Args:
        data_path: Path to the inference data
        
    Returns:
        DataFrame with predictions
    """
    df = ingest_df(data_path=data_path)
    X = preprocess_inference_df(df)
    preds = predict_from_deployed_model(X)
    return preds