import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreprocessingStrategy 


@step
def preprocess_inference_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess inference data using the same preprocessing strategy as training.
    
    Args:
        df: Raw inference data
        
    Returns:
        Preprocessed data ready for prediction
    """
    processed = DataCleaning(df, DataPreprocessingStrategy()).handle_data()

    if "review_score" in processed.columns:
        processed = processed.drop(columns=["review_score"])

    return processed