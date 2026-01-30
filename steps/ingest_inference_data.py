import pandas as pd
from zenml import step


@step
def ingest_inference_df(data_path: str) -> pd.DataFrame:
    """Load data for inference."""
    return pd.read_csv(data_path)