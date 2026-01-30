from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.preprocess_inference import preprocess_inference_df
from steps.inference import predict_from_deployed_model


@pipeline(enable_cache=False)
def inference_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    X = preprocess_inference_df(df)
    preds = predict_from_deployed_model(X)
    return preds