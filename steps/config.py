from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Configuration for model selection"""
    model_name: str = "LinearRegression"