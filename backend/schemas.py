from pydantic import BaseModel


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    threshold_used: float
    model_version: str
    mask_path: str
    mask_url: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_path: str
    weights_found: bool
    allow_untrained_model: bool
