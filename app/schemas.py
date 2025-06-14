from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum

class HealthStatus(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"

class HealthResponse(BaseModel):
    status: str  # Cambiado de HealthStatus a str para flexibilidad
    message: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    filename: Optional[str]
    predictions: List[str]
    confidence_scores: List[float]
    processing_time: float

class ModelInfo(BaseModel):
    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int

class PredictionResponse(BaseModel):
    filename: Optional[str]
    predictions: List[str]
    confidence_scores: List[float]
    processing_time: float

class ModelInfo(BaseModel):
    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int