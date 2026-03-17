from typing import List, Optional

from pydantic import BaseModel, Field


class CategoryScore(BaseModel):
    prdtypecode: int
    category_name: Optional[str] = None
    proba: float


class PredictionRequest(BaseModel):
    designation: str = Field(..., min_length=10)
    top_k: int = Field(1, ge=1, le=10)


class PredictionResponse(BaseModel):
    designation: str
    predictions: List[CategoryScore]

class ModelReloadResponse(BaseModel):
    status: str
    message: Optional[str] = None
    reload_time_seconds: Optional[float] = None