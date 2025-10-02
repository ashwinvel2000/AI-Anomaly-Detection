from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from pydantic import RootModel
from typing import List, Dict, Any, Optional
import pandas as pd

from .utils import schema_hash_for_df as _schema_hash_for_df


class Row(RootModel[Dict[str, Any]]):
    # Flexible row: dict of feature -> value as RootModel
    pass


class BatchPredictRequest(BaseModel):
    rows: List[Row] = Field(default_factory=list)


class RetrainConfig(BaseModel):
    model_name: str = Field("default", description="Logical model name")
    n_estimators: int = 100
    max_samples: Optional[float] = None
    contamination: Optional[float] = None
    random_state: int = 42
    thresholds: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: datetime
    details: Dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    started_at: datetime


def compute_schema_hash_df(df: pd.DataFrame) -> str:
    """Stable schema hash based on column order and dtypes."""
    return _schema_hash_for_df(df)
