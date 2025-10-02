from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .versioning import read_pointer, load_meta
from .models.base_model import BaseAnomalyModel, IsolationForestModel
from .models.registry import create_model


class Predictor:
    def __init__(self) -> None:
        self.model: Optional[BaseAnomalyModel] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.model_dir: Optional[Path] = None

    def load_current(self) -> None:
        cur = read_pointer()
        # For now we assume IsolationForest; meta can store model_name later.
        if cur and (cur / "model.pkl").exists():
            self.model_dir = cur
            m = IsolationForestModel()
            m.load(cur)
            self.model = m
            self.meta = load_meta(cur)

    def schema_hash(self) -> Optional[str]:
        if self.meta:
            return self.meta.get("schema_hash")
        return None

    def predict(self, df: pd.DataFrame) -> List[float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        preds = self.model.predict(df)
        return [float(p) for p in preds]


predictor = Predictor()
