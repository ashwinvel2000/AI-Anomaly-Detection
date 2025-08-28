from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import onnxruntime as ort
from ...config import EXTERNAL_MODEL_DIR


class OnnxRegistry:
    def __init__(self, models_dir: Path | str = EXTERNAL_MODEL_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._sess: Dict[str, ort.InferenceSession] = {}
        self._mad: Dict[str, dict] | None = None

    def get_session(self, file: str) -> ort.InferenceSession:
        if file in self._sess:
            return self._sess[file]
        full = self.models_dir / file
        if not full.exists():
            raise FileNotFoundError(f"Model not found: {full}")
        # Session options can be tuned here
        so = ort.SessionOptions()
        sess = ort.InferenceSession(full.as_posix(), sess_options=so)
        self._sess[file] = sess
        return sess

    @property
    def residual_mad(self) -> Dict[str, dict]:
        if self._mad is None:
            p = self.models_dir / "residual_mad.json"
            self._mad = json.loads(p.read_text()) if p.exists() else {}
        return self._mad


registry = OnnxRegistry()
