"""ONNX inference with feature order and transforms matching C# FeatureTensorBuilder.

C# references:
- FeatureTensorBuilder.cs: lines 6-63 (single row), 65-111 (batch), 114-125 (FilterByFeatures)
- AnomalyRouter.cs: helpers around batch IF/residuals (for output handling)
- Scripts/ModelRegistry.cs: residual_mad.json reading
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Iterable

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .joins import JoinedRow


def _feature_names(sess: ort.InferenceSession) -> List[str]:
    md = sess.get_modelmeta()
    feats = md.custom_metadata_map.get("feature_names", "")
    return [s.strip() for s in feats.split(",") if s.strip()]


def _transform_value(name: str, v: float) -> float:
    # log1p for any feature containing "Pressure" or "Diff"
    lname = name.lower()
    if "pressure" in lname or "diff" in lname:
        v = np.log1p(max(v, 0.0))
    return float(v)


def build_tensor(sess: ort.InferenceSession, row: JoinedRow) -> np.ndarray:
    feats = _feature_names(sess)
    data = np.zeros((1, len(feats)), dtype=np.float32)
    for j, name in enumerate(feats):
        v = row.Values.get(name, 0.0)
        data[0, j] = _transform_value(name, v)
    return data


def build_tensor_batch(sess: ort.InferenceSession, rows: List[JoinedRow]) -> np.ndarray:
    feats = _feature_names(sess)
    n = len(feats)
    m = len(rows)
    data = np.zeros((m, n), dtype=np.float32)
    feat_idx = {name: j for j, name in enumerate(feats)}
    for i, row in enumerate(rows):
        for tag, v in row.Values.items():
            j = feat_idx.get(tag)
            if j is None:
                continue
            data[i, j] = _transform_value(tag, v)
    return data


def filter_by_features(sess: ort.InferenceSession, rows: List[JoinedRow]) -> List[JoinedRow]:
    feats = set(_feature_names(sess))
    return [r for r in rows if feats.issubset(r.Values.keys())]


def run_if(sess: ort.InferenceSession, rows: List[JoinedRow]) -> Tuple[np.ndarray, np.ndarray]:
    # Expect ONNX with label + score or just score
    input_name = next(iter(sess.get_inputs())).name
    tensor = build_tensor_batch(sess, rows)
    outputs = sess.run(None, {input_name: tensor})
    if len(outputs) == 1:
        scores = outputs[0].reshape(-1)
        labels = (scores < 0).astype(np.int64) * -1 + (scores >= 0).astype(np.int64)
    else:
        labels = outputs[0].reshape(-1)
        scores = outputs[1].reshape(-1)
    return labels, scores


def run_regression(sess: ort.InferenceSession, rows: List[JoinedRow]) -> np.ndarray:
    input_name = next(iter(sess.get_inputs())).name
    tensor = build_tensor_batch(sess, rows)
    outputs = sess.run(None, {input_name: tensor})
    return outputs[0].reshape(-1)


def load_manifest_and_mad(models_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((models_dir / "model_manifest.json").read_text()) if (models_dir / "model_manifest.json").exists() else {}
    mad = json.loads((models_dir / "residual_mad.json").read_text()) if (models_dir / "residual_mad.json").exists() else {}
    return manifest, mad
