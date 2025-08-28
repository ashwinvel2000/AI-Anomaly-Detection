"""Event mapping to severities consistent with C# AnomalyRouter.

C# references:
- AnomalyRouter.cs: lines 21-112, 144-262

Rules extracted:
- IF models: severity thresholds differ by model (examples in code)
- Residuals: compare resid to cutoff from residual_mad.json; ratio >= 2 => High, >= 1 => Medium, else Low
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AnomalyEvent:
    Timestamp: object
    Detector: str
    RawValue: float
    Score: float
    Threshold: float
    Severity: str
    Predicted: float | None = None
    Observed: float | None = None


def severity_if(detector: str, score: float) -> str:
    if detector == "choke_position_if":
        return "High" if score < -0.5 else ("Medium" if score < -0.2 else "Low")
    if detector == "full_vectors_if":
        return "High" if score < -0.7 else ("Medium" if score < -0.4 else "Low")
    # default
    return "High" if score < 0 else "Low"


def severity_residual_ratio(ratio: float) -> str:
    return "High" if ratio >= 2 else ("Medium" if ratio >= 1 else "Low")


def make_residual_event(ts, tag: str, resid: float, pred: float, obs: float, cut: float) -> AnomalyEvent:
    sev = severity_residual_ratio(resid / cut) if cut > 0 else "Low"
    return AnomalyEvent(ts, f"residual_{tag.lower()}", resid, resid, cut, sev, pred, obs)
