"""Data-quality checks replicated from _reference-csharp/Services/Anomaly/DataQualityWatchdog.cs

C# reference:
- DataQualityWatchdog.cs: lines 9-78 (DQ states and rules)

Rules:
- NaN/Inf => Bad
- Hard limits as per HARD_LIMITS
- Flat-line(20) for P/T tags using equality up to 6 decimals
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import math

from .joins import JoinedRow


HARD_LIMITS = {
    "Battery-Voltage": (10, 16),
    "Upstream-Pressure": (0, 7000),
    "Downstream-Pressure": (0, 7000),
    "Upstream-Temperature": (0, 150),
    "Downstream-Temperature": (0, 150),
    "Choke-Position": (-1, 101),
    "Target-Position": (0, 100),
    "Tool-State": (0, 20),
}

FLATLINE_TAGS = {"Upstream-Pressure", "Downstream-Pressure", "Upstream-Temperature", "Downstream-Temperature"}
FLAT_WINDOW = 20


@dataclass(frozen=True)
class DqEvent:
    tag: str
    rule: str  # "nan" | "out_of_range" | "flatline"
    state: str  # "Good" | "Warning" | "Bad"


def scan_window(window: List[JoinedRow]) -> Dict[str, str]:
    latest = window[-1]
    states: Dict[str, str] = {}
    for tag, val in latest.Values.items():
        if pd_is_nan_or_inf(val):
            states[tag] = "Bad"; continue
        if tag in HARD_LIMITS:
            lo, hi = HARD_LIMITS[tag]
            if val < lo or val > hi:
                states[tag] = "Bad"; continue
        if tag in FLATLINE_TAGS and len(window) == FLAT_WINDOW:
            if all(f"{r.Values.get(tag, float('nan')):.6f}" == f"{val:.6f}" for r in window):
                states[tag] = "Warning"; continue
        states[tag] = "Good"
    return states


def pd_is_nan_or_inf(x: float) -> bool:
    # Lightweight check without importing numpy globally
    return math.isnan(x) or math.isinf(x)


def get_breaches(window: List[JoinedRow]) -> List[DqEvent]:
    out: List[DqEvent] = []
    latest = window[-1]
    for tag, val in latest.Values.items():
        if pd_is_nan_or_inf(val):
            out.append(DqEvent(tag, "nan", "Bad")); continue
        if tag in HARD_LIMITS:
            lo, hi = HARD_LIMITS[tag]
            if val < lo or val > hi:
                out.append(DqEvent(tag, "out_of_range", "Bad")); continue
        if tag in FLATLINE_TAGS and len(window) == FLAT_WINDOW:
            if all(f"{r.Values.get(tag, float('nan')):.6f}" == f"{val:.6f}" for r in window):
                out.append(DqEvent(tag, "flatline", "Warning"))
    return out
