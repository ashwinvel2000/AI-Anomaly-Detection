"""Feature derivations mirroring MultiCsvJoiner.cs and comments in C#.

C# reference:
- MultiCsvJoiner.cs: lines 34-54 (ToolStateNum, DeltaTemperature, IsOpen)
"""
from __future__ import annotations

from typing import Dict


def derive_features(values: Dict[str, float]) -> Dict[str, float]:
    out = dict(values)
    if "Tool-State" in out:
        out["ToolStateNum"] = out["Tool-State"]
    if "Upstream-Temperature" in out and "Downstream-Temperature" in out:
        out["DeltaTemperature"] = out["Upstream-Temperature"] - out["Downstream-Temperature"]
    choke = out.get("Choke-Position", 0.0)
    out["IsOpen"] = 1.0 if choke > 10 else 0.0
    return out
