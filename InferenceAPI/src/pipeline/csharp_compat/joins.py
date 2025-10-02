"""Join logic mirroring _reference-csharp/Services/Anomaly/MultiCsvJoiner.cs

C# reference:
- MultiCsvJoiner.cs: lines 8-66
  - Takes streams: tag -> IEnumerable<(DateTime ts, double val)>
  - Assumes identical timestamps for inner-join; throws if misaligned
  - Derives ToolStateNum (from 'Tool-State'), DeltaTemperature, IsOpen
  - Emits JoinedRow(ts, dict, dict['IsOpen'] == 1)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, Tuple


@dataclass(frozen=True)
class JoinedRow:
    Timestamp: datetime
    Values: Dict[str, float]
    IsOpen: bool


def inner_join_identical_timestamps(streams: Dict[str, Iterable[tuple[datetime, float]]]) -> Iterator[JoinedRow]:
    enums = {k: iter(v) for k, v in streams.items()}
    # prime
    try:
        cur = {k: next(it) for k, it in enums.items()}
    except StopIteration:
        return

    while True:
        ts = next(iter(cur.values()))[0]  # ts from first
        if any(v[0] != ts for v in cur.values()):
            raise ValueError("Timestamps not aligned.")

        values = {k: float(v[1]) for k, v in cur.items()}

        if "Tool-State" in values:
            values["ToolStateNum"] = values["Tool-State"]

        if "Upstream-Temperature" in values and "Downstream-Temperature" in values:
            values["DeltaTemperature"] = values["Upstream-Temperature"] - values["Downstream-Temperature"]

        choke = values.get("Choke-Position", 0.0)
        values["IsOpen"] = 1.0 if choke > 10 else 0.0
        is_open = values["IsOpen"] == 1.0

        yield JoinedRow(ts, values, is_open)

        # advance
        ended = False
        for k, it in enums.items():
            try:
                cur[k] = next(it)
            except StopIteration:
                ended = True
        if ended:
            return
