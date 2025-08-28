"""CSV loaders that mirror _reference-csharp/Services/Anomaly/CsvStreamReader.cs

C# reference:
- CsvStreamReader.cs: lines 6-33
  - HasHeaderRecord = true
  - BadDataFound = null (ignore junk lines)
  - Read header, then for each row: GetField<DateTime>(0), GetField<double>(1)

We replicate this strictly: two columns [Timestamp, value], header present, parse
Timestamp with pandas to datetime64[ns] using ISO-compatible parser.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def read_two_col_csv(path: str | Path) -> Iterable[tuple[pd.Timestamp, float]]:
    # Header required, read strictly two columns
    df = pd.read_csv(path, usecols=[0, 1])
    # Coerce dtypes to mirror CsvHelper GetField<DateTime>, GetField<double>
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="raise", utc=False)
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    # Drop junk (NaN in value) akin to BadDataFound = null then numeric coercion
    df = df.dropna(subset=[df.columns[1]])
    for _, r in df.iterrows():
        yield (pd.Timestamp(r.iloc[0]), float(r.iloc[1]))
