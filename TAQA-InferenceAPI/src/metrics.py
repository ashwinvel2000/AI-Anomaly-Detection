from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .versioning import read_pointer, load_meta


class MetricsStore:
    def __init__(self) -> None:
        self.latest: Optional[Dict[str, Any]] = None

    def update(self, meta: Dict[str, Any]) -> None:
        self.latest = meta

    def get(self) -> Optional[Dict[str, Any]]:
        if self.latest:
            return self.latest
        cur = read_pointer()
        if cur:
            m = load_meta(cur)
            if m:
                self.latest = m
                return m
        return None


metrics_store = MetricsStore()
