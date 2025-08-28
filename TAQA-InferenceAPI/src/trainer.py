from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from .schemas import RetrainConfig
from .versioning import model_dir, save_meta, atomic_swap, compute_schema_hash
from .metrics import metrics_store


@dataclass
class TrainJob:
    csv_path: Path
    config: RetrainConfig
    model_name: str
    job_id: str


class Trainer:
    def __init__(self) -> None:
        self.queue: "asyncio.Queue[TrainJob]" = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def enqueue(self, job: TrainJob) -> None:
        await self.queue.put(job)

    @staticmethod
    def _schema_hash(df: pd.DataFrame) -> str:
        return compute_schema_hash(df)

    async def _run(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                df = pd.read_csv(job.csv_path)
                schema_hash = self._schema_hash(df)

                model = IsolationForest(
                    n_estimators=job.config.n_estimators,
                    max_samples=job.config.max_samples,
                    contamination=job.config.contamination,
                    random_state=job.config.random_state,
                )
                model.fit(df)

                # prepare target dir
                tdir = model_dir(job.model_name)
                os.makedirs(tdir, exist_ok=True)
                joblib.dump(model, tdir / "model.pkl")

                # warmup: ensure the saved artifact can be loaded and used before swap
                loaded = joblib.load(tdir / "model.pkl")
                _ = loaded.predict(df.head(5))

                meta = {
                    "model_name": job.model_name,
                    "model_version": tdir.name,
                    "trained_at": datetime.utcnow().isoformat() + "Z",
                    "features": list(df.columns),
                    "schema_hash": schema_hash,
                    "n_rows": int(len(df)),
                    "features_dtypes": {c: str(df[c].dtype) for c in df.columns},
                    "thresholds": job.config.thresholds or {},
                    "training_args": {
                        "n_estimators": job.config.n_estimators,
                        "max_samples": job.config.max_samples,
                        "contamination": job.config.contamination,
                        "random_state": job.config.random_state,
                    },
                    "job_id": job.job_id,
                }
                save_meta(tdir, meta)

                # swap current
                atomic_swap(tdir)

                # update metrics
                metrics_store.update(meta)
            except Exception as e:
                # minimal logging; errors swallowed to keep worker alive
                print(f"Training job {job.job_id} failed: {e}")
            finally:
                self.queue.task_done()


trainer = Trainer()
