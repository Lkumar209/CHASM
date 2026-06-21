"""Versioned trajectory store: parquet + SQLite index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class TrajectoryStore:
    """
    Persist and query trajectories.
    Parquet for bulk data; SQLite index for fast lookup by task/condition/split.
    """

    def __init__(self, store_dir: str | Path = "data/trajectories") -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.db"
        self._records: list[dict[str, Any]] = []
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3
        con = sqlite3.connect(self._index_path)
        con.execute(
            """CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                task_id TEXT,
                family TEXT,
                split TEXT,
                condition TEXT,
                seed INTEGER,
                model_name TEXT,
                covert_enacted INTEGER,
                primary_score REAL,
                parquet_file TEXT
            )"""
        )
        con.commit()
        con.close()

    def save(self, trajectory: Any) -> None:
        """Append a Trajectory; flush to parquet in batches of 100."""
        self._records.append(trajectory.model_dump())
        self._index_one(trajectory)
        if len(self._records) >= 100:
            self._flush()

    def flush(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        batch_id = len(list(self._dir.glob("batch_*.parquet")))
        path = self._dir / f"batch_{batch_id:04d}.parquet"
        df = pd.json_normalize(self._records)
        df.to_parquet(path, index=False)
        self._records = []

    def _index_one(self, trajectory: Any) -> None:
        import sqlite3
        con = sqlite3.connect(self._index_path)
        con.execute(
            "INSERT OR REPLACE INTO trajectories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                trajectory.trajectory_id,
                trajectory.meta.task_id,
                trajectory.meta.family,
                trajectory.meta.split,
                trajectory.meta.condition,
                trajectory.meta.seed,
                trajectory.meta.model_name,
                int(trajectory.ground_truth.covert_enacted),
                trajectory.primary_score,
                "",
            ),
        )
        con.commit()
        con.close()

    def query(self, split: str | None = None, condition: str | None = None) -> pd.DataFrame:
        import sqlite3
        con = sqlite3.connect(self._index_path)
        clauses = []
        params: list[Any] = []
        if split:
            clauses.append("split = ?")
            params.append(split)
        if condition:
            clauses.append("condition = ?")
            params.append(condition)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        df = pd.read_sql_query(f"SELECT * FROM trajectories {where}", con, params=params)
        con.close()
        return df

    def count(self) -> int:
        import sqlite3
        con = sqlite3.connect(self._index_path)
        n = con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        con.close()
        return int(n)
