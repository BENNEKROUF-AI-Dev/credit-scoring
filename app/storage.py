"""Stockage des données de production.

Double écriture :
- logs structurés JSON (une ligne par appel) -> collectables par Fluentd/Logstash
- base SQLite `predictions` -> requêtable pour le dashboard et l'analyse de drift

En production réelle, SQLite se remplace par PostgreSQL en changeant
uniquement cette classe (même schéma de table).
Conformité RGPD : aucune donnée directement identifiante (nom, ID client)
n'est stockée, seulement les variables du modèle.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    endpoint      TEXT NOT NULL,
    status_code   INTEGER NOT NULL,
    inputs        TEXT,
    score         REAL,
    decision      TEXT,
    threshold     REAL,
    model_version TEXT,
    backend       TEXT,
    inference_ms  REAL,
    latency_ms    REAL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp);
"""


class PredictionStore:
    def __init__(self, db_path: Path, log_path: Path):
        self.db_path = Path(db_path)
        self.log_path = Path(log_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

        self._json_logger = logging.getLogger("predictions")
        self._json_logger.setLevel(logging.INFO)
        self._json_logger.propagate = False
        if not any(getattr(h, "baseFilename", None) == str(self.log_path.resolve())
                   for h in self._json_logger.handlers):
            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._json_logger.addHandler(handler)

    def log(self, **record) -> None:
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._json_logger.info(json.dumps(record, default=str))
        row = dict(record)
        row["inputs"] = json.dumps(row.get("inputs")) if row.get("inputs") is not None else None
        cols = ["request_id", "timestamp", "endpoint", "status_code", "inputs", "score",
                "decision", "threshold", "model_version", "backend", "inference_ms",
                "latency_ms", "error"]
        values = [row.get(c) for c in cols]
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO predictions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                values,
            )
