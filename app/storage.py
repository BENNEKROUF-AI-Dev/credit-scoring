"""Stockage des données de production.

Chaque appel à l'API est écrit à deux endroits :
- des logs structurés JSON (une ligne par appel), collectables par Fluentd ou Logstash ;
- une table `predictions`, requêtable en SQL pour le dashboard et l'analyse du drift.

Deux bases possibles pour la table, choisies par la variable d'environnement DATABASE_URL :
- DATABASE_URL renseignée (ex. NeonDB) -> PostgreSQL. La couche qui STOCKE est séparée de
  la couche qui CALCULE (Render) : les données survivent aux redéploiements et se consultent
  indépendamment de l'API. Les écritures partent dans un fil d'exécution en arrière-plan,
  pour ne pas ajouter la latence réseau de la base au temps de réponse de l'API.
- sinon -> SQLite, un simple fichier local (développement, tests).

Conformité RGPD : aucune donnée directement identifiante (nom, ID client) n'est stockée,
seulement les variables du modèle.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

COLUMNS = ["request_id", "timestamp", "endpoint", "status_code", "inputs", "score",
           "decision", "threshold", "model_version", "backend", "inference_ms",
           "latency_ms", "error"]

SQLITE_SCHEMA = """
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

POSTGRES_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS predictions (
        id            BIGSERIAL PRIMARY KEY,
        request_id    TEXT NOT NULL,
        timestamp     TEXT NOT NULL,
        endpoint      TEXT NOT NULL,
        status_code   INTEGER NOT NULL,
        inputs        TEXT,
        score         DOUBLE PRECISION,
        decision      TEXT,
        threshold     DOUBLE PRECISION,
        model_version TEXT,
        backend       TEXT,
        inference_ms  DOUBLE PRECISION,
        latency_ms    DOUBLE PRECISION,
        error         TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp)",
]


def _row_values(record: dict) -> list:
    row = dict(record)
    row["inputs"] = json.dumps(row.get("inputs")) if row.get("inputs") is not None else None
    return [row.get(c) for c in COLUMNS]


class _SQLiteBackend:
    name = "sqlite"

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SQLITE_SCHEMA)

    def write(self, record: dict) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO predictions ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                _row_values(record),
            )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _PostgresBackend:
    """PostgreSQL (ex. NeonDB), avec écriture en arrière-plan et reconnexion automatique."""
    name = "postgres"

    def __init__(self, url: str):
        import psycopg

        self._psycopg = psycopg
        self.url = url
        self._conn = None
        with self._connect() as conn:  # crée la table au démarrage (échoue tôt si l'URL est fausse)
            for stmt in POSTGRES_SCHEMA:
                conn.execute(stmt)
        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._worker = threading.Thread(target=self._run, name="predictions-writer", daemon=True)
        self._worker.start()

    def _connect(self):
        return self._psycopg.connect(self.url, autocommit=True, connect_timeout=10)

    def _insert(self, record: dict) -> None:
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        self._conn.execute(
            f"INSERT INTO predictions ({','.join(COLUMNS)}) VALUES ({','.join(['%s'] * len(COLUMNS))})",
            _row_values(record),
        )

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                try:
                    self._insert(record)
                except self._psycopg.OperationalError:
                    # connexion coupée (ex. base Neon mise en veille) : on reconnecte et on réessaie une fois
                    self._conn = None
                    self._insert(record)
            except Exception:
                logger.exception("Échec d'écriture d'une prédiction dans PostgreSQL")
            finally:
                self._queue.task_done()

    def write(self, record: dict) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            logger.error("File d'écriture PostgreSQL pleine : prédiction non stockée")

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self.flush()
        self._queue.put(None)
        self._worker.join(timeout=5)
        if self._conn is not None:
            self._conn.close()


class PredictionStore:
    def __init__(self, db_path: Path, log_path: Path, database_url: str | None = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._backend = _PostgresBackend(database_url) if database_url else _SQLiteBackend(db_path)
        logger.info("Stockage des prédictions : %s", self._backend.name)

        self._json_logger = logging.getLogger("predictions")
        self._json_logger.setLevel(logging.INFO)
        self._json_logger.propagate = False
        if not any(getattr(h, "baseFilename", None) == str(self.log_path.resolve())
                   for h in self._json_logger.handlers):
            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._json_logger.addHandler(handler)

    @property
    def backend(self) -> str:
        return self._backend.name

    def log(self, **record) -> None:
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._json_logger.info(json.dumps(record, default=str))
        self._backend.write(record)

    def flush(self) -> None:
        """Attend que toutes les écritures en attente soient faites (utile pour les tests)."""
        self._backend.flush()

    def close(self) -> None:
        self._backend.close()
