"""Tests du stockage des prédictions : SQLite (par défaut) et PostgreSQL (NeonDB).

Le test PostgreSQL tourne si TEST_DATABASE_URL est définie (c'est le cas dans la CI,
qui démarre un conteneur PostgreSQL) ; sinon il est ignoré.
"""
import json
import os

import pytest

from app.storage import PredictionStore
from monitoring.analysis import expand_inputs, load_logs

RECORD = {"request_id": "r1", "endpoint": "/predict", "status_code": 200,
          "inputs": {"AMT_CREDIT": 600000.0, "EXT_SOURCE_3": 0.51}, "score": 0.42,
          "decision": "ACCORDE", "threshold": 0.52, "model_version": "v32", "backend": "onnx",
          "inference_ms": 0.4, "latency_ms": 0.8}


def test_sqlite_is_the_default(tmp_path):
    store = PredictionStore(tmp_path / "p.db", tmp_path / "p.jsonl")
    assert store.backend == "sqlite"
    store.log(**RECORD)
    store.log(**{**RECORD, "request_id": "r2", "status_code": 422, "score": None,
                 "decision": None, "inputs": None, "error": "AMT_CREDIT invalide"})
    logs = load_logs(tmp_path / "p.db")
    assert list(logs["status_code"]) == [200, 422]
    assert json.loads(logs["inputs"].iloc[0])["AMT_CREDIT"] == 600000.0
    assert (tmp_path / "p.jsonl").read_text().count("\n") == 2


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL non définie")
def test_postgres_backend_roundtrip(tmp_path):
    import psycopg

    url = os.environ["TEST_DATABASE_URL"]
    store = PredictionStore(tmp_path / "unused.db", tmp_path / "p.jsonl", database_url=url)
    assert store.backend == "postgres"
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("TRUNCATE predictions")
    for i in range(3):
        store.log(**{**RECORD, "request_id": f"r{i}", "score": 0.1 * (i + 1)})
    store.flush()  # les écritures PostgreSQL partent en arrière-plan
    logs = load_logs(url)
    assert len(logs) == 3
    assert logs["score"].round(2).tolist() == [0.1, 0.2, 0.3]
    current = expand_inputs(logs)
    assert current["EXT_SOURCE_3"].tolist() == [0.51, 0.51, 0.51]
    store.close()
