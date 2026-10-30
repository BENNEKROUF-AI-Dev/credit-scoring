import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _isolated_env(tmp_path_factory):
    """Les tests écrivent dans un dossier temporaire, jamais dans data/."""
    tmp = tmp_path_factory.mktemp("data")
    os.environ["PREDICTIONS_DB"] = str(tmp / "test.db")
    os.environ["PREDICTIONS_LOG"] = str(tmp / "test.jsonl")
    os.environ.setdefault("ENABLE_UI", "0")
    yield


def _clean(row: dict) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if pd.notna(v)}


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Dossiers du jeu de test du projet précédent + score attendu (calculé à l'export)."""
    model_dir = Path(os.getenv("MODEL_DIR", ROOT / "model"))
    return pd.read_parquet(model_dir / "sample_clients.parquet")


@pytest.fixture(scope="session")
def sample_records(sample_df) -> list[dict]:
    return [_clean(r) for r in sample_df.drop(columns=["SK_ID_CURR", "expected_score"]).to_dict("records")]


@pytest.fixture
def valid_client(sample_records) -> dict:
    """Un dossier complet et valide (copie modifiable)."""
    rec = next(r for r in sample_records
               if r.get("AMT_ANNUITY") and r.get("DAYS_EMPLOYED", 0) != 365243)
    return dict(rec)


@pytest.fixture(scope="session")
def client():
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.main
    importlib.reload(app.main)
    from fastapi.testclient import TestClient
    with TestClient(app.main.app) as c:
        yield c
