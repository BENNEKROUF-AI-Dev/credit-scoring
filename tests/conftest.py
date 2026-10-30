import os
import sys
from pathlib import Path

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


@pytest.fixture
def valid_client() -> dict:
    return {
        "EXT_SOURCE_1": 0.45, "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.51,
        "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 600000, "AMT_ANNUITY": 28000,
        "AMT_GOODS_PRICE": 540000, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500,
        "CNT_CHILDREN": 1, "CODE_GENDER": "F", "NAME_EDUCATION_TYPE": "Higher education",
    }


@pytest.fixture(scope="session")
def client():
    # import tardif : les variables d'environnement ci-dessus doivent être posées avant
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.main
    importlib.reload(app.main)
    from fastapi.testclient import TestClient
    with TestClient(app.main.app) as c:
        yield c
