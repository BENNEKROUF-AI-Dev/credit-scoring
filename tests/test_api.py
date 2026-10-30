"""Tests d'intégration de l'API (FastAPI TestClient)."""
import sqlite3

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model-info").json()
    assert 0 < body["threshold"] < 1
    assert "AMT_CREDIT" in body["features"]


def test_predict_valid(client, valid_client):
    r = client.post("/predict", json=valid_client)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"ACCORDE", "REFUSE"}
    assert (body["decision"] == "REFUSE") == (body["score"] >= body["threshold"])


def test_predict_is_deterministic(client, valid_client):
    s1 = client.post("/predict", json=valid_client).json()["score"]
    s2 = client.post("/predict", json=valid_client).json()["score"]
    assert s1 == s2


def test_optional_fields_can_be_missing(client, valid_client):
    for f in ("EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_GOODS_PRICE", "DAYS_EMPLOYED"):
        valid_client.pop(f)
    r = client.post("/predict", json=valid_client)
    assert r.status_code == 200


# ----- Cas critiques demandés : données manquantes, hors plage, mauvais types
@pytest.mark.parametrize("field", ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                                   "DAYS_BIRTH", "CODE_GENDER", "NAME_EDUCATION_TYPE"])
def test_missing_required_field(client, valid_client, field):
    valid_client.pop(field)
    assert client.post("/predict", json=valid_client).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("AMT_INCOME_TOTAL", 0),          # revenu nul
    ("AMT_INCOME_TOTAL", -1000),      # revenu négatif
    ("AMT_CREDIT", -5),
    ("DAYS_BIRTH", 1825),             # âge de -5 ans
    ("DAYS_BIRTH", -3000),            # mineur (8 ans)
    ("EXT_SOURCE_2", 1.5),            # score externe > 1
    ("CNT_CHILDREN", -1),
    ("DAYS_EMPLOYED", 365243),        # anomalie connue du jeu Home Credit
])
def test_out_of_range(client, valid_client, field, value):
    valid_client[field] = value
    assert client.post("/predict", json=valid_client).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("AMT_CREDIT", "beaucoup"),
    ("DAYS_BIRTH", "trente ans"),
    ("EXT_SOURCE_1", [0.5]),
    ("CODE_GENDER", "X"),
    ("NAME_EDUCATION_TYPE", "PhD"),
])
def test_wrong_type_or_category(client, valid_client, field, value):
    valid_client[field] = value
    assert client.post("/predict", json=valid_client).status_code == 422


def test_unknown_field_rejected(client, valid_client):
    valid_client["SK_ID_CURR"] = 100002
    assert client.post("/predict", json=valid_client).status_code == 422


def test_incoherent_annuity(client, valid_client):
    valid_client["AMT_ANNUITY"] = valid_client["AMT_CREDIT"] * 2
    assert client.post("/predict", json=valid_client).status_code == 422


def test_empty_body(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predictions_are_stored(client, valid_client):
    import app.config as cfg
    client.post("/predict", json=valid_client)
    client.post("/predict", json={"bad": 1})
    with sqlite3.connect(cfg.PREDICTIONS_DB) as conn:
        rows = conn.execute("SELECT status_code, score, latency_ms, inputs FROM predictions").fetchall()
    codes = {r[0] for r in rows}
    assert 200 in codes and 422 in codes
    ok = [r for r in rows if r[0] == 200][-1]
    assert ok[1] is not None and ok[2] > 0 and "AMT_CREDIT" in ok[3]


def test_internal_error_returns_500_without_leaking(client, valid_client, monkeypatch):
    import app.main as m
    def boom(_):
        raise RuntimeError("secret stack detail")
    monkeypatch.setattr(m.STATE["model"], "predict_one", boom)
    r = client.post("/predict", json=valid_client)
    assert r.status_code == 500
    assert "secret" not in r.text


def test_gradio_predict_function(client):
    import app.main as m
    from app.ui import make_predict_fn
    run = make_predict_fn(m.score_client)
    md, probs = run(0.5, 0.6, 0.5, 180000, 600000, 28000, 540000, 38, 7, 1, "F", "Higher education")
    assert "Crédit" in md and abs(sum(probs.values()) - 1) < 1e-9
    md, probs = run(0.5, 0.6, 0.5, -5, 600000, 28000, 540000, 38, 7, 1, "F", "Higher education")
    assert "invalides" in md and probs is None
