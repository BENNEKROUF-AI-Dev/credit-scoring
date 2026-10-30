"""Tests d'intégration de l'API (FastAPI TestClient)."""
import sqlite3

import pytest

MINIMAL = {"AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 600000, "DAYS_BIRTH": -14000, "CODE_GENDER": "F"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model-info").json()
    assert 0 < body["threshold"] < 1
    assert "AMT_CREDIT" in body["input_fields"]
    assert body["n_model_columns"] >= len(body["input_fields"]) - 10


def test_sample_clients_endpoints(client):
    ids = client.get("/clients").json()
    assert ids
    rec = client.get(f"/clients/{ids[0]}").json()
    assert "AMT_CREDIT" in rec
    assert client.get("/clients/1").status_code == 404


def test_predict_valid(client, valid_client):
    r = client.post("/predict", json=valid_client)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"ACCORDE", "REFUSE"}
    assert (body["decision"] == "REFUSE") == (body["score"] >= body["threshold"])


def test_api_reproduces_notebook_scores(client, sample_df, sample_records):
    """Non-régression : l'API redonne les scores du modèle MLflow d'origine."""
    for rec, expected in list(zip(sample_records, sample_df["expected_score"]))[:50]:
        r = client.post("/predict", json=rec)
        assert r.status_code == 200, r.text
        assert r.json()["score"] == pytest.approx(expected, abs=1e-4)


def test_minimal_file_without_history(client):
    """Client sans aucun historique (cœur de cible) : seuls les champs obligatoires."""
    r = client.post("/predict", json=MINIMAL)
    assert r.status_code == 200


def test_unemployed_sentinel_is_accepted(client, valid_client):
    valid_client["DAYS_EMPLOYED"] = 365243  # code « sans emploi » du jeu Home Credit
    assert client.post("/predict", json=valid_client).status_code == 200


# ----- Cas critiques demandés : données manquantes, hors plage, mauvais types
@pytest.mark.parametrize("field", ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "CODE_GENDER"])
def test_missing_required_field(client, field):
    body = dict(MINIMAL)
    body.pop(field)
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert field in r.text


@pytest.mark.parametrize("field,value", [
    ("AMT_INCOME_TOTAL", 0),          # revenu nul
    ("AMT_INCOME_TOTAL", -1000),      # revenu négatif
    ("AMT_CREDIT", -5),
    ("DAYS_BIRTH", 1825),             # âge de -5 ans
    ("DAYS_BIRTH", -3000),            # mineur (8 ans)
    ("EXT_SOURCE_2", 1.5),            # score externe > 1
    ("CNT_CHILDREN", -1),
    ("DAYS_EMPLOYED", 500),           # date d'embauche dans le futur
])
def test_out_of_range(client, valid_client, field, value):
    valid_client[field] = value
    assert client.post("/predict", json=valid_client).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("AMT_CREDIT", "beaucoup"),
    ("DAYS_BIRTH", "trente ans"),
    ("EXT_SOURCE_1", [0.5]),
    ("CODE_GENDER", "X"),             # catégorie inconnue
    ("NAME_CONTRACT_TYPE", "Leasing"),
])
def test_wrong_type_or_category(client, valid_client, field, value):
    valid_client[field] = value
    assert client.post("/predict", json=valid_client).status_code == 422


def test_unknown_field_rejected(client, valid_client):
    valid_client["SK_ID_CURR"] = 100002          # identifiant : refusé (RGPD)
    assert client.post("/predict", json=valid_client).status_code == 422


def test_derived_field_rejected(client, valid_client):
    valid_client["DAYS_EMPLOYED_ANOM"] = 1        # calculé par l'API, pas par l'appelant
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


def test_gradio_functions(client):
    import app.main as m
    from app.ui import make_load_fn, make_predict_fn
    clients = m.STATE["clients"]
    sk = sorted(clients)[0]
    values = make_load_fn(clients)(sk)
    md, probs = make_predict_fn(m.score_client, clients)(sk, *values)
    assert "Crédit" in md and abs(sum(probs.values()) - 1) < 1e-9
    values[3] = -5  # revenu négatif
    md, probs = make_predict_fn(m.score_client, clients)(sk, *values)
    assert "invalides" in md and probs is None
