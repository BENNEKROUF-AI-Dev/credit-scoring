"""Tests unitaires du modèle et du feature engineering."""
import numpy as np
import pandas as pd
import pytest

from app.features import MODEL_FEATURES, build_features, build_features_fast
from app.model import ScoringModel
from scripts.synthetic import generate_clients


@pytest.fixture(scope="module")
def sample():
    return generate_clients(500, seed=1)


def test_build_features_shape_and_order(sample):
    X = build_features(sample)
    assert list(X.columns) == MODEL_FEATURES
    assert len(X) == len(sample)
    assert not np.isinf(X.to_numpy()).any()


def test_build_features_ratios(valid_client):
    X = build_features(pd.DataFrame([valid_client]))
    assert X["CREDIT_INCOME_PERCENT"].iloc[0] == pytest.approx(600000 / 180000)
    assert X["GENDER_M"].iloc[0] == 0.0
    assert X["EDUCATION_ORD"].iloc[0] == 3.0


def test_fast_features_match_pandas(sample):
    """L'optimisation NumPy doit reproduire exactement le feature engineering pandas."""
    records = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
               for r in sample.to_dict("records")]
    fast = build_features_fast(records, dtype=np.float64)
    ref = build_features(sample).to_numpy(np.float64)
    np.testing.assert_allclose(fast, ref, rtol=1e-12, equal_nan=True)


def test_predict_one_fast_equals_slow(valid_client):
    fast = ScoringModel(backend="lightgbm", fast_features=True).predict_one(valid_client)[0]
    slow = ScoringModel(backend="lightgbm", fast_features=False).predict_one(valid_client)[0]
    assert fast == pytest.approx(slow, abs=1e-9)


@pytest.mark.parametrize("backend", ["lightgbm", "onnx"])
def test_scores_are_probabilities(backend, sample):
    m = ScoringModel(backend=backend)
    p = m.predict_proba(sample)
    assert p.shape == (len(sample),)
    assert ((p >= 0) & (p <= 1)).all()


def test_onnx_matches_lightgbm(sample):
    """Non-régression de l'optimisation : ONNX doit donner les mêmes scores."""
    ref = ScoringModel(backend="lightgbm").predict_proba(sample)
    onx = ScoringModel(backend="onnx")
    if onx.backend != "onnx":
        pytest.skip("model.onnx absent")
    p = onx.predict_proba(sample)
    assert np.max(np.abs(ref - p)) < 1e-4
    # aucune décision ne doit changer (hors cas à 1e-4 du seuil)
    t = onx.threshold
    far = np.abs(ref - t) > 1e-4
    assert ((ref[far] >= t) == (p[far] >= t)).all()


def test_decision_threshold():
    m = ScoringModel(backend="lightgbm")
    assert m.decide(m.threshold) == "REFUSE"
    assert m.decide(m.threshold - 0.01) == "ACCORDE"


def test_model_discriminates():
    """Garde-fou qualité : l'AUC sur données fraîches doit rester correcte."""
    from sklearn.metrics import roc_auc_score
    df = generate_clients(4000, seed=7, with_target=True)
    p = ScoringModel(backend="lightgbm").predict_proba(df)
    assert roc_auc_score(df["TARGET"], p) > 0.70
