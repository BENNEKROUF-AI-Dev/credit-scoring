"""Tests unitaires de la préparation des données et du modèle."""
import numpy as np
import pytest

from app.model import ScoringModel
from app.preprocessing import DERIVED, derive, noms_propres, underive


@pytest.fixture(scope="module")
def model():
    return ScoringModel(backend="onnx")


def test_noms_propres_like_notebook():
    assert noms_propres(["ORGANIZATION_TYPE_Trade: type 7", "a b", "a-b", "1X"]) == \
        ["ORGANIZATION_TYPE_Trade_type_7", "a_b", "a_b_1", "c_1X"]


def test_derive_recomputes_notebook1_flags():
    raw = {"DAYS_EMPLOYED": 365243, "EXT_SOURCE_1": None, "EXT_SOURCE_2": 0.5}
    cols = {"DAYS_EMPLOYED_ANOM", "EXT_SOURCE_1_MISSING", "EXT_SOURCE_2_MISSING", "EXT_SOURCE_3_MISSING",
            "OWN_CAR_AGE", "BUREAU_MISSING", "CC_MISSING"}
    r = derive(raw, cols)
    assert r["DAYS_EMPLOYED_ANOM"] == 1 and r["DAYS_EMPLOYED"] is None
    assert r["EXT_SOURCE_1_MISSING"] == 1 and r["EXT_SOURCE_2_MISSING"] == 0
    assert r["OWN_CAR_AGE"] == -1
    assert r["BUREAU_MISSING"] == 1 and r["BUREAU_COUNT"] == 0
    assert r["CC_MISSING"] == 1


def test_underive_is_inverse_of_derive(model, sample_records):
    cols = model.preprocessor.raw_columns
    for rec in sample_records[:50]:
        back = {k: v for k, v in underive(derive(rec, cols)).items() if v is not None}
        assert back == rec


def test_derived_fields_not_in_inputs(model):
    assert not set(DERIVED) & set(model.preprocessor.input_columns)


def test_fast_transform_matches_pandas(model, sample_records):
    """L'encodage NumPy optimisé == l'encodage pandas calqué sur le notebook 2."""
    prep = model.preprocessor
    fast = prep.transform(sample_records, dtype=np.float64)
    ref = prep.transform_pandas(sample_records).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(fast, ref, rtol=1e-12, equal_nan=True)


def test_unknown_category_encoded_as_all_zero(model, sample_records):
    prep = model.preprocessor
    if not prep.categorical:
        pytest.skip("pas de variable texte")
    col = next(iter(prep.categorical))
    rec = dict(sample_records[0], **{col: "__inconnue__"})
    X = prep.transform([rec], dtype=np.float64)[0]
    assert all(X[j] == 0 for j in prep._cat_idx[col].values())


def test_scores_reproduce_mlflow_model(model, sample_df, sample_records):
    p = model.predict_records(sample_records)
    assert np.max(np.abs(p - sample_df["expected_score"].to_numpy())) < 1e-4
    t = model.threshold
    exp = sample_df["expected_score"].to_numpy()
    far = np.abs(exp - t) > 1e-4
    assert ((p[far] >= t) == (exp[far] >= t)).all()


def test_onnx_matches_lightgbm(sample_records):
    lgb = ScoringModel(backend="lightgbm")
    if lgb.backend != "lightgbm":
        pytest.skip("model.txt absent (modèle non LightGBM)")
    onx = ScoringModel(backend="onnx")
    assert np.max(np.abs(lgb.predict_records(sample_records) - onx.predict_records(sample_records))) < 1e-4


def test_pandas_path_equals_fast_path(sample_records):
    fast = ScoringModel(backend="onnx", fast_features=True).predict_one(sample_records[0])[0]
    slow = ScoringModel(backend="onnx", fast_features=False).predict_one(sample_records[0])[0]
    assert fast == pytest.approx(slow, abs=1e-6)


def test_decision_threshold(model):
    assert model.decide(model.threshold) == "REFUSE"
    assert model.decide(model.threshold - 0.01) == "ACCORDE"
