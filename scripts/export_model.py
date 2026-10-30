"""Exporte le modèle du projet précédent (notebooks 1 et 2) vers l'API.

À lancer UNE fois, avec le Python de votre projet précédent (celui où mlflow et
lightgbm sont installés), depuis la racine de ce dépôt :

    pip install onnxmltools skl2onnx onnxruntime pyarrow
    python -m scripts.export_model --features "CHEMIN/app_features.parquet" \
           --tracking-uri "file:CHEMIN/mlruns"

Options :
    --model-uri  (défaut models:/scoring_credit/latest)
    --seuil      force le seuil métier (sinon lu dans la métrique 'seuil' du run MLflow)
    --retrain    pas de modèle MLflow ? ré-entraîne un LightGBM avec les réglages du notebook 2

Produit dans model/ :
    model.onnx               modèle converti (servi par l'API, indépendant des versions)
    model.txt                booster LightGBM natif (si le modèle est un LightGBM)
    model_meta.json          seuil, colonnes, catégories, correspondance de l'encodage
    sample_clients.parquet   1000 dossiers du jeu de test (démo, tests, simulation)
    reference_data.parquet   référence pour le data drift (variables surveillées + score)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.preprocessing import DERIVED, Preprocessor, noms_propres, underive

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # anciens dossiers mlruns/

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
SEUILS = np.arange(0.02, 0.99, 0.02)      # même grille que le notebook 2
KEY_FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED",
                "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "CODE_GENDER"]


def cout(vrai, predit) -> int:
    fn = int(((predit == 0) & (vrai == 1)).sum())
    fp = int(((predit == 1) & (vrai == 0)).sum())
    return 10 * fn + fp


def meilleur_seuil(vrai, proba):
    couts = [cout(vrai, (proba >= s).astype(int)) for s in SEUILS]
    i = int(np.argmin(couts))
    return round(float(SEUILS[i]), 2), couts[i]


def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)


def strip_samplers(model):
    """Un pipeline imblearn (SMOTE) ne sert qu'à l'entraînement : on retire l'étape."""
    if hasattr(model, "steps"):
        from sklearn.pipeline import Pipeline
        steps = [(n, s) for n, s in model.steps if not hasattr(s, "fit_resample")]
        if len(steps) != len(model.steps):
            print("  étape de ré-échantillonnage (SMOTE) retirée pour l'inférence")
            return Pipeline(steps)
    return model


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def model_columns_of(model) -> list[str]:
    for obj in (model, final_estimator(model)):
        for attr in ("feature_names_in_", "feature_name_"):
            cols = getattr(obj, attr, None)
            if cols is not None:
                return [str(c) for c in cols]
    raise ValueError("Impossible de lire les noms de colonnes du modèle.")


def load_from_mlflow(args):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    print(f"Chargement de {args.model_uri} depuis {mlflow.get_tracking_uri()}")
    model = mlflow.sklearn.load_model(args.model_uri)
    info = {"model_uri": args.model_uri}
    try:  # retrouver le run d'origine pour lire seuil / AUC / coût
        client = MlflowClient()
        name, ver = args.model_uri.replace("models:/", "").split("/", 1)
        if ver == "latest":
            mv = max(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
        else:
            mv = client.get_model_version(name, ver)
        run = client.get_run(mv.run_id)
        info.update({"registry_version": str(mv.version), "run_id": mv.run_id,
                     "run_name": run.data.tags.get("mlflow.runName"),
                     "metrics": run.data.metrics})
    except Exception as exc:  # noqa: BLE001
        print(f"  (métriques du run non lues : {exc})")
    return model, info


def retrain(X_train, y_train):
    from lightgbm import LGBMClassifier
    print("Ré-entraînement LightGBM (réglages du notebook 2)…")
    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                       is_unbalance=True, random_state=42, verbose=-1)
    return m.fit(X_train, y_train), {"model_uri": "retrained", "metrics": {}}


def to_onnx(model, n_features: int):
    est = final_estimator(model)
    if type(est).__name__ == "LGBMClassifier" and not hasattr(model, "steps"):
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        return convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                                zipmap=False, target_opset=15)
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    try:  # permet de convertir un pipeline qui contient un LightGBM
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        update_registered_converter(LGBMClassifier, "LightGbmLGBMClassifier",
                                    calculate_linear_classifier_output_shapes, convert_lightgbm,
                                    options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    except Exception:  # noqa: BLE001
        pass
    return convert_sklearn(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                           options={id(est): {"zipmap": False}}, target_opset={"": 15, "ai.onnx.ml": 3})


def importance_by_raw(model, prep: Preprocessor) -> list[str]:
    est = final_estimator(model)
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "coef_"):
        imp = np.abs(np.ravel(est.coef_))
    if imp is None:
        return [c for c in KEY_FEATURES if c in prep.raw_columns]
    by_col = dict(zip(prep.model_columns, np.asarray(imp, dtype=float)))
    inverse = {v: k for k, v in prep.rename_map.items()}
    scores: dict[str, float] = {}
    for mc, val in by_col.items():
        orig = inverse.get(mc, mc)
        raw = next((c for c in prep.categorical if orig.startswith(f"{c}_")), orig)
        if raw in DERIVED:
            continue
        scores[raw] = scores.get(raw, 0.0) + val
    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def main():
    global MODEL_DIR
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="app_features.parquet (notebook 1)")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--model-uri", default="models:/scoring_credit/latest")
    p.add_argument("--seuil", type=float, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--version", default=None)
    p.add_argument("--n-monitored", type=int, default=15)
    p.add_argument("--n-sample", type=int, default=1000, help="nb de dossiers d'exemple")
    p.add_argument("--output", default=None, help="dossier de sortie (défaut model/)")
    args = p.parse_args()
    if args.output:
        MODEL_DIR = Path(args.output)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    app = load_features(args.features)
    print(f"Données : {app.shape}")
    raw = app.drop(columns=["TARGET", "SK_ID_CURR"])
    y = app["TARGET"].to_numpy()

    # --- 1. Colonnes et catégories (comme pd.get_dummies du notebook 2)
    cat_cols = list(raw.select_dtypes(include=["object", "string", "category"]).columns)
    num_cols = [c for c in raw.columns if c not in cat_cols]
    categories = {c: sorted(str(v) for v in raw[c].dropna().unique()) for c in cat_cols}

    # Reproduit get_dummies + noms_propres pour obtenir la correspondance des noms
    n_demo = max([len(v) for v in categories.values()] + [1])
    demo = pd.DataFrame({c: np.zeros(n_demo) for c in num_cols})
    for c in cat_cols:
        vals = categories[c] or [None]
        demo[c] = [vals[i % len(vals)] for i in range(n_demo)]
    demo = demo[list(raw.columns)]
    before = list(pd.get_dummies(demo, dtype=int).columns)
    rename_map = dict(zip(before, noms_propres(before)))

    # --- 2. Le modèle
    X_all = pd.get_dummies(raw, dtype=int)
    X_all.columns = noms_propres(X_all.columns)
    idx_train, idx_test = train_test_split(np.arange(len(app)), test_size=0.2, stratify=y,
                                           random_state=42)  # même découpage que le notebook 2
    if args.retrain:
        model, info = retrain(X_all.iloc[idx_train], y[idx_train])
    else:
        model, info = load_from_mlflow(args)
    model = strip_samplers(model)
    model_columns = model_columns_of(model)

    missing = set(model_columns) - set(rename_map.values())
    if missing:
        raise SystemExit(f"Colonnes du modèle introuvables dans les données : {sorted(missing)[:10]}…\n"
                         "Vérifiez que --features est bien le fichier utilisé pour entraîner ce modèle.")

    meta = {"model_columns": model_columns, "raw_numeric": num_cols,
            "raw_categorical": categories, "rename_map": rename_map}
    prep = Preprocessor(meta)

    # --- 3. Vérifier que l'encodage de l'API == celui du notebook
    X_test_nb = X_all.iloc[idx_test].reindex(columns=model_columns)
    proba_test = model.predict_proba(X_test_nb)[:, 1]
    sample_idx = idx_test[:2000]  # contrôle sur 2000 dossiers
    records = [underive(r) for r in raw.iloc[sample_idx].to_dict("records")]
    X_api = prep.transform(records, dtype=np.float64)
    diff = float(np.nanmax(np.abs(model.predict_proba(pd.DataFrame(X_api, columns=model_columns))[:, 1]
                                  - proba_test[:2000])))
    print(f"Contrôle encodage API vs notebook : écart max de score = {diff:.2e}")
    assert diff < 1e-6, "L'encodage de l'API ne reproduit pas celui du notebook."

    # --- 4. Seuil et métriques
    auc = float(roc_auc_score(y[idx_test], proba_test))
    seuil_calc, cout_calc = meilleur_seuil(y[idx_test], proba_test)
    seuil = args.seuil or info.get("metrics", {}).get("seuil") or seuil_calc
    print(f"AUC test = {auc:.4f} | seuil MLflow = {info.get('metrics', {}).get('seuil')} "
          f"| seuil recalculé = {seuil_calc} | seuil retenu = {seuil}")

    # --- 5. ONNX + vérification de non-régression
    MODEL_DIR.mkdir(exist_ok=True)
    for old in ("model.joblib", "model.onnx", "model.txt"):
        (MODEL_DIR / old).unlink(missing_ok=True)
    onx = to_onnx(model, len(model_columns))
    (MODEL_DIR / "model.onnx").write_bytes(onx.SerializeToString())
    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    p_onnx = np.asarray(sess.run(None, {sess.get_inputs()[0].name: X_api.astype(np.float32)})[1])[:, 1]
    d_onnx = float(np.max(np.abs(p_onnx - proba_test[:2000])))
    same = float(((p_onnx >= seuil) == (proba_test[:2000] >= seuil)).mean())
    print(f"ONNX : écart max {d_onnx:.2e}, décisions identiques {same:.2%}")
    assert d_onnx < 1e-4, "Régression après conversion ONNX"
    est = final_estimator(model)
    if hasattr(est, "booster_") and not hasattr(model, "steps"):
        est.booster_.save_model(str(MODEL_DIR / "model.txt"))

    # --- 6. Exemples et référence pour le drift
    monitored = importance_by_raw(model, prep)[: args.n_monitored]
    ns = min(args.n_sample, len(idx_test))
    sample = [underive(r) for r in raw.iloc[idx_test[:ns]].to_dict("records")]
    sample_df = pd.DataFrame(sample)
    sample_df.insert(0, "SK_ID_CURR", app["SK_ID_CURR"].iloc[idx_test[:ns]].to_numpy())
    sample_df["expected_score"] = proba_test[:ns]
    sample_df.to_parquet(MODEL_DIR / "sample_clients.parquet", index=False)

    ref_idx = np.random.default_rng(0).choice(idx_train, min(5000, len(idx_train)), replace=False)
    ref = pd.DataFrame([underive(r) for r in raw.iloc[ref_idx].to_dict("records")])
    ref = ref.reindex(columns=monitored)
    ref["score"] = model.predict_proba(X_all.iloc[ref_idx].reindex(columns=model_columns))[:, 1]
    ref.to_parquet(MODEL_DIR / "reference_data.parquet", index=False)

    version = args.version or (f"v{info['registry_version']}" if info.get("registry_version")
                               else datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M"))
    meta.update({
        "version": version,
        "threshold": float(seuil),
        "source": info,
        "algorithm": type(est).__name__,
        "monitored_features": monitored,
        "metrics": {"auc_test": round(auc, 4), "cout_test": int(cout(y[idx_test], (proba_test >= seuil).astype(int))),
                    "cout_fn": 10, "cout_fp": 1, "n_test": int(len(idx_test))},
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"\n✅ Export terminé ({len(model_columns)} colonnes modèle, {len(prep.input_columns)} champs d'entrée).")
    print(f"   Variables surveillées : {', '.join(monitored)}")
    print("   Lancez maintenant : python -m pytest")


if __name__ == "__main__":
    main()
