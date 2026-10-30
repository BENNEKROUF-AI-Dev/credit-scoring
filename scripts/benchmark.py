"""Benchmark et profiling des stratégies d'optimisation de l'inférence.

    python -m scripts.benchmark

Écrit docs/benchmark_results.json, docs/benchmark_results.md et
docs/profiling_v1.txt / docs/profiling_final.txt (sorties cProfile).
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import platform
import pstats
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.features import MODEL_FEATURES, build_features, build_features_fast
from app.model import ScoringModel
from scripts.synthetic import generate_clients

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
N_SINGLE, N_WARMUP = 1000, 50


def clean(records):
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            for r in records]


def timeit(fn, inputs):
    for x in inputs[:N_WARMUP]:
        fn(x)
    t = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"mean_ms": t.mean(), "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95),
            "p99_ms": np.percentile(t, 99)}


def profile(fn, inputs, path):
    pr = cProfile.Profile()
    pr.enable()
    for x in inputs[:300]:
        fn(x)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    path.write_text(s.getvalue())


def main():
    data = generate_clients(N_SINGLE, seed=99)
    records = clean(data.to_dict("records"))
    sk_model = joblib.load(ROOT / "model" / "model.joblib")
    lgb = ScoringModel(backend="lightgbm")
    onx = ScoringModel(backend="onnx")

    configs = {
        "v1 — pandas + sklearn predict_proba": lambda r: sk_model.predict_proba(
            build_features(pd.DataFrame([r])))[0, 1],
        "pandas + booster LightGBM natif": lambda r: lgb._predict_array(
            build_features(pd.DataFrame([r])).to_numpy()),
        "pandas + ONNX Runtime": lambda r: onx._predict_array(
            build_features(pd.DataFrame([r])).to_numpy()),
        "NumPy + booster LightGBM natif": lambda r: lgb._predict_array(build_features_fast([r])),
        "final — NumPy + ONNX Runtime": lambda r: onx._predict_array(build_features_fast([r])),
    }
    single = {name: timeit(fn, records) for name, fn in configs.items()}

    # Décomposition : feature engineering vs modèle seul
    X1 = [build_features_fast([r]) for r in records]
    breakdown = {
        "features pandas": timeit(lambda r: build_features(pd.DataFrame([r])), records),
        "features NumPy": timeit(lambda r: build_features_fast([r]), records),
        "modèle sklearn predict_proba (DataFrame 1 ligne)": timeit(
            lambda x: sk_model.predict_proba(pd.DataFrame(x, columns=MODEL_FEATURES)), X1),
        "modèle booster natif": timeit(lgb._predict_array, X1),
        "modèle ONNX Runtime": timeit(onx._predict_array, X1),
    }

    # Batch (scoring de masse, 10 000 clients)
    big = generate_clients(10000, seed=5)
    Xb = build_features(big).to_numpy()
    batch = {}
    for name, fn in {"sklearn": lambda: sk_model.predict_proba(pd.DataFrame(Xb, columns=MODEL_FEATURES)),
                     "booster natif": lambda: lgb._predict_array(Xb),
                     "ONNX Runtime (1 thread)": lambda: onx._predict_array(Xb)}.items():
        fn()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        batch[name] = {"ms_per_10k": (time.perf_counter() - t0) / 5 * 1000}

    # Non-régression : précision identique ?
    labelled = generate_clients(20000, seed=11, with_target=True)
    p_ref = sk_model.predict_proba(build_features(labelled))[:, 1]
    p_onx = onx._predict_array(build_features_fast(clean(labelled.drop(columns="TARGET").to_dict("records"))))
    t = onx.threshold
    quality = {
        "auc_sklearn": roc_auc_score(labelled["TARGET"], p_ref),
        "auc_final": roc_auc_score(labelled["TARGET"], p_onx),
        "max_abs_score_diff": float(np.max(np.abs(p_ref - p_onx))),
        "decision_agreement": float(((p_ref >= t) == (p_onx >= t)).mean()),
    }

    # Latence de bout en bout via l'API (FastAPI TestClient)
    os.environ["ENABLE_PREDICTION_LOGGING"] = "0"
    os.environ["ENABLE_UI"] = "0"
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    payloads = [{k: v for k, v in r.items() if v is not None} for r in records[:500]]
    api = {}
    with TestClient(main_mod.app) as c:
        for name, model in {"v1 (pandas + LightGBM)": ScoringModel(backend="lightgbm", fast_features=False),
                            "final (NumPy + ONNX)": ScoringModel(backend="onnx", fast_features=True)}.items():
            main_mod.STATE["model"] = model
            api[name] = timeit(lambda b: c.post("/predict", json=b), payloads)

    # Temps de chargement et taille
    load = {}
    for name, kw in {"LightGBM (joblib)": {"backend": "lightgbm"}, "ONNX": {"backend": "onnx"}}.items():
        t0 = time.perf_counter()
        ScoringModel(**kw)
        load[name] = (time.perf_counter() - t0) * 1000
    sizes = {f: (ROOT / "model" / f).stat().st_size / 1e6 for f in ("model.joblib", "model.onnx")}

    profile(configs["v1 — pandas + sklearn predict_proba"], records, DOCS / "profiling_v1.txt")
    profile(configs["final — NumPy + ONNX Runtime"], records, DOCS / "profiling_final.txt")

    env = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count()}
    results = {"env": env, "single": single, "breakdown": breakdown, "batch": batch,
               "quality": quality, "api": api, "load_ms": load, "size_mb": sizes}
    (DOCS / "benchmark_results.json").write_text(json.dumps(results, indent=2, default=float))

    base = single["v1 — pandas + sklearn predict_proba"]["p50_ms"]
    lines = ["## Inférence unitaire (1 client, 1000 appels)", "",
             "| Configuration | p50 (ms) | p95 (ms) | p99 (ms) | Gain p50 |", "|---|---|---|---|---|"]
    for n, r in single.items():
        lines.append(f"| {n} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | ×{base / r['p50_ms']:.1f} |")
    lines += ["", "## Décomposition du temps (p50, ms)", "", "| Étape | p50 (ms) |", "|---|---|"]
    lines += [f"| {n} | {r['p50_ms']:.4f} |" for n, r in breakdown.items()]
    lines += ["", "## Latence API de bout en bout (500 requêtes HTTP)", "",
              "| Version | p50 (ms) | p95 (ms) | p99 (ms) |", "|---|---|---|---|"]
    lines += [f"| {n} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} |" for n, r in api.items()]
    lines += ["", "## Scoring en masse (10 000 clients)", "", "| Moteur | ms |", "|---|---|"]
    lines += [f"| {n} | {r['ms_per_10k']:.1f} |" for n, r in batch.items()]
    lines += ["", "## Non-régression", "",
              f"- AUC sklearn : {quality['auc_sklearn']:.4f} — AUC version finale : {quality['auc_final']:.4f}",
              f"- Écart max de score : {quality['max_abs_score_diff']:.2e}",
              f"- Décisions identiques : {quality['decision_agreement']:.2%}",
              "", "## Chargement et taille", ""]
    lines += [f"- {n} : chargement {v:.0f} ms" for n, v in load.items()]
    lines += [f"- {n} : {v:.2f} Mo" for n, v in sizes.items()]
    lines += ["", f"Environnement : Python {env['python']}, {env['cpu_count']} vCPU ({env['cpu']})"]
    (DOCS / "benchmark_results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
