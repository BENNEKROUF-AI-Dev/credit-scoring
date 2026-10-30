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

import numpy as np
import pandas as pd

from app.model import ScoringModel

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
N_WARMUP = 50


def load_records():
    df = pd.read_parquet(ROOT / "model" / "sample_clients.parquet")
    recs = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if pd.notna(v)}
            for r in df.drop(columns=["SK_ID_CURR", "expected_score"]).to_dict("records")]
    return recs, df["expected_score"].to_numpy()


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
    records, expected = load_records()
    onx = ScoringModel(backend="onnx")
    lgb = ScoringModel(backend="lightgbm")
    has_lgb = lgb.backend == "lightgbm"
    prep = onx.preprocessor
    native = lgb if has_lgb else onx
    native_name = "booster LightGBM natif" if has_lgb else "ONNX Runtime"

    configs = {
        f"v1 — préparation pandas + {native_name}":
            lambda r: native.predict_array(prep.transform_pandas([r]).to_numpy(np.float64)),
        "préparation pandas + ONNX Runtime":
            lambda r: onx.predict_array(prep.transform_pandas([r]).to_numpy(np.float64)),
        f"préparation NumPy + {native_name}":
            lambda r: native.predict_array(prep.transform([r])),
        "final — préparation NumPy + ONNX Runtime":
            lambda r: onx.predict_array(prep.transform([r])),
    }
    single = {name: timeit(fn, records) for name, fn in configs.items()}

    X1 = [prep.transform([r]) for r in records]
    breakdown = {
        "préparation pandas (get_dummies…)": timeit(lambda r: prep.transform_pandas([r]), records),
        "préparation NumPy": timeit(lambda r: prep.transform([r]), records),
        "modèle ONNX Runtime": timeit(onx.predict_array, X1),
    }
    if has_lgb:
        breakdown["modèle booster LightGBM natif"] = timeit(lgb.predict_array, X1)

    Xb = prep.transform(records * 10)
    batch = {}
    engines = {"ONNX Runtime (1 thread)": lambda: onx.predict_array(Xb)}
    if has_lgb:
        engines["booster LightGBM natif"] = lambda: lgb.predict_array(Xb)
    for name, fn in engines.items():
        fn()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        batch[name] = {"ms": (time.perf_counter() - t0) / 5 * 1000, "n": len(Xb)}

    p_final = onx.predict_records(records)
    t = onx.threshold
    quality = {
        "max_abs_score_diff_vs_mlflow": float(np.max(np.abs(p_final - expected))),
        "decision_agreement": float(((p_final >= t) == (expected >= t)).mean()),
        "n": len(records),
    }

    os.environ["ENABLE_PREDICTION_LOGGING"] = "0"
    os.environ["ENABLE_UI"] = "0"
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.main as main_mod
    importlib.reload(main_mod)
    from fastapi.testclient import TestClient
    api = {}
    with TestClient(main_mod.app) as c:
        versions = {
            "v1 (préparation pandas)": ScoringModel(backend="lightgbm" if has_lgb else "onnx",
                                                    fast_features=False),
            "final (NumPy + ONNX)": ScoringModel(backend="onnx", fast_features=True),
        }
        for name, model in versions.items():
            main_mod.STATE["model"] = model
            api[name] = timeit(lambda b: c.post("/predict", json=b), records[:500])

    load = {}
    for name, kw in {"ONNX": {"backend": "onnx"}, **({"LightGBM": {"backend": "lightgbm"}} if has_lgb else {})}.items():
        t0 = time.perf_counter()
        ScoringModel(**kw)
        load[name] = (time.perf_counter() - t0) * 1000
    sizes = {f: (ROOT / "model" / f).stat().st_size / 1e6
             for f in ("model.onnx", "model.txt") if (ROOT / "model" / f).exists()}

    profile(configs[next(iter(configs))], records, DOCS / "profiling_v1.txt")
    profile(configs["final — préparation NumPy + ONNX Runtime"], records, DOCS / "profiling_final.txt")

    env = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(), "algorithm": onx.meta.get("algorithm"),
           "n_model_columns": len(prep.model_columns)}
    results = {"env": env, "single": single, "breakdown": breakdown, "batch": batch,
               "quality": quality, "api": api, "load_ms": load, "size_mb": sizes}
    (DOCS / "benchmark_results.json").write_text(json.dumps(results, indent=2, default=float))

    base = single[next(iter(single))]["p50_ms"]
    lines = [f"Modèle : {env['algorithm']}, {env['n_model_columns']} colonnes", "",
             f"## Inférence unitaire (1 client, {len(records)} appels)", "",
             "| Configuration | p50 (ms) | p95 (ms) | p99 (ms) | Gain p50 |", "|---|---|---|---|---|"]
    for n, r in single.items():
        lines.append(f"| {n} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | ×{base / r['p50_ms']:.1f} |")
    lines += ["", "## Décomposition du temps (p50, ms)", "", "| Étape | p50 (ms) |", "|---|---|"]
    lines += [f"| {n} | {r['p50_ms']:.4f} |" for n, r in breakdown.items()]
    lines += ["", "## Latence API de bout en bout (500 requêtes HTTP)", "",
              "| Version | p50 (ms) | p95 (ms) | p99 (ms) |", "|---|---|---|---|"]
    lines += [f"| {n} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} |" for n, r in api.items()]
    lines += ["", f"## Scoring en masse ({len(Xb)} clients déjà encodés)", "", "| Moteur | ms |", "|---|---|"]
    lines += [f"| {n} | {r['ms']:.1f} |" for n, r in batch.items()]
    lines += ["", "## Non-régression (vs modèle MLflow d'origine)", "",
              f"- Écart max de score : {quality['max_abs_score_diff_vs_mlflow']:.2e}",
              f"- Décisions identiques : {quality['decision_agreement']:.2%} ({quality['n']} dossiers)",
              "", "## Chargement et taille", ""]
    lines += [f"- {n} : chargement {v:.0f} ms" for n, v in load.items()]
    lines += [f"- {n} : {v:.2f} Mo" for n, v in sizes.items()]
    lines += ["", f"Environnement : Python {env['python']}, {env['cpu_count']} vCPU ({env['cpu']})"]
    (DOCS / "benchmark_results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
