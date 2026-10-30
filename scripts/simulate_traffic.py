"""Simule du trafic de production sur l'API pour alimenter le monitoring.

Scénario par défaut (3 phases) :
  1. trafic « normal » (même distribution que l'entraînement)
  2. dérive progressive de la clientèle (plus jeune, plus endettée, EXT_SOURCE_3 plus bas)
  3. quelques requêtes invalides (taux d'erreur)

    # contre une API qui tourne (local, Docker ou Hugging Face) :
    python -m scripts.simulate_traffic --url http://localhost:7860 --n 1500
    # sans serveur (appel in-process, écrit dans data/predictions.db) :
    python -m scripts.simulate_traffic --in-process --n 1500
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from scripts.synthetic import generate_clients


def to_payload(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[k] = int(v) if k in ("DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN") else (
            float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
    return out


INVALID = [
    {"AMT_CREDIT": "abc"},
    {"AMT_INCOME_TOTAL": 0},
    {"DAYS_BIRTH": 1825},
    {"CODE_GENDER": "X"},
    {"EXT_SOURCE_2": 3.2},
]


def build_requests(n: int, seed: int, error_rate: float):
    n_norm = n // 2
    normal = generate_clients(n_norm, seed=seed, drift=0.0)
    drifted = [generate_clients(1, seed=seed + 10_000 + i, drift=min(1.0, i / (n - n_norm) * 1.5))
               for i in range(n - n_norm)]
    rows = [to_payload(r) for r in normal.to_dict("records")]
    rows += [to_payload(d.iloc[0].to_dict()) for d in drifted]
    rng = random.Random(seed)
    for i in range(len(rows)):
        if rng.random() < error_rate:
            rows[i] = {**rows[i], **rng.choice(INVALID)}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:7860")
    p.add_argument("--in-process", action="store_true")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--error-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    payloads = build_requests(args.n, args.seed, args.error_rate)

    if args.in_process:
        from fastapi.testclient import TestClient
        from app.main import app
        ctx = TestClient(app)
        client = ctx.__enter__()
        post = lambda body: client.post("/predict", json=body)
    else:
        import requests
        session = requests.Session()
        post = lambda body: session.post(f"{args.url}/predict", json=body, timeout=10)

    codes: dict[int, int] = {}
    t0 = time.perf_counter()
    for i, body in enumerate(payloads, 1):
        code = post(body).status_code
        codes[code] = codes.get(code, 0) + 1
        if i % 250 == 0:
            print(f"{i}/{len(payloads)} requêtes envoyées")
    print(f"Terminé en {time.perf_counter() - t0:.1f}s — codes HTTP : {json.dumps(codes)}")
    if args.in_process:
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
