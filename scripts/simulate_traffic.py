"""Simule du trafic de production sur l'API pour alimenter le monitoring.

Les requêtes sont des dossiers réels du jeu de test (model/sample_clients.parquet).
Scénario par défaut :
  1. première moitié : dossiers tels quels (même distribution que l'entraînement)
  2. seconde moitié : dérive progressive de la clientèle
     (plus jeune, revenus plus faibles, crédits plus élevés, EXT_SOURCE_2/3 plus bas)
  3. ~3 % de requêtes invalides (taux d'erreur)

    # contre une API qui tourne (local, Docker ou Hugging Face) :
    python -m scripts.simulate_traffic --url http://localhost:7860 --n 2000
    # sans serveur (appel in-process, écrit dans data/predictions.db) :
    python -m scripts.simulate_traffic --in-process --n 2000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INVALID = [
    {"AMT_CREDIT": "abc"},
    {"AMT_INCOME_TOTAL": 0},
    {"DAYS_BIRTH": 1825},
    {"CODE_GENDER": "X"},
    {"EXT_SOURCE_2": 3.2},
]


def load_clients() -> list[dict]:
    df = pd.read_parquet(ROOT / "model" / "sample_clients.parquet").drop(
        columns=["SK_ID_CURR", "expected_score"], errors="ignore")
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if pd.notna(v)}
            for r in df.to_dict("records")]


def apply_drift(rec: dict, level: float, rng: random.Random) -> dict:
    """level ∈ [0, 1] : intensité de la dérive."""
    r = dict(rec)
    for col in ("EXT_SOURCE_2", "EXT_SOURCE_3"):  # scores externes plus bas
        if col in r:
            r[col] = round(max(0.0, r[col] * (1 - 0.5 * level * rng.uniform(0.5, 1))), 6)
    if "DAYS_BIRTH" in r:
        r["DAYS_BIRTH"] = int(min(-6570, r["DAYS_BIRTH"] + level * rng.uniform(0, 4500)))
        if isinstance(r.get("DAYS_EMPLOYED"), (int, float)) and r["DAYS_EMPLOYED"] != 365243:
            r["DAYS_EMPLOYED"] = int(max(r["DAYS_EMPLOYED"], r["DAYS_BIRTH"]))
    if "AMT_INCOME_TOTAL" in r:
        r["AMT_INCOME_TOTAL"] = round(r["AMT_INCOME_TOTAL"] * (1 - 0.3 * level), 0)
    if "AMT_CREDIT" in r:
        r["AMT_CREDIT"] = round(r["AMT_CREDIT"] * (1 + 0.5 * level), 0)
    return r


def build_requests(n: int, seed: int, error_rate: float) -> list[dict]:
    rng = random.Random(seed)
    clients = load_clients()
    half = n // 2
    out = []
    for i in range(n):
        rec = rng.choice(clients)
        if i >= half:
            rec = apply_drift(rec, min(1.0, (i - half) / (n - half) * 1.5), rng)
        if rng.random() < error_rate:
            rec = {**rec, **rng.choice(INVALID)}
        out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:7860")
    p.add_argument("--in-process", action="store_true")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--error-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    payloads = build_requests(args.n, args.seed, args.error_rate)

    if args.in_process:
        from fastapi.testclient import TestClient
        from app.main import app
        ctx = TestClient(app)
        client = ctx.__enter__()
        post = lambda body: client.post("/predict", json=body)  # noqa: E731
    else:
        import requests
        session = requests.Session()
        post = lambda body: session.post(f"{args.url}/predict", json=body, timeout=10)  # noqa: E731

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
