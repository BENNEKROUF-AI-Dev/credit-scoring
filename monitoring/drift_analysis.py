"""Analyse automatique des données de production.

- Data drift (Evidently) : données de production vs données d'entraînement
- Dérive des scores (PSI) et du taux de refus
- Anomalies opérationnelles : taux d'erreur, latence anormale
- Analyse par fenêtres temporelles pour voir QUAND la dérive apparaît

    python -m monitoring.drift_analysis [--db data/predictions.db] [--windows 4]

Sorties : monitoring/reports/drift_report.html, monitoring/reports/summary.json
Code retour 1 si une alerte est levée (utilisable dans un job planifié / CI).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from monitoring.analysis import (ALERTS, CATEGORICAL, DEFAULT_DB, ROOT, expand_inputs,  # noqa: E402
                                 latency_anomalies, load_logs, load_reference,
                                 operational_metrics, psi)

REPORTS = ROOT / "monitoring" / "reports"


def run_evidently(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path | None = None) -> dict:
    cols = [c for c in reference.columns if c in current.columns]
    num = [c for c in cols if c not in CATEGORICAL]
    dd = DataDefinition(numerical_columns=num, categorical_columns=[c for c in CATEGORICAL if c in cols])
    snap = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current[cols], data_definition=dd),
        Dataset.from_pandas(reference[cols], data_definition=dd),
    )
    if html_path:
        snap.save_html(str(html_path))
    columns, share = {}, None
    for m in snap.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            share = m["value"]["share"]
        elif cfg["type"].endswith("ValueDrift"):
            method, thr, val = cfg["method"], cfg["threshold"], float(m["value"])
            is_pvalue = "p_value" in method.lower()
            columns[cfg["column"]] = {
                "method": method, "value": round(val, 4), "threshold": thr,
                "drift": bool(val < thr) if is_pvalue else bool(val >= thr),
            }
    return {"drift_share": share, "columns": columns}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--windows", type=int, default=4, help="nb de fenêtres temporelles")
    args = p.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    logs = load_logs(args.db)
    if logs.empty:
        sys.exit("Aucune donnée de production dans la base.")
    reference = load_reference()
    current = expand_inputs(logs)

    ops = operational_metrics(logs)
    ref_refusal = None
    try:
        thr = float(logs.loc[logs["threshold"].notna(), "threshold"].iloc[-1])
        ref_refusal = float((reference["score"] >= thr).mean())
    except IndexError:
        pass

    drift = run_evidently(reference, current, REPORTS / "drift_report.html")
    score_psi = psi(reference["score"].to_numpy(), current["score"].to_numpy())

    # Fenêtres temporelles : la dérive est-elle progressive ?
    windows = []
    for i, chunk in enumerate(pd.Series(range(len(current))).groupby(
            pd.qcut(range(len(current)), args.windows, labels=False))):
        w = current.iloc[chunk[1].values]
        d = run_evidently(reference, w)
        windows.append({
            "window": i + 1,
            "start": str(w["timestamp"].min()), "end": str(w["timestamp"].max()),
            "n": int(len(w)), "drift_share": round(d["drift_share"], 3),
            "score_psi": round(psi(reference["score"].to_numpy(), w["score"].to_numpy()), 4),
            "mean_score": round(float(w["score"].mean()), 4),
            "drifted_columns": [c for c, v in d["columns"].items() if v["drift"]],
        })

    alerts = []
    if ops["error_rate"] > ALERTS["error_rate"]:
        alerts.append(f"Taux d'erreur élevé : {ops['error_rate']:.1%}")
    if ops["latency_p95_ms"] and ops["latency_p95_ms"] > ALERTS["latency_p95_ms"]:
        alerts.append(f"Latence p95 élevée : {ops['latency_p95_ms']:.1f} ms")
    if score_psi > ALERTS["score_psi"]:
        alerts.append(f"Dérive de la distribution des scores : PSI = {score_psi:.3f}")
    last = windows[-1]
    if last["score_psi"] > ALERTS["score_psi"] and score_psi <= ALERTS["score_psi"]:
        alerts.append(f"Dérive RÉCENTE des scores (dernière fenêtre) : PSI = {last['score_psi']:.3f} "
                      "— masquée dans l'agrégat global")
    if drift["drift_share"] and drift["drift_share"] > ALERTS["drift_share"]:
        drifted = [c for c, v in drift["columns"].items() if v["drift"]]
        alerts.append(f"Data drift sur {drift['drift_share']:.0%} des variables : {', '.join(drifted)}")
    if ref_refusal is not None and ops["refusal_rate"] is not None and \
            abs(ops["refusal_rate"] - ref_refusal) > ALERTS["refusal_rate_delta"]:
        alerts.append(f"Taux de refus {ops['refusal_rate']:.1%} vs {ref_refusal:.1%} en référence")
    lat_anom = latency_anomalies(logs)

    summary = {
        "operational": ops,
        "reference_refusal_rate": ref_refusal,
        "score_psi": round(score_psi, 4),
        "data_drift": drift,
        "windows": windows,
        "latency_anomalies": int(len(lat_anom)),
        "alerts": alerts,
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"Requêtes : {ops['n_requests']} | erreurs : {ops['error_rate']:.1%} | "
          f"latence p50/p95 : {ops['latency_p50_ms']:.1f}/{ops['latency_p95_ms']:.1f} ms")
    print(f"PSI score : {score_psi:.3f} | variables en dérive : {drift['drift_share']:.0%}")
    for w in windows:
        print(f"  fenêtre {w['window']} : drift {w['drift_share']:.0%}, PSI {w['score_psi']:.3f}, "
              f"score moyen {w['mean_score']:.3f}")
    print("ALERTES :" if alerts else "Aucune alerte.")
    for a in alerts:
        print(f"  ⚠ {a}")
    print(f"Rapport : {REPORTS / 'drift_report.html'}")
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
