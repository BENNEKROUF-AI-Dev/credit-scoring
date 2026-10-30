"""Dashboard de monitoring Streamlit.

    streamlit run monitoring/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.analysis import (ALERTS, CATEGORICAL, DEFAULT_DB, ROOT, expand_inputs,  # noqa: E402
                                 latency_anomalies, load_logs, load_reference,
                                 operational_metrics, psi)

REPORTS = ROOT / "monitoring" / "reports"
REF_COLOR, PROD_COLOR = "#8a94a6", "#2563eb"

st.set_page_config(page_title="Monitoring — Scoring crédit", layout="wide")
st.title("Monitoring du modèle de scoring — Prêt à Dépenser")

db_path = st.sidebar.text_input("Base des prédictions", str(DEFAULT_DB))
if not Path(db_path).exists():
    st.warning("Aucune base trouvée. Lancez l'API puis `python -m scripts.simulate_traffic`.")
    st.stop()


@st.cache_data(ttl=30)
def get_data(path):
    logs = load_logs(path)
    return logs, expand_inputs(logs), load_reference()


logs, current, reference = get_data(db_path)
if st.sidebar.button("Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

rolling = st.sidebar.slider("Fenêtre glissante (requêtes)", 20, 500, 100, 10)
ops = operational_metrics(logs)
threshold = float(logs["threshold"].dropna().iloc[-1]) if logs["threshold"].notna().any() else 0.5
ref_refusal = float((reference["score"] >= threshold).mean())
score_psi = psi(reference["score"].to_numpy(), current["score"].to_numpy())

# ----------------------------------------------------------------- KPIs
c = st.columns(6)
c[0].metric("Requêtes", f"{ops['n_requests']:,}".replace(",", " "))
c[1].metric("Taux d'erreur", f"{ops['error_rate']:.1%}",
            delta="alerte" if ops["error_rate"] > ALERTS["error_rate"] else None, delta_color="inverse")
c[2].metric("Latence API p95", f"{ops['latency_p95_ms']:.1f} ms",
            delta=f"p50 {ops['latency_p50_ms']:.1f} ms", delta_color="off")
c[3].metric("Inférence p95", f"{ops['inference_p95_ms']:.2f} ms",
            delta=f"p50 {ops['inference_p50_ms']:.2f} ms", delta_color="off")
c[4].metric("Taux de refus", f"{ops['refusal_rate']:.1%}",
            delta=f"{(ops['refusal_rate'] - ref_refusal) * 100:+.1f} pts vs réf.", delta_color="off")
c[5].metric("PSI des scores", f"{score_psi:.3f}",
            delta="dérive" if score_psi > ALERTS["score_psi"] else "stable", delta_color="off")

summary_path = REPORTS / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    for a in summary.get("alerts", []):
        st.error(a)

tab_scores, tab_perf, tab_drift, tab_raw = st.tabs(
    ["Scores", "Performance", "Data drift", "Données brutes"])

# ----------------------------------------------------------------- Scores
with tab_scores:
    col1, col2 = st.columns(2)
    fig = go.Figure()
    fig.add_histogram(x=reference["score"], name="Référence (entraînement)", histnorm="probability",
                      marker_color=REF_COLOR, opacity=0.6, nbinsx=40)
    fig.add_histogram(x=current["score"], name="Production", histnorm="probability",
                      marker_color=PROD_COLOR, opacity=0.6, nbinsx=40)
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"seuil {threshold:.2f}")
    fig.update_layout(barmode="overlay", title="Distribution des scores prédits",
                      xaxis_title="Probabilité de défaut", yaxis_title="Proportion")
    col1.plotly_chart(fig, use_container_width=True)

    ts = current[["timestamp", "score"]].reset_index(drop=True)
    ts["score moyen (glissant)"] = ts["score"].rolling(rolling, min_periods=10).mean()
    ts["taux de refus (glissant)"] = (ts["score"] >= threshold).rolling(rolling, min_periods=10).mean()
    fig = px.line(ts, x=ts.index, y=["score moyen (glissant)", "taux de refus (glissant)"],
                  title="Évolution dans le temps", labels={"index": "n° de requête", "value": ""})
    fig.add_hline(y=reference["score"].mean(), line_dash="dot", line_color=REF_COLOR,
                  annotation_text="score moyen réf.")
    col2.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- Performance
with tab_perf:
    ok = logs[logs["status_code"] == 200].reset_index(drop=True)
    col1, col2 = st.columns(2)
    fig = px.line(ok, x=ok.index, y=["latency_ms", "inference_ms"],
                  title="Latence API et temps d'inférence (ms)",
                  labels={"index": "n° de requête", "value": "ms"})
    col1.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(ok, x="latency_ms", nbins=60, title="Distribution de la latence")
    col2.plotly_chart(fig, use_container_width=True)

    err = logs.copy()
    err["erreur"] = (err["status_code"] != 200).astype(int)
    err["taux d'erreur (glissant)"] = err["erreur"].rolling(rolling, min_periods=10).mean()
    st.plotly_chart(px.line(err.reset_index(), x="index", y="taux d'erreur (glissant)",
                            title="Taux d'erreur"), use_container_width=True)
    st.caption("Codes HTTP : " + ", ".join(f"{k} → {v}" for k, v in
                                          logs["status_code"].value_counts().items()))
    anom = latency_anomalies(logs)
    st.subheader(f"Latences anormales ({len(anom)})")
    st.dataframe(anom[["timestamp", "latency_ms", "inference_ms", "backend"]], hide_index=True)
    errs = logs[logs["status_code"] != 200]
    st.subheader(f"Dernières erreurs ({len(errs)})")
    st.dataframe(errs[["timestamp", "status_code", "error"]].tail(20), hide_index=True)

# ----------------------------------------------------------------- Drift
with tab_drift:
    features = [c for c in reference.columns if c != "score"]
    feat = st.selectbox("Variable", features,
                        index=features.index("EXT_SOURCE_3") if "EXT_SOURCE_3" in features else 0)
    if feat in CATEGORICAL:
        r = reference[feat].value_counts(normalize=True).rename("Référence")
        p = current[feat].value_counts(normalize=True).rename("Production")
        dfc = pd.concat([r, p], axis=1).fillna(0).reset_index()
        fig = px.bar(dfc, x=feat, y=["Référence", "Production"], barmode="group",
                     color_discrete_sequence=[REF_COLOR, PROD_COLOR])
    else:
        fig = go.Figure()
        fig.add_histogram(x=reference[feat], name="Référence", histnorm="probability",
                          marker_color=REF_COLOR, opacity=0.6, nbinsx=40)
        fig.add_histogram(x=current[feat], name="Production", histnorm="probability",
                          marker_color=PROD_COLOR, opacity=0.6, nbinsx=40)
        fig.update_layout(barmode="overlay")
        st.caption(f"PSI {feat} : {psi(pd.to_numeric(reference[feat]).to_numpy(float), pd.to_numeric(current[feat]).to_numpy(float)):.3f} "
                   f"| valeurs manquantes réf. {reference[feat].isna().mean():.1%} / prod. "
                   f"{current[feat].isna().mean():.1%}")
    fig.update_layout(title=f"{feat} : référence vs production")
    st.plotly_chart(fig, use_container_width=True)

    if summary_path.exists():
        st.subheader("Résultat de la dernière analyse automatique (Evidently)")
        cols = pd.DataFrame(summary["data_drift"]["columns"]).T.reset_index(names="variable")
        st.dataframe(cols.sort_values("drift", ascending=False), hide_index=True)
        st.subheader("Dérive par fenêtre temporelle")
        st.dataframe(pd.DataFrame(summary["windows"]), hide_index=True)
    html = REPORTS / "drift_report.html"
    if html.exists() and st.checkbox("Afficher le rapport Evidently complet"):
        components.html(html.read_text(), height=1000, scrolling=True)
    if not summary_path.exists():
        st.info("Lancez `python -m monitoring.drift_analysis` pour générer le rapport Evidently.")

# ----------------------------------------------------------------- Raw
with tab_raw:
    st.dataframe(logs.tail(500), hide_index=True)
    st.download_button("Télécharger les logs (CSV)", logs.to_csv(index=False), "predictions.csv")
