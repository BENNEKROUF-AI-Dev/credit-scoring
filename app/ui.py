"""Interface Gradio de démonstration, montée sur l'API FastAPI (/ui).

On choisit un dossier d'exemple (jeu de test du projet précédent), on peut modifier
les variables les plus influentes, puis on évalue. La requête passe par la même
validation et la même journalisation que /predict."""
from __future__ import annotations

import time

import gradio as gr
from pydantic import ValidationError

from app.schemas import ClientData

EDITABLE = [  # (variable, libellé, type)
    ("EXT_SOURCE_1", "Score externe 1 (0-1)", "num"),
    ("EXT_SOURCE_2", "Score externe 2 (0-1)", "num"),
    ("EXT_SOURCE_3", "Score externe 3 (0-1)", "num"),
    ("AMT_INCOME_TOTAL", "Revenu annuel", "num"),
    ("AMT_CREDIT", "Montant du crédit", "num"),
    ("AMT_ANNUITY", "Annuité", "num"),
    ("AGE", "Âge (années)", "age"),
]


def make_predict_fn(score_fn, clients: dict):
    """Fonction appelée par le bouton de l'interface (testable sans navigateur)."""
    def run(sk_id, *values):
        record = dict(clients.get(int(sk_id), {})) if sk_id is not None else {}
        for (col, _, kind), v in zip(EDITABLE, values):
            if kind == "age":
                if v is not None:
                    record["DAYS_BIRTH"] = int(-float(v) * 365.25)
            elif v is None or v == "":
                record.pop(col, None)
            else:
                record[col] = float(v)
        try:
            client = ClientData.model_validate(record)
        except ValidationError as exc:
            msgs = "\n".join(f"- {'.'.join(map(str, e['loc'])) or 'dossier'} : {e['msg']}"
                             for e in exc.errors())
            return f"### ⚠️ Données invalides\n{msgs}", None
        res = score_fn(client, "/ui", time.perf_counter())
        verdict = "✅ Crédit ACCORDÉ" if res.decision == "ACCORDE" else "❌ Crédit REFUSÉ"
        md = (f"## {verdict}\n"
              f"Probabilité de défaut : **{res.score:.1%}** (seuil métier {res.threshold:.0%})\n\n"
              f"Modèle `{res.model_version}` · inférence {res.inference_ms:.2f} ms")
        return md, {"Risque de défaut": res.score, "Remboursement": 1 - res.score}
    return run


def make_load_fn(clients: dict):
    def load(sk_id):
        rec = clients.get(int(sk_id), {}) if sk_id is not None else {}
        out = []
        for col, _, kind in EDITABLE:
            if kind == "age":
                out.append(round(-rec["DAYS_BIRTH"] / 365.25, 1) if "DAYS_BIRTH" in rec else None)
            else:
                v = rec.get(col)
                out.append(round(v, 4) if isinstance(v, float) else v)
        return out
    return load


def build_ui(score_fn, clients: dict) -> gr.Blocks:
    run, load = make_predict_fn(score_fn, clients), make_load_fn(clients)
    ids = sorted(clients)[:200]
    with gr.Blocks(title="Prêt à Dépenser — Scoring") as demo:
        gr.Markdown("# Prêt à Dépenser — Scoring crédit\n"
                    "Choisissez un dossier client, ajustez éventuellement les variables clés, puis évaluez.")
        sk = gr.Dropdown(ids, value=ids[0] if ids else None, label="Dossier (SK_ID_CURR)")
        inputs = []
        with gr.Row():
            with gr.Column():
                for col, label, _ in EDITABLE[:3]:
                    inputs.append(gr.Number(label=label, minimum=0, maximum=1, step=0.01))
            with gr.Column():
                for col, label, _ in EDITABLE[3:]:
                    inputs.append(gr.Number(label=label))
        btn = gr.Button("Évaluer le client", variant="primary")
        out_md = gr.Markdown()
        out_label = gr.Label(label="Probabilités")
        sk.change(load, sk, inputs)
        demo.load(load, sk, inputs)
        btn.click(run, [sk, *inputs], [out_md, out_label])
    return demo
