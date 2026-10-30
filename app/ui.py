"""Interface Gradio de démonstration, montée sur l'API FastAPI (/ui).
Elle passe par la même validation et la même journalisation que /predict."""
from __future__ import annotations

import time

import gradio as gr
from pydantic import ValidationError

from app.schemas import ClientData

EDUCATION = ["Lower secondary", "Secondary / secondary special", "Incomplete higher",
             "Higher education", "Academic degree"]


def make_predict_fn(score_fn):
    """Fonction appelée par le bouton de l'interface (testable sans navigateur)."""
    def run(ext1, ext2, ext3, income, credit, annuity, goods, age, years_emp,
            children, gender, education):
        payload = {
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods or None,
            "DAYS_BIRTH": int(-age * 365.25) if age else None,
            "DAYS_EMPLOYED": int(-years_emp * 365.25) if years_emp is not None else None,
            "CNT_CHILDREN": int(children or 0),
            "CODE_GENDER": gender, "NAME_EDUCATION_TYPE": education,
        }
        try:
            client = ClientData(**payload)
        except ValidationError as exc:
            msgs = "\n".join(f"- {'.'.join(map(str, e['loc'])) or 'données'} : {e['msg']}"
                             for e in exc.errors())
            return f"### ⚠️ Données invalides\n{msgs}", None
        res = score_fn(client, "/ui", time.perf_counter())
        verdict = "✅ Crédit ACCORDÉ" if res.decision == "ACCORDE" else "❌ Crédit REFUSÉ"
        md = (f"## {verdict}\n"
              f"Probabilité de défaut : **{res.score:.1%}** (seuil métier {res.threshold:.1%})\n\n"
              f"Modèle `{res.model_version}` · inférence {res.inference_ms:.2f} ms")
        return md, {"Risque de défaut": res.score, "Remboursement": 1 - res.score}
    return run


def build_ui(score_fn) -> gr.Blocks:
    run = make_predict_fn(score_fn)

    with gr.Blocks(title="Prêt à Dépenser — Scoring") as demo:
        gr.Markdown("# Prêt à Dépenser — Scoring crédit\nSaisissez les informations du client.")
        with gr.Row():
            with gr.Column():
                ext1 = gr.Slider(0, 1, 0.5, label="EXT_SOURCE_1")
                ext2 = gr.Slider(0, 1, 0.6, label="EXT_SOURCE_2")
                ext3 = gr.Slider(0, 1, 0.5, label="EXT_SOURCE_3")
                gender = gr.Radio(["F", "M"], value="F", label="Genre")
                education = gr.Dropdown(EDUCATION, value="Higher education", label="Éducation")
            with gr.Column():
                income = gr.Number(180000, label="Revenu annuel")
                credit = gr.Number(600000, label="Montant du crédit")
                annuity = gr.Number(28000, label="Annuité")
                goods = gr.Number(540000, label="Prix du bien")
                age = gr.Number(38, label="Âge (années)")
                years_emp = gr.Number(7, label="Ancienneté emploi (années)")
                children = gr.Number(1, label="Nombre d'enfants", precision=0)
        btn = gr.Button("Évaluer le client", variant="primary")
        out_md = gr.Markdown()
        out_label = gr.Label(label="Probabilités")
        btn.click(run, [ext1, ext2, ext3, income, credit, annuity, goods, age, years_emp,
                        children, gender, education], [out_md, out_label])
    return demo
