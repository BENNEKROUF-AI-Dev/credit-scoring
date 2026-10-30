"""API de scoring crédit « Prêt à Dépenser ».

Lancer :  uvicorn app.main:app --host 0.0.0.0 --port 7860
Docs Swagger : http://localhost:7860/docs   |   Interface Gradio : /ui
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app import config
from app.model import ScoringModel
from app.schemas import ClientData, HealthResponse, PredictionResponse
from app.storage import PredictionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement UNIQUE du modèle au démarrage
    STATE["model"] = ScoringModel()
    STATE["store"] = (
        PredictionStore(config.PREDICTIONS_DB, config.PREDICTIONS_LOG)
        if config.ENABLE_PREDICTION_LOGGING else None
    )
    yield
    STATE.clear()


app = FastAPI(
    title="Prêt à Dépenser — API de scoring crédit",
    description="Retourne la probabilité de défaut d'un client et la décision d'octroi.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model() -> ScoringModel:
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return model


def _log(**record) -> None:
    store = STATE.get("store")
    if store is not None:
        try:
            store.log(**record)
        except Exception:  # le logging ne doit jamais casser une prédiction
            logger.exception("Échec d'écriture du log de prédiction")


def score_client(client: ClientData, endpoint: str, t_start: float) -> PredictionResponse:
    model = get_model()
    request_id = str(uuid.uuid4())
    inputs = client.model_dump()
    try:
        score, inference_ms = model.predict_one(inputs)
    except Exception as exc:
        logger.exception("Erreur d'inférence")
        _log(request_id=request_id, endpoint=endpoint, status_code=500, inputs=inputs,
             model_version=model.version, backend=model.backend,
             latency_ms=(time.perf_counter() - t_start) * 1000, error=str(exc))
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction")

    decision = model.decide(score)
    _log(request_id=request_id, endpoint=endpoint, status_code=200, inputs=inputs,
         score=score, decision=decision, threshold=model.threshold,
         model_version=model.version, backend=model.backend, inference_ms=inference_ms,
         latency_ms=(time.perf_counter() - t_start) * 1000)
    return PredictionResponse(request_id=request_id, score=round(score, 6),
                              threshold=model.threshold, decision=decision,
                              model_version=model.version, inference_ms=round(inference_ms, 3))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Les requêtes invalides sont aussi journalisées (suivi du taux d'erreur)."""
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
              for e in exc.errors()]
    model = STATE.get("model")
    _log(request_id=str(uuid.uuid4()), endpoint=request.url.path, status_code=422,
         model_version=getattr(model, "version", None), backend=getattr(model, "backend", None),
         latency_ms=None, error=str(errors)[:1000])
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui" if config.ENABLE_UI else "/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    model = STATE.get("model")
    return HealthResponse(status="ok" if model else "degraded", model_loaded=model is not None,
                          backend=getattr(model, "backend", "none"),
                          model_version=getattr(model, "version", "none"))


@app.get("/model-info")
def model_info():
    model = get_model()
    return {"version": model.version, "backend": model.backend,
            "threshold": model.threshold, "metrics": model.meta.get("metrics", {}),
            "features": model.meta.get("raw_features", [])}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    return score_client(client, "/predict", time.perf_counter())


# ---------------------------------------------------------------- Interface Gradio
if config.ENABLE_UI:
    from app.ui import build_ui
    import gradio as gr

    app = gr.mount_gradio_app(app, build_ui(score_client), path="/ui")
