FROM python:3.12-slim

# libgomp : requis par LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (uid 1000, exigé par Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser app/ app/
COPY --chown=appuser model/ model/
RUN mkdir -p /app/data && chown appuser /app/data

USER appuser
ENV PYTHONUNBUFFERED=1 \
    MODEL_BACKEND=onnx \
    DATA_DIR=/app/data \
    GRADIO_ANALYTICS_ENABLED=False

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fs http://localhost:7860/health || exit 1

# 1 worker = 1 copie du modèle en mémoire ; augmenter --workers selon les CPU dispo
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
