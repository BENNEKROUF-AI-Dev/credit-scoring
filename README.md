# Prêt à Dépenser — API de scoring crédit (MLOps)

Mise en production du modèle de scoring : API **FastAPI** avec une interface **Gradio**, image
**Docker**, pipeline **CI/CD GitHub Actions** (tests, build, déploiement sur Hugging Face Spaces),
stockage des données de production, **monitoring et détection de data drift** (Evidently,
Streamlit) et **optimisation de l'inférence** (profiling, ONNX Runtime).

![CI/CD](https://github.com/<votre-compte>/<votre-repo>/actions/workflows/ci-cd.yml/badge.svg)

## Architecture

```
          ┌────────────── GitHub Actions (push sur main) ───────────────┐
          │  1. pytest (39 tests)  →  2. docker build + smoke test  →   │
          │  3. push image GHCR    →  4. déploiement Hugging Face Space │
          └─────────────────────────────────────────────────────────────┘
                                         │
 Client ──HTTP──► FastAPI /predict ──► Pydantic (validation) ──► features NumPy ──► ONNX Runtime
                     │  /ui (Gradio)                                                   │
                     ▼                                                                 ▼
          logs JSON (predictions.jsonl) + SQLite (predictions.db)  ◄──── score, décision, latences
                     │
                     ▼
     monitoring/drift_analysis.py (Evidently, PSI, alertes)  ──►  dashboard Streamlit
```

## Structure du dépôt

```
app/                    code de l'API (embarqué dans l'image Docker)
  main.py               endpoints FastAPI, chargement unique du modèle (lifespan), logging
  schemas.py            validation des entrées (types, plages, cohérence)
  features.py           feature engineering (pandas pour l'entraînement, NumPy pour l'inférence)
  model.py              chargement et inférence (ONNX Runtime ou LightGBM)
  storage.py            stockage des données de production (JSON + SQLite)
  ui.py                 interface Gradio
model/                  artefacts : model.joblib, model.onnx, model_meta.json, reference_data.parquet
scripts/
  prepare_model.py      exporte le modèle (MLflow, Home Credit ou données synthétiques)
  convert_to_onnx.py    conversion ONNX et vérification de non-régression
  simulate_traffic.py   simulation de trafic de production (avec dérive)
  benchmark.py          profiling cProfile et comparaison des optimisations
  deploy_hf_space.py    déploiement Hugging Face (appelé par la CI/CD)
monitoring/
  drift_analysis.py     analyse automatique : drift, PSI, erreurs, latences, alertes
  dashboard.py          dashboard Streamlit
tests/                  tests unitaires et d'intégration (pytest)
docs/                   rapport d'optimisation, étude du drift, captures d'écran
.github/workflows/      pipeline CI/CD
Dockerfile
```

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt

# (facultatif : les artefacts sont déjà versionnés dans model/)
python -m scripts.prepare_model && python -m scripts.convert_to_onnx

uvicorn app.main:app --port 7860
```

- Swagger (documentation interactive) : http://localhost:7860/docs
- Interface Gradio : http://localhost:7860/ui

### Avec Docker

```bash
docker build -t credit-scoring-api .
docker run -p 7860:7860 -v $(pwd)/data:/app/data credit-scoring-api
```

Le volume `data/` conserve les logs de production hors du conteneur.

## Utiliser l'API

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | État de l'API et du modèle |
| GET | `/model-info` | Version, seuil métier, métriques, variables attendues |
| POST | `/predict` | Score d'un client |

```bash
curl -X POST http://localhost:7860/predict -H "Content-Type: application/json" -d '{
  "EXT_SOURCE_1": 0.45, "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.51,
  "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 600000, "AMT_ANNUITY": 28000,
  "AMT_GOODS_PRICE": 540000, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500,
  "CNT_CHILDREN": 1, "CODE_GENDER": "F", "NAME_EDUCATION_TYPE": "Higher education"}'
```
```json
{"request_id": "…", "score": 0.124, "threshold": 0.49, "decision": "ACCORDE",
 "model_version": "v1.0.0", "inference_ms": 0.21}
```

- `score` est la probabilité de défaut.
- `decision` vaut `REFUSE` si le score est supérieur ou égal au seuil métier. Ce seuil minimise un
  coût où un faux négatif coûte 10 fois plus qu'un faux positif.

**Validation et erreurs :** une requête invalide renvoie le code **422** avec le détail. C'est le cas
si un champ obligatoire manque, si une valeur est hors plage (revenu ≤ 0, âge < 18 ans,
`EXT_SOURCE` > 1…), si le type est faux (texte à la place d'un nombre), si la catégorie est inconnue,
si un champ inconnu est présent ou si une annuité dépasse le crédit. Une erreur interne renvoie
**500** et le message n'expose rien de l'implémentation. Les `EXT_SOURCE_*`, `AMT_GOODS_PRICE` et
`DAYS_EMPLOYED` sont facultatifs : le modèle gère les valeurs manquantes.

**Variables d'environnement :**

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_BACKEND` | `onnx` | `onnx` ou `lightgbm` |
| `FAST_FEATURES` | `1` | `0` pour revenir aux features pandas (v1) |
| `DATA_DIR` | `data/` | Emplacement des logs |
| `ENABLE_UI` | `1` | Monte l'interface Gradio |

## Tests

```bash
pytest -v --cov=app
```

39 tests :
- endpoints de l'API ;
- cas critiques : données manquantes, valeurs hors plage, types incorrects ;
- stockage des prédictions ;
- feature engineering ;
- non-régression ONNX par rapport à LightGBM ;
- interface Gradio et erreur interne (500) ;
- garde-fou d’AUC.

## CI/CD (`.github/workflows/ci-cd.yml`)

Déclenché à chaque push et pull request sur `main`.

1. **test** : installation, puis `pytest` avec une couverture minimale de 80 %.
2. **build** (seulement si les tests passent) : construction de l'image Docker, puis *smoke test* du
   conteneur (`/health`, une prédiction valide, une requête invalide qui doit renvoyer 422). Sur
   `main`, l'image est publiée sur GitHub Container Registry.
3. **deploy** (seulement sur `main`) : déploiement sur un Hugging Face Space (SDK Docker).

**Configuration du déploiement :**
1. Créez un token *write* sur huggingface.co (Settings → Access Tokens).
2. Dans le dépôt GitHub, allez dans Settings → Secrets and variables → Actions.
3. Ajoutez le **secret** `HF_TOKEN`.
4. Ajoutez la **variable** `HF_SPACE` (par exemple `mon-pseudo/credit-scoring-api`).

Aucun identifiant n'est écrit dans le code. Sans ces réglages, le job de déploiement est simplement
ignoré.

## Monitoring : lancer et interpréter

```bash
python -m scripts.simulate_traffic --url http://localhost:7860 --n 2000   # ou --in-process
python -m monitoring.drift_analysis      # rapport Evidently + summary.json + alertes
streamlit run monitoring/dashboard.py    # http://localhost:8501
```

**Comment lire le dashboard :**

| Indicateur | Lecture | Alerte si |
|---|---|---|
| Taux d'erreur | Part des requêtes en 4xx ou 5xx. Un pic de 422 signale souvent un client de l'API mal intégré | > 5 % |
| Latence API / inférence (p50, p95) | p95 = temps sous lequel passent 95 % des requêtes. Inférence = modèle et features seuls | p95 > 200 ms |
| Distribution des scores | Histogramme production contre entraînement. Un décalage vers la droite signifie plus de clients jugés risqués | PSI > 0,2 |
| Taux de refus | À comparer avec celui de la référence | ± 10 points |
| PSI | < 0,1 : stable ; 0,1 à 0,2 : à surveiller ; > 0,2 : dérive significative | > 0,2 |
| Onglet Data drift | Distribution de chaque variable (référence contre production), tableau Evidently, **dérive par fenêtre temporelle** | > 30 % de variables en dérive |

L'analyse par fenêtres est essentielle : une dérive récente peut être noyée dans l'agrégat global.

Pour aller plus loin :
- [`docs/etude_data_drift.md`](docs/etude_data_drift.md) : étude de la dérive et points de vigilance ;
- [`docs/rapport_optimisation.md`](docs/rapport_optimisation.md) : profiling et optimisations
  (latence API divisée par 6, inférence 100 fois plus rapide, décisions identiques à 100 %).

## Brancher le modèle de la partie 1 (MLflow)

```bash
# modèle LightGBM enregistré dans MLflow + données Home Credit
python -m scripts.prepare_model --mlflow-model-uri "models:/credit_scoring/Production" \
       --data data/application_train.csv --version v2.0.0
python -m scripts.convert_to_onnx
pytest
```

Si votre modèle utilise d'autres variables, adaptez `RAW_FEATURES` et `build_features` dans
`app/features.py`, puis `build_features_fast` (le test d'équivalence vous signalera tout écart), et
`app/schemas.py`. Les artefacts de `model/` sont versionnés avec le code. Pour des modèles plus
lourds, utilisez Git LFS ou un téléchargement depuis le registre MLflow au build.

## Choix techniques

- **FastAPI** : validation par Pydantic, Swagger automatique, performances. **Gradio** est monté sur
  la même application pour une démo sans front-end.
- **SQLite + JSON lines** : un PoC sans infrastructure. Le schéma est directement transposable à
  PostgreSQL, et les logs JSON peuvent être collectés par Fluentd ou Logstash.
- **Evidently** : tests statistiques adaptés au type de chaque variable, et rapport HTML partageable.
  **Streamlit** : dashboard interactif écrit en Python.
- **ONNX Runtime** : le moteur d'inférence le plus rapide mesuré, portable.
- **Hugging Face Spaces** : hébergement Docker gratuit, déploiement par API depuis la CI.
