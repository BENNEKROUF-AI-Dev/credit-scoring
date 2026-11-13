# Prêt à Dépenser — API de scoring crédit (MLOps)

Mise en production du modèle de scoring développé au projet précédent (notebooks
`01_exploration_feature_engineering` et `02_modelisation`, modèle enregistré dans MLflow sous
`scoring_credit`). Le projet comprend :
- une API **FastAPI** avec une interface **Gradio** ;
- une image **Docker** ;
- un pipeline **CI/CD GitHub Actions** (tests, build, déploiement sur Render) ;
- le stockage des données de production ;
- le **monitoring et la détection de data drift** (Evidently, Streamlit) ;
- l'**optimisation de l'inférence** (profiling, ONNX Runtime).

![CI/CD](https://github.com/BENNEKROUF-AI-Dev/credit-scoring/actions/workflows/ci-cd.yml/badge.svg)

## Architecture

```
 Projet précédent                         Ce dépôt
 ───────────────                          ────────
 notebook 1 → app_features.parquet ─┐
 notebook 2 → MLflow scoring_credit ─┴─► scripts/export_model.py ─► model/ (ONNX, métadonnées)
                                                                        │
          ┌────────────── GitHub Actions (push sur main) ───────────────┤
          │  1. pytest  →  2. docker build + smoke test  →  3. GHCR  →  4. Render
          └─────────────────────────────────────────────────────────────┘
                                         │
 Client ──HTTP──► FastAPI /predict ──► validation Pydantic ──► variables dérivées (notebook 1)
                     │  /ui (Gradio)       ──► encodage get_dummies/noms_propres (notebook 2, en NumPy)
                     │                     ──► ONNX Runtime ──► score + décision (seuil métier)
                     ▼
          logs JSON (predictions.jsonl) + SQLite (predictions.db)
                     │
                     ▼
     monitoring/drift_analysis.py (Evidently, PSI, alertes)  ──►  dashboard Streamlit
```

## Structure du dépôt

```
app/                    code de l'API (embarqué dans l'image Docker)
  main.py               endpoints FastAPI, chargement unique du modèle (lifespan), logging
  schemas.py            validation des entrées (générée depuis les métadonnées du modèle)
  preprocessing.py      transformations des notebooks 1 et 2 (versions pandas et NumPy)
  model.py              chargement et inférence (ONNX Runtime ou LightGBM)
  storage.py            stockage des données de production (JSON + SQLite)
  ui.py                 interface Gradio
model/                  artefacts produits par scripts/export_model.py
scripts/
  export_model.py       récupère le modèle MLflow, vérifie l'encodage, convertit en ONNX
  simulate_traffic.py   simulation de trafic de production (avec dérive)
  benchmark.py          profiling cProfile et comparaison des optimisations
monitoring/
  drift_analysis.py     analyse automatique : drift, PSI, erreurs, latences, alertes
  dashboard.py          dashboard Streamlit
tests/                  tests unitaires et d'intégration (pytest)
docs/                   rapport d'optimisation, étude du drift, captures d'écran
.github/workflows/      pipeline CI/CD
Dockerfile
```

## 1. Exporter le modèle du projet précédent

Cette étape se fait une seule fois, et à nouveau à chaque nouveau modèle. Utilisez le **Python du
projet précédent** : il contient déjà mlflow et lightgbm dans les bonnes versions.

```bash
pip install onnxmltools skl2onnx onnxruntime pyarrow     # dans l'environnement du projet précédent
cd chemin/vers/credit-scoring-mlops
python -m scripts.export_model --features "chemin/vers/app_features.parquet" \
       --tracking-uri "file:chemin/vers/mlruns"
```

Le script :
1. charge `models:/scoring_credit/latest` et lit le seuil métier dans la métrique `seuil` du run ;
2. **vérifie** que l'encodage de l'API (variables dérivées, `get_dummies`, `noms_propres`) donne
   exactement les mêmes scores que le notebook, sur 2 000 dossiers du jeu de test ;
3. convertit le modèle en **ONNX** et vérifie la non-régression (100 % de décisions identiques) ;
4. écrit dans `model/` :
   - `model.onnx` et `model.txt` (booster LightGBM) ;
   - `model_meta.json` (seuil, colonnes, catégories) ;
   - 1 000 dossiers d'exemple ;
   - la référence pour le drift.

Options utiles :
- `--model-uri models:/scoring_credit/3` pour choisir une version précise ;
- `--seuil 0.4` pour forcer le seuil ;
- `--retrain` si vous n'avez plus le dossier `mlruns` : le script ré-entraîne avec les réglages du
  notebook 2.

## 2. Lancer l'API

```bash
python -m venv .venv && .venv\Scripts\activate      # Mac/Linux : source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
uvicorn app.main:app --port 7860
```

- Swagger (documentation interactive) : http://localhost:7860/docs
- Interface Gradio : http://localhost:7860/ui

### Avec Docker

```bash
docker build -t credit-scoring-api .
docker run -p 7860:7860 -v "$(pwd)/data:/app/data" credit-scoring-api
```

Le volume `data/` conserve les logs de production hors du conteneur.

## Utiliser l'API

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | État de l'API et du modèle |
| GET | `/model-info` | Version, algorithme, seuil, métriques, champs acceptés |
| GET | `/clients`, `/clients/{id}` | Dossiers d'exemple (jeu de test), prêts à envoyer |
| POST | `/predict` | Score d'un dossier client |

**Format d'entrée.** Le corps de `/predict` est le dossier client au format du notebook 1 : les
variables de la demande et l'historique agrégé (`BUREAU_*`, `PREV_*`, `INST_*`…), sous leurs noms
d'origine.
- **Champs obligatoires** : `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `DAYS_BIRTH`, `CODE_GENDER`.
- **Tous les autres sont facultatifs** : un champ absent est traité comme une information manquante,
  comme pour un client sans historique.
- **Variables dérivées** (`DAYS_EMPLOYED_ANOM`, `*_MISSING`) : l'API les recalcule elle-même.
- **Code `DAYS_EMPLOYED = 365243`** (sans emploi) : accepté et traité comme dans le notebook 1.

```bash
curl -X POST http://localhost:7860/predict -H "Content-Type: application/json" -d '{
  "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 600000, "AMT_ANNUITY": 28000,
  "DAYS_BIRTH": -14000, "CODE_GENDER": "F", "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.51}'
```
```json
{"request_id": "…", "score": 0.21, "threshold": 0.44, "decision": "ACCORDE",
 "model_version": "v1", "inference_ms": 0.2}
```

- `score` est la probabilité de défaut.
- `decision` vaut `REFUSE` si le score est supérieur ou égal au seuil métier. Ce seuil minimise un
  coût où un défaut non détecté coûte 10 fois plus qu'un refus injustifié (notebook 2).

**Validation et erreurs :** une requête invalide renvoie le code **422** avec le détail. C'est le cas :
- si un champ obligatoire manque ;
- si une valeur est hors plage (revenu ≤ 0, âge < 18 ans, `EXT_SOURCE` > 1, date d'embauche dans le
  futur…) ;
- si le type est faux (texte à la place d'un nombre) ;
- si une catégorie est inconnue de l'entraînement ;
- si un champ inconnu est présent (dont `SK_ID_CURR`, refusé pour le RGPD) ou si une variable
  dérivée est envoyée ;
- si une annuité dépasse le crédit.

Une erreur interne renvoie **500**, sans détail d'implémentation.

**Variables d'environnement :**

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_BACKEND` | `onnx` | `onnx` ou `lightgbm` |
| `FAST_FEATURES` | `1` | `0` pour revenir à la préparation pandas (v1) |
| `DATA_DIR` | `data/` | Emplacement des logs |
| `ENABLE_UI` | `1` | Monte l'interface Gradio |
| `DATABASE_URL` | vide | Base PostgreSQL (NeonDB) des prédictions ; vide = SQLite local `data/predictions.db` |

## Tests

```bash
python -m pytest -v --cov=app
```

41 tests, qui couvrent notamment :
- **la non-régression** : l'API redonne les scores du modèle MLflow d'origine sur les dossiers
  d'exemple, et l'encodage NumPy est identique à l'encodage pandas du notebook ;
- **les cas critiques** : données manquantes, valeurs hors plage, mauvais types, catégories
  inconnues ;
- les variables dérivées, le stockage, l'interface Gradio et les erreurs 500.

## CI/CD (`.github/workflows/ci-cd.yml`)

Déclenché à chaque push et pull request sur `main`.

1. **test** : installation, puis `pytest` avec une couverture minimale de 80 %.
2. **build** (seulement si les tests passent) : construction de l'image Docker, puis *smoke test* du
   conteneur. Il appelle `/health`, envoie un dossier minimal et un dossier complet, puis vérifie
   qu'une requête invalide renvoie 422. Sur `main`, l'image est publiée sur GitHub Container Registry.
3. **deploy** (seulement sur `main`) : déclenche le déploiement sur **Render** (offre gratuite,
   image construite depuis le `Dockerfile`), puis vérifie que l'API en ligne répond sur `/health`.

**Configuration du déploiement :**
1. Sur render.com, créez un *Web Service* relié à ce dépôt : runtime **Docker**, instance **Free**,
   variable d'environnement `ENABLE_UI=0`, *Health Check Path* `/health`, *Auto-Deploy* **Off**
   (c'est la CI qui déploie, seulement si les tests passent).
2. Copiez le *Deploy Hook* du service (Settings → Deploy Hook).
3. Dans GitHub (Settings → Secrets and variables → Actions) : **secret** `RENDER_DEPLOY_HOOK`
   = le Deploy Hook, **variable** `RENDER_URL` = l'adresse du service (ex. `https://credit-scoring-api.onrender.com`).

Aucun identifiant n'est écrit dans le code. Sans ces réglages, le job de déploiement est ignoré.

**Pourquoi Render ?** Hugging Face Spaces exige désormais un abonnement payant pour les Spaces
Docker. L'offre gratuite de Render suffit pour un PoC, avec deux limites à connaître : 512 Mo de
RAM (d'où l'interface Gradio désactivée en ligne, Swagger `/docs` reste disponible) et une mise en
veille après 15 min d'inactivité (premier appel ≈ 1 min ensuite).

## Stockage des prédictions : SQLite ou NeonDB

Chaque appel à l'API enregistre le dossier reçu (`inputs`), le score, la décision, le seuil, la version
du modèle, le temps d'inférence, la latence et l'erreur éventuelle, dans la table `predictions`
(et dans les logs JSON `data/predictions.jsonl`).

- **En local** (par défaut) : un fichier SQLite, `data/predictions.db`.
- **En production** : une base PostgreSQL **NeonDB**, activée par la variable `DATABASE_URL`. La couche
  qui stocke est ainsi **séparée de la couche qui calcule** : les données survivent aux redéploiements
  de Render et se consultent indépendamment de l'API. Les écritures partent en arrière-plan, sans
  ralentir les réponses.

**Mise en place de NeonDB :**
1. Sur neon.tech, créez un projet (région Europe) et copiez la *connection string*
   (`postgresql://…neon.tech/neondb?sslmode=require`).
2. Dans Render, service `credit-scoring-api` → *Environment* : ajoutez `DATABASE_URL` = cette URL.
   La table est créée automatiquement au démarrage. `/health` affiche alors `"storage": "postgres"`.
3. Pour analyser les vraies requêtes en ligne, définissez la même variable en local
   (PowerShell : `$env:DATABASE_URL="postgresql://…"`), puis lancez le notebook, le dashboard ou
   `python -m monitoring.drift_analysis` : ils lisent alors NeonDB au lieu du fichier local.

## Monitoring : lancer et interpréter

```bash
python -m scripts.simulate_traffic --url http://localhost:7860 --n 2000   # ou --in-process
python -m monitoring.drift_analysis      # rapport Evidently + summary.json + alertes
streamlit run monitoring/dashboard.py    # http://localhost:8501
```

Le drift est suivi sur les **15 variables les plus importantes du modèle** et sur le score. Cette
liste est calculée à l'export, à partir de l'importance des variables du LightGBM.

**Comment lire le dashboard :**

| Indicateur | Lecture | Alerte si |
|---|---|---|
| Taux d'erreur | Part des requêtes en 4xx ou 5xx. Un pic de 422 signale souvent un client de l'API mal intégré | > 5 % |
| Latence API / inférence (p50, p95) | p95 = temps sous lequel passent 95 % des requêtes | p95 > 200 ms |
| Distribution des scores | Histogramme production contre entraînement. Un décalage vers la droite signifie plus de clients jugés risqués | PSI > 0,2 |
| Taux de refus | À comparer avec celui de la référence | ± 10 points |
| PSI | < 0,1 : stable ; 0,1 à 0,2 : à surveiller ; > 0,2 : dérive significative | > 0,2 |
| Onglet Data drift | Distribution de chaque variable surveillée, tableau Evidently, **dérive par fenêtre temporelle** | > 30 % de variables en dérive |

L'analyse par fenêtres est essentielle : une dérive récente peut être noyée dans l'agrégat global.

Pour aller plus loin :
- [`docs/etude_data_drift.md`](docs/etude_data_drift.md) : étude de la dérive et points de vigilance ;
- [`docs/rapport_optimisation.md`](docs/rapport_optimisation.md) : profiling et optimisations.

## Choix techniques

- **FastAPI** : validation par Pydantic, Swagger automatique. **Gradio** est monté sur la même
  application pour une démo sans front-end.
- **Schéma de validation généré depuis les métadonnées du modèle** : avec des centaines de
  variables, un schéma écrit à la main serait impossible à maintenir. Les règles métier ne portent
  que sur les variables clés.
- **ONNX** : le modèle servi ne dépend plus des versions de scikit-learn ou LightGBM utilisées à
  l'entraînement, et c'est le moteur le plus rapide mesuré.
- **NeonDB (PostgreSQL) en production, SQLite en local** : stockage séparé du calcul, consultable à tout moment ; SQLite garde un lancement local sans infrastructure.
- **SQLite + JSON lines** : un PoC sans infrastructure. Le schéma est directement transposable à
  PostgreSQL, et les logs peuvent être collectés par Fluentd ou Logstash.
- **Evidently + Streamlit** : tests statistiques adaptés au type de chaque variable, et dashboard
  interactif écrit en Python.
- **Render** : hébergement Docker gratuit, déploiement déclenché par la CI après les tests.
