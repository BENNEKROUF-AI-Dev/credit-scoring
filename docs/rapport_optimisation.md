# Rapport d'optimisation post-déploiement

## 1. Point de départ : ce que montrait le monitoring

Première version déployée (v1) : feature engineering avec **pandas**, puis modèle LightGBM appelé
via `predict_proba` de scikit-learn. Le dashboard de monitoring montrait alors, sur 1 500 requêtes
simulées :

| Indicateur (v1) | Valeur |
|---|---|
| Latence API p50 / p95 | 7,9 ms / 10,9 ms |
| Temps d'inférence p50 | 7,2 ms |

**Constat :** le temps d'inférence représentait plus de 90 % de la latence de l'API. Le goulot
d'étranglement est donc dans le code Python de prédiction, pas dans le réseau ni dans FastAPI.

## 2. Profiling (cProfile)

`python -m scripts.benchmark` lance cProfile sur 300 prédictions unitaires (sortie complète :
`docs/profiling_v1.txt`).

| Fonction | Temps cumulé | Part |
|---|---|---|
| Chemin complet d'une prédiction | 5,38 s | 100 % |
| `build_features` (pandas) | 4,26 s | **79 %** |
| `LGBMClassifier.predict_proba` (sklearn) | 0,73 s | 14 % |

La décomposition mesurée confirme (p50 pour **un** client) :

| Étape | p50 |
|---|---|
| Features avec pandas | 6,41 ms |
| Features avec NumPy | 0,055 ms |
| Modèle via `predict_proba` sklearn (DataFrame d'une ligne) | 1,48 ms |
| Modèle via le booster LightGBM natif | 0,050 ms |
| Modèle via ONNX Runtime | 0,030 ms |

**Goulots d'étranglement identifiés :**
1. **pandas sur une seule ligne** : chaque création ou sélection de colonne coûte environ 50 à 100 µs
   d'overhead (index, vérifications de type, options). C'est négligeable en batch mais domine en
   temps réel.
2. **Le wrapper sklearn de LightGBM** : il valide le DataFrame et convertit les données à chaque appel.
   Le calcul sur les arbres ne prend lui-même que quelques dizaines de µs.

## 3. Stratégies testées

| # | Stratégie | p50 (ms) | p95 (ms) | Gain p50 |
|---|---|---|---|---|
| v1 | pandas + `predict_proba` sklearn | 7,68 | 11,00 | ×1 |
| A | pandas + booster LightGBM natif | 6,58 | 9,87 | ×1,2 |
| B | pandas + ONNX Runtime | 5,82 | 9,07 | ×1,3 |
| C | NumPy + booster LightGBM natif | 0,113 | 0,226 | ×68 |
| **Finale** | **NumPy + ONNX Runtime (1 thread)** | **0,072** | **0,136** | **×106** |

- **Optimisation de code** (`build_features_fast`) : réécriture du feature engineering en NumPy pur,
  sans DataFrame. C'est le gain principal. La version pandas reste utilisée pour l'entraînement et le
  scoring en batch.
- **ONNX Runtime** : conversion avec `onnxmltools` (sortie sous forme de tenseur, sans ZipMap). On gagne
  encore environ 40 % sur le calcul du modèle. Le fichier est aussi plus léger (0,92 Mo contre 1,39 Mo).
- **Configuration du runtime** : `intra_op_num_threads=1`. Pour une requête unitaire, le
  parallélisme coûte plus en synchronisation qu'il ne rapporte, et il crée de la contention quand
  plusieurs workers uvicorn tournent sur la même machine.
- **Quantification : écartée.** La quantification INT8 d'ONNX Runtime s'applique aux opérateurs
  linéaires et convolutifs (MatMul, Conv), pas aux ensembles d'arbres (`TreeEnsembleClassifier`).
  Il n'y a rien à quantifier, et passer les seuils des arbres en basse précision risquerait de
  modifier des décisions.
- **Scoring en masse (10 000 clients)** : sklearn 142 ms, booster natif 140 ms, ONNX 149 ms. En batch,
  le calcul vectorisé domine et les trois moteurs se valent. **L'optimisation ne vise donc que le
  temps réel**, qui est le besoin exprimé par « Crédit Express ».

## 4. Impact sur l'API (de bout en bout, 500 requêtes HTTP)

| Version | p50 | p95 | p99 |
|---|---|---|---|
| v1 (pandas + LightGBM) | 10,97 ms | 12,89 ms | 14,90 ms |
| **Finale (NumPy + ONNX)** | **1,77 ms** | **2,20 ms** | **2,69 ms** |

La latence est **divisée par 6**. Ce qui reste est surtout de l'overhead HTTP, de la validation
Pydantic et de l'écriture du log. En production (données de monitoring après redéploiement), le
temps d'inférence p50 est passé de **7,2 ms à 0,28 ms**.

## 5. Non-régression (précision, biais)

Mesurée sur 20 000 clients :
- AUC identique : 0,7734 contre 0,7734
- Écart maximal de score : 4,6·10⁻⁷ (conversion en float32)
- **100 % des décisions identiques** au seuil métier

Le modèle n'a pas changé, seul le moteur d'exécution a changé, donc aucun biais nouveau ne peut être
introduit. Des tests automatiques (`tests/test_model.py`) bloquent le pipeline CI/CD si :
- les features NumPy diffèrent des features pandas ;
- les scores ONNX s'écartent de plus de 1e-4 de ceux de LightGBM ;
- une décision change.

## 6. Configuration finale retenue et justification

| Élément | Choix | Justification |
|---|---|---|
| Moteur d'inférence | ONNX Runtime 1.29 (CPUExecutionProvider) | Le plus rapide en unitaire, portable, sans dépendance à LightGBM pour servir |
| Features en ligne | NumPy (`build_features_fast`) | Supprime 79 % du temps d'inférence |
| Threads ONNX | 1 par session | Meilleure latence unitaire, pas de contention |
| Serveur | uvicorn, 1 worker par vCPU | Le modèle (environ 1 Mo) est chargé une fois par worker au démarrage (`lifespan`) |
| Hardware | CPU seulement (2 vCPU suffisent) | Le modèle à base d'arbres ne profite pas d'un GPU ; p99 inférieur à 3 ms |
| Image | `python:3.12-slim` + utilisateur non-root | Image légère, compatible Hugging Face Spaces |

Retour arrière possible sans reconstruire l'image : `MODEL_BACKEND=lightgbm` et/ou `FAST_FEATURES=0`.

*Chiffres obtenus avec Python 3.13 sur 2 vCPU x86_64. Pour les reproduire : `python -m scripts.benchmark`.*
