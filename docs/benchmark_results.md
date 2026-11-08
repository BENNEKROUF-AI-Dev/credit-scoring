Modèle : LGBMClassifier, 1017 colonnes

## Inférence unitaire (1 client, 1000 appels)

| Configuration | p50 (ms) | p95 (ms) | p99 (ms) | Gain p50 |
|---|---|---|---|---|
| v1 — préparation pandas + booster LightGBM natif | 156.120 | 288.865 | 919.978 | ×1.0 |
| préparation pandas + ONNX Runtime | 170.171 | 322.751 | 1020.326 | ×0.9 |
| préparation NumPy + booster LightGBM natif | 0.191 | 0.312 | 0.343 | ×818.2 |
| final — préparation NumPy + ONNX Runtime | 0.112 | 0.161 | 0.253 | ×1389.7 |

## Décomposition du temps (p50, ms)

| Étape | p50 (ms) |
|---|---|
| préparation pandas (get_dummies…) | 161.3358 |
| préparation NumPy | 0.0963 |
| modèle ONNX Runtime | 0.0113 |
| modèle booster LightGBM natif | 0.0294 |

## Latence API de bout en bout (500 requêtes HTTP)

| Version | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| v1 (préparation pandas) | 186.21 | 779.93 | 1061.12 |
| final (NumPy + ONNX) | 4.86 | 5.97 | 6.96 |

## Scoring en masse (10000 clients déjà encodés)

| Moteur | ms |
|---|---|
| ONNX Runtime (1 thread) | 56.7 |
| booster LightGBM natif | 76.4 |

## Non-régression (vs modèle MLflow d'origine)

- Écart max de score : 1.61e-07
- Décisions identiques : 100.00% (1000 dossiers)

## Chargement et taille

- ONNX : chargement 6 ms
- LightGBM : chargement 10 ms
- model.onnx : 0.11 Mo
- model.txt : 0.23 Mo

Environnement : Python 3.12.3, 2 vCPU (x86_64)