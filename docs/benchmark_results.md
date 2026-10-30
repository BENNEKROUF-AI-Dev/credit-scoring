Modèle : LGBMClassifier, 63 colonnes

## Inférence unitaire (1 client, 1000 appels)

| Configuration | p50 (ms) | p95 (ms) | p99 (ms) | Gain p50 |
|---|---|---|---|---|
| v1 — préparation pandas + booster LightGBM natif | 15.624 | 24.345 | 27.182 | ×1.0 |
| préparation pandas + ONNX Runtime | 15.972 | 24.479 | 28.289 | ×1.0 |
| préparation NumPy + booster LightGBM natif | 0.064 | 0.142 | 0.388 | ×245.6 |
| final — préparation NumPy + ONNX Runtime | 0.035 | 0.052 | 0.064 | ×442.5 |

## Décomposition du temps (p50, ms)

| Étape | p50 (ms) |
|---|---|
| préparation pandas (get_dummies…) | 14.8163 |
| préparation NumPy | 0.0146 |
| modèle ONNX Runtime | 0.0185 |
| modèle booster LightGBM natif | 0.0359 |

## Latence API de bout en bout (500 requêtes HTTP)

| Version | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| v1 (préparation pandas) | 22.71 | 28.16 | 32.62 |
| final (NumPy + ONNX) | 2.00 | 2.40 | 3.17 |

## Scoring en masse (10000 clients déjà encodés)

| Moteur | ms |
|---|---|
| ONNX Runtime (1 thread) | 74.8 |
| booster LightGBM natif | 79.8 |

## Non-régression (vs modèle MLflow d'origine)

- Écart max de score : 2.16e-07
- Décisions identiques : 100.00% (1000 dossiers)

## Chargement et taille

- ONNX : chargement 9 ms
- LightGBM : chargement 12 ms
- model.onnx : 0.45 Mo
- model.txt : 0.70 Mo

Environnement : Python 3.13.18, 2 vCPU (x86_64)