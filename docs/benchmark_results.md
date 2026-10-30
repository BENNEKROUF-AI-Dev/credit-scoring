## Inférence unitaire (1 client, 1000 appels)

| Configuration | p50 (ms) | p95 (ms) | p99 (ms) | Gain p50 |
|---|---|---|---|---|
| v1 — pandas + sklearn predict_proba | 7.684 | 11.004 | 11.543 | ×1.0 |
| pandas + booster LightGBM natif | 6.582 | 9.869 | 11.222 | ×1.2 |
| pandas + ONNX Runtime | 5.816 | 9.071 | 9.866 | ×1.3 |
| NumPy + booster LightGBM natif | 0.113 | 0.226 | 0.274 | ×68.3 |
| final — NumPy + ONNX Runtime | 0.072 | 0.136 | 0.174 | ×106.2 |

## Décomposition du temps (p50, ms)

| Étape | p50 (ms) |
|---|---|
| features pandas | 6.4125 |
| features NumPy | 0.0545 |
| modèle sklearn predict_proba (DataFrame 1 ligne) | 1.4821 |
| modèle booster natif | 0.0499 |
| modèle ONNX Runtime | 0.0300 |

## Latence API de bout en bout (500 requêtes HTTP)

| Version | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| v1 (pandas + LightGBM) | 10.97 | 12.89 | 14.90 |
| final (NumPy + ONNX) | 1.77 | 2.20 | 2.69 |

## Scoring en masse (10 000 clients)

| Moteur | ms |
|---|---|
| sklearn | 142.3 |
| booster natif | 139.7 |
| ONNX Runtime (1 thread) | 148.8 |

## Non-régression

- AUC sklearn : 0.7734 — AUC version finale : 0.7734
- Écart max de score : 4.56e-07
- Décisions identiques : 100.00%

## Chargement et taille

- LightGBM (joblib) : chargement 7 ms
- ONNX : chargement 22 ms
- model.joblib : 1.39 Mo
- model.onnx : 0.92 Mo

Environnement : Python 3.13.18, 2 vCPU (x86_64)