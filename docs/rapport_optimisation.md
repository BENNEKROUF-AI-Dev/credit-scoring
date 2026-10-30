# Rapport d'optimisation post-déploiement

> ⚠️ **À mettre à jour avec vos chiffres.** Les valeurs ci-dessous ont été obtenues sur un jeu de
> données de démonstration au format de vos notebooks : 63 colonnes, contre plusieurs centaines pour
> votre vrai modèle. Après l'export de votre modèle, relancez `python -m scripts.benchmark` et
> reportez les résultats de `docs/benchmark_results.md`. Plus votre modèle a de colonnes, plus le
> gain de l'étape 3 sera important.

## 1. Point de départ : ce que montrait le monitoring

Première version (v1) : la préparation des données reproduit le notebook 2 avec **pandas**
(`pd.DataFrame` d'une ligne, puis `get_dummies`, renommage et réalignement des colonnes). Le
modèle est ensuite appelé. Mesures sur l'API :

| Indicateur (v1) | Valeur |
|---|---|
| Latence API p50 / p95 | 22,7 ms / 28,2 ms |
| Temps d'inférence p50 (préparation + modèle) | environ 15,6 ms |

**Constat :** le temps d'inférence représente environ 70 % de la latence. Le goulot d'étranglement
est dans notre code, pas dans le réseau ni dans FastAPI.

## 2. Profiling (cProfile)

`python -m scripts.benchmark` lance cProfile sur 300 prédictions (sortie complète :
`docs/profiling_v1.txt`).

| Fonction | Temps cumulé | Part |
|---|---|---|
| Chemin complet d'une prédiction | 11,6 s | 100 % |
| `transform_pandas` (préparation) | 11,4 s | **98 %** |
| dont `pd.get_dummies` | 3,7 s | 32 % |
| dont création et affectation de colonnes (`__setitem__`) | 4,2 s | 36 % |

Décomposition mesurée (p50, un client) :

| Étape | p50 |
|---|---|
| Préparation pandas | 14,8 ms |
| Préparation NumPy | 0,015 ms |
| Modèle, booster LightGBM natif | 0,036 ms |
| Modèle, ONNX Runtime | 0,019 ms |

**Goulot d'étranglement :** pandas, sur une seule ligne. Chaque création de colonne et chaque appel
à `get_dummies` coûte des dizaines de µs d'overhead (index, vérifications de type). Avec des
centaines de colonnes, ce coût se multiplie. Le calcul sur les arbres du modèle ne pèse presque
rien.

## 3. Stratégies testées

| # | Stratégie | p50 (ms) | p95 (ms) | Gain p50 |
|---|---|---|---|---|
| v1 | Préparation pandas + booster LightGBM | 15,62 | 24,35 | ×1 |
| A | Préparation pandas + ONNX Runtime | 15,97 | 24,48 | ×1 |
| B | Préparation NumPy + booster LightGBM | 0,064 | 0,142 | ×246 |
| **Finale** | **Préparation NumPy + ONNX Runtime (1 thread)** | **0,035** | **0,052** | **×443** |

- **Optimisation de code** (`Preprocessor.transform`) : la correspondance entre une variable (ou une
  catégorie) et une colonne du modèle est calculée **une seule fois**, à l'export. À chaque requête,
  on remplit simplement un tableau NumPy pré-alloué. C'est le gain principal. La version pandas est
  conservée comme référence lisible, et un test vérifie que les deux sont identiques.
- **ONNX Runtime** : il divise encore le temps du modèle par deux. Il rend aussi l'API indépendante
  des versions de scikit-learn et LightGBM utilisées à l'entraînement, ce qui élimine les problèmes
  de compatibilité au déploiement.
- **Changer seulement le moteur (stratégie A) n'apporte rien** tant que la préparation domine. Il
  faut d'abord mesurer avant d'optimiser.
- **Configuration du runtime** : `intra_op_num_threads=1`. Pour une requête unitaire, le
  parallélisme coûte plus qu'il ne rapporte, et il crée de la contention avec plusieurs workers.
- **Quantification : écartée.** La quantification INT8 s'applique aux opérateurs linéaires et
  convolutifs, pas aux ensembles d'arbres. Il n'y a rien à quantifier, et réduire la précision des
  seuils des arbres risquerait de modifier des décisions.
- **Scoring en masse (10 000 dossiers)** : ONNX 75 ms, LightGBM 80 ms. En batch, les moteurs se
  valent. L'optimisation vise le temps réel, qui est le besoin de « Crédit Express ».

## 4. Impact sur l'API (de bout en bout, 500 requêtes HTTP)

| Version | p50 | p95 | p99 |
|---|---|---|---|
| v1 (préparation pandas) | 22,71 ms | 28,16 ms | 32,62 ms |
| **Finale (NumPy + ONNX)** | **2,00 ms** | **2,40 ms** | **3,17 ms** |

La latence est **divisée par 11**. Ce qui reste est surtout de l'overhead HTTP, de la validation
Pydantic (plusieurs centaines de champs) et de l'écriture du log.

## 5. Non-régression (précision, biais)

- Écart maximal de score par rapport au modèle MLflow d'origine : 2·10⁻⁷ (conversion en float32).
- **100 % des décisions identiques** au seuil métier, sur 1 000 dossiers du jeu de test.
- Le modèle n'a pas changé, seul le moteur d'exécution a changé : aucun biais nouveau n'est
  introduit.
- Garde-fous automatiques dans la CI : les tests `test_scores_reproduce_mlflow_model`,
  `test_fast_transform_matches_pandas` et `test_api_reproduces_notebook_scores` bloquent le
  déploiement en cas d'écart.

## 6. Configuration finale retenue

| Élément | Choix | Justification |
|---|---|---|
| Moteur | ONNX Runtime (CPUExecutionProvider) | Le plus rapide, indépendant des versions d'entraînement |
| Préparation en ligne | NumPy, correspondance des colonnes pré-calculée | Supprime 98 % du temps |
| Threads ONNX | 1 par session | Meilleure latence unitaire |
| Serveur | uvicorn, 1 worker par vCPU | Modèle chargé une fois par worker (`lifespan`) |
| Hardware | CPU seulement, 2 vCPU | Les modèles à base d'arbres ne profitent pas d'un GPU ; p99 inférieur à 4 ms |

Retour arrière possible sans reconstruire l'image : `MODEL_BACKEND=lightgbm` et/ou `FAST_FEATURES=0`.
