# Rapport d'optimisation post-déploiement

Modèle optimisé : **LightGBM `scoring_credit` v32** (MLflow), 1 017 colonnes après encodage,
887 champs en entrée de l'API.

**Résultat en une phrase :** la latence de l'API passe de **186 ms à 4,9 ms** (38 fois plus rapide),
et la prédiction seule devient **1 390 fois plus rapide**, avec **100 % de décisions identiques**.

## 1. Point de départ

Première version (v1) : la préparation des données reproduit le notebook 2 avec **pandas**. Pour un
client, on crée un `DataFrame` d'une ligne, on applique `get_dummies`, on renomme les colonnes puis
on les réaligne sur les 1 017 colonnes du modèle. Le modèle est ensuite appelé.

| Indicateur (v1) | Valeur |
|---|---|
| Latence API p50 / p95 / p99 | **186 ms / 780 ms / 1 061 ms** |
| Préparation + modèle, p50 | 156 ms |

**Constat :** une demande sur vingt met près de 0,8 seconde, et la pire dépasse une seconde. Pour un
service de crédit en temps réel, c'est trop lent et imprévisible. La quasi-totalité du temps est passée
dans notre code Python, pas dans le réseau ni dans FastAPI.

## 2. Profiling (cProfile)

`python -m scripts.benchmark` lance cProfile sur 300 prédictions de la version v1 (sortie complète :
`docs/profiling_v1.txt`).

| Fonction | Temps cumulé (300 appels) | Part |
|---|---|---|
| Chemin complet d'une prédiction | 120,4 s | 100 % |
| `transform_pandas` (préparation des données) | 118,8 s | **99 %** |
| dont ajout et modification de colonnes (`DataFrame.__setitem__`) | 78,1 s | 65 % |
| dont `pd.get_dummies` | 12,6 s | 10 % |

Décomposition mesurée (p50, un client) :

| Étape | p50 |
|---|---|
| Préparation avec pandas | 161,3 ms |
| Préparation avec NumPy | 0,096 ms |
| Modèle, booster LightGBM natif | 0,029 ms |
| Modèle, ONNX Runtime | 0,011 ms |

**Goulot d'étranglement : la préparation des données avec pandas.** Pandas est conçu pour traiter
beaucoup de lignes à la fois. Sur **une seule ligne**, chaque colonne ajoutée coûte des dizaines de
microsecondes de gestion interne (index, types, réorganisation de la mémoire). Avec plus de 1 000
colonnes, ces petits coûts s'additionnent : pandas affiche même un avertissement de fragmentation
(`PerformanceWarning: DataFrame is highly fragmented`). Le calcul du modèle, lui, ne pèse presque rien
(0,01 à 0,03 ms).

## 3. Stratégies testées

| # | Stratégie | p50 (ms) | p95 (ms) | p99 (ms) | Gain p50 |
|---|---|---|---|---|---|
| v1 | Préparation pandas + booster LightGBM | 156,12 | 288,87 | 919,98 | ×1 |
| A | Préparation pandas + ONNX Runtime | 170,17 | 322,75 | 1 020,33 | ×0,9 |
| B | Préparation NumPy + booster LightGBM | 0,191 | 0,312 | 0,343 | ×818 |
| **Finale** | **Préparation NumPy + ONNX Runtime (1 thread)** | **0,112** | **0,161** | **0,253** | **×1 390** |

- **Optimisation du code : la préparation en NumPy.** À l'export du modèle, je calcule **une seule
  fois** la correspondance entre chaque champ d'entrée (ou chaque catégorie) et sa position dans les
  1 017 colonnes du modèle. À chaque requête, il suffit alors de remplir un tableau NumPy déjà créé,
  sans `DataFrame` ni `get_dummies`. C'est le gain principal. La version pandas est gardée comme
  référence lisible, et un test vérifie que les deux donnent exactement le même résultat.
- **Changer seulement le moteur (stratégie A) n'apporte rien**, et c'est même un peu plus lent, tant
  que pandas domine. C'est la leçon principale : **il faut mesurer avant d'optimiser**, sinon on
  optimise la mauvaise étape.
- **ONNX Runtime** : une fois la préparation corrigée, il calcule environ 2,6 fois plus vite que
  LightGBM (0,011 contre 0,029 ms). Il rend aussi l'API indépendante des versions de scikit-learn et
  LightGBM utilisées à l'entraînement, ce qui évite les problèmes de compatibilité au déploiement.
- **Configuration du runtime** : `intra_op_num_threads=1`. Pour une requête unitaire, le
  parallélisme coûte plus en coordination qu'il ne rapporte, et il crée de la concurrence entre
  processus quand plusieurs workers tournent.
- **Quantification : écartée.** La quantification INT8 réduit la précision des opérations de calcul
  matriciel (réseaux de neurones). Un modèle LightGBM est un ensemble d'arbres de décision : il n'y a
  rien à quantifier, et arrondir les seuils des arbres risquerait de changer des décisions.
- **Scoring en masse (10 000 clients déjà encodés)** : ONNX 57 ms, LightGBM 76 ms. En batch, les deux
  moteurs restent proches. L'optimisation vise surtout le **temps réel**, qui est le besoin du service.

## 4. Impact sur l'API (de bout en bout, 500 requêtes HTTP)

| Version | p50 | p95 | p99 |
|---|---|---|---|
| v1 (préparation pandas) | 186,21 ms | 779,93 ms | 1 061,12 ms |
| **Finale (NumPy + ONNX)** | **4,86 ms** | **5,97 ms** | **6,96 ms** |

- La latence médiane est **divisée par 38**, et la latence p99 par **152**.
- L'API devient surtout **prévisible** : l'écart entre p50 et p99 passe de 875 ms à 2 ms.
- Ce qui reste (environ 5 ms) vient surtout de la validation Pydantic des 887 champs, du protocole
  HTTP et de l'écriture du log, et non plus du modèle.
- Le monitoring en production le confirme : temps d'inférence p50 de 0,57 ms sur 2 000 requêtes
  (`docs/etude_data_drift.md`).

## 5. Non-régression : précision et biais

- Écart maximal de score par rapport au modèle MLflow d'origine : **1,6 × 10⁻⁷**, dû au passage des
  nombres en 32 bits pour ONNX.
- **100 % des décisions identiques** au seuil métier de 0,52, sur 1 000 dossiers du jeu de test.
- Le modèle lui-même n'a pas changé, seul le moteur d'exécution a changé : aucun biais nouveau n'est
  introduit.
- **Garde-fous automatiques** dans la CI : les tests `test_scores_reproduce_mlflow_model`,
  `test_fast_transform_matches_pandas` et `test_api_reproduces_notebook_scores` bloquent le
  déploiement en cas d'écart.

## 6. Configuration finale retenue

| Élément | Choix | Justification |
|---|---|---|
| Moteur d'inférence | ONNX Runtime (CPUExecutionProvider) | Le plus rapide, indépendant des versions d'entraînement |
| Préparation en ligne | NumPy, correspondance des colonnes calculée à l'export | Supprime 99 % du temps de prédiction |
| Threads ONNX | 1 par session | Meilleure latence unitaire, pas de concurrence entre processus |
| Serveur | uvicorn, 1 worker | Modèle chargé une seule fois au démarrage (`lifespan`) |
| Hébergement | Render, offre gratuite (512 Mo, CPU partagé) | Environ 300 Mo de RAM utilisés, interface Gradio désactivée en ligne |
| Matériel | CPU seulement | Un modèle à base d'arbres ne profite pas d'un GPU ; p99 sous 7 ms |

Retour arrière possible sans reconstruire l'image : variables `MODEL_BACKEND=lightgbm` et/ou
`FAST_FEATURES=0`.

*Mesures réalisées avec `python -m scripts.benchmark` sous Python 3.12, sur une machine Linux à
2 vCPU, comparable au serveur de production. Résultats bruts : `docs/benchmark_results.md` et
`docs/benchmark_results.json`.*
