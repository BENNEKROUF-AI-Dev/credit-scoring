# Étude de la dérive des données (data drift) en production

Modèle analysé : **LightGBM `scoring_credit` v32** (MLflow), 1 017 colonnes, seuil métier 0,52.

## 1. Dispositif

| Élément | Choix |
|---|---|
| Données stockées | Pour chaque appel : request_id, horodatage, endpoint, code HTTP, **dossier reçu**, **score**, décision, seuil, version du modèle, moteur, **temps d'inférence**, **latence totale**, erreur |
| Stockage | Logs JSON structurés (`data/predictions.jsonl`), prêts pour Fluentd ou Logstash, **et** table SQLite `predictions` requêtable en SQL (schéma transposable à PostgreSQL) |
| Référence | 5 000 dossiers du jeu d'entraînement avec leurs scores (`model/reference_data.parquet`) |
| Variables surveillées | Les **15 plus importantes** pour le modèle (importance LightGBM), plus le score : `EXT_SOURCE_1/2/3`, `AMT_CREDIT`, `DAYS_BIRTH`, `PREV_CNT_PAYMENT_MEAN`, `BUREAU_DAYS_CREDIT_MAX/MIN`, `INST_PAYMENT_DIFF_MEAN`, `AMT_GOODS_PRICE`, `DAYS_EMPLOYED`, `AMT_ANNUITY`, `INST_AMT_PAYMENT_MEAN`, `POS_COUNT`, `INST_DAYS_LATE_MAX` |
| Détection | Evidently `DataDriftPreset` (distance de Wasserstein normalisée, seuil 0,1), complété par le **PSI** du score et l'écart du taux de refus |
| Fenêtrage | Analyse globale **et** découpage en 4 fenêtres temporelles successives (environ 485 requêtes chacune) |
| Opérationnel | Taux d'erreur, latence p50/p95/p99, latences anormales (médiane + 4 × MAD) |
| Automatisation | `python -m monitoring.drift_analysis` renvoie le code 1 en cas d'alerte, donc il peut tourner en cron ou en job planifié |

**RGPD :** l'API refuse tout identifiant (`SK_ID_CURR`). Seules les variables du modèle sont
stockées. La durée de conservation est à fixer avec le DPO (90 jours glissants proposés).

## 2. Scénario simulé

2 000 appels construits à partir de 1 000 dossiers réels du jeu de test :
- **requêtes 1 à 1 000** : les dossiers sont envoyés tels quels ;
- **requêtes 1 001 à 2 000** : la clientèle dérive progressivement. `EXT_SOURCE_2` et
  `EXT_SOURCE_3` baissent, les clients rajeunissent, leurs revenus baissent de 30 % au maximum et le
  crédit demandé augmente de 50 % au maximum ;
- environ 3 % de requêtes invalides (texte au lieu d'un nombre, revenu nul, âge négatif…).

## 3. Résultats

### Indicateurs opérationnels

| Indicateur | Valeur |
|---|---|
| Requêtes | 2 000, dont 1 942 réussies |
| Taux d'erreur | **2,9 %**, uniquement des 422 (données refusées), aucune erreur 500 |
| Latence API p50 / p95 / p99 | 0,80 / 1,82 / 2,72 ms |
| Temps d'inférence p50 / p95 | 0,57 / 1,28 ms |
| Taux de refus | **36,7 %**, contre **27,5 %** sur les données d'entraînement (+9,2 points) |
| Score moyen | 0,436 |

Le taux d'erreur correspond exactement aux requêtes invalides injectées : la validation des
entrées fonctionne. La latence est très faible. La simulation tourne sur la même machine que l'API,
donc ce chiffre n'inclut pas le temps réseau.

### Dérive globale (Evidently)

5 colonnes sur 16 dérivent (**31 %**).

| Variable | Distance | Dérive |
|---|---|---|
| EXT_SOURCE_3 | 0,358 | ✅ forte |
| EXT_SOURCE_2 | 0,330 | ✅ forte |
| AMT_CREDIT | 0,275 | ✅ |
| score | 0,274 | ✅ |
| DAYS_BIRTH | 0,215 | ✅ |
| POS_COUNT | 0,088 | ❌ (proche du seuil) |
| AMT_ANNUITY | 0,077 | ❌ |
| Autres variables (historique bureau, paiements, `EXT_SOURCE_1`…) | 0,028 à 0,056 | ❌ |

**PSI global du score : 0,088.**

### Dérive par fenêtre temporelle

| Fenêtre | Requêtes | Variables en dérive | PSI du score | Score moyen |
|---|---|---|---|---|
| 1 | 1 à 486 (normales) | 6 % (`POS_COUNT`) | 0,025 | 0,391 |
| 2 | 487 à 971 (normales) | 19 % (`EXT_SOURCE_1`, `BUREAU_DAYS_CREDIT_MAX`, `INST_PAYMENT_DIFF_MEAN`) | 0,050 | 0,371 |
| 3 | 972 à 1 456 (début de la dérive) | 38 % | 0,183 | 0,457 |
| 4 | 1 457 à 2 000 (dérive forte) | 44 % | **1,618** | **0,527** |

### Alertes levées automatiquement
- Dérive **récente** des scores : PSI = 1,618 sur la dernière fenêtre, masquée dans l'agrégat global.
- Data drift sur 31 % des variables : `EXT_SOURCE_3`, `EXT_SOURCE_2`, `AMT_CREDIT`, `DAYS_BIRTH`, score.

## 4. Interprétation

1. **La dérive est bien détectée, et sur les bonnes variables.** Les quatre variables modifiées
   par la simulation et surveillées (`EXT_SOURCE_2`, `EXT_SOURCE_3`, `AMT_CREDIT`, `DAYS_BIRTH`)
   sont exactement celles qui ressortent. Les variables d'historique (bureau, paiements, demandes
   précédentes), qui n'ont pas été modifiées, restent stables.

2. **Le PSI global (0,088) cache le problème.** Sur toute la période, on conclurait à une population
   stable (PSI inférieur à 0,1). La dernière fenêtre montre pourtant une dérive massive (1,618).
   **Le monitoring doit donc se faire sur des fenêtres glissantes**, pas sur le cumul.

3. **L'impact métier est direct.** Le score moyen passe de 0,39 à 0,53, **au-dessus du seuil de
   décision (0,52)**. Le taux de refus monte de 27,5 % à 36,7 % sur l'ensemble, et bien plus sur
   la dernière fenêtre. Le métier le ressentirait tout de suite : beaucoup plus de clients refusés.

4. **Les fenêtres 1 et 2 montrent de fausses alertes.** Une à trois variables y sont marquées « en
   dérive » alors qu'aucune donnée n'a été modifiée. Avec environ 485 requêtes par fenêtre, et
   `EXT_SOURCE_1` vide dans plus de la moitié des dossiers, le bruit d'échantillonnage suffit à
   franchir le seuil. Il faut donc des fenêtres assez grandes, et **ne pas alerter sur une variable
   isolée** : il vaut mieux exiger plusieurs variables, ou une confirmation sur deux fenêtres
   successives.

5. **Une limite du choix des 15 variables.** La simulation a aussi baissé les revenus
   (`AMT_INCOME_TOTAL`), mais cette variable ne fait pas partie des 15 surveillées, donc la baisse
   n'est pas détectée. Surveiller seulement les variables les plus importantes évite les fausses
   alertes, mais peut laisser passer des changements sur les autres.

6. **Des « latences anormales » sans gravité.** 136 requêtes (7 %) sont signalées. À l'échelle de
   la milliseconde, le critère statistique (médiane + 4 × MAD) est trop sensible. Il faudrait
   ajouter un seuil absolu, par exemple n'alerter qu'au-dessus de 200 ms.

7. **Data drift ne veut pas dire baisse de performance.** Sans la vraie cible (défaut constaté
   plusieurs mois plus tard), on mesure un **risque**, pas une erreur du modèle. Il faudra joindre
   les remboursements observés aux `request_id` pour suivre l'AUC réelle et le coût métier.

## 5. Points de vigilance et plan d'action

| Signal | Seuil d'alerte | Action |
|---|---|---|
| PSI du score sur une fenêtre | > 0,2 | Analyse des variables en dérive, alerte au Lead DS |
| Part de variables surveillées en dérive | > 30 %, confirmé sur 2 fenêtres | Vérifier les sources de données, envisager un ré-entraînement |
| Écart du taux de refus | ± 10 points | Prévenir le métier (impact commercial). Ici +9,2 points : juste sous le seuil |
| Valeurs manquantes d'`EXT_SOURCE_*` | ± 5 points vs référence | Incident possible chez le fournisseur des scores externes |
| Taux d'erreur | > 5 % | Vérifier les applications qui appellent l'API |
| Latence p95 | > 200 ms | Vérifier la charge et le dimensionnement |
| Variables hors des 15 surveillées | Contrôle mensuel complet | Analyse Evidently sur toutes les variables d'origine, une fois par mois |

**Ré-entraînement :** déclenchement conditionnel, et non à date fixe. On ré-entraîne quand le PSI
du score dépasse 0,2 pendant deux semaines **et** que l'on dispose de défauts réellement observés.
Le nouveau modèle est comparé à l'actuel dans MLflow, puis exporté (`scripts/export_model.py`) et
déployé par le même pipeline CI/CD.
