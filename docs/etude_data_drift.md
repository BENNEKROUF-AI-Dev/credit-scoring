# Étude de la dérive des données (data drift) en production

## 1. Dispositif

| Élément | Choix |
|---|---|
| Données stockées | Pour chaque appel : request_id, horodatage, endpoint, code HTTP, **inputs**, **score**, décision, seuil, version du modèle, moteur, **temps d'inférence**, **latence totale**, erreur |
| Stockage | Logs JSON structurés (`data/predictions.jsonl`), une ligne par appel, prêts pour Fluentd ou Logstash, **et** table SQLite `predictions` requêtable en SQL (même schéma que PostgreSQL en cible) |
| Référence | 5 000 clients tirés du jeu d'entraînement, avec leurs scores (`model/reference_data.parquet`) |
| Détection | Evidently `DataDriftPreset` : distance de Wasserstein pour les variables numériques, Jensen-Shannon pour les catégorielles, seuil 0,1. Complété par le **PSI** du score et l'écart du taux de refus |
| Fenêtrage | Analyse globale **et** découpage en 4 fenêtres temporelles successives |
| Opérationnel | Taux d'erreur (422 et 500), latence p50/p95/p99, latences anormales (médiane + 4 × MAD) |
| Automatisation | `python -m monitoring.drift_analysis` renvoie le code 1 en cas d'alerte, donc il peut tourner en cron ou en job CI planifié |

**RGPD :** aucun identifiant client (SK_ID_CURR, nom) n'est accepté par l'API (`extra="forbid"`)
ni stocké. Seules les variables du modèle sont conservées. La durée de conservation est à fixer avec
le DPO (90 jours glissants proposés), et les logs bruts peuvent être agrégés au-delà.

**Coût de stockage :** environ 600 octets par appel en JSON, soit environ 0,6 Go pour un million de
demandes. SQLite suffit pour le PoC. En cible : PostgreSQL partitionné par mois, ou Elasticsearch
avec Kibana si le volume augmente.

## 2. Scénario simulé

2 000 appels :
- les 1 000 premiers suivent la distribution d'entraînement ;
- les 1 000 suivants dérivent progressivement : clientèle plus jeune, revenus plus faibles,
  crédits plus élevés, `EXT_SOURCE_3` plus bas, plus d'enfants ;
- 3 % des requêtes sont invalides.

## 3. Résultats

**Indicateurs opérationnels :** 3,0 % d'erreurs (toutes en 422, aucune en 500), latence
p50/p95 = 0,3/0,4 ms. Aucune alerte de ce côté.

**Dérive globale (Evidently) :** 5 colonnes sur 13 dérivent (38 %).

| Variable | Distance | Dérive |
|---|---|---|
| EXT_SOURCE_3 | 0,297 | ✅ forte |
| score | 0,248 | ✅ |
| DAYS_BIRTH | 0,132 | ✅ |
| CNT_CHILDREN | 0,129 | ✅ |
| AMT_INCOME_TOTAL | 0,111 | ✅ |
| AMT_CREDIT, AMT_ANNUITY, EXT_SOURCE_1/2, … | < 0,1 | ❌ |

**Dérive par fenêtre :**

| Fenêtre | Variables en dérive | PSI du score | Score moyen |
|---|---|---|---|
| 1 | 0 % | 0,024 | 0,295 |
| 2 | 15 % | 0,013 | 0,314 |
| 3 | 46 % | 0,105 | 0,390 |
| 4 | **77 %** | **0,498** | **0,496** |

## 4. Interprétation

1. **Le PSI global (0,059) aurait masqué le problème.** Sur l'ensemble de la période, la population
   paraît stable. Le découpage en fenêtres montre au contraire que la dernière fenêtre a un PSI de
   0,50, très au-dessus du seuil usuel de 0,2. **Le monitoring doit donc se faire sur des fenêtres
   glissantes**, pas en cumul.
2. **EXT_SOURCE_3 est la variable la plus importante qui dérive.** C'est l'une des variables les plus
   prédictives du modèle : sa baisse fait mécaniquement monter les scores (de 0,30 à 0,50 en moyenne)
   et le taux de refus (+8 points par rapport à la référence).
3. **Il faut distinguer deux cas :**
   - la population a réellement changé (nouveau canal d'acquisition, ciblage plus jeune). Le modèle
     peut rester juste, mais le risque du portefeuille augmente ;
   - la source externe a changé (nouveau fournisseur, recalibrage). C'est un **problème de données**
     et le modèle devient biaisé.

   Seul un échange avec le métier ou le fournisseur permet de trancher.
4. **Data drift ne veut pas dire baisse de performance.** Sans la vraie cible (défaut constaté
   plusieurs mois plus tard), on ne mesure qu'un risque. Il faudra joindre les remboursements
   observés aux `request_id` pour suivre l'AUC réelle et le coût métier (concept drift).

## 5. Points de vigilance et plan d'action

| Signal | Seuil d'alerte | Action |
|---|---|---|
| PSI du score sur une fenêtre | > 0,2 | Analyse des variables en dérive, alerte au Lead DS |
| Part de variables en dérive | > 30 % | Vérifier les pipelines amont, envisager un ré-entraînement |
| Écart du taux de refus | ± 10 points | Prévenir le métier (impact commercial) |
| Taux d'erreur | > 5 % | Vérifier les clients de l'API et l'évolution du format d'entrée |
| Latence p95 | > 200 ms | Vérifier la charge et le dimensionnement (workers, CPU) |
| Valeurs manquantes d'EXT_SOURCE_* | ± 5 points vs référence | Incident chez le fournisseur de données |

**Stratégie de ré-entraînement proposée :** déclenchement conditionnel, et non périodique aveugle,
quand le PSI du score dépasse 0,2 pendant 2 semaines **et** que l'on dispose d'au moins 3 mois de
cibles observées. Le nouveau modèle est comparé au champion dans MLflow, puis déployé par le même
pipeline CI/CD.
