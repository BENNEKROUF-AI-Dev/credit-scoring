# Étude de la dérive des données (data drift) en production

> ⚠️ **À mettre à jour avec vos chiffres.** Les résultats ci-dessous viennent d'un jeu de
> démonstration. Après l'export de votre modèle, relancez la simulation et l'analyse (README,
> section Monitoring), puis reportez les valeurs de `monitoring/reports/summary.json`. Rédigez
> l'interprétation avec vos propres mots : c'est elle que l'évaluateur jugera.

## 1. Dispositif

| Élément | Choix |
|---|---|
| Données stockées | Pour chaque appel : request_id, horodatage, endpoint, code HTTP, **dossier reçu**, **score**, décision, seuil, version du modèle, moteur, **temps d'inférence**, **latence totale**, erreur |
| Stockage | Logs JSON structurés (`data/predictions.jsonl`), prêts pour Fluentd ou Logstash, **et** table SQLite `predictions` requêtable en SQL (schéma transposable à PostgreSQL) |
| Référence | 5 000 dossiers du jeu d'entraînement avec leurs scores (`model/reference_data.parquet`) |
| Variables surveillées | Les **15 plus importantes** pour le modèle (importance LightGBM, catégories regroupées par variable d'origine), plus le score. Surveiller les centaines de variables ferait exploser les fausses alertes |
| Détection | Evidently `DataDriftPreset` : distance de Wasserstein pour les variables numériques, Jensen-Shannon pour les catégorielles, seuil 0,1. Complété par le **PSI** du score et l'écart du taux de refus |
| Fenêtrage | Analyse globale **et** découpage en 4 fenêtres temporelles successives |
| Opérationnel | Taux d'erreur (422 et 500), latence p50/p95/p99, latences anormales (médiane + 4 × MAD) |
| Automatisation | `python -m monitoring.drift_analysis` renvoie le code 1 en cas d'alerte, donc il peut tourner en cron ou en job planifié |

**RGPD :** l'API refuse tout identifiant (`SK_ID_CURR`). Seules les variables du modèle sont
stockées. La durée de conservation est à fixer avec le DPO (90 jours glissants proposés).

**Coût de stockage :** un dossier complet fait plusieurs Ko en JSON, soit quelques Go par million
de demandes. En cible, on peut ne stocker que les variables surveillées et le score, et archiver les
dossiers complets en stockage froid.

## 2. Scénario simulé

2 000 appels construits à partir de dossiers réels du jeu de test :
- les 1 000 premiers sont envoyés tels quels ;
- les 1 000 suivants dérivent progressivement : `EXT_SOURCE_2` et `EXT_SOURCE_3` plus bas, clients
  plus jeunes, revenus plus faibles, crédits plus élevés ;
- 3 % des requêtes sont invalides.

## 3. Résultats (jeu de démonstration)

**Indicateurs opérationnels :** 3,0 % d'erreurs (toutes en 422, aucune en 500), latence
p50/p95 = 0,2/0,3 ms.

**Dérive globale (Evidently) :** 3 colonnes sur 16 dérivent (19 %).

| Variable | Distance | Dérive |
|---|---|---|
| EXT_SOURCE_3 | 0,447 | ✅ forte |
| EXT_SOURCE_2 | 0,447 | ✅ forte |
| score | 0,362 | ✅ |
| DAYS_BIRTH | 0,075 | ❌ (proche du seuil) |
| Variables d'historique (INST, PREV, BB…) | < 0,09 | ❌ |

**Dérive par fenêtre :**

| Fenêtre | Variables en dérive | PSI du score | Score moyen |
|---|---|---|---|
| 1 | 12 % (bruit) | 0,038 | 0,300 |
| 2 | 12 % (bruit) | 0,008 | 0,311 |
| 3 | 25 % | 0,232 | 0,410 |
| 4 | 25 % | **1,409** | **0,553** |

**Alertes levées :** dérive récente des scores (PSI 1,41 sur la dernière fenêtre) et taux de refus
de 43 % contre 28 % en référence.

## 4. Interprétation

1. **Le PSI global (0,14) sous-estime le problème.** Sur l'ensemble de la période, on serait
   seulement en zone « à surveiller ». La dernière fenêtre montre pourtant une dérive massive. **Le
   monitoring doit se faire sur des fenêtres glissantes.**
2. **Les variables en dérive sont les plus importantes du modèle.** Une baisse des `EXT_SOURCE` fait
   mécaniquement monter les scores (de 0,30 à 0,55) et le taux de refus (+15 points). Le métier le
   ressentirait immédiatement.
3. **Les variables d'historique (bureau, demandes précédentes) restent stables.** Cela oriente le
   diagnostic vers les scores externes : changement de fournisseur ? recalibrage ? évolution réelle
   de la clientèle ?
4. **Les fenêtres 1 et 2 montrent 1 à 2 variables « en dérive » sans aucune perturbation.** C'est du
   bruit d'échantillonnage. Il faut donc des fenêtres assez grandes, et ne pas alerter sur une seule
   variable isolée.
5. **Data drift ne veut pas dire baisse de performance.** Sans la vraie cible (défaut constaté
   plusieurs mois plus tard), on ne mesure qu'un risque. Il faudra joindre les remboursements
   observés aux `request_id` pour suivre l'AUC réelle et le coût métier.

## 5. Points de vigilance et plan d'action

| Signal | Seuil d'alerte | Action |
|---|---|---|
| PSI du score sur une fenêtre | > 0,2 | Analyse des variables en dérive, alerte au Lead DS |
| Part de variables surveillées en dérive | > 30 % | Vérifier les pipelines amont, envisager un ré-entraînement |
| Écart du taux de refus | ± 10 points | Prévenir le métier (impact commercial) |
| Taux d'erreur | > 5 % | Vérifier les clients de l'API et l'évolution du format d'entrée |
| Latence p95 | > 200 ms | Vérifier la charge et le dimensionnement |
| Part de dossiers sans historique (`BUREAU_COUNT` absent) | ± 5 points | Changement du canal d'acquisition |

**Ré-entraînement :** déclenchement conditionnel (PSI > 0,2 pendant 2 semaines **et** cibles
observées disponibles). Le nouveau modèle est comparé au champion dans MLflow, puis exporté
(`scripts/export_model.py`) et déployé par le même pipeline CI/CD.
