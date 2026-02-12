# FFCK Slalom – Évaluation relative (prototype)

Application Streamlit d’aide à l’évaluation des performances en slalom, orientée sélection / commission.
L’objectif est de comparer les performances **entre courses** et **entre catégories** en tenant compte du contexte (qualité et densité de course), à partir de données de résultats.

Ce dépôt utilise uniquement des **données simulées** (aucune donnée athlète réelle).

## Fonctionnalités

- Sélection d’une course via `event_id`, `run_id`, `bassin`
- Filtres au même niveau : **Embarcation** (Tous/Kayak/Canoë) et **Sexe** (Tous/Homme/Femme)
- Indicateurs (par catégorie) :
  - **Temps final** (affiché en mm:ss.cc)
  - **Écart au meilleur** : en secondes et en %
  - **IQC** (Indice de Qualité de Course) : compare la performance du meilleur temps à une référence bassin (si fournie)
  - **Densité de course** (normalisée) : donne une idée de la “compétitivité” (temps serrés vs dispersés)
  - **Ratios inter-catégories** : ex. K1H/C1H, K1F/C1H, C1F/C1H, avec comparaison à un historique si fourni
- Onglet **Visuels** (Plotly) :
  - Distribution des temps
  - “Top N vs reste” (écart % au meilleur)
  - Ratios today vs historique (+ delta)

## Fichiers attendus (CSV)

### 1) Résultats courses (obligatoire)
Colonnes minimales :
- `event_id`, `date`, `bassin`, `run_id`, `categorie`, `athlete`, `time_final`
Optionnel :
- `status` (si absent, l’app considère OK quand `time_final` est renseigné)

Valeurs de `categorie` (prototype) :
- `K1H`, `K1F`, `C1H`, `C1F`

### 2) Références bassin (optionnel)
Colonnes :
- `bassin`, `categorie_ref`, `time_ref`
Optionnel :
- `year_ref`, `source`

### 3) Ratios historiques (optionnel)
Colonnes :
- `bassin`, `ratio_name`, `ratio_value`
Optionnel :
- `ratio_sd`, `n_events`

## Lancer en local

### Prérequis
- Python 3.10+ recommandé
- Dépendances : `streamlit`, `pandas`, `numpy`, `plotly`

### Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt
