# Courbes d'apprentissage — Mémoire ML

Étude comparative de 6 algorithmes de machine learning via des courbes d'apprentissage,
sur des tâches de **classification** (3 datasets) et de **régression** (3 datasets).

---

## Structure du projet

```
ml code/
├── classification/
│   ├── main.py          ← point d'entrée classification
│   ├── config.py        ← paramètres globaux (seed, splits, CV, seuil de convergence)
│   ├── data.py          ← chargement des datasets (UCI + sklearn)
│   ├── models.py        ← algorithmes et grilles d'hyperparamètres
│   ├── experiment.py    ← boucle expérimentale principale
│   ├── metrics.py       ← calcul des métriques
│   ├── visualization.py ← génération des figures et tableaux CSV
│   ├── results/         ← résultats mis en cache (.pkl) + CSV de synthèse
│   └── figures/         ← courbes d'apprentissage générées (.png)
│
├── regression/
│   ├── main.py          ← point d'entrée régression
│   └── ...              ← même structure que classification/
│
├── requirements.txt
└── README.md
```

---

## Prérequis

- **Python 3.9 ou supérieur**
- Connexion internet (téléchargement des datasets UCI et authentification TabPFN)

---

## Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Authentification TabPFN

TabPFN est un modèle fondationnel développé par **PriorLabs**. Il nécessite une
authentification via le site **https://ux.priorlabs.ai**.

### Étapes

1. Rendez-vous sur **https://ux.priorlabs.ai**
2. Créez un compte (ou connectez-vous)
3. Acceptez les conditions de licence dans l'onglet **License**
4. Récupérez votre token depuis votre profil

### Configurer le token

**Option A — Variable d'environnement (recommandé)**

```bash
# Windows (PowerShell)
$env:TABPFN_TOKEN = "votre_token_ici"

# Windows (CMD)
set TABPFN_TOKEN=votre_token_ici

# macOS / Linux
export TABPFN_TOKEN="votre_token_ici"
```

**Option B — Connexion interactive (première utilisation)**

Au premier lancement du code, TabPFN ouvre automatiquement un navigateur pour
vous authentifier. Le token est ensuite mis en cache localement et n'est plus
demandé pour les exécutions suivantes.

---

## Lancer les expériences

### Classification

```bash
cd classification
python main.py
```

Datasets traités : **Ionosphere**, **Sonar**, **BloodTransfusion**

Algorithmes : Logistic Regression, SVM RBF, KNN, Random Forest, XGBoost, **TabPFN**

### Régression

```bash
cd regression
python main.py
```

Datasets traités : **Diabetes**, **AutoMPG**, **Abalone**

Algorithmes : Ridge, SVR RBF, KNN, Random Forest, XGBoost, Gaussian Process

---

## Résultats produits

Pour chaque dataset, les fichiers suivants sont générés automatiquement :

| Fichier | Description |
|---------|-------------|
| `figures/learning_curves_auc.png` (classif) / `rmse.png` (régr) | Courbe principale |
| `figures/learning_curves_all_metrics.png` | Grille 2×2 des 4 métriques |
| `results/{dataset}/comparison_table_N{max}.csv` | Tableau comparatif au N max |
| `results/{dataset}/convergence_analysis.csv` | Point de convergence N* par modèle |

### Métriques

**Classification** : AUC-ROC, F1-macro, Accuracy, Cohen's Kappa

**Régression** : RMSE, MAE, R², R² ajusté

---

## Cache et relance

Les résultats sont mis en cache dans `results/{dataset}/results.pkl`.
Si ce fichier existe, les expériences sont sautées et les résultats en cache
sont réutilisés directement.

Pour **relancer** un dataset depuis zéro :

```bash
# Exemple : supprimer le cache Ionosphere
del classification\results\ionosphere\results.pkl   # Windows
rm classification/results/ionosphere/results.pkl    # macOS / Linux
```

---

## Protocole expérimental (résumé)

| Paramètre | Valeur |
|-----------|--------|
| Split train/test | 80 / 20 |
| Répétitions par taille | 20 |
| Validation croisée | 3-fold (2-fold si N_train < 30) |
| Scoring CV (classif) | AUC-ROC |
| Scoring CV (régr) | RMSE négatif |
| Seuil de convergence | 2 % de gain relatif |
| Seed de base | 42 |

---

## Dépendances principales

| Package | Usage |
|---------|-------|
| `scikit-learn` | Logistic Regression, SVM, KNN, Random Forest, Gaussian Process, métriques, preprocessing |
| `xgboost` | XGBClassifier, XGBRegressor |
| `tabpfn` | TabPFNClassifier (foundation model, classification uniquement) |
| `ucimlrepo` | Chargement des datasets UCI (Ionosphere, Sonar, BloodTransfusion, AutoMPG, Abalone) |
| `numpy` | Calculs numériques |
| `pandas` | Tableaux de résultats |
| `matplotlib` | Génération des figures |
