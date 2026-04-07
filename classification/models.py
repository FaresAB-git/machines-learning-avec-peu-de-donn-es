"""
Définition des algorithmes et de leurs grilles d'hyperparamètres.
Chaque entrée : (estimateur_sklearn, param_grid_pour_GridSearchCV).

Algorithmes retenus pour la classification (cf. protocole §4) :
  - Logistic Regression     (linéaire régularisée)
  - SVM kernel RBF          (noyau)
  - K-Nearest Neighbors     (non-paramétrique)
  - Random Forest           (ensemble bagging)
  - XGBoost                 (ensemble boosting)
  - TabPFN                  (foundation model — classification uniquement)

Note TabPFN : pas de GridSearchCV (aucun hyperparamètre à sélectionner via CV).
              L'entrée param_grid est un dict vide {} ; experiment.py détecte
              ce cas et entraîne directement le modèle sans GridSearchCV.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier


def get_models_and_grids() -> dict:
    """
    Retourne un dict ordonné :
        { nom_modèle: (estimateur, param_grid) }
    """
    return {
        # ── Logistic Regression ───────────────────────────────────────────────
        "LogisticReg": (
            LogisticRegression(max_iter=1000, solver="lbfgs"),
            {
                "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"],
            },
        ),

        # ── SVM RBF ───────────────────────────────────────────────────────────
        # probability=True requis pour predict_proba → calcul AUC-ROC
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True),
            {
                "C":     [0.1, 1, 10, 100],
                "gamma": ["scale", "auto"],
            },
        ),

        # ── KNN ───────────────────────────────────────────────────────────────
        # La grille est filtrée dynamiquement dans experiment.py pour garantir
        # n_neighbors <= N_train (cf. contrainte sklearn).
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 10, 15],
                "weights":     ["uniform", "distance"],
            },
        ),

        # ── Random Forest ─────────────────────────────────────────────────────
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators":      [50, 100, 200],
                "max_depth":         [None, 3, 5, 10],
                "min_samples_split": [2, 5],
            },
        ),

        # ── XGBoost ───────────────────────────────────────────────────────────
        "XGBoost": (
            XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
            {
                "n_estimators":  [50, 100],
                "max_depth":     [3, 5],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample":     [0.8, 1.0],
            },
        ),

        # ── TabPFN ────────────────────────────────────────────────────────────
        # Foundation model conçu pour le small data tabulaire (Hollmann et al., 2025).
        # Pas d'hyperparamètres à tuner via CV : param_grid vide → fit direct.
        "TabPFN": (
            TabPFNClassifier(),
            {},  # pas de GridSearchCV
        ),
    }
