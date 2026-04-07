"""
Définition des algorithmes et de leurs grilles d'hyperparamètres.
Chaque entrée : (estimateur_sklearn, param_grid_pour_GridSearchCV).

Algorithmes retenus pour la régression (cf. protocole §4) :
  - Ridge Regression        (linéaire régularisée)
  - SVR kernel RBF          (noyau)
  - K-Nearest Neighbors     (non-paramétrique)
  - Random Forest           (ensemble bagging)
  - XGBoost                 (ensemble boosting)
  - Gaussian Process Reg.   (non-paramétrique bayésien)
"""

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from xgboost import XGBRegressor


def get_models_and_grids() -> dict:
    """
    Retourne un dict ordonné :
        { nom_modèle: (estimateur, param_grid) }
    """
    return {
        # ── Ridge ─────────────────────────────────────────────────────────────
        "Ridge": (
            Ridge(),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),

        # ── SVR RBF ───────────────────────────────────────────────────────────
        "SVR_RBF": (
            SVR(kernel="rbf"),
            {
                "C":       [0.1, 1, 10, 100],
                "gamma":   ["scale", "auto"],
                "epsilon": [0.01, 0.1, 0.5],
            },
        ),

        # ── KNN ───────────────────────────────────────────────────────────────
        # La grille est filtrée dynamiquement dans experiment.py pour garantir
        # n_neighbors <= N_train (cf. contrainte sklearn).
        "KNN": (
            KNeighborsRegressor(),
            {
                "n_neighbors": [3, 5, 7, 10, 15],
                "weights":     ["uniform", "distance"],
            },
        ),

        # ── Random Forest ─────────────────────────────────────────────────────
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators":    [50, 100, 200],
                "max_depth":       [None, 3, 5, 10],
                "min_samples_split": [2, 5],
            },
        ),

        # ── XGBoost ───────────────────────────────────────────────────────────
        "XGBoost": (
            XGBRegressor(random_state=42, verbosity=0),
            {
                "n_estimators":  [50, 100],
                "max_depth":     [3, 5],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample":     [0.8, 1.0],
            },
        ),

        # ── GPR ───────────────────────────────────────────────────────────────
        # normalize_y=True : recommandé pour la stabilité numérique en petit N.
        # Les hyperparamètres du kernel (longueur d'échelle, etc.) sont optimisés
        # automatiquement par sklearn via log-vraisemblance marginale ; GridSearchCV
        # sélectionne uniquement la famille de noyau et le niveau de bruit alpha.
        "GPR": (
            GaussianProcessRegressor(normalize_y=True),
            {
                "kernel": [
                    RBF(),
                    Matern(nu=1.5),
                    Matern(nu=2.5),
                    RationalQuadratic(),
                ],
                "alpha": [1e-5, 1e-2, 0.1],
            },
        ),
    }
