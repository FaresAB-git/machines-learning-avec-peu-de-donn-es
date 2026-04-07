"""
Calcul des métriques d'évaluation pour la régression (cf. protocole §6.2).

Métriques :
  - RMSE     : métrique principale des learning curves
  - MAE      : robuste aux outliers, complète le RMSE
  - R²       : proportion de variance expliquée
  - R²_adj   : pénalise le surparamétrage ; interprétation nuancée en petit N
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

METRIC_NAMES = ["RMSE", "MAE", "R2", "R2_adj"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> dict:
    """
    Calcule les quatre métriques sur le test set fixe.

    Parameters
    ----------
    y_true     : valeurs réelles
    y_pred     : prédictions du modèle
    n_features : nombre de features du dataset (p = 10 pour Diabetes)

    Returns
    -------
    dict avec clés RMSE, MAE, R2, R2_adj
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # R² ajusté : R²_adj = 1 - (1 - R²)(n-1)/(n-p-1)
    # n = taille du test set (fixe) ; p = nombre de features
    n = len(y_true)
    p = n_features
    if n > p + 1:
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = np.nan  # indéfini si n ≤ p+1

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "R2_adj": r2_adj}
