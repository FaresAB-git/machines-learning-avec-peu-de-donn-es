"""
Calcul des métriques d'évaluation pour la classification (cf. protocole §6.1).

Métriques :
  - AUC-ROC  : métrique principale des learning curves
  - F1 macro : tient compte du déséquilibre de classes (64/36)
  - Accuracy : taux de classification global
  - Kappa    : accord corrigé du hasard (Cohen, 1960)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, cohen_kappa_score

METRIC_NAMES = ["AUC", "F1", "Accuracy", "Kappa"]


def compute_metrics(
    y_true:       np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred:       np.ndarray,
) -> dict:
    """
    Calcule les quatre métriques sur le test set fixe.

    Parameters
    ----------
    y_true       : étiquettes réelles (0/1)
    y_pred_proba : probabilités estimées pour la classe positive (colonne 1)
    y_pred       : étiquettes prédites (seuil 0.5)

    Returns
    -------
    dict avec clés AUC, F1, Accuracy, Kappa
    """
    auc      = roc_auc_score(y_true, y_pred_proba)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)

    return {"AUC": auc, "F1": f1, "Accuracy": accuracy, "Kappa": kappa}
