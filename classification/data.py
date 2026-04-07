"""
Chargement et préparation des datasets de classification.
Split train/test stratifié, effectué une seule fois par dataset (cf. protocole §3).

Datasets :
  - Ionosphere      : ucimlrepo id=52  (N=351, p=34)
  - Sonar           : ucimlrepo id=151 (N=208, p=60)
  - BloodTransfusion: ucimlrepo id=176 (N=748, p=4)
"""

import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Loaders individuels
# ─────────────────────────────────────────────────────────────────────────────

def _load_ionosphere():
    dataset = fetch_ucirepo(id=52)
    X = dataset.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(dataset.data.targets.values.ravel())
    return X, y


def _load_sonar():
    dataset = fetch_ucirepo(id=151)
    X = dataset.data.features.values.astype(float)
    # Cibles : 'R' (roche) et 'M' (mine)
    y = LabelEncoder().fit_transform(dataset.data.targets.values.ravel())
    return X, y


def _load_blood_transfusion():
    dataset = fetch_ucirepo(id=176)
    X = dataset.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(dataset.data.targets.values.ravel())
    return X, y


_LOADERS = {
    "ionosphere":       _load_ionosphere,
    "sonar":            _load_sonar,
    "blood_transfusion": _load_blood_transfusion,
}


# ─────────────────────────────────────────────────────────────────────────────
# Interface publique
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(loader_key: str, random_state: int = 42, test_size: float = 0.20):
    """
    Charge le dataset identifié par loader_key et effectue le split unique
    train/test stratifié (préserve la distribution des classes).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray (non standardisés)
    """
    X, y = _LOADERS[loader_key]()
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )


def describe_dataset(loader_key: str, dataset_name: str):
    """Affiche un résumé du dataset pour vérification initiale."""
    X, y = _LOADERS[loader_key]()
    n, p = X.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    print("=" * 55)
    print(f"Dataset : {dataset_name}")
    print(f"  Observations  : {n}")
    print(f"  Features      : {p}")
    print(f"  Ratio N/p     : {n / p:.1f}")
    print(f"  Valeurs manq. : {int(np.isnan(X).sum())}")
    print(f"  Classe 1      : {n_pos} ({n_pos/n*100:.1f} %)  "
          f"|  Classe 0 : {n_neg} ({n_neg/n*100:.1f} %)")
    print("=" * 55)
