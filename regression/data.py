"""
Chargement et préparation des datasets de régression.
Split train/test simple (non stratifié), effectué une seule fois (cf. protocole §3).

Datasets :
  - Diabetes : sklearn (N=442, p=10)
  - AutoMPG  : ucimlrepo id=9  (N=392 après nettoyage, p=7)
  - Abalone  : ucimlrepo id=1  (N=1000 sous-échantillonnés, p=8)
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from config import RANDOM_SEED_BASE


# ─────────────────────────────────────────────────────────────────────────────
# Loaders individuels
# ─────────────────────────────────────────────────────────────────────────────

def _load_diabetes():
    data = load_diabetes()
    return data.data, data.target


def _load_autompg():
    """
    Auto MPG (UCI id=9) : 398 obs. brutes, 6 lignes avec horsepower manquant → 392.
    Features : cylinders, displacement, horsepower, weight, acceleration,
               model_year, origin (7 features numériques).
    Cible : mpg (miles per gallon).
    """
    dataset = fetch_ucirepo(id=9)
    df = dataset.data.features.copy()
    y  = dataset.data.targets.values.ravel().astype(float)

    # Suppression des lignes avec valeurs manquantes (horsepower)
    mask = ~np.isnan(df.values.astype(float)).any(axis=1)
    X = df.values.astype(float)[mask]
    y = y[mask]
    return X, y


def _load_abalone():
    """
    Abalone (UCI id=1) : 4177 obs., sous-échantillonnage à N=1000 (seed fixe).
    Sex (M/F/I) encodé par label (0/1/2) → p=8 features numériques.
    Cible : Rings (nombre d'anneaux, proxy de l'âge).
    """
    dataset = fetch_ucirepo(id=1)
    df  = dataset.data.features.copy()
    y   = dataset.data.targets.values.ravel().astype(float)

    # Encodage de la colonne Sex (catégorielle)
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"].values)
    X = df.values.astype(float)

    # Sous-échantillonnage reproductible à N=1000
    rng  = np.random.RandomState(RANDOM_SEED_BASE)
    idx  = rng.choice(len(X), size=1000, replace=False)
    X, y = X[idx], y[idx]
    return X, y


_LOADERS = {
    "diabetes": _load_diabetes,
    "autompg":  _load_autompg,
    "abalone":  _load_abalone,
}


# ─────────────────────────────────────────────────────────────────────────────
# Interface publique
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(loader_key: str, random_state: int = 42, test_size: float = 0.20):
    """
    Charge le dataset et effectue le split unique train/test (tirage simple).

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
    )


def describe_dataset(loader_key: str, dataset_name: str):
    """Affiche un résumé du dataset pour vérification initiale."""
    X, y = _LOADERS[loader_key]()
    n, p = X.shape
    print("=" * 55)
    print(f"Dataset : {dataset_name}")
    print(f"  Observations  : {n}")
    print(f"  Features      : {p}")
    print(f"  Ratio N/p     : {n / p:.1f}")
    print(f"  Valeurs manq. : {int(np.isnan(X).sum())}")
    print(f"  Cible — min : {y.min():.2f}  max : {y.max():.2f}  "
          f"mean : {y.mean():.2f}  std : {y.std():.2f}")
    print("=" * 55)
