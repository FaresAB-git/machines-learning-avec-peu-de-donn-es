"""
Boucle expérimentale principale — courbes d'apprentissage en classification.

Protocole (cf. §5) :
  Pour chaque taille N_train dans la grille :
    Pour chaque répétition k in [0, K) :
      1. Tirage stratifié de N_train obs. depuis le train set fixe
      2. Standardisation : StandardScaler fitté sur le sous-échantillon (no leakage)
      3. GridSearchCV (3-fold stratifié, ou 2-fold si N_train < 30) — scoring = roc_auc
         Exception TabPFN : param_grid vide → fit direct sans GridSearchCV
      4. Évaluation sur le test set fixe → AUC, F1, Accuracy, Kappa
  → Moyenne et écart-type sur K répétitions
"""

import copy
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import (
    N_REPETITIONS,
    CV_FOLDS_DEFAULT,
    CV_FOLDS_SMALL,
    CV_SMALL_THRESHOLD,
    RANDOM_SEED_BASE,
)
from metrics import compute_metrics, METRIC_NAMES


def _filter_knn_grid(param_grid: dict, n_train: int, cv: int) -> dict:
    if "n_neighbors" not in param_grid:
        return param_grid
    min_fold_train = int(n_train * (cv - 1) / cv)
    filtered = dict(param_grid)
    filtered["n_neighbors"] = [k for k in param_grid["n_neighbors"]
                                if k < min_fold_train]
    if not filtered["n_neighbors"]:
        filtered["n_neighbors"] = [1]
    return filtered


def run_experiment(
    X_train:          np.ndarray,
    X_test:           np.ndarray,
    y_train:          np.ndarray,
    y_test:           np.ndarray,
    models_and_grids: dict,
    train_sizes:      list,
) -> dict:
    """
    Lance l'expérimentation complète pour un dataset et retourne les résultats.

    Structure de retour pour chaque modèle :
        results[model_name] = {
            "train_sizes" : liste des tailles testées,
            "raw"  : { métrique -> array (n_sizes, K) },
            "mean" : { métrique -> array (n_sizes,)   },
            "std"  : { métrique -> array (n_sizes,)   },
        }
    """
    results = {}

    for model_name, (base_model, param_grid) in models_and_grids.items():
        print(f"\n{'=' * 60}")
        print(f"  Modèle : {model_name}")
        print(f"{'=' * 60}")

        raw = {m: np.full((len(train_sizes), N_REPETITIONS), np.nan)
               for m in METRIC_NAMES}

        for size_idx, n_train in enumerate(train_sizes):
            n_train = min(n_train, len(X_train))
            cv      = CV_FOLDS_SMALL if n_train < CV_SMALL_THRESHOLD else CV_FOLDS_DEFAULT
            grid    = _filter_knn_grid(param_grid, n_train, cv)
            use_gs  = bool(grid)

            print(f"  N_train={n_train:4d}  cv={cv}", end="", flush=True)

            for rep in range(N_REPETITIONS):
                seed = RANDOM_SEED_BASE + rep * 1000 + size_idx

                # 1. Tirage stratifié
                if n_train >= len(X_train):
                    X_sub, y_sub = X_train, y_train
                else:
                    sss = StratifiedShuffleSplit(
                        n_splits=1, train_size=n_train, random_state=seed)
                    idx, _ = next(sss.split(X_train, y_train))
                    X_sub, y_sub = X_train[idx], y_train[idx]

                # 2. Standardisation sans fuite
                scaler    = StandardScaler()
                X_sub_sc  = scaler.fit_transform(X_sub)
                X_test_sc = scaler.transform(X_test)

                # 3a. GridSearchCV
                if use_gs:
                    gs = GridSearchCV(
                        estimator  = copy.deepcopy(base_model),
                        param_grid = grid,
                        cv         = cv,
                        scoring    = "roc_auc",
                        refit      = True,
                        n_jobs     = -1,
                        error_score= "raise",
                    )
                    gs.fit(X_sub_sc, y_sub)
                    best_model = gs.best_estimator_

                # 3b. Fit direct (TabPFN — param_grid vide)
                else:
                    try:
                        best_model = copy.deepcopy(base_model)
                        best_model.fit(X_sub_sc, y_sub)
                    except Exception as e:
                        if rep == 0 and size_idx == 0:
                            print(f"\n  [AVERTISSEMENT] TabPFN indisponible : {e}\n")
                        continue

                # Vérification que le modèle est bien fitté
                if not hasattr(best_model, "classes_"):
                    continue

                # 4. Évaluation
                y_pred       = best_model.predict(X_test_sc)
                y_pred_proba = best_model.predict_proba(X_test_sc)[:, 1]
                metrics      = compute_metrics(y_test, y_pred_proba, y_pred)
                for m in METRIC_NAMES:
                    raw[m][size_idx, rep] = metrics[m]

            auc_mean = np.nanmean(raw["AUC"][size_idx])
            auc_std  = np.nanstd(raw["AUC"][size_idx])
            print(f"  ->  AUC = {auc_mean:.4f} +/- {auc_std:.4f}")

        results[model_name] = {
            "train_sizes": train_sizes,
            "raw":  raw,
            "mean": {m: np.nanmean(raw[m], axis=1) for m in METRIC_NAMES},
            "std":  {m: np.nanstd(raw[m],  axis=1) for m in METRIC_NAMES},
        }

    return results
