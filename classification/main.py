"""
Point d'entrée — Classification (tous datasets).

Usage :
    cd classification/
    python main.py

Chaque dataset est mis en cache séparément dans results/<dataset>/results.pkl.
Supprime le fichier d'un dataset pour relancer uniquement celui-là.
"""

import os
import pickle
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATASETS, RANDOM_SEED_BASE, TEST_SIZE,
    RESULTS_BASE_DIR, FIGURES_BASE_DIR,
)
from data import load_and_split, describe_dataset
from models import get_models_and_grids
from experiment import run_experiment
from visualization import (
    plot_learning_curves,
    plot_all_metrics,
    print_comparison_table,
    analyze_convergence,
)


def run_dataset(dataset_name: str, cfg: dict):
    """Lance le protocole complet pour un dataset de classification."""
    loader      = cfg["loader"]
    train_sizes = cfg["train_sizes"]

    # Dossiers propres à ce dataset
    results_dir = os.path.join(RESULTS_BASE_DIR, dataset_name.lower())
    figures_dir = os.path.join(FIGURES_BASE_DIR, dataset_name.lower())

    print(f"\n{'#' * 65}")
    print(f"#  DATASET : {dataset_name}")
    print(f"{'#' * 65}")

    # 1. Description
    describe_dataset(loader, dataset_name)

    # 2. Split train/test
    print(f"\nSplit train/test (80/20, stratifié, seed={RANDOM_SEED_BASE})...")
    X_train, X_test, y_train, y_test = load_and_split(
        loader, random_state=RANDOM_SEED_BASE, test_size=TEST_SIZE)
    print(f"  Train : {X_train.shape[0]} obs.  |  Test : {X_test.shape[0]} obs.")

    # 3. Modèles
    models_and_grids = get_models_and_grids()
    print(f"  Modèles : {list(models_and_grids.keys())}")
    print(f"  Grille  : {train_sizes}")

    # 4. Expérimentation (ou cache)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.pkl")

    if os.path.exists(results_path):
        print(f"\nCache trouvé → {results_path}")
        print("(Supprimer ce fichier pour relancer.)")
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        print(f"\nLancement : {len(models_and_grids)} modèles × "
              f"{len(train_sizes)} tailles × 20 répétitions")
        results = run_experiment(
            X_train, X_test, y_train, y_test,
            models_and_grids, train_sizes,
        )
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nRésultats sauvegardés → {results_path}")

    # 5. Figures
    print("\nGénération des figures...")
    plot_learning_curves(results, dataset_name, train_sizes, figures_dir, metric="AUC")
    plot_all_metrics(results, dataset_name, train_sizes, figures_dir)

    # 6. Tableau comparatif
    print_comparison_table(results, train_sizes, results_dir)

    # 7. Convergence
    analyze_convergence(results, train_sizes, results_dir, metric="AUC")

    print(f"\n→ Figures : {os.path.abspath(figures_dir)}")
    print(f"→ Résultats : {os.path.abspath(results_dir)}")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="lbfgs failed")

    for dataset_name, cfg in DATASETS.items():
        run_dataset(dataset_name, cfg)

    print(f"\n{'=' * 65}")
    print("  Tous les datasets traités.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
