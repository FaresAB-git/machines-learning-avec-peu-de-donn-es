"""
Visualisation — courbes d'apprentissage, tableaux comparatifs, convergence.
Toutes les fonctions acceptent dataset_name, train_sizes, results_dir, figures_dir.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import CONVERGENCE_THRESHOLD

PALETTE = {
    "Ridge":        "#1f77b4",
    "SVR_RBF":      "#ff7f0e",
    "KNN":          "#2ca02c",
    "RandomForest": "#d62728",
    "XGBoost":      "#9467bd",
    "GPR":          "#8c564b",
}

METRIC_LABELS = {
    "RMSE":   "RMSE",
    "MAE":    "MAE",
    "R2":     "R²",
    "R2_adj": "R² ajusté",
}

LOWER_IS_BETTER = {"RMSE", "MAE"}


# ─────────────────────────────────────────────────────────────────────────────
# Courbe d'apprentissage — métrique unique
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    results:      dict,
    dataset_name: str,
    train_sizes:  list,
    figures_dir:  str,
    metric:       str  = "RMSE",
    save:         bool = True,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in results.items():
        sizes = np.array(train_sizes)
        mean  = data["mean"][metric]
        std   = data["std"][metric]
        color = PALETTE.get(model_name)
        ax.plot(sizes, mean, marker="o", label=model_name,
                color=color, linewidth=2, markersize=5)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.12, color=color)

    ax.set_xlabel("Taille du sous-échantillon d'entraînement (N)", fontsize=12)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    ax.set_title(
        f"Courbes d'apprentissage — {dataset_name}\n"
        f"Métrique : {METRIC_LABELS.get(metric, metric)}  |  K=20  |  ±1σ",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, f"learning_curves_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegardé : {path}")
    plt.show()
    plt.close('all')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Grille 2×2 — toutes métriques
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_metrics(
    results:      dict,
    dataset_name: str,
    train_sizes:  list,
    figures_dir:  str,
    save:         bool = True,
):
    metrics = ["RMSE", "MAE", "R2", "R2_adj"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for model_name, data in results.items():
            sizes = np.array(train_sizes)
            mean  = data["mean"][metric]
            std   = data["std"][metric]
            color = PALETTE.get(model_name)
            ax.plot(sizes, mean, marker="o", label=model_name,
                    color=color, linewidth=1.8, markersize=4)
            ax.fill_between(sizes, mean - std, mean + std, alpha=0.10, color=color)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xlabel("N_train", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Courbes d'apprentissage — Toutes métriques\n"
        f"{dataset_name}  |  K=20  |  ±1σ",
        fontsize=13,
    )
    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, "learning_curves_all_metrics.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegardé : {path}")
    plt.show()
    plt.close('all')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tableau comparatif à taille maximale
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    results:     dict,
    train_sizes: list,
    results_dir: str,
    save:        bool = True,
) -> pd.DataFrame:
    n_max = train_sizes[-1]
    rows  = []
    for model_name, data in results.items():
        row = {"Modèle": model_name}
        for m in ["RMSE", "MAE", "R2", "R2_adj"]:
            row[f"{m}_mean"] = round(float(data["mean"][m][-1]), 4)
            row[f"{m}_std"]  = round(float(data["std"][m][-1]),  4)
        rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values("RMSE_mean", ascending=True)
          .reset_index(drop=True))
    df.index += 1

    print(f"\n{'=' * 70}")
    print(f"Tableau comparatif à taille maximale (N = {n_max})")
    print(f"{'=' * 70}")
    print(df.to_string())

    if save:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"comparison_table_N{n_max}.csv")
        df.to_csv(path, index=True)
        print(f"  Sauvegardé : {path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analyse du point de convergence
# ─────────────────────────────────────────────────────────────────────────────

def analyze_convergence(
    results:     dict,
    train_sizes: list,
    results_dir: str,
    metric:      str   = "RMSE",
    threshold:   float = CONVERGENCE_THRESHOLD,
    save:        bool  = True,
) -> pd.DataFrame:
    """
    N* = premier N_i où la réduction relative de RMSE vers N_{i+1} < threshold.
    RMSE : lower is better → gain = (mean[i] - mean[i+1]) / mean[i].
    """
    print(f"\n{'=' * 70}")
    print(f"Convergence — {metric} (seuil = {threshold*100:.0f} %)")
    print(f"{'=' * 70}")

    sizes = np.array(train_sizes)
    rows  = []

    for model_name, data in results.items():
        mean   = data["mean"][metric]
        conv_n = int(sizes[-1])

        for i in range(len(sizes) - 1):
            denom    = abs(mean[i]) + 1e-10
            if metric in LOWER_IS_BETTER:
                rel_gain = (mean[i] - mean[i + 1]) / denom
            else:
                rel_gain = (mean[i + 1] - mean[i]) / denom

            if rel_gain < threshold:
                conv_n = int(sizes[i])
                break

        idx_conv = list(sizes).index(conv_n)
        print(f"  {model_name:20s}  N* = {conv_n:4d}  "
              f"({metric} @ N* = {mean[idx_conv]:.4f})")
        rows.append({"Modèle": model_name, "N_convergence": conv_n})

    df = pd.DataFrame(rows)
    if save:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "convergence_analysis.csv")
        df.to_csv(path, index=False)
        print(f"  Sauvegardé : {path}")

    return df
