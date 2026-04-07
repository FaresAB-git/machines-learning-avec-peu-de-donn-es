"""
Configuration centrale — Régression (tous datasets).
"""

# ── Reproductibilité ─────────────────────────────────────────────────────────
RANDOM_SEED_BASE = 42

# ── Split train / test ───────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Répétitions ──────────────────────────────────────────────────────────────
N_REPETITIONS = 20

# ── Validation croisée ───────────────────────────────────────────────────────
CV_FOLDS_DEFAULT   = 3
CV_FOLDS_SMALL     = 2
CV_SMALL_THRESHOLD = 30

# ── Convergence ──────────────────────────────────────────────────────────────
CONVERGENCE_THRESHOLD = 0.02   # 2 % de RMSE (cf. protocole §8)

# ── Dossiers de sortie ───────────────────────────────────────────────────────
RESULTS_BASE_DIR = "results"
FIGURES_BASE_DIR = "figures"

# ── Datasets ─────────────────────────────────────────────────────────────────
DATASETS = {
    "Diabetes": {
        "loader":      "diabetes",
        "train_sizes": [20, 35, 50, 75, 100, 150, 200, 250, 300, 354],
        # N=442, test≈88, train≈354 — ratio N/p=44.2
    },
    "AutoMPG": {
        "loader":      "autompg",
        "train_sizes": [20, 35, 50, 75, 100, 150, 200, 250, 300, 314],
        # N=392 (après suppression de 6 lignes manquantes), test≈78, train≈314
        # ratio N/p=56.0
    },
    "Abalone": {
        "loader":      "abalone",
        "train_sizes": [20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800],
        # N=1000 (sous-échantillonnage depuis 4177, seed fixe), test≈200, train≈800
        # ratio N/p=125.0
    },
}
