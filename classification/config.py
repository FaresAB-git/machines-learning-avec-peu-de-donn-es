"""
Configuration centrale — Classification (tous datasets).
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
CONVERGENCE_THRESHOLD = 0.02   # +2 % AUC (cf. protocole §8)

# ── Dossiers de sortie (relatifs au répertoire d'exécution) ─────────────────
RESULTS_BASE_DIR = "results"
FIGURES_BASE_DIR = "figures"

# ── Datasets ─────────────────────────────────────────────────────────────────
# train_sizes : grille non-linéaire, dense sous N<100, max = 80 % de N_total
DATASETS = {
    "Ionosphere": {
        "loader":      "ionosphere",   # clé dans data.py
        "train_sizes": [20, 35, 50, 75, 100, 140, 180, 220, 260, 281],
        # N=351, test≈70, train≈281 — ratio N/p=10.3 (difficile)
    },
    "Sonar": {
        "loader":      "sonar",
        "train_sizes": [20, 35, 50, 75, 100, 130, 166],
        # N=208, test≈42, train≈166 — ratio N/p=3.5 (très difficile)
    },
    "BloodTransfusion": {
        "loader":      "blood_transfusion",
        "train_sizes": [20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 598],
        # N=748, test≈150, train≈598 — ratio N/p=187.0 (très favorable)
    },
}
