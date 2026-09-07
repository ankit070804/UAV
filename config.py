# =====================================================
# config.py
#
# Single source of truth for paths and constants used
# across the pipeline. Previously every script hardcoded
# its own copy of DATASET_ROOT (as a Windows-only path),
# which made the project impossible to run on any other
# machine. Now everything reads from here, and the root
# can be overridden with an environment variable so the
# code works on Windows, Linux, and CI alike.
# =====================================================

import os

# Root folder that contains the per-sequence subfolders
# (P000, P001, ...). Override with:
#   export UAV_DATASET_ROOT=/path/to/ArchVizTinyHouseDay/Data_easy   (Linux/Mac)
#   set UAV_DATASET_ROOT=E:\...\ArchVizTinyHouseDay\Data_easy         (Windows)
DATASET_ROOT = os.environ.get(
    "UAV_DATASET_ROOT",
    r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy",
)

# Sequences that make up the training data.
SEQUENCES = ["P000", "P001", "P002", "P003", "P004", "P005", "P006"]

# Sequence held out for the single-frame demo / explanation script.
# (Change this, or override with UAV_DEMO_SEQUENCE, to try a different one.)
DEMO_SEQUENCE = os.environ.get("UAV_DEMO_SEQUENCE", "P006")
DEMO_FRAME = int(os.environ.get("UAV_DEMO_FRAME", "20"))

# Sequence used as the dedicated test set in sequence_holdout_evaluation.py
# (train on every other sequence, test only on this one).
HOLDOUT_TEST_SEQUENCE = os.environ.get("UAV_HOLDOUT_TEST_SEQUENCE", "P000")

# Fraction of frames (from the END of each sequence, in frame order) held
# out as test data in time_split_evaluation.py.
TIME_SPLIT_TEST_FRACTION = float(os.environ.get("UAV_TIME_SPLIT_TEST_FRACTION", "0.20"))

# Target column and columns that must never be used as model inputs
# (they either leak the label or are identifiers, not features).
TARGET_COLUMN = "localization_error"
NON_FEATURE_COLUMNS = [
    "frame",
    "sequence",
    "ground_truth_distance",
    "estimated_distance",
    "translation_distance",
    "matches",
    TARGET_COLUMN,
]

# Shared filenames so every script agrees on where artifacts live.
ALL_FEATURES_CSV = "all_features.csv"
ALL_LABELS_CSV = "all_localization_labels.csv"
TRAINING_DATASET_CSV = "training_dataset_all.csv"
MODEL_PATH = "rf_localization.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
SPLIT_INDEX_PATH = "test_split_indices.pkl"

# Train/test split settings — kept in one place so train_model.py and
# evaluation_model.py can never drift out of sync with each other.
TEST_SIZE = 0.20
RANDOM_STATE = 42


def drop_columns(df):
    """Return only the columns of df that are safe to train/predict on."""
    return [c for c in NON_FEATURE_COLUMNS if c in df.columns]
