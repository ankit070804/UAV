import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    TRAINING_DATASET_CSV,
    TARGET_COLUMN,
    HOLDOUT_TEST_SEQUENCE,
    RANDOM_STATE,
    drop_columns,
)

# ============================================
# TRAIN ON EVERYTHING EXCEPT ONE SEQUENCE,
# TEST ONLY ON THAT SEQUENCE
#
# This is different from:
#   - train_model.py       -> random 80/20 split across ALL sequences
#                              (rows from the same flight can leak
#                              between train and test)
#   - loso_evaluation.py    -> loops through every sequence as the
#                              test set, one at a time, and reports
#                              the average
#
# Here you pick ONE specific sequence (config.HOLDOUT_TEST_SEQUENCE,
# override with UAV_HOLDOUT_TEST_SEQUENCE) and train on the rest.
# Useful when you want to simulate "the UAV has never seen this
# environment before" for a specific trajectory you care about.
# ============================================

df = pd.read_csv(TRAINING_DATASET_CSV)

TARGET = TARGET_COLUMN
TEST_SEQ = HOLDOUT_TEST_SEQUENCE

available = sorted(df["sequence"].unique())
if TEST_SEQ not in available:
    raise ValueError(
        f"HOLDOUT_TEST_SEQUENCE='{TEST_SEQ}' not found in the dataset. "
        f"Available sequences: {available}"
    )

print("=" * 60)
print(f"Train on {[s for s in available if s != TEST_SEQ]}")
print(f"Test on  {TEST_SEQ}")
print("=" * 60)

train_df = df[df["sequence"] != TEST_SEQ]
test_df = df[df["sequence"] == TEST_SEQ]

DROP_COLUMNS = drop_columns(df)

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]

print(f"\nTraining rows : {len(X_train)}  (from {len(available) - 1} sequences)")
print(f"Testing rows  : {len(X_test)}  (all from {TEST_SEQ})")

# ============================================
# TRAIN
# ============================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================================
# EVALUATE
# ============================================

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\n" + "=" * 60)
print(f"RESULTS — tested only on {TEST_SEQ} (never seen during training)")
print("=" * 60)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================
# SAVE PREDICTIONS
# ============================================

results = pd.DataFrame({
    "frame": test_df["frame"].values,
    "Actual": y_test.values,
    "Predicted": pred,
})
results["Absolute Error"] = (results["Actual"] - results["Predicted"]).abs()

out_csv = f"holdout_{TEST_SEQ}_predictions.csv"
results.to_csv(out_csv, index=False)
print(f"\nSaved : {out_csv}")

# ============================================
# SAVE MODEL (kept separate from rf_localization.pkl so it
# doesn't silently overwrite the model the rest of the pipeline
# — graphs.py, test_system.py — expects to use)
# ============================================

model_path = f"rf_localization_holdout_{TEST_SEQ}.pkl"
joblib.dump(model, model_path)
print(f"Saved : {model_path}")
