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
    TIME_SPLIT_TEST_FRACTION,
    RANDOM_STATE,
    drop_columns,
)

# ============================================
# TEMPORAL (WITHIN-SEQUENCE) SPLIT
#
# A third way to split this data, alongside:
#   - train_model.py                 -> random 80/20 rows across ALL
#                                        sequences (can leak near-duplicate
#                                        adjacent frames between train/test)
#   - sequence_holdout_evaluation.py -> train on every sequence but one,
#                                        test only on that one (tests
#                                        generalization to a brand-new
#                                        environment)
#   - loso_evaluation.py             -> the above, looped over every
#                                        sequence, averaged
#
# This instead keeps every sequence's EARLY frames for training and its
# LATE frames for testing. It answers a different question: "if the UAV
# has already flown the first part of THIS route, how well can it predict
# its own localization error on the rest of the route?" It's a different
# kind of split from sequence_holdout/loso (same environment in both train
# and test) — but empirically it does NOT score higher than those. On this
# dataset it lands close to the LOSO average (~0.77 vs ~0.79), which
# suggests the model's accuracy isn't really coming from memorizing
# "this specific room" — motion dynamics and per-frame texture matter
# more than which environment it is.
# ============================================

df = pd.read_csv(TRAINING_DATASET_CSV)

TARGET = TARGET_COLUMN
TEST_FRACTION = TIME_SPLIT_TEST_FRACTION

print("=" * 60)
print(f"Temporal split: last {TEST_FRACTION:.0%} of frames in EACH "
      f"sequence held out for testing")
print("=" * 60)

train_parts = []
test_parts = []

for seq, group in df.groupby("sequence"):
    group = group.sort_values("frame")
    n_test = max(1, int(round(len(group) * TEST_FRACTION)))
    train_parts.append(group.iloc[:-n_test])
    test_parts.append(group.iloc[-n_test:])
    print(f"  {seq}: {len(group) - n_test} train / {n_test} test")

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

DROP_COLUMNS = drop_columns(df)

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]

print(f"\nTotal training rows : {len(X_train)}")
print(f"Total testing rows  : {len(X_test)}")

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
print("RESULTS — trained on early frames, tested on later frames")
print("(same environments in both train and test)")
print("=" * 60)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================
# SAVE PREDICTIONS
# ============================================

results = test_df[["sequence", "frame"]].copy()
results["Actual"] = y_test.values
results["Predicted"] = pred
results["Absolute Error"] = (results["Actual"] - results["Predicted"]).abs()

results.to_csv("time_split_predictions.csv", index=False)
print("\nSaved : time_split_predictions.csv")

# ============================================
# SAVE MODEL (kept separate — same reasoning as
# sequence_holdout_evaluation.py, doesn't touch rf_localization.pkl)
# ============================================

joblib.dump(model, "rf_localization_time_split.pkl")
print("Saved : rf_localization_time_split.pkl")
