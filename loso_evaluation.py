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
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    RANDOM_STATE,
    drop_columns,
)

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv(TRAINING_DATASET_CSV)

print("="*60)
print("Leave-One-Sequence-Out Evaluation")
print("="*60)
print(
    "This is the most trustworthy accuracy number in the project: "
    "each sequence is held out completely, so it measures how well "
    "the model generalizes to an environment it never trained on "
    "(unlike train_model.py's random row split, which can leak "
    "similar frames from the same sequence into both train and test)."
)

TARGET = TARGET_COLUMN

DROP_COLUMNS = drop_columns(df)

# ============================================
# ALL SEQUENCES
# ============================================

sequences = sorted(df["sequence"].unique())

results = []

# ============================================
# LOOP OVER EACH SEQUENCE
# ============================================

for seq in sequences:

    print("\n---------------------------------------")
    print(f"Testing on {seq}")
    print("---------------------------------------")

    train_df = df[df["sequence"] != seq]
    test_df = df[df["sequence"] == seq]

    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    print(f"Frames : {len(test_df)}")
    print(f"MAE    : {mae:.4f}")
    print(f"RMSE   : {rmse:.4f}")
    print(f"R²     : {r2:.4f}")

    results.append({
        "Sequence": seq,
        "Frames": len(test_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# ============================================
# RESULTS TABLE
# ============================================

results = pd.DataFrame(results)

print("\n")
print("="*60)
print("FINAL RESULTS")
print("="*60)

print(results)

print("\nAverage Performance")

print(f"MAE  : {results['MAE'].mean():.4f}")
print(f"RMSE : {results['RMSE'].mean():.4f}")
print(f"R²   : {results['R2'].mean():.4f}")

results.to_csv(
    "loso_results.csv",
    index=False
)

print("\nSaved : loso_results.csv")

# ============================================
# BEST MODEL
# ============================================

print("\nTraining final model on ALL data...")

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]

final_model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_model.fit(X, y)

joblib.dump(final_model, MODEL_PATH)
joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

print("Saved:")
print("rf_localization.pkl")
print("feature_names.pkl")