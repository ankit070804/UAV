import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    TRAINING_DATASET_CSV,
    TARGET_COLUMN,
    MODEL_PATH,
    SPLIT_INDEX_PATH,
    drop_columns,
)

# ============================================
# LOAD DATASET
# ============================================
#
# NOTE ON A BUG THAT USED TO BE HERE:
# This script previously evaluated the model on the ENTIRE
# training_dataset_all.csv, which includes the exact rows the
# model was trained on. That gave an inflated R^2 (~0.97) that
# looked better than train_model.py's own honest test-split
# score (~0.92). It now loads the row indices train_model.py
# held out and scores ONLY on those, so the number reported
# here is a real out-of-sample metric.
# ============================================

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

df = pd.read_csv(TRAINING_DATASET_CSV)

print("Dataset Shape :", df.shape)

TARGET = TARGET_COLUMN

# ============================================
# RESTRICT TO THE HELD-OUT TEST ROWS
# ============================================

try:
    test_indices = joblib.load(SPLIT_INDEX_PATH)
    df_eval = df.loc[test_indices]
    print(f"\nEvaluating on the {len(df_eval)} held-out test rows "
          f"saved by train_model.py ({SPLIT_INDEX_PATH}).")
except FileNotFoundError:
    df_eval = df
    print(
        f"\nWARNING: {SPLIT_INDEX_PATH} not found (run train_model.py first). "
        "Falling back to evaluating on the full dataset — this INCLUDES "
        "training rows and will overstate performance."
    )

# ============================================
# FEATURES / LABELS
# ============================================

X = df_eval.drop(columns=drop_columns(df_eval))
y = df_eval[TARGET]

print("\nNumber of Features :", X.shape[1])

# ============================================
# LOAD MODEL
# ============================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Done!")

# ============================================
# PREDICTION
# ============================================

pred = model.predict(X)

# ============================================
# METRICS
# ============================================

mae = mean_absolute_error(y, pred)

rmse = mean_squared_error(
    y,
    pred
) ** 0.5

r2 = r2_score(
    y,
    pred
)

print("\n" + "=" * 50)
print("MODEL EVALUATION (held-out test rows only)")
print("=" * 50)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# ============================================
# SAVE RESULTS
# ============================================

results = df_eval.copy()

results["predicted_error"] = pred

results["absolute_error"] = (
    results[TARGET] - pred
).abs()

results.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nSaved : evaluation_results.csv")

# ============================================
# ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y,
    pred,
    alpha=0.7
)

minimum = min(y.min(), pred.min())
maximum = max(y.max(), pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--'
)

plt.xlabel("Actual Localization Error")

plt.ylabel("Predicted Localization Error")

plt.title("Actual vs Predicted (Held-Out Test Set)")

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

# ============================================
# ERROR HISTOGRAM
# ============================================

plt.figure(figsize=(7, 5))

plt.hist(
    results["absolute_error"],
    bins=20
)

plt.xlabel("Absolute Error (m)")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution (Held-Out Test Set)")

plt.tight_layout()

plt.savefig(
    "error_distribution.png",
    dpi=300
)

# ============================================
# ERROR PER SAMPLE
# ============================================

plt.figure(figsize=(12, 5))

plt.plot(
    results["absolute_error"].values
)

plt.xlabel("Sample (held-out test set order)")

plt.ylabel("Absolute Error")

plt.title("Localization Error per Sample (Held-Out Test Set)")

plt.tight_layout()

plt.savefig(
    "sample_error.png",
    dpi=300
)

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "evaluation_feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 8))

top20 = importance.head(20)

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Top 20 Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

print("\nSaved:")
print("actual_vs_predicted.png")
print("error_distribution.png")
print("sample_error.png")
print("feature_importance.png")
print("evaluation_feature_importance.csv")

print("\n" + "=" * 50)
print("Evaluation Completed Successfully")
print("=" * 50)
