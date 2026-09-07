import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
    SPLIT_INDEX_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    drop_columns,
)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv(TRAINING_DATASET_CSV)

print("=" * 50)
print("Training Dataset")
print("=" * 50)

print("Shape :", df.shape)

TARGET = TARGET_COLUMN

X = df.drop(columns=drop_columns(df))
y = df[TARGET]

print("\nNumber of Features :", X.shape[1])

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Save the held-out row indices so evaluation_model.py can score the
# model on the SAME rows it never saw during training, instead of
# accidentally re-scoring on the full (train+test) dataset.
joblib.dump(list(X_test.index), SPLIT_INDEX_PATH)

# =====================================
# MODEL
# =====================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================
# TEST PREDICTION
# =====================================

pred = model.predict(X_test)

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(y_test, pred)

rmse = mean_squared_error(
    y_test,
    pred
) ** 0.5

r2 = r2_score(
    y_test,
    pred
)

print("\n==============================")
print("TEST SET RESULTS")
print("==============================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features\n")
print(importance.head(20))

importance.to_csv(
    "feature_importance.csv",
    index=False
)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    list(X.columns),
    FEATURE_NAMES_PATH
)

print("\nModel Saved")

# =====================================
# SAVE TEST PREDICTIONS
# =====================================

results = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": pred

})

results["Absolute Error"] = (
    results["Actual"] - results["Predicted"]
).abs()

results.to_csv(
    "test_predictions.csv",
    index=False
)

# =====================================
# ACTUAL vs PREDICTED
# =====================================

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    pred,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Localization Error")

plt.ylabel("Predicted Localization Error")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

# =====================================
# ERROR HISTOGRAM
# =====================================

plt.figure(figsize=(7,5))

plt.hist(
    results["Absolute Error"],
    bins=20
)

plt.xlabel("Absolute Error (m)")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    "error_distribution.png",
    dpi=300
)

# =====================================
# FEATURE IMPORTANCE PLOT
# =====================================

plt.figure(figsize=(10,8))

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

print("\nSaved Files")
print("----------------------------")
print("rf_localization.pkl")
print("feature_names.pkl")
print("feature_importance.csv")
print("test_predictions.csv")
print("actual_vs_predicted.png")
print("error_distribution.png")
print("feature_importance.png")