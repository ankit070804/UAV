import os
import pandas as pd
import joblib
import shap

from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor
from config import DATASET_ROOT, DEMO_SEQUENCE, DEMO_FRAME, MODEL_PATH, FEATURE_NAMES_PATH
from class_names import get_class_name

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
explainer = shap.TreeExplainer(model)

# =====================================================
# DATASET
#
# Was hardcoded to a Windows path + a fixed sequence (P006).
# Now driven by config.py — override with the UAV_DATASET_ROOT
# / UAV_DEMO_SEQUENCE / UAV_DEMO_FRAME environment variables.
# =====================================================

dataset = TartanAirDatasetLoader(
    os.path.join(DATASET_ROOT, DEMO_SEQUENCE)
)

feature_extractor = FeatureExtractor()
semantic_extractor = SemanticExtractor()
depth_extractor = DepthFeatureExtractor()

# =====================================================
# SELECT FRAME
# =====================================================

frame = DEMO_FRAME
sample = dataset.get_sample(frame)

# =====================================================
# FEATURE EXTRACTION
# =====================================================

features = {}

features.update(feature_extractor.extract(sample["rgb"]))
features.update(semantic_extractor.extract(sample["segmentation"]))
features.update(depth_extractor.extract(sample["depth"]))

# ---------------- Motion ----------------

try:

    pose1 = dataset.poses[frame]
    pose2 = dataset.poses[frame + 1]

    features["dx"] = pose2[0] - pose1[0]
    features["dy"] = pose2[1] - pose1[1]
    features["dz"] = pose2[2] - pose1[2]

except:

    features["dx"] = 0
    features["dy"] = 0
    features["dz"] = 0

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame([features])

for col in feature_names:

    if col not in df.columns:
        df[col] = 0

df = df[feature_names]

# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(df)[0]

print("\n====================================================")
print(" UAV LOCALIZATION EXPLANATION SYSTEM")
print("====================================================")

print(f"\nPredicted Localization Error : {prediction:.3f} meters\n")

# =====================================================
# SHAP
# =====================================================

shap_values = explainer.shap_values(df)

importance = pd.DataFrame({

    "Feature": feature_names,
    "Value": df.iloc[0].values,
    "SHAP": shap_values[0]

})

importance["ABS"] = importance["SHAP"].abs()
importance = importance.sort_values("ABS", ascending=False)

top = importance.head(5)

# The raw feature/value/SHAP numbers below are computed for the LLM's
# internal understanding only (see llm_engine.build_prompt) — they are
# intentionally NOT printed to the console. The user-facing output is
# the plain-English report from generate_report() further down.

# =====================================================
# LLM REPORT
#
# The LLM reads the top SHAP features for THIS frame (name, value,
# signed contribution — translated to human-readable form via
# class_names.py) and synthesizes them into a single plain-English
# explanation plus recommendations. See generate_report() in
# llm_engine.py.
# =====================================================

from llm_engine import generate_report

top_features_for_llm = [
    {
        "name": get_class_name(row["Feature"]),
        "value": float(row["Value"]),
        "shap": float(row["SHAP"]),
        "raw_name": row["Feature"],
    }
    for _, row in top.iterrows()
]

report = generate_report(
    prediction,
    top_features_for_llm
)

print(report)

print("\n====================================================")
print("Analysis Completed Successfully")
print("====================================================")