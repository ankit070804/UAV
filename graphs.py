"""
generate_publication_figures.py

Generates every publication-quality figure needed for the IEEE paper,
PPT, and thesis, from the EXISTING trained pipeline. Does not retrain
or modify any existing project file.

Inputs (already in the repo, untouched):
    - training_dataset_all.csv   (859 rows x 72 cols, target = localization_error)
    - rf_localization.pkl
    - feature_names.pkl

Output:
    graphs/  (all PNGs, 300 DPI)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    TRAINING_DATASET_CSV,
    TARGET_COLUMN,
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    drop_columns,
)

# ======================================================================
# STYLE
# ======================================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

PRIMARY = "#1f5fa8"
ACCENT = "#d9534f"
GREEN = "#3f9142"
PALETTE = sns.color_palette("crest", as_cmap=False)

OUT = "graphs"
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# ======================================================================
# LOAD DATA + MODEL (exactly as train_model.py does)
# ======================================================================

print("=" * 60)
print("Loading dataset and trained model")
print("=" * 60)

df = pd.read_csv(TRAINING_DATASET_CSV)
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

TARGET = TARGET_COLUMN

DROP_COLUMNS = drop_columns(df)

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]
X = X[feature_names]  # enforce training column order

print("Dataset shape:", df.shape)
print("Feature matrix:", X.shape)

# Recreate the same train/test split used in train_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
rmse = mean_squared_error(y_test, pred_test) ** 0.5
r2 = r2_score(y_test, pred_test)

print(f"\nMAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

# Semantic class label lookup — now the single shared mapping in
# class_names.py instead of a fourth local copy of it.
from class_names import CLASS_NAMES

VISUAL_FEATS = ["brightness", "contrast", "blur", "orb_features", "edge_density"]
DEPTH_FEATS = ["mean_depth", "std_depth", "min_depth", "max_depth",
               "depth_range", "depth_entropy", "valid_depth_ratio"]
MOTION_FEATS = ["dx", "dy", "dz"]
SEMANTIC_FEATS = [c for c in X.columns if c.startswith("class_")]


# ======================================================================
# 1. DATASET / TARGET STATISTICS
# ======================================================================

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(y, bins=30, kde=True, color=PRIMARY, ax=ax)
ax.axvline(y.mean(), color=ACCENT, ls="--", lw=1.5, label=f"Mean = {y.mean():.3f} m")
ax.set_xlabel("Localization Error (m)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Localization Error (Full Dataset)")
ax.legend()
save(fig, "01_localization_error_distribution.png")

# Per-sequence error
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="sequence", y=TARGET, hue="sequence",
            order=sorted(df["sequence"].unique()), palette="crest",
            legend=False, ax=ax)
ax.set_xlabel("TartanAir Sequence")
ax.set_ylabel("Localization Error (m)")
ax.set_title("Localization Error by Trajectory Sequence")
save(fig, "02_error_by_sequence.png")

# Sample counts per sequence
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["sequence"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color=PRIMARY, edgecolor="white")
ax.set_xlabel("Sequence")
ax.set_ylabel("Number of Samples")
ax.set_title("Dataset Composition Across Trajectories")
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha="center", fontsize=10)
save(fig, "03_dataset_composition.png")


# ======================================================================
# 2. VISUAL FEATURE STATISTICS
# ======================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(VISUAL_FEATS):
    sns.histplot(df[feat], bins=25, kde=True, color=PALETTE[i % len(PALETTE)], ax=axes[i])
    axes[i].set_title(feat.replace("_", " ").title())
    axes[i].set_xlabel("")
axes[-1].axis("off")
fig.suptitle("Visual Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "04_visual_feature_distributions.png")


# ======================================================================
# 3. DEPTH FEATURE STATISTICS
# ======================================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(DEPTH_FEATS):
    sns.histplot(df[feat], bins=25, kde=True, color=PALETTE[i % len(PALETTE)], ax=axes[i])
    axes[i].set_title(feat.replace("_", " ").title())
    axes[i].set_xlabel("")
for j in range(len(DEPTH_FEATS), len(axes)):
    axes[j].axis("off")
fig.suptitle("Depth Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "05_depth_feature_distributions.png")


# ======================================================================
# 4. SEMANTIC CLASS DISTRIBUTION
# ======================================================================

mean_sem = df[SEMANTIC_FEATS].mean().sort_values(ascending=False).head(15)
labels = [CLASS_NAMES.get(c, c) for c in mean_sem.index]

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(labels[::-1], mean_sem.values[::-1], color=PRIMARY)
ax.set_xlabel("Mean Scene Coverage (%)")
ax.set_title("Top 15 Semantic Classes (Average Scene Coverage)")
save(fig, "06_semantic_class_distribution.png")


# ======================================================================
# 5. MOTION FEATURE DISTRIBUTIONS
# ======================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for i, feat in enumerate(MOTION_FEATS):
    sns.histplot(df[feat], bins=25, kde=True, color=PALETTE[i % len(PALETTE)], ax=axes[i])
    axes[i].set_title(f"{feat.upper()} (Frame-to-Frame Motion)")
fig.tight_layout()
save(fig, "07_motion_feature_distributions.png")


# ======================================================================
# 6. CORRELATION HEATMAP (numeric, non-semantic core features + target)
# ======================================================================

core_cols = VISUAL_FEATS + DEPTH_FEATS + MOTION_FEATS + [TARGET]
corr = df[core_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Heatmap: Core Features vs Localization Error")
save(fig, "08_correlation_heatmap.png")


# ======================================================================
# 7. RANDOM FOREST PERFORMANCE (test set, same split as training)
# ======================================================================

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred_test, alpha=0.6, color=PRIMARY, edgecolor="white", s=40)
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Ideal (y = x)")
ax.set_xlabel("Actual Localization Error (m)")
ax.set_ylabel("Predicted Localization Error (m)")
ax.set_title(f"Actual vs Predicted (R² = {r2:.3f})")
ax.legend()
save(fig, "09_actual_vs_predicted.png")

residuals = y_test.values - pred_test
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(pred_test, residuals, alpha=0.6, color=GREEN, edgecolor="white", s=40)
ax.axhline(0, color=ACCENT, ls="--", lw=1.5)
ax.set_xlabel("Predicted Localization Error (m)")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Residual Plot")
save(fig, "10_residual_plot.png")

abs_err = np.abs(residuals)
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(abs_err, bins=20, color=PRIMARY, kde=True, ax=ax)
ax.axvline(mae, color=ACCENT, ls="--", lw=1.5, label=f"MAE = {mae:.3f} m")
ax.set_xlabel("Absolute Error (m)")
ax.set_ylabel("Frequency")
ax.set_title("Prediction Error Distribution")
ax.legend()
save(fig, "11_error_histogram.png")

# Metrics summary card
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")
metrics_text = (
    f"Random Forest Regressor\n"
    f"{'-'*30}\n"
    f"MAE   : {mae:.4f} m\n"
    f"RMSE  : {rmse:.4f} m\n"
    f"R\u00b2    : {r2:.4f}\n"
    f"Train samples : {len(X_train)}\n"
    f"Test samples  : {len(X_test)}\n"
    f"Trees (n_estimators) : {model.n_estimators}\n"
    f"Total features : {len(feature_names)}"
)
ax.text(0.05, 0.95, metrics_text, va="top", ha="left", fontsize=13,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f2f2f2", edgecolor=PRIMARY))
save(fig, "12_performance_summary.png")


# ======================================================================
# 8. FEATURE IMPORTANCE (Random Forest native)
# ======================================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

top20 = importance.head(20).copy()
top20["Label"] = top20["Feature"].map(lambda c: CLASS_NAMES.get(c, c))

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top20["Label"][::-1], top20["Importance"][::-1], color=PRIMARY)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top 20 Feature Importances (Random Forest)")
save(fig, "13_feature_importance.png")


# ======================================================================
# 9. SHAP EXPLAINABILITY
# ======================================================================

print("\nComputing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(model)

# Use a manageable sample for readable plots
sample_idx = X_test.sample(n=min(200, len(X_test)), random_state=42).index
X_shap = X_test.loc[sample_idx]
shap_values = explainer.shap_values(X_shap)

X_shap_labeled = X_shap.rename(columns=lambda c: CLASS_NAMES.get(c, c))

# SHAP summary (beeswarm)
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap_labeled, show=False, max_display=15)
plt.title("SHAP Summary Plot (Feature Impact on Predicted Error)", fontsize=14, fontweight="bold")
fig = plt.gcf()
save(fig, "14_shap_summary.png")

# SHAP bar (mean |SHAP|)
fig = plt.figure(figsize=(9, 8))
shap.summary_plot(shap_values, X_shap_labeled, plot_type="bar", show=False, max_display=15)
plt.title("Mean |SHAP Value| per Feature", fontsize=14, fontweight="bold")
fig = plt.gcf()
save(fig, "15_shap_bar.png")

# SHAP waterfall for a single representative sample
single_idx = 0
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(np.array(base_val).ravel()[0])
else:
    base_val = float(base_val)

expl = shap.Explanation(
    values=shap_values[single_idx],
    base_values=base_val,
    data=X_shap.iloc[single_idx].values,
    feature_names=X_shap_labeled.columns.tolist()
)
fig = plt.figure(figsize=(10, 7))
shap.plots.waterfall(expl, max_display=12, show=False)
plt.title("SHAP Waterfall: Single-Sample Explanation", fontsize=13, fontweight="bold")
fig = plt.gcf()
save(fig, "16_shap_waterfall.png")


# ======================================================================
# 10. PIPELINE DIAGRAM
# ======================================================================

def box(ax, xy, w, h, text, color=PRIMARY, fontsize=10, textcolor="white"):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.2, edgecolor="#333333", facecolor=color)
    ax.add_patch(b)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold", wrap=True)

def arrow(ax, start, end):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                         color="#444444", linewidth=1.4)
    ax.add_patch(a)

fig, ax = plt.subplots(figsize=(6, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 26)
ax.axis("off")

stages = [
    ("RGB / Depth / Segmentation\n/ Pose (TartanAir)", "#6c757d"),
    ("Feature Extraction\n(Visual, Semantic, Depth, Motion)", "#6c757d"),
    ("Feature Vector\n(65 features)", "#6c757d"),
    ("Random Forest Regressor", PRIMARY),
    ("Predicted Localization Error", PRIMARY),
    ("SHAP TreeExplainer", GREEN),
    ("Top Contributing Features", GREEN),
    ("Reasoning Engine\n(Verified Facts)", "#e08a1e"),
    ("LLM Report Generation\n(Llama 3.2)", ACCENT),
    ("Natural-Language\nLocalization Report", ACCENT),
]

y = 24
positions = []
for label, color in stages:
    box(ax, (1, y), 8, 2, label, color=color, fontsize=10.5)
    positions.append(y)
    y -= 2.7

for i in range(len(positions) - 1):
    arrow(ax, (5, positions[i]), (5, positions[i+1] + 2))

ax.set_title("UAV Localization Error – XAI Pipeline", fontsize=15, fontweight="bold", pad=15)
save(fig, "17_pipeline_diagram.png")


# ======================================================================
# 11. MODEL ARCHITECTURE FIGURE (Random Forest concept diagram)
# ======================================================================

fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 17)
ax.set_ylim(0, 8)
ax.axis("off")

# Input
box(ax, (0.3, 3.3), 1.6, 1.4, "Feature\nVector\n(65)", color="#6c757d", fontsize=9)

n_trees = 5
tree_x_start = 2.6
tree_gap = 1.6
for i in range(n_trees):
    x = tree_x_start + i * tree_gap
    box(ax, (x, 5.2), 1.35, 1.1, f"Tree {i+1}", color=PRIMARY, fontsize=9)
    arrow(ax, (1.9, 4.0), (x + 0.05, 5.2))

box(ax, (tree_x_start, 3.3), 1.35, 1.1, "...", color="#adb5bd", fontsize=11)
arrow(ax, (1.9, 4.0), (tree_x_start + 0.65, 4.4))

box(ax, (tree_x_start + tree_gap * (n_trees - 1), 1.4), 1.55, 1.1, f"Tree {model.n_estimators}",
    color=PRIMARY, fontsize=9)
arrow(ax, (1.9, 3.9), (tree_x_start + tree_gap * (n_trees - 1) + 0.1, 2.5))

# Aggregation
agg_x = tree_x_start + tree_gap * n_trees + 0.5
box(ax, (agg_x, 3.3), 1.9, 1.4, "Average\nPrediction", color=GREEN, fontsize=9)
for i in range(n_trees):
    x = tree_x_start + i * tree_gap
    arrow(ax, (x + 0.65, 5.2), (agg_x + 0.3, 4.0))
arrow(ax, (tree_x_start + tree_gap * (n_trees - 1) + 0.8, 2.5), (agg_x + 1.0, 3.3))

box(ax, (agg_x + 2.4, 3.3), 2.1, 1.4, "Predicted\nLocalization\nError (m)", color=ACCENT, fontsize=9)
arrow(ax, (agg_x + 1.9, 4.0), (agg_x + 2.4, 4.0))

ax.set_title(f"Random Forest Regressor Architecture ({model.n_estimators} Trees, "
             f"{len(feature_names)} Input Features)", fontsize=13, fontweight="bold")
save(fig, "18_model_architecture.png")


print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
for f in sorted(os.listdir(OUT)):
    print(" -", f)