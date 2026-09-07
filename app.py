"""
UAV-XAI Localization Console
=============================
A mission-console style Streamlit dashboard for the UAV-Localization
explainable-AI pipeline. It wraps your existing modules — dataset_loader,
feature_extractor, semantic_extractor, depth_features, pose_analyser,
reasoning_engine, and llm_engine — in a proper front end instead of
console prints.

Place this file in the same directory as the rest of the repo
(dataset_loader.py, feature_extractor.py, rf_localization.pkl, etc.)
and run:

    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor
from pose_analyser import PoseAnalyzer

try:
    from reasoning_engine import explain
    HAS_REASONING = True
except Exception:
    HAS_REASONING = False

try:
    from llm_engine import generate_report
    HAS_LLM = True
except Exception:
    HAS_LLM = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="UAV-XAI Localization Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS  — flight-instrumentation palette
# ============================================================
CYAN = "#4FD1C5"
AMBER = "#F0A94E"
RED = "#E5636B"
BG = "#0B111C"
PANEL = "#121A28"
BORDER = "#223047"
TEXT = "#E7ECF3"
MUTED = "#7C8DB0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {TEXT}; }}

.stApp {{ background: radial-gradient(circle at 12% 0%, #101A2A 0%, {BG} 55%); }}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}

h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }}

.console-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 14px;
    margin-bottom: 22px;
}}
.console-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    margin: 0;
}}
.console-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: {MUTED};
    margin-top: 2px;
    letter-spacing: 0.03em;
}}
.status-pill {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    padding: 5px 11px;
    border-radius: 3px;
    border: 1px solid {CYAN}55;
    color: {CYAN};
    background: {CYAN}14;
}}

.metric-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-left: 3px solid {CYAN};
    border-radius: 4px;
    padding: 14px 18px;
    height: 100%;
}}
.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.06em;
    color: {MUTED};
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.metric-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: {TEXT};
}}
.metric-unit {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {MUTED}; margin-left: 4px; }}

.risk-low {{ color: {CYAN} !important; border-left-color: {CYAN} !important; }}
.risk-medium {{ color: {AMBER} !important; border-left-color: {AMBER} !important; }}
.risk-high {{ color: {RED} !important; border-left-color: {RED} !important; }}

.panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 20px 22px;
}}
.panel-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.panel-title:before {{ content: ""; width: 6px; height: 6px; background: {CYAN}; display: inline-block; border-radius: 1px; }}

.rec-item {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: {TEXT};
    border-bottom: 1px dashed {BORDER};
    padding: 9px 2px;
}}
.rec-item:last-child {{ border-bottom: none; }}

.report-box {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; line-height: 1.6; color: {TEXT}; white-space: pre-wrap; }}

button[data-baseweb="tab"] {{ font-family: 'Space Grotesk', sans-serif; font-size: 13px; }}

.stButton>button {{
    background: {CYAN};
    color: #06110F;
    border: none;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.02em;
    border-radius: 3px;
}}
.stButton>button:hover {{ background: #6EE0D6; color: #06110F; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED PIPELINE LOADER
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pipeline(dataset_path: str):
    dataset = TartanAirDatasetLoader(dataset_path)
    return {
        "dataset": dataset,
        "feature_extractor": FeatureExtractor(),
        "semantic_extractor": SemanticExtractor(),
        "depth_extractor": DepthFeatureExtractor(),
        "pose_analyzer": PoseAnalyzer(),
        "model": joblib.load("rf_localization.pkl"),
        "feature_names": joblib.load("feature_names.pkl"),
        "explainer": shap.TreeExplainer(joblib.load("rf_localization.pkl")),
    }


CLASS_NAME_MAP = {
    28: "Furniture", 229: "Shelf", 180: "Carpet", 157: "Wall",
    199: "Door", 74: "Window", 87: "Ceiling", 64: "Floor",
}


def risk_from_error(err: float):
    if err < 0.30:
        return "LOW", "risk-low"
    if err < 0.80:
        return "MEDIUM", "risk-medium"
    return "HIGH", "risk-high"


def run_inference(pipeline, frame: int, matches: int):
    dataset = pipeline["dataset"]
    sample = dataset.get_sample(frame)

    features = {}
    features.update(pipeline["feature_extractor"].extract(sample["rgb"]))
    features.update(pipeline["semantic_extractor"].extract(sample["segmentation"]))
    features.update(pipeline["depth_extractor"].extract(sample["depth"]))

    pose1 = sample["pose"]
    pose2 = dataset.get_sample(frame + 1)["pose"]
    features.update(pipeline["pose_analyzer"].motion_features(pose1, pose2))
    features["matches"] = matches

    feature_names = pipeline["feature_names"]
    row = pd.DataFrame([features])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]

    prediction = pipeline["model"].predict(row)[0]

    values = pipeline["explainer"].shap_values(row)
    if isinstance(values, list):
        values = values[0]
    shap_values = values[0]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Value": row.iloc[0].values,
        "SHAP": shap_values,
        "Importance": np.abs(shap_values),
    }).sort_values("Importance", ascending=False)

    return sample, prediction, importance


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Mission Parameters")
    dataset_path = st.text_input(
        "Dataset path",
        value=r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000",
        help="Path to a TartanAir sequence folder (e.g. .../Data_easy/P000)",
    )
    frame = st.number_input("Frame index", min_value=0, value=20, step=1)
    matches = st.number_input("ORB feature matches", min_value=0, value=1800, step=50)
    top_n = st.slider("Top SHAP features to show", 3, 15, 5)
    generate_llm = st.checkbox("Generate AI mission report (requires Ollama)", value=False)
    st.markdown("---")
    run = st.button("▶  RUN ANALYSIS", use_container_width=True)
    st.markdown("---")
    st.caption("UAV-XAI · Explainable Localization Error Estimation · Random Forest + SHAP")


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="console-header">
    <div>
        <p class="console-title">UAV-XAI Localization Console</p>
        <p class="console-sub">EXPLAINABLE LOCALIZATION ERROR ESTIMATION · TARTANAIR PIPELINE</p>
    </div>
    <div class="status-pill">● SYSTEM READY</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
if "result" not in st.session_state:
    st.session_state.result = None

if run:
    try:
        with st.spinner("Loading pipeline and dataset..."):
            pipeline = load_pipeline(dataset_path)
        with st.spinner(f"Extracting features and running inference on frame {frame}..."):
            sample, prediction, importance = run_inference(pipeline, frame, matches)
        st.session_state.result = {
            "sample": sample,
            "prediction": prediction,
            "importance": importance,
            "frame": frame,
        }
        st.session_state.pop("recommendations", None)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.session_state.result = None

result = st.session_state.result

if result is None:
    st.markdown(f"""
    <div class="panel">
        <p class="panel-title">Awaiting Run</p>
        <p style="color:{MUTED}; font-family:'IBM Plex Mono',monospace; font-size:13px;">
        Set a dataset path and frame index in the sidebar, then press RUN ANALYSIS
        to extract features, predict localization error, and generate a SHAP-based
        explanation for the selected frame.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

prediction = result["prediction"]
importance = result["importance"]
sample = result["sample"]
risk_label, risk_class = risk_from_error(prediction)

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card {risk_class}">
        <div class="metric-label">Predicted Localization Error</div>
        <div class="metric-value {risk_class}">{prediction:.3f}<span class="metric-unit">m</span></div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card {risk_class}">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value {risk_class}">{risk_label}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Frame Index</div>
        <div class="metric-value">{result['frame']:04d}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ORB Matches</div>
        <div class="metric-value">{matches}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

tab_sensor, tab_xai, tab_report, tab_recs = st.tabs(
    ["◧ Sensor Feed", "◨ Explainability", "◩ Mission Report", "◪ Recommendations"]
)

# ---- Sensor Feed ----
with tab_sensor:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="panel-title">RGB</p>', unsafe_allow_html=True)
        st.image(sample["rgb"], use_container_width=True)
    with col2:
        st.markdown('<p class="panel-title">Depth</p>', unsafe_allow_html=True)
        st.image(sample["depth"], use_container_width=True)
    with col3:
        st.markdown('<p class="panel-title">Segmentation</p>', unsafe_allow_html=True)
        st.image(sample["segmentation"], use_container_width=True)

# ---- Explainability ----
with tab_xai:
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(top) + 1))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    colors = [CYAN if v >= 0 else RED for v in top["SHAP"]]
    ax.barh(top["Feature"], top["SHAP"], color=colors, height=0.55)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xlabel("SHAP contribution to predicted error", color=MUTED, fontsize=9)
    fig.tight_layout()

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<p class="panel-title">Feature Contribution (SHAP)</p>', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
    with right:
        st.markdown('<p class="panel-title">Top Feature Table</p>', unsafe_allow_html=True)
        st.dataframe(
            importance.head(top_n)[["Feature", "Value", "SHAP"]].style.format(
                {"Value": "{:.3f}", "SHAP": "{:+.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<p class="panel-title">Plain-Language Observations</p>', unsafe_allow_html=True)
    lines = []
    for _, r in importance.head(top_n).iterrows():
        f = r["Feature"]
        if f.startswith("class_"):
            cid = int(f.split("_")[1])
            name = CLASS_NAME_MAP.get(cid, f"Class {cid}")
            lines.append(f"Large presence of {name} in view influenced the localization estimate.")
        elif f == "contrast":
            lines.append("Low image contrast reduced reliable feature detection.")
        elif f == "brightness":
            lines.append("Illumination levels affected localization quality.")
        elif f == "matches":
            lines.append("Number of ORB feature matches affected pose estimation confidence.")
        elif "depth" in f:
            lines.append("Scene depth variation influenced the localization estimate.")
        elif "orb" in f:
            lines.append("ORB keypoint density affected pose estimation accuracy.")
        else:
            lines.append(f"'{f}' contributed to the localization estimate.")
    st.markdown(
        '<div class="panel">' + "".join(f'<div class="rec-item">• {l}</div>' for l in lines) + "</div>",
        unsafe_allow_html=True,
    )

# ---- Mission Report ----
with tab_report:
    st.markdown('<p class="panel-title">AI-Generated Mission Report</p>', unsafe_allow_html=True)
    if not (HAS_REASONING and HAS_LLM):
        st.info("reasoning_engine.py and/or llm_engine.py were not importable — "
                "make sure both are in the app directory to enable AI report generation.")
    elif not generate_llm:
        st.info("Enable 'Generate AI mission report' in the sidebar, then re-run analysis.")
    else:
        with st.spinner("Generating report via local LLM (Ollama)..."):
            try:
                facts, recommendations = explain(importance)
                report_text = generate_report(prediction, facts, recommendations)
                st.session_state["recommendations"] = recommendations
                st.markdown(f'<div class="panel"><div class="report-box">{report_text}</div></div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Report generation failed (is Ollama running?): {e}")

# ---- Recommendations ----
with tab_recs:
    st.markdown('<p class="panel-title">Operational Recommendations</p>', unsafe_allow_html=True)
    recs = st.session_state.get("recommendations")
    if not recs:
        recs = [
            "Reduce UAV speed",
            "Increase illumination",
            "Enable Visual-Inertial Localization",
            "Avoid texture-poor regions",
        ]
        st.caption("Showing default recommendations — run the AI report for context-specific guidance.")
    st.markdown(
        '<div class="panel">' + "".join(f'<div class="rec-item">✓ {r}</div>' for r in recs) + "</div>",
        unsafe_allow_html=True,
    )