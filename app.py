"""
UAV-XAI Localization Console
=============================
A mission-console style Streamlit dashboard for the UAV-Localization
pipeline. Mirrors test_system.py exactly (same feature extraction, same
model, same SHAP ranking, same llm_engine.generate_report() call) and
renders it as a UI instead of console prints.

Place this file in the repo root, next to config.py, dataset_loader.py,
feature_extractor.py, semantic_extractor.py, depth_features.py,
class_names.py, llm_engine.py, and rf_localization.pkl / feature_names.pkl.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor
from class_names import get_class_name
import config
from llm_engine import generate_report, compute_risk_level


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
# THEME / CSS — flight-instrumentation palette
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
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }}

.console-header {{
    display: flex; justify-content: space-between; align-items: flex-end;
    border-bottom: 1px solid {BORDER}; padding-bottom: 14px; margin-bottom: 22px;
}}
.console-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; margin: 0; }}
.console-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: {MUTED}; margin-top: 2px; letter-spacing: 0.03em; }}
.status-pill {{ font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; padding: 5px 11px; border-radius: 3px; border: 1px solid {CYAN}55; color: {CYAN}; background: {CYAN}14; }}

.metric-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-left: 3px solid {CYAN}; border-radius: 4px; padding: 14px 18px; height: 100%; }}
.metric-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.06em; color: {MUTED}; text-transform: uppercase; margin-bottom: 6px; }}
.metric-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 600; color: {TEXT}; }}
.metric-unit {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {MUTED}; margin-left: 4px; }}

.risk-low {{ color: {CYAN} !important; border-left-color: {CYAN} !important; }}
.risk-medium {{ color: {AMBER} !important; border-left-color: {AMBER} !important; }}
.risk-high {{ color: {RED} !important; border-left-color: {RED} !important; }}

.panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 20px 22px; }}
.panel-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 600; color: {TEXT}; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
.panel-title:before {{ content: ""; width: 6px; height: 6px; background: {CYAN}; display: inline-block; border-radius: 1px; }}

.rec-item {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: {TEXT}; border-bottom: 1px dashed {BORDER}; padding: 9px 2px; }}
.rec-item:last-child {{ border-bottom: none; }}

.report-box {{ font-family: 'Inter', sans-serif; font-size: 14.5px; line-height: 1.7; color: {TEXT}; }}

.caveat {{ font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: {MUTED}; line-height: 1.6; border-left: 2px solid {AMBER}; padding: 4px 0 4px 10px; margin-top: 4px; }}

button[data-baseweb="tab"] {{ font-family: 'Space Grotesk', sans-serif; font-size: 13px; }}
.stButton>button {{ background: {CYAN}; color: #06110F; border: none; font-family: 'IBM Plex Mono', monospace; font-weight: 600; letter-spacing: 0.02em; border-radius: 3px; }}
.stButton>button:hover {{ background: #6EE0D6; color: #06110F; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(config.MODEL_PATH)
    feature_names = joblib.load(config.FEATURE_NAMES_PATH)
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


@st.cache_resource(show_spinner=False)
def get_extractors():
    return FeatureExtractor(), SemanticExtractor(), DepthFeatureExtractor()


@st.cache_resource(show_spinner=False)
def load_dataset(dataset_root: str, sequence: str):
    return TartanAirDatasetLoader(os.path.join(dataset_root, sequence))


def to_displayable(img):
    """Percentile-normalize any depth/segmentation array to a viewable uint8 image."""
    arr = np.nan_to_num(np.asarray(img).astype(np.float64))
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def risk_class(prediction: float) -> str:
    if prediction < 0.30:
        return "risk-low"
    if prediction <= 0.80:
        return "risk-medium"
    return "risk-high"


def run_inference(dataset, frame, model, feature_names, explainer, extractors):
    """Mirrors test_system.py exactly: same feature dict, same dx/dy/dz
    fallback, same column alignment, same SHAP ranking."""
    feature_extractor, semantic_extractor, depth_extractor = extractors
    sample = dataset.get_sample(frame)

    features = {}
    features.update(feature_extractor.extract(sample["rgb"]))
    features.update(semantic_extractor.extract(sample["segmentation"]))
    features.update(depth_extractor.extract(sample["depth"]))

    try:
        pose1 = dataset.poses[frame]
        pose2 = dataset.poses[frame + 1]
        features["dx"] = pose2[0] - pose1[0]
        features["dy"] = pose2[1] - pose1[1]
        features["dz"] = pose2[2] - pose1[2]
    except Exception:
        features["dx"] = 0
        features["dy"] = 0
        features["dz"] = 0

    df = pd.DataFrame([features])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    prediction = model.predict(df)[0]

    shap_values = explainer.shap_values(df)
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Value": df.iloc[0].values,
        "SHAP": shap_values[0],
    })
    importance["ABS"] = importance["SHAP"].abs()
    importance = importance.sort_values("ABS", ascending=False).reset_index(drop=True)
    importance["Readable"] = importance["Feature"].apply(get_class_name)

    return sample, prediction, importance


def parse_report(report_text: str):
    """generate_report() always returns
    'Risk Level: ...\\n\\nExplanation\\n...\\n\\nRecommendations\\n...' —
    this split is deterministic (built in Python), so it's safe to parse."""
    try:
        risk_part, rest = report_text.split("\n\nExplanation\n", 1)
        explanation, recommendations = rest.split("\n\nRecommendations\n", 1)
        risk_level = risk_part.replace("Risk Level:", "").strip()
        return risk_level, explanation.strip(), recommendations.strip()
    except ValueError:
        return None, report_text.strip(), ""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Mission Parameters")

    dataset_root = st.text_input(
        "Dataset root (UAV_DATASET_ROOT)",
        value=config.DATASET_ROOT,
        help="Folder containing per-sequence subfolders (P000, P001, ...).",
    )

    default_seq_index = (
        config.SEQUENCES.index(config.DEMO_SEQUENCE)
        if config.DEMO_SEQUENCE in config.SEQUENCES else 0
    )
    sequence = st.selectbox("Sequence", config.SEQUENCES, index=default_seq_index)

    frame = st.number_input("Frame index", min_value=0, value=config.DEMO_FRAME, step=1)
    top_n = st.slider("Top SHAP factors", 3, 10, 5)
    generate_llm = st.checkbox("Generate AI explanation (requires Ollama)", value=False)

    st.markdown("---")
    run = st.button("▶  RUN ANALYSIS", use_container_width=True)
    st.markdown("---")

    st.caption("Known caveats")
    st.markdown(f"""
    <div class="caveat">
    orb_features dominates feature importance (~73%) — the label itself
    is partly derived from ORB match count, so treat this as a limitation
    of the current labeling approach, not a validated causal finding.<br><br>
    Segmentation class names are unverified placeholders pending the
    official TartanAir label legend for this scene.
    </div>
    """, unsafe_allow_html=True)


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
        with st.spinner("Loading model and dataset..."):
            model, feature_names, explainer = load_model()
            dataset = load_dataset(dataset_root, sequence)
            extractors = get_extractors()
        with st.spinner(f"Extracting features and running inference on {sequence} / frame {frame}..."):
            sample, prediction, importance = run_inference(
                dataset, frame, model, feature_names, explainer, extractors
            )
        st.session_state.result = {
            "sample": sample,
            "prediction": prediction,
            "importance": importance,
            "sequence": sequence,
            "frame": frame,
        }
        st.session_state.pop("report", None)
    except FileNotFoundError as e:
        st.error(f"Dataset or model file not found: {e}")
        st.session_state.result = None
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.session_state.result = None

result = st.session_state.result

if result is None:
    st.markdown(f"""
    <div class="panel">
        <p class="panel-title">Awaiting Run</p>
        <p style="color:{MUTED}; font-family:'IBM Plex Mono',monospace; font-size:13px;">
        Set the dataset root, sequence, and frame index in the sidebar, then
        press RUN ANALYSIS to extract features, predict localization error,
        and rank the SHAP factors behind that prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

prediction = result["prediction"]
importance = result["importance"]
sample = result["sample"]
rclass = risk_class(prediction)
risk_label = compute_risk_level(prediction)

# ---- KPI row ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card {rclass}">
        <div class="metric-label">Predicted Localization Error</div>
        <div class="metric-value {rclass}">{prediction:.3f}<span class="metric-unit">m</span></div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card {rclass}">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value {rclass}" style="font-size:16px;">{risk_label}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sequence</div>
        <div class="metric-value">{result['sequence']}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Frame Index</div>
        <div class="metric-value">{result['frame']:04d}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

tab_sensor, tab_xai, tab_report = st.tabs(
    ["◧ Sensor Feed", "◨ Explainability", "◩ AI Explanation"]
)

# ---- Sensor Feed ----
with tab_sensor:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="panel-title">RGB</p>', unsafe_allow_html=True)
        st.image(sample["rgb"], use_container_width=True)
    with col2:
        st.markdown('<p class="panel-title">Depth</p>', unsafe_allow_html=True)
        st.image(to_displayable(sample["depth"]), use_container_width=True)
    with col3:
        st.markdown('<p class="panel-title">Segmentation</p>', unsafe_allow_html=True)
        st.image(to_displayable(sample["segmentation"]), use_container_width=True)

# ---- Explainability ----
with tab_xai:
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(top) + 1))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    colors = [CYAN if v >= 0 else RED for v in top["SHAP"]]
    ax.barh(top["Readable"], top["SHAP"], color=colors, height=0.55)
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
            importance.head(top_n)[["Readable", "Feature", "Value", "SHAP"]].style.format(
                {"Value": "{:.3f}", "SHAP": "{:+.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

# ---- AI Explanation ----
with tab_report:
    st.markdown('<p class="panel-title">AI-Generated Explanation</p>', unsafe_allow_html=True)

    if not generate_llm:
        st.info("Enable 'Generate AI explanation' in the sidebar, then re-run analysis.")
    elif "report" in st.session_state:
        risk_level, explanation, recommendations = st.session_state["report"]
        st.markdown(f'<div class="panel"><div class="report-box">{explanation}</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<p class="panel-title" style="margin-top:18px;">Recommendations</p>', unsafe_allow_html=True)
        rec_lines = [l.strip() for l in recommendations.splitlines() if l.strip()]
        st.markdown(
            '<div class="panel">' + "".join(f'<div class="rec-item">✓ {l}</div>' for l in rec_lines) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Generating explanation via local LLM (Ollama)..."):
            try:
                top = importance.head(top_n)
                top_features_for_llm = [
                    {
                        "name": row["Readable"],
                        "value": float(row["Value"]),
                        "shap": float(row["SHAP"]),
                        "raw_name": row["Feature"],
                    }
                    for _, row in top.iterrows()
                ]
                report_text = generate_report(prediction, top_features_for_llm)
                st.session_state["report"] = parse_report(report_text)
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Report generation failed: {e}")