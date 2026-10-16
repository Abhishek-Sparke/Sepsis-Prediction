"""
app.py
======
Main Streamlit Application for Sepsis Early Warning Prediction.
A professional, responsive healthcare analytics dashboard featuring:
  - Multi-page navigation (Home, Dataset Management, Model Evaluation, Prediction Lab, Clinical Education)
  - Interactive Plotly visualizations (ROC Curves, PR Curves, Confusion Matrices, Feature Importance, Patient Trajectories)
  - Time-aware Machine Learning pipelines (Logistic Regression, Random Forest, XGBoost)
  - Schema validation, leak-free patient grouping, and real-time inference
"""

import os
import streamlit as st
import numpy as np
import pandas as pd

# Modular project imports
from data_generator import generate_synthetic_sepsis_data
from preprocessing import (
    validate_dataset,
    extract_time_aware_features,
    split_patient_data,
    REQUIRED_COLUMNS,
    VITAL_AND_LAB_COLS,
    SINGLE_MEASUREMENT_FEATURES,
    CLINICAL_BOUNDS,
    validate_single_measurement,
    prepare_single_measurement_dataset,
    build_single_measurement_feature_vector
)
from model_training import (
    train_all_models,
    predict_patient_record,
    predict_single_measurement,
    predict_batch,
    init_or_load_all_models,
    save_models,
    load_models
)
from evaluation import evaluate_models, extract_feature_importances
from visualizations import (
    plot_roc_curves,
    plot_pr_curves,
    plot_model_comparison_bar,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_patient_trajectory,
    plot_patient_prediction_comparison,
    plot_batch_predictions_summary
)

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & CUSTOM HEALTHCARE STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SepsisGuard | Early Warning Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected CSS for Healthcare Dashboard Theme: Navy sidebar, light main background, and high-contrast typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #0f172a;
        font-weight: 700;
    }

    /* ==========================================================================
       SIDEBAR HIGH-CONTRAST NAVY THEME
       ========================================================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b132b 0%, #1e293b 100%) !important;
        border-right: 1px solid #1e293b !important;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Radio button container & labels */
    [data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: none !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.28), rgba(13, 148, 136, 0.35)) !important;
        border: 1.5px solid #0ea5e9 !important;
        box-shadow: 0 0 10px rgba(14, 165, 233, 0.3) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] input[type="radio"] {
        accent-color: #0ea5e9 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.15) !important;
        margin: 16px 0 !important;
    }

    .sidebar-status-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.8rem;
        line-height: 1.55;
        color: #cbd5e1 !important;
    }

    .sidebar-status-card b {
        color: #38bdf8 !important;
    }

    .sidebar-disclaimer-card {
        background: rgba(245, 158, 11, 0.12);
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 0.74rem;
        line-height: 1.45;
        color: #fde68a !important;
        margin-top: 14px;
    }

    /* ==========================================================================
       HOMEPAGE WORKFLOW & ALGORITHM CARDS
       ========================================================================== */
    .workflow-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 18px;
        height: 100%;
        min-height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-top: 4px solid #0d9488;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .workflow-card:hover {
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    .workflow-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #0d9488;
        background: #f0fdfa;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 10px;
        width: fit-content;
    }
    .workflow-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
        font-family: 'Outfit', sans-serif;
    }
    .workflow-desc {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
    }

    .algo-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px 18px;
        height: 100%;
        min-height: 175px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-top: 4px solid #0d9488;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .algo-card:hover {
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    .algo-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
        font-family: 'Outfit', sans-serif;
    }
    .algo-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }

    .disclaimer-banner {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #92400e;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .clinical-callout {
        background-color: #f0fdfa;
        border-left: 4px solid #0d9488;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 16px 0;
        color: #115e59;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. APPLICATION STATE MANAGEMENT & DEFAULT DATA
# ------------------------------------------------------------------------------
@st.cache_data
def get_default_dataset():
    """Generates and caches baseline clinical dataset."""
    os.makedirs("data", exist_ok=True)
    cache_path = os.path.join("data", "sample_sepsis_data.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        df = generate_synthetic_sepsis_data(n_patients=120, seed=42)
        df.to_csv(cache_path, index=False)
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
    return df


if "raw_df" not in st.session_state:
    st.session_state.raw_df = get_default_dataset()

# Automatically load persisted models or initialize on first load
if "all_models_bundle" not in st.session_state:
    st.session_state.all_models_bundle = init_or_load_all_models(st.session_state.raw_df)

if "trained_models" not in st.session_state or st.session_state.trained_models is None:
    st.session_state.trained_models = st.session_state.all_models_bundle["long_models"]
    st.session_state.eval_results = st.session_state.all_models_bundle["long_meta"]["eval_results"]
    st.session_state.importances = st.session_state.all_models_bundle["long_meta"]["importances"]
    st.session_state.feature_names = st.session_state.all_models_bundle["long_meta"]["features"]

if "single_models" not in st.session_state:
    st.session_state.single_models = st.session_state.all_models_bundle["single_models"]
    st.session_state.single_meta = st.session_state.all_models_bundle["single_meta"]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Project Overview"

# Interactive Prediction Lab session states
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if "submitted_inputs" not in st.session_state:
    st.session_state.submitted_inputs = None

if "preset_selection" not in st.session_state:
    st.session_state.preset_selection = "Custom Input"


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION (HIGH CONTRAST & READABLE)
# ------------------------------------------------------------------------------
NAV_PAGES = [
    "Project Overview",
    "Dataset Management",
    "Model Training & Evaluation",
    "Patient Prediction Lab",
    "Clinical & ML Guide"
]

NAV_ICONS = {
    "Project Overview": "🏠",
    "Dataset Management": "📊",
    "Model Training & Evaluation": "⚙️",
    "Patient Prediction Lab": "🧪",
    "Clinical & ML Guide": "📖"
}

with st.sidebar:
    st.markdown("### 🩺 SepsisGuard AI")
    st.markdown("<div style='font-size: 0.82rem; color: #94a3b8; margin-top: -6px;'>Critical Care ML Research Platform</div>", unsafe_allow_html=True)
    
    current_idx = NAV_PAGES.index(st.session_state.current_page) if st.session_state.current_page in NAV_PAGES else 0

    selected_page = st.radio(
        "Navigation Menu",
        NAV_PAGES,
        index=current_idx,
        format_func=lambda p: f"{NAV_ICONS.get(p, '')}  {p}",
        key="nav_radio"
    )
    st.session_state.current_page = selected_page

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-status-card">
            <b>Status:</b> Demonstration Mode<br>
            <b>Pipeline:</b> Time-Aware Classification<br>
            <b>Prediction Horizon:</b> Defined by the selected evaluation configuration.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-disclaimer-card">
            <strong>⚠️ Research Demonstration:</strong> Synthetic dataset provided for algorithm evaluation. Results are not clinically certified.
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------------------
# 4. PAGE 1: HOME PAGE (PROJECT OVERVIEW)
# ------------------------------------------------------------------------------
if selected_page == "Project Overview":
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ Research Demonstration Notice:</strong> This web application demonstrates machine learning methodologies 
            for early warning prediction of sepsis using vital signs and laboratory measurements. 
            The current synthetic dataset is for demonstration and model results are <strong>not clinically validated</strong> or approved for diagnostic use.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("Sepsis Early Warning Prediction")
    st.markdown(
        "##### Developing a time-aware machine-learning model to study early warning patterns associated with sepsis using patient vital signs and laboratory measurements."
    )
    st.markdown(
        """
        Sepsis represents a critical medical emergency characterized by a dysregulated systemic host response to infection. 
        Because physiological deterioration can occur rapidly, developing models that capture subtle hemodynamic and laboratory changes 
        over time provides vital study of early warning signals before overt clinical complications develop.
        """
    )

    # Prominent Call-to-Action to open analysis dashboard
    c_cta1, c_cta2 = st.columns([1.6, 1])
    with c_cta1:
        if st.button("📊 Open Model Training & Evaluation Dashboard", type="primary", use_container_width=True):
            st.session_state.current_page = "Model Training & Evaluation"
            st.rerun()
    with c_cta2:
        if st.button("📋 Inspect Dataset & Quality Assurance", use_container_width=True):
            st.session_state.current_page = "Dataset Management"
            st.rerun()

    st.markdown("---")
    st.subheader("Machine Learning Workflow")
    st.markdown("<p style='font-size:0.9rem; color:#475569; margin-top:-8px;'>Structured pipeline enforcing prediction boundaries and patient isolation.</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-badge">Step 1</div>
                <div class="workflow-title">Time-Aware Data Ingestion</div>
                <div class="workflow-desc">
                    Processes longitudinal patient measurements available strictly before the prediction time, establishing a defined retrospective observation window without future leakage.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-badge">Step 2</div>
                <div class="workflow-title">Feature Engineering</div>
                <div class="workflow-desc">
                    Computes historical means, minimums, maximums, dynamic trends (last minus first), and derived clinical indicators (e.g., Shock Index, Pulse Pressure).
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-badge">Step 3</div>
                <div class="workflow-title">Patient-Level Validation</div>
                <div class="workflow-desc">
                    Applies grouped patient splitting (GroupShuffleSplit) to prevent patient overlap between training and testing sets, ensuring zero inter-patient data leakage.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-badge">Step 4</div>
                <div class="workflow-title">Model Benchmarking</div>
                <div class="workflow-desc">
                    Trains and compares Logistic Regression, Random Forest, and XGBoost using clinically aligned metrics: ROC-AUC, PR-AUC, Sensitivity, and Specificity.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Candidate Algorithms")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="algo-card" style="border-top-color: #0284c7;">
                <div class="algo-title">Logistic Regression</div>
                <div class="algo-desc">
                    <strong>Clinical Linear Baseline:</strong> Standardized features with L2 regularization and class weighting. 
                    Offers transparent coefficients, fast inference, and an essential baseline for non-linear models.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="algo-card" style="border-top-color: #0d9488;">
                <div class="algo-title">Random Forest</div>
                <div class="algo-desc">
                    <strong>Ensemble Decision Trees:</strong> 120 randomized trees capturing non-linear physiological interactions 
                    and multi-variate biomarker thresholds with high robustness against outliers.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="algo-card" style="border-top-color: #1e3a8a;">
                <div class="algo-title">XGBoost</div>
                <div class="algo-desc">
                    <strong>Gradient Boosted Trees:</strong> Iteratively minimizes log-loss with positive weight scaling 
                    to maximize sensitivity and recall on imbalanced clinical outcome events.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Current Demonstration Dataset Summary")
    st.caption("Demonstration cohort reflecting authentic ICU ranges, multi-hour sampling, and intermittent laboratory draws.")
    df = st.session_state.raw_df
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Unique Patients", f"{df['patient_id'].nunique()}")
    m3.metric("Sepsis Cases", f"{int(df['sepsis_label'].sum())} ({df['sepsis_label'].mean():.1%})")
    m4.metric("Features per Record", f"{len(df.columns)}")


# ------------------------------------------------------------------------------
# 5. PAGE 2: DATASET MANAGEMENT & VALIDATION
# ------------------------------------------------------------------------------
elif selected_page == "Dataset Management":
    st.title("Dataset Management & Quality Assurance")
    st.markdown("Upload clinical time-series cohorts, inspect missingness, validate schema integrity, and export synthetic datasets.")

    st.markdown(
        """
        <div class="clinical-callout">
            <strong>Expected Clinical Schema:</strong> <code>patient_id, timestamp, heart_rate, resp_rate, temperature, sbp, dbp, map, oxygen_saturation, wbc, lactate, creatinine, platelets, sepsis_label</code>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_inspect, tab_upload, tab_patient_view = st.tabs(["📋 Current Dataset Preview", "📤 Upload Custom CSV", "📈 Patient Trajectory Explorer"])

    with tab_inspect:
        df = st.session_state.raw_df
        is_valid, errors, meta = validate_dataset(df)

        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(f"**Showing First 15 Records of {len(df):,} total observations:**")
            st.dataframe(df.head(15), use_container_width=True)

        with col_right:
            st.markdown("### Quick Statistics")
            st.write(f"• **Records:** {meta.get('total_records', 0):,}")
            st.write(f"• **Patients:** {meta.get('unique_patients', 0)}")
            st.write(f"• **Sepsis Positive:** {meta.get('sepsis_cases', 0)}")
            st.write(f"• **Prevalence:** {meta.get('sepsis_prevalence', 0):.1%}")

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dataset (CSV)",
                data=csv_data,
                file_name="sepsis_dataset_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("Field Metadata & Missing-Value Summary")
        null_counts = df.isnull().sum()
        null_pct = (df.isnull().mean() * 100).round(1)
        dtypes_series = df.dtypes.astype(str)

        summary_table = pd.DataFrame({
            "Data Type": dtypes_series,
            "Missing Count": null_counts,
            "Missing (%)": null_pct.astype(str) + "%"
        })
        st.dataframe(summary_table.T, use_container_width=True)

    with tab_upload:
        st.subheader("Upload Custom Clinical CSV")
        uploaded_file = st.file_uploader("Upload patient dataset (CSV format)", type=["csv"])

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if "patient_id" in uploaded_df.columns:
                    uploaded_df["patient_id"] = uploaded_df["patient_id"].astype(str)
                is_valid, errors, meta = validate_dataset(uploaded_df)

                if is_valid:
                    st.success(f"✅ Schema validation passed! Loaded {len(uploaded_df):,} records across {meta['unique_patients']} patients.")
                    st.dataframe(uploaded_df.head(10), use_container_width=True)
                    if st.button("Apply Uploaded Dataset to Active Session"):
                        st.session_state.raw_df = uploaded_df
                        st.session_state.trained_models = None
                        st.session_state.eval_results = None
                        st.rerun()
                else:
                    st.error("❌ Dataset Schema Validation Failed:")
                    for err in errors:
                        st.write(f"  • {err}")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

        st.markdown("---")
        if st.button("🔄 Reset to Default Synthetic Dataset"):
            st.session_state.raw_df = get_default_dataset()
            st.session_state.trained_models = None
            st.session_state.eval_results = None
            st.success("Reverted to default synthetic ICU cohort.")
            st.rerun()

    with tab_patient_view:
        st.subheader("Patient Longitudinal Trajectory Viewer")
        df = st.session_state.raw_df
        patient_list = sorted(df["patient_id"].unique())
        selected_pid = st.selectbox("Select Patient to View Hemodynamic Trajectory:", patient_list)

        if selected_pid:
            traj_fig = plot_patient_trajectory(df, selected_pid)
            st.plotly_chart(traj_fig, use_container_width=True)
            p_subset = df[df["patient_id"] == selected_pid]
            has_sepsis = int(p_subset["sepsis_label"].max()) == 1
            if has_sepsis:
                st.warning(f"Patient {selected_pid} experienced sepsis deterioration during ICU stay.")
            else:
                st.info(f"Patient {selected_pid} maintained stable vitals with no sepsis onset.")


# ------------------------------------------------------------------------------
# 6. PAGE 3: MODEL TRAINING & EVALUATION DASHBOARD
# ------------------------------------------------------------------------------
elif selected_page == "Model Training & Evaluation":
    st.title("Model Training & Validation Dashboard")
    st.markdown("Time-aware feature engineering, patient-level train/test splitting, and multi-model benchmarking.")

    df = st.session_state.raw_df
    
    # Training Execution Section
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.caption("Candidate pipelines (Logistic Regression, Random Forest, XGBoost) with leak-free GroupShuffleSplit on patient_id.")
    with col_t2:
        re_train_btn = st.button("🚀 Retrain Models", use_container_width=True)

    if re_train_btn or st.session_state.trained_models is None:
        with st.spinner("Fitting pipelines on patient-level splits and persisting to disk..."):
            bundle = init_or_load_all_models(df, force_retrain=True)
            st.session_state.all_models_bundle = bundle
            st.session_state.trained_models = bundle["long_models"]
            st.session_state.eval_results = bundle["long_meta"]["eval_results"]
            st.session_state.importances = bundle["long_meta"]["importances"]
            st.session_state.feature_names = bundle["long_meta"]["features"]
            st.session_state.single_models = bundle["single_models"]
            st.session_state.single_meta = bundle["single_meta"]
            st.success(f"Successfully trained and persisted models ({bundle['long_meta']['n_train_patients']} train patients; {bundle['long_meta']['n_test_patients']} test patients).")
            st.rerun()

    eval_data = st.session_state.eval_results
    metrics_df = eval_data["metrics_df"]

    # 1. Top Metrics Cards
    st.markdown("### Test Cohort Performance Summary")
    best_model_name = metrics_df["ROC-AUC"].idxmax()
    best_row = metrics_df.loc[best_model_name]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Top Model ({best_model_name})", f"{best_row['ROC-AUC']:.3f}", "ROC-AUC")
    c2.metric("Accuracy", f"{best_row['Accuracy']:.1%}")
    c3.metric("Sensitivity (Recall)", f"{best_row['Recall (Sensitivity)']:.1%}")
    c4.metric("Specificity", f"{best_row['Specificity']:.1%}")
    c5.metric("F1-Score", f"{best_row['F1-Score']:.3f}")

    st.markdown("---")

    # 2. Benchmark Table
    st.subheader("Performance Comparison Table")
    formatted_df = metrics_df.copy()
    for col in ["Accuracy", "ROC-AUC", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "PR-AUC"]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(formatted_df, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Visualizations Tabs
    tab_roc, tab_bar, tab_cm, tab_pr, tab_feat = st.tabs([
        "📈 Detailed ROC Curves",
        "📊 Bar Chart Comparison",
        "🗂️ Confusion Matrices",
        "🎯 Precision-Recall Curves",
        "⭐ Feature Importance"
    ])

    with tab_roc:
        st.subheader("Receiver Operating Characteristic (ROC) Analysis")
        roc_fig = plot_roc_curves(eval_data["roc_curves"])
        st.plotly_chart(roc_fig, use_container_width=True)
        st.markdown(
            """
            **Clinical Meaning of ROC Curves:**
            - **X-axis (False Positive Rate):** Probability of falsely alerting sepsis for a healthy patient (1 - Specificity). Minimizing this prevents ICU alarm fatigue.
            - **Y-axis (True Positive Rate):** Probability of correctly detecting sepsis onset before clinical shock (Sensitivity / Recall).
            - **AUC (Area Under Curve):** Overall discrimination capability. A perfect test has $\\text{AUC} = 1.0$; random guessing has $\\text{AUC} = 0.50$ (dashed line).
            """
        )

    with tab_bar:
        st.subheader("Comprehensive Metric Comparison Across Models")
        bar_fig = plot_model_comparison_bar(metrics_df)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption("Grouped performance comparison demonstrating trade-offs between precision and recall across linear and tree-based architectures.")

    with tab_cm:
        st.subheader("Confusion Matrices (Test Set Patients)")
        cm_fig = plot_confusion_matrices(eval_data["confusion_matrices"])
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown(
            """
            **Diagnostic Breakdown:**
            - **True Positive (TP):** Correctly predicted sepsis onset.
            - **False Negative (FN):** Missed sepsis events (highest clinical danger).
            - **False Positive (FP):** Unnecessary alarms on non-septic patients.
            - **True Negative (TN):** Correctly classified stable patients.
            """
        )

    with tab_pr:
        st.subheader("Precision-Recall (PR) Curves")
        pr_fig = plot_pr_curves(eval_data["pr_curves"])
        st.plotly_chart(pr_fig, use_container_width=True)
        st.caption("Because ICU sepsis prevalence is low (~15-20%), PR-AUC is more informative than ROC-AUC for evaluating positive predictive power.")

    with tab_feat:
        st.subheader("Top Physiological & Laboratory Early Warning Biomarkers")
        feat_fig = plot_feature_importance(st.session_state.importances)
        st.plotly_chart(feat_fig, use_container_width=True)
        st.markdown(
            """
            **Key Predictive Signals Identified:**
            - **Lactate trend / peak:** Indicator of anaerobic metabolism and occult tissue hypoperfusion.
            - **Shock Index (HR / SBP):** Early sign of circulatory collapse before overt blood pressure drops.
            - **Respiratory Rate:** Tachypnea is one of the earliest compensation mechanisms in systemic inflammation.
            - **Platelet count drops:** Early marker of coagulopathy and endothelial activation.
            """
        )


# ------------------------------------------------------------------------------
# 7. PAGE 4: PREDICTION LAB (INTERACTIVE CALCULATOR)
# ------------------------------------------------------------------------------
elif selected_page == "Patient Prediction Lab":
    st.title("Patient Early Warning Prediction Lab")
    st.markdown("Interactive inference system to evaluate prospective patient measurements using trained machine learning models.")

    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ Demonstration Output Notice:</strong> Model-estimated probabilities and class outputs are mathematical demonstrations 
            derived from statistical training data. They are <strong>not clinical diagnostic assessments</strong>, validated risk scores, or approved clinical alerts. 
            Do not use these outputs for patient triage or therapeutic decisions.
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_single, tab_batch = st.tabs([
        "📋 Single Patient Measurement Analysis",
        "📂 Prospective Batch CSV Prediction Mode"
    ])

    # Preset clinical scenarios
    PRESETS = {
        "Custom Input": {
            "pid": "PT-2026-001", "timestamp": "Hour 12.0",
            "hr": 80.0, "rr": 18.0, "temp": 37.0, "spo2": 97.0,
            "sbp": 118.0, "dbp": 76.0, "map": 90.0,
            "wbc": 7.5, "lactate": 1.2, "creatinine": 0.9, "creat": 0.9, "platelets": 240.0
        },
        "Scenario 1: Stable Patient (Normal Physiological Baseline)": {
            "pid": "PT-STABLE-101", "timestamp": "Hour 06.0",
            "hr": 72.0, "rr": 15.0, "temp": 36.8, "spo2": 98.5,
            "sbp": 120.0, "dbp": 78.0, "map": 92.0,
            "wbc": 6.8, "lactate": 1.0, "creatinine": 0.8, "creat": 0.8, "platelets": 260.0
        },
        "Scenario 2: Compensated Systemic Response (Early Warning Signs)": {
            "pid": "PT-COMP-204", "timestamp": "Hour 14.0",
            "hr": 104.0, "rr": 23.0, "temp": 38.3, "spo2": 94.0,
            "sbp": 98.0, "dbp": 62.0, "map": 74.0,
            "wbc": 14.8, "lactate": 2.6, "creatinine": 1.4, "creat": 1.4, "platelets": 155.0
        },
        "Scenario 3: Severe Hemodynamic Decompensation (Septic Shock Profile)": {
            "pid": "PT-SHOCK-309", "timestamp": "Hour 22.0",
            "hr": 128.0, "rr": 29.0, "temp": 39.2, "spo2": 90.5,
            "sbp": 82.0, "dbp": 46.0, "map": 58.0,
            "wbc": 21.4, "lactate": 4.8, "creatinine": 2.4, "creat": 2.4, "platelets": 72.0
        }
    }

    with tab_single:
        st.subheader("Patient Measurement Analysis")
        st.markdown(
            "<p style='font-size:0.9rem; color:#475569; margin-top:-8px;'>"
            "Enter current physiological parameters and laboratory values. Measurements are validated against physiological bounds before model evaluation."
            "</p>",
            unsafe_allow_html=True
        )

        col_preset, col_clear = st.columns([3, 1])
        with col_preset:
            preset_choice = st.selectbox(
                "Load Clinical Scenario Preset:",
                list(PRESETS.keys()),
                index=list(PRESETS.keys()).index(st.session_state.get("preset_selection", "Custom Input")) if st.session_state.get("preset_selection") in PRESETS else 0,
                key="preset_selector_dropdown"
            )

        if preset_choice != st.session_state.get("preset_selection"):
            st.session_state.preset_selection = preset_choice
            p_data = PRESETS[preset_choice]
            st.session_state["f_pid"] = p_data["pid"]
            st.session_state["f_ts"] = p_data["timestamp"]
            st.session_state["f_hr"] = p_data["hr"]
            st.session_state["f_rr"] = p_data["rr"]
            st.session_state["f_temp"] = p_data["temp"]
            st.session_state["f_spo2"] = p_data["spo2"]
            st.session_state["f_sbp"] = p_data["sbp"]
            st.session_state["f_dbp"] = p_data["dbp"]
            st.session_state["f_map"] = p_data["map"]
            st.session_state["f_wbc"] = p_data["wbc"]
            st.session_state["f_lactate"] = p_data["lactate"]
            st.session_state["f_creatinine"] = p_data.get("creatinine", p_data.get("creat", 0.9))
            st.session_state["f_creat"] = st.session_state["f_creatinine"]
            st.session_state["f_plat"] = p_data["platelets"]
            st.session_state.analysis_completed = False
            st.session_state.prediction_results = None

        # Fallback initialization
        def_vals = PRESETS[st.session_state.get("preset_selection", "Custom Input")]

        # Patient Metadata Section
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            in_pid = st.text_input("Patient Identifier", value=st.session_state.get("f_pid", def_vals["pid"]), key="input_patient_id")
        with col_m2:
            in_ts = st.text_input("Measurement Timestamp / Observation Hour", value=st.session_state.get("f_ts", def_vals["timestamp"]), key="input_timestamp")

        st.markdown("#### Vital Signs")
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            in_hr = st.number_input("Heart Rate (bpm)", min_value=20.0, max_value=300.0, value=float(st.session_state.get("f_hr", def_vals["hr"])), step=1.0, format="%.1f", help="Physiological range: 20 - 300 bpm")
        with col_v2:
            in_rr = st.number_input("Respiratory Rate (breaths/min)", min_value=4.0, max_value=80.0, value=float(st.session_state.get("f_rr", def_vals["rr"])), step=1.0, format="%.1f", help="Physiological range: 4 - 80 breaths/min")
        with col_v3:
            in_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=float(st.session_state.get("f_temp", def_vals["temp"])), step=0.1, format="%.1f", help="Physiological range: 30.0 - 45.0 °C")
        with col_v4:
            in_spo2 = st.number_input("Oxygen Saturation - SpO2 (%)", min_value=50.0, max_value=100.0, value=float(st.session_state.get("f_spo2", def_vals["spo2"])), step=0.5, format="%.1f", help="Physiological range: 50.0 - 100.0 %")

        st.markdown("#### Blood Pressure & Perfusion")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            in_sbp = st.number_input("Systolic BP (mmHg)", min_value=30.0, max_value=300.0, value=float(st.session_state.get("f_sbp", def_vals["sbp"])), step=1.0, format="%.1f", help="Must be greater than Diastolic BP. Range: 30 - 300 mmHg")
        with col_b2:
            in_dbp = st.number_input("Diastolic BP (mmHg)", min_value=10.0, max_value=200.0, value=float(st.session_state.get("f_dbp", def_vals["dbp"])), step=1.0, format="%.1f", help="Range: 10 - 200 mmHg")
        with col_b3:
            auto_map_calc = round((2.0 * float(st.session_state.get("f_dbp", def_vals["dbp"])) + float(st.session_state.get("f_sbp", def_vals["sbp"]))) / 3.0, 1)
            in_map = st.number_input("Mean Arterial Pressure - MAP (mmHg)", min_value=20.0, max_value=250.0, value=float(st.session_state.get("f_map", auto_map_calc)), step=1.0, format="%.1f", help="Must lie between DBP and SBP.")

        # Real-time calculated bedside indices feedback
        curr_si = in_hr / max(1.0, in_sbp)
        curr_pp = in_sbp - in_dbp
        st.markdown(
            f"""
            <div style="background: rgba(14, 165, 233, 0.05); border: 1px solid rgba(14, 165, 233, 0.2); border-radius: 6px; padding: 8px 12px; margin: 8px 0 16px 0; font-size: 0.85rem; color: #334155;">
                <strong>Bedside Perfusion Indicators:</strong> &nbsp;
                Shock Index (HR/SBP): <strong style="color: {'#b91c1c' if curr_si > 0.85 else '#0284c7'};">{curr_si:.2f}</strong> &nbsp;|&nbsp;
                Pulse Pressure (SBP-DBP): <strong>{curr_pp:.1f} mmHg</strong> &nbsp;|&nbsp;
                Reference MAP: <strong>{((2*in_dbp + in_sbp)/3):.1f} mmHg</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Laboratory Measurements")
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            in_wbc = st.number_input("WBC Count (x10³/µL)", min_value=0.1, max_value=150.0, value=float(st.session_state.get("f_wbc", def_vals["wbc"])), step=0.1, format="%.1f", help="Range: 0.1 - 150.0 x10³/µL")
        with col_l2:
            in_lactate = st.number_input("Blood Lactate (mmol/L)", min_value=0.1, max_value=30.0, value=float(st.session_state.get("f_lactate", def_vals["lactate"])), step=0.1, format="%.1f", help="Range: 0.1 - 30.0 mmol/L")
        with col_l3:
            def_creat_val = def_vals.get("creatinine", def_vals.get("creat", 0.9))
            in_creatinine = st.number_input(
                "Serum Creatinine (mg/dL)",
                min_value=0.1,
                max_value=25.0,
                value=float(st.session_state.get("f_creatinine", st.session_state.get("f_creat", def_creat_val))),
                step=0.1,
                format="%.1f",
                help="Range: 0.1 - 25.0 mg/dL"
            )
            in_creat = in_creatinine
        with col_l4:
            in_plat = st.number_input("Platelets (x10³/µL)", min_value=5.0, max_value=1500.0, value=float(st.session_state.get("f_plat", def_vals["platelets"])), step=1.0, format="%.1f", help="Range: 5.0 - 1500.0 x10³/µL")

        st.markdown("<br>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([1.5, 1])
        with col_act1:
            btn_analyze = st.button("🔍 Analyze Measurements", type="primary", use_container_width=True)
        with col_act2:
            btn_reset = st.button("🔄 Reset Form", use_container_width=True)

        # Reset button execution
        if btn_reset:
            st.session_state.analysis_completed = False
            st.session_state.prediction_results = None
            st.session_state.submitted_inputs = None
            st.session_state.preset_selection = "Custom Input"
            # Reset values to default custom baseline
            c_base = PRESETS["Custom Input"]
            st.session_state["f_pid"] = c_base["pid"]
            st.session_state["f_ts"] = c_base["timestamp"]
            st.session_state["f_hr"] = c_base["hr"]
            st.session_state["f_rr"] = c_base["rr"]
            st.session_state["f_temp"] = c_base["temp"]
            st.session_state["f_spo2"] = c_base["spo2"]
            st.session_state["f_sbp"] = c_base["sbp"]
            st.session_state["f_dbp"] = c_base["dbp"]
            st.session_state["f_map"] = c_base["map"]
            st.session_state["f_wbc"] = c_base["wbc"]
            st.session_state["f_lactate"] = c_base["lactate"]
            st.session_state["f_creatinine"] = c_base.get("creatinine", c_base.get("creat", 0.9))
            st.session_state["f_creat"] = st.session_state["f_creatinine"]
            st.session_state["f_plat"] = c_base["platelets"]
            st.success("Form inputs and previous analysis results reset.")
            st.rerun()

        # Analysis button execution
        if btn_analyze:
            input_payload = {
                "heart_rate": in_hr,
                "resp_rate": in_rr,
                "temperature": in_temp,
                "sbp": in_sbp,
                "dbp": in_dbp,
                "map": in_map,
                "oxygen_saturation": in_spo2,
                "wbc": in_wbc,
                "lactate": in_lactate,
                "creatinine": in_creatinine,
                "platelets": in_plat
            }

            # Strict Physiological Validation
            is_valid, validation_errors = validate_single_measurement(input_payload)
            if not is_valid:
                st.error("### ⚠️ Measurement Validation Error")
                for err in validation_errors:
                    st.error(f"• {err}")
            else:
                single_models = st.session_state.get("single_models")
                if single_models is None:
                    st.error("⚠️ Trained models are not loaded. Please initialize models via the 'Model Training & Evaluation' page.")
                else:
                    with st.spinner("Executing candidate machine-learning pipelines (Logistic Regression, Random Forest, XGBoost)..."):
                        feature_vector = build_single_measurement_feature_vector(input_payload)
                        preds = predict_single_measurement(single_models, feature_vector)

                        # Store in session state so inputs & outputs persist
                        st.session_state.prediction_results = preds
                        st.session_state.submitted_inputs = {
                            "patient_id": in_pid,
                            "timestamp": in_ts,
                            **input_payload,
                            "shock_index": in_hr / max(1.0, in_sbp),
                            "pulse_pressure": in_sbp - in_dbp
                        }
                        # Save current field values into session state
                        for k, v in input_payload.items():
                            st.session_state[f"f_{k}"] = v
                        st.session_state["f_creat"] = in_creatinine
                        st.session_state["f_pid"] = in_pid
                        st.session_state["f_ts"] = in_ts
                        st.session_state.analysis_completed = True

        # Render Results Section (only displayed after successful submission)
        if st.session_state.analysis_completed and st.session_state.prediction_results is not None:
            preds = st.session_state.prediction_results
            sub_inputs = st.session_state.submitted_inputs

            st.markdown("---")
            st.subheader("Model Classification Results")
            st.markdown(
                """
                <div style="background: rgba(14, 165, 233, 0.08); border-left: 3px solid #0ea5e9; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; color: #0f172a; margin-bottom: 20px;">
                    <strong>Output Interpretation:</strong> The results below reflect the mathematical classifications produced by the three candidate models 
                    for the submitted patient observation vector. The output represents the model's classification of the dataset-defined outcome. 
                    It is an algorithmic demonstration output and does <strong>not</strong> constitute a confirmed medical diagnosis or clinically validated patient risk.
                </div>
                """,
                unsafe_allow_html=True
            )

            # Model Output Cards
            c_lr, c_rf, c_xgb = st.columns(3)

            with c_lr:
                lr_data = preds["Logistic Regression"]
                badge_bg = "rgba(239, 68, 68, 0.12)" if lr_data["is_positive"] else "rgba(2, 132, 199, 0.12)"
                badge_border = "#ef4444" if lr_data["is_positive"] else "#0284c7"
                badge_text = "#dc2626" if lr_data["is_positive"] else "#0369a1"
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                        <div style="font-size:0.82rem; font-weight:700; color:#64748b; letter-spacing:0.5px;">LOGISTIC REGRESSION</div>
                        <div style="display:inline-block; margin:8px 0; padding:4px 10px; border-radius:20px; background:{badge_bg}; border:1px solid {badge_border}; font-size:0.88rem; font-weight:700; color:{badge_text};">
                            {lr_data['label']}
                        </div>
                        <div style="font-size:2rem; font-weight:800; color:#0f172a; line-height:1.2;">{lr_data['prob_percent']}</div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">Estimated Positive Class Probability</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c_rf:
                rf_data = preds["Random Forest"]
                badge_bg = "rgba(239, 68, 68, 0.12)" if rf_data["is_positive"] else "rgba(13, 148, 136, 0.12)"
                badge_border = "#ef4444" if rf_data["is_positive"] else "#0d9488"
                badge_text = "#dc2626" if rf_data["is_positive"] else "#0f766e"
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                        <div style="font-size:0.82rem; font-weight:700; color:#64748b; letter-spacing:0.5px;">RANDOM FOREST</div>
                        <div style="display:inline-block; margin:8px 0; padding:4px 10px; border-radius:20px; background:{badge_bg}; border:1px solid {badge_border}; font-size:0.88rem; font-weight:700; color:{badge_text};">
                            {rf_data['label']}
                        </div>
                        <div style="font-size:2rem; font-weight:800; color:#0f172a; line-height:1.2;">{rf_data['prob_percent']}</div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">Estimated Positive Class Probability</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c_xgb:
                xgb_data = preds["XGBoost"]
                badge_bg = "rgba(239, 68, 68, 0.12)" if xgb_data["is_positive"] else "rgba(30, 58, 138, 0.12)"
                badge_border = "#ef4444" if xgb_data["is_positive"] else "#1e3a8a"
                badge_text = "#dc2626" if xgb_data["is_positive"] else "#1e3a8a"
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                        <div style="font-size:0.82rem; font-weight:700; color:#64748b; letter-spacing:0.5px;">XGBOOST CLASSIFIER</div>
                        <div style="display:inline-block; margin:8px 0; padding:4px 10px; border-radius:20px; background:{badge_bg}; border:1px solid {badge_border}; font-size:0.88rem; font-weight:700; color:{badge_text};">
                            {xgb_data['label']}
                        </div>
                        <div style="font-size:2rem; font-weight:800; color:#0f172a; line-height:1.2;">{xgb_data['prob_percent']}</div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">Estimated Positive Class Probability</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Model Output Comparison Chart
            st.markdown("<br>", unsafe_allow_html=True)
            comp_chart = plot_patient_prediction_comparison(preds)
            st.plotly_chart(comp_chart, use_container_width=True)

            # Submitted Inputs Verification Table
            st.markdown("#### Submitted Input Verification Table")
            st.caption("Detailed record of parameters passed to candidate pipelines for audit and clinical review.")

            verif_rows = [
                {"Category": "Patient Identification", "Clinical Parameter": "Patient ID", "Value": str(sub_inputs.get("patient_id", "N/A")), "Unit": "Identifier"},
                {"Category": "Patient Identification", "Clinical Parameter": "Observation Timestamp", "Value": str(sub_inputs.get("timestamp", "N/A")), "Unit": "ICU Time"},
                {"Category": "Vital Signs", "Clinical Parameter": "Heart Rate", "Value": f"{sub_inputs['heart_rate']:.1f}", "Unit": "bpm"},
                {"Category": "Vital Signs", "Clinical Parameter": "Respiratory Rate", "Value": f"{sub_inputs['resp_rate']:.1f}", "Unit": "breaths/min"},
                {"Category": "Vital Signs", "Clinical Parameter": "Body Temperature", "Value": f"{sub_inputs['temperature']:.1f}", "Unit": "°C"},
                {"Category": "Vital Signs", "Clinical Parameter": "Oxygen Saturation (SpO2)", "Value": f"{sub_inputs['oxygen_saturation']:.1f}", "Unit": "%"},
                {"Category": "Blood Pressure", "Clinical Parameter": "Systolic BP", "Value": f"{sub_inputs['sbp']:.1f}", "Unit": "mmHg"},
                {"Category": "Blood Pressure", "Clinical Parameter": "Diastolic BP", "Value": f"{sub_inputs['dbp']:.1f}", "Unit": "mmHg"},
                {"Category": "Blood Pressure", "Clinical Parameter": "Mean Arterial Pressure (MAP)", "Value": f"{sub_inputs['map']:.1f}", "Unit": "mmHg"},
                {"Category": "Hemodynamics", "Clinical Parameter": "Shock Index (HR / SBP)", "Value": f"{sub_inputs['shock_index']:.2f}", "Unit": "ratio"},
                {"Category": "Hemodynamics", "Clinical Parameter": "Pulse Pressure (SBP - DBP)", "Value": f"{sub_inputs['pulse_pressure']:.1f}", "Unit": "mmHg"},
                {"Category": "Laboratory", "Clinical Parameter": "White Blood Cell (WBC) Count", "Value": f"{sub_inputs['wbc']:.1f}", "Unit": "x10³/µL"},
                {"Category": "Laboratory", "Clinical Parameter": "Blood Lactate", "Value": f"{sub_inputs['lactate']:.1f}", "Unit": "mmol/L"},
                {"Category": "Laboratory", "Clinical Parameter": "Serum Creatinine", "Value": f"{sub_inputs['creatinine']:.1f}", "Unit": "mg/dL"},
                {"Category": "Laboratory", "Clinical Parameter": "Platelet Count", "Value": f"{sub_inputs['platelets']:.1f}", "Unit": "x10³/µL"}
            ]
            st.dataframe(pd.DataFrame(verif_rows), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: PROSPECTIVE BATCH CSV PREDICTION MODE
    # --------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Patient Measurement Prediction")
        st.markdown(
            """
            Upload a prospective CSV dataset containing multiple patient records to perform automated batch inference.
            Each record is validated and processed through the exact candidate model pipelines.
            """
        )

        # Download Template Button
        sample_batch_df = pd.DataFrame([
            {
                "patient_id": f"PT-BATCH-{i:03d}",
                "timestamp": f"Hour {i*2:02d}.0",
                "heart_rate": 72.0 + (i % 5) * 11.0,
                "resp_rate": 15.0 + (i % 4) * 3.0,
                "temperature": 36.7 + (i % 3) * 0.7,
                "sbp": 124.0 - (i % 4) * 11.0,
                "dbp": 78.0 - (i % 4) * 6.0,
                "map": 93.0 - (i % 4) * 7.5,
                "oxygen_saturation": 98.0 - (i % 3) * 2.0,
                "wbc": 7.2 + (i % 4) * 2.4,
                "lactate": 1.1 + (i % 3) * 0.9,
                "creatinine": 0.8 + (i % 3) * 0.4,
                "platelets": 255.0 - (i % 5) * 32.0
            }
            for i in range(1, 11)
        ])

        col_t1, col_t2 = st.columns([1.5, 2])
        with col_t1:
            st.download_button(
                "📄 Download Sample Batch CSV Template",
                data=sample_batch_df.to_csv(index=False).encode("utf-8"),
                file_name="sample_patient_batch_template.csv",
                mime="text/csv",
                help="Download a properly formatted CSV template with required clinical columns."
            )

        uploaded_batch = st.file_uploader(
            "Upload Patient Measurements CSV",
            type=["csv"],
            key="batch_csv_uploader"
        )

        if uploaded_batch is not None:
            try:
                batch_df = pd.read_csv(uploaded_batch)
                if "patient_id" in batch_df.columns:
                    batch_df["patient_id"] = batch_df["patient_id"].astype(str)
                st.success(f"Loaded CSV file with {len(batch_df)} patient records and {len(batch_df.columns)} columns.")

                # Validate required measurement columns
                missing_cols = [c for c in VITAL_AND_LAB_COLS if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing required clinical columns in uploaded CSV: {', '.join(missing_cols)}")
                    st.info(f"Required columns are: {', '.join(VITAL_AND_LAB_COLS)}")
                else:
                    st.markdown("##### Uploaded Data Preview (First 5 Rows)")
                    st.dataframe(batch_df.head(5), use_container_width=True)

                    if st.button("⚡ Run Batch Model Analysis", type="primary", use_container_width=True):
                        single_models = st.session_state.get("single_models")
                        if single_models is None:
                            st.error("Trained models not available. Please train models first.")
                        else:
                            with st.spinner(f"Evaluating {len(batch_df)} records across candidate models..."):
                                proc_df = batch_df.copy()
                                proc_df["shock_index"] = proc_df["heart_rate"] / proc_df["sbp"].clip(lower=1.0)
                                proc_df["pulse_pressure"] = proc_df["sbp"] - proc_df["dbp"]

                                # Run batch inference
                                results_df = predict_batch(single_models, proc_df[SINGLE_MEASUREMENT_FEATURES])

                                # Attach identification fields if available
                                for id_col in ["patient_id", "timestamp"]:
                                    if id_col in batch_df.columns:
                                        results_df.insert(0, id_col, batch_df[id_col])

                                st.session_state.batch_results = results_df

                    if "batch_results" in st.session_state and st.session_state.batch_results is not None:
                        res_df = st.session_state.batch_results
                        st.markdown("---")
                        st.subheader("Batch Prediction Results")

                        # Summary metric cards
                        b1, b2, b3 = st.columns(3)
                        total_rec = len(res_df)
                        pos_consensus = int((res_df["consensus_output"] == "Positive class").sum())
                        neg_consensus = total_rec - pos_consensus

                        b1.metric("Total Records Evaluated", total_rec)
                        b2.metric("Consensus: Positive Class", f"{pos_consensus} ({pos_consensus/total_rec:.1%})")
                        b3.metric("Consensus: Negative Class", f"{neg_consensus} ({neg_consensus/total_rec:.1%})")

                        # Distribution Chart
                        dist_fig = plot_batch_predictions_summary(res_df)
                        st.plotly_chart(dist_fig, use_container_width=True)

                        # Preview results table
                        st.markdown("##### Detailed Predictions Preview")
                        st.dataframe(res_df.head(10), use_container_width=True)

                        # Download Predictions CSV
                        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Batch Predictions CSV",
                            data=csv_bytes,
                            file_name="sepsisguard_batch_model_outputs.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
            except Exception as ex:
                st.error(f"Error processing uploaded CSV: {ex}")


# ------------------------------------------------------------------------------
# 8. PAGE 5: CLINICAL & ML EDUCATIONAL INFORMATION
# ------------------------------------------------------------------------------
elif selected_page == "Clinical & ML Guide":
    st.title("Clinical & Machine Learning Guide")
    st.markdown("Comprehensive educational background designed for college presentations and clinical data science coursework.")

    e_tab1, e_tab2, e_tab3, e_tab4 = st.tabs([
        "🩺 Understanding Sepsis",
        "🧪 Clinical Features & Biomarkers",
        "🤖 Algorithms Explained",
        "📊 Evaluation Metrics & Limitations"
    ])

    with e_tab1:
        st.subheader("What is Sepsis?")
        st.markdown(
            """
            **Definition (Sepsis-3 International Consensus):**
            > Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection.

            **Why Sepsis Progresses Rapidly:**
            1. **Infection Trigger:** Bacteria or viruses enter the bloodstream, triggering immune activation.
            2. **Systemic Inflammation (SIRS):** Cytokines cause widespread endothelial leakage and vasodilation.
            3. **Tissue Hypoperfusion:** Blood pressure drops, leading to microcirculatory failure and cellular hypoxia.
            4. **Septic Shock:** Severe metabolic acidosis, hyperlactatemia, and refractory hypotension requiring vasopressors.
            5. **The Value of Early Warning:** Mortality increases by approximately **7.6% for every hour** antibiotic administration is delayed following septic shock onset.
            """
        )

    with e_tab2:
        st.subheader("Clinical Features & Physiological Meaning")
        st.markdown(
            """
            | Parameter | Normal Range | Clinical Role in Sepsis Detection |
            | :--- | :--- | :--- |
            | **Heart Rate** | 60–100 bpm | Tachycardia (>90 bpm) is a primary SIRS compensatory response to infection and fever. |
            | **Respiratory Rate** | 12–20 /min | Tachypnea (≥22 /min) is often the earliest warning sign of metabolic acidosis compensation. |
            | **Temperature** | 36.5–37.5 °C | Both hyperthermia (>38.3°C) and hypothermia (<36.0°C) reflect severe systemic dysregulation. |
            | **Blood Pressure (SBP/DBP)** | 120/80 mmHg | Progressive hypotension indicates loss of systemic vascular resistance. |
            | **MAP** | 70–100 mmHg | Mean Arterial Pressure measures organ perfusion; $\\text{MAP} < 65\\text{ mmHg}$ defines shock. |
            | **Oxygen Saturation** | 95–100% | Hypoxemia indicates acute lung injury or secondary respiratory failure. |
            | **Blood Lactate** | 0.5–2.0 mmol/L | Hallmark marker of cellular hypoxia and anaerobic glycolysis ($>2\\text{ mmol/L}$ is high risk). |
            | **White Blood Cells (WBC)** | 4.5–11.0 k/µL | Leukocytosis (>12 k/µL) or leukopenia (<4 k/µL) reflects severe immune mobilization. |
            | **Serum Creatinine** | 0.7–1.3 mg/dL | Acute rise signals early acute kidney injury (AKI) from hypoperfusion. |
            | **Platelet Count** | 150–450 k/µL | Thrombocytopenia (<100 k/µL) reflects consumption coagulopathy and microthrombi (DIC). |
            """
        )

    with e_tab3:
        st.subheader("Machine Learning Architectures")
        st.markdown(
            """
            #### 1. Logistic Regression
            - **How it works:** Models log-odds of sepsis as a weighted linear combination of standardized features:
              $$\\log\\left(\\frac{p}{1-p}\\right) = \\beta_0 + \\sum_{i=1}^n \\beta_i x_i$$
            - **Advantages:** Highly transparent, interpretable coefficients, fast inference.
            - **Limitations:** Cannot automatically capture complex non-linear interactions without explicit feature engineering.

            #### 2. Random Forest
            - **How it works:** An ensemble of $B$ de-correlated decision trees trained on bootstrap samples with random feature subsampling. Predictions are averaged:
              $$\\hat{p} = \\frac{1}{B} \\sum_{b=1}^B T_b(x)$$
            - **Advantages:** Handles non-linear biomarker interactions, robust against outliers, provides feature importance.

            #### 3. XGBoost (Extreme Gradient Boosting)
            - **How it works:** Iteratively adds decision trees, where each subsequent tree minimizes the residual loss gradient of the ensemble:
              $$\\mathcal{L}^{(t)} = \\sum_i l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i)) + \\Omega(f_t)$$
            - **Advantages:** State-of-the-art performance on tabular clinical data, supports scale_pos_weight to counter class imbalance.
            """
        )

    with e_tab4:
        st.subheader("Evaluation Metrics & Clinical Research Limitations")
        st.markdown(
            """
            #### Metrics Guide:
            - **Sensitivity (Recall):** $\\frac{TP}{TP + FN}$ — Critical in sepsis: what percentage of sick patients did the algorithm catch?
            - **Specificity:** $\\frac{TN}{TN + FP}$ — What percentage of healthy patients were spared false alarms?
            - **Precision (PPV):** $\\frac{TP}{TP + FP}$ — When an alarm sounds, what is the probability the patient truly has sepsis?
            - **ROC-AUC:** Area under the sensitivity vs. (1-specificity) curve across all possible classification thresholds.

            #### Why Synthetic Data & What Real Clinical Validation Requires:
            - **Synthetic Data Utility:** Allows rapid algorithm prototyping, pedagogical demonstrations, and software verification without risking protected health information (PHI) under HIPAA.
            - **Real-World Translation Requirements:**
              1. **Prospective Clinical Trials:** Retrospective high AUC does not guarantee improved patient survival.
              2. **External Hospital Validation:** Models must be tested on different EHR platforms (Epic, Cerner, MIMIC-IV) and patient demographics to detect distribution shift.
              3. **Alarm Fatigue Mitigation:** High false alert rates cause nurses and intensivists to silence warnings, defeating the purpose of early warning.
            """
        )
