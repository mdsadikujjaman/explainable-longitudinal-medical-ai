"""Streamlit web application for Explainable Longitudinal Medical AI.

Implements all six Workstream 5 user interfaces:
1. Login
2. Patient Search
3. Patient Overview
4. Prediction
5. Explainability
6. Dataset Audit
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure local 'src' directory takes priority in sys.path when executed via Streamlit CLI
SRC_PATH = str(Path(__file__).resolve().parents[2])
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pandas as pd
import streamlit as st

try:
    from nacc_progression.frontend.api_client import APIClient
except ImportError:
    from api_client import APIClient  # type: ignore[no-redef]

try:
    from nacc_progression.frontend.components import (
        render_disclaimer_notice,
        render_header,
        render_patient_timeline,
        render_prediction_badge,
        render_synthetic_notice,
        render_trajectory_chart,
        render_xai_dashboard,
    )
except ImportError:
    try:
        from components import (  # type: ignore[no-redef]
            render_disclaimer_notice,
            render_header,
            render_patient_timeline,
            render_prediction_badge,
            render_synthetic_notice,
            render_trajectory_chart,
            render_xai_dashboard,
        )
    except ImportError:
        # Emergency fallback for cached environments
        from nacc_progression.frontend.components import (  # type: ignore[no-redef]
            render_disclaimer_notice,
            render_header,
            render_patient_timeline,
            render_prediction_badge,
            render_trajectory_chart,
            render_xai_dashboard,
        )

        def render_synthetic_notice(subject_key: str | None = None) -> None:  # type: ignore[no-redef]
            """Display prominent synthetic simulation badge and research boundary notice."""
            key_text = f" for `{subject_key}`" if subject_key else ""
            st.warning(
                f"🧪 **SYNTHETIC DEMO RECORD{key_text}**\n\n"
                "This patient trajectory, imaging measurements, clinical scores, and risk projections are generated entirely "
                "by an *in silico* synthetic simulator for interface demonstration. They do not represent real human participants "
                "or actual clinical observations from any private medical registry.",
                icon="🧪",
            )

from nacc_progression.platform.schemas import (
    ExplanationRequest,
    PredictionRequest,
)

# Page configuration
st.set_page_config(
    page_title="Explainable Longitudinal Medical AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_client() -> APIClient:
    """Retrieve or initialize persistent API client in session state."""
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = APIClient()
    return st.session_state["api_client"]


def init_session() -> None:
    """Initialize session state defaults."""
    if "authenticated" not in st.session_state:
        # Default authenticated for smooth demonstration review
        st.session_state["authenticated"] = True
        st.session_state["username"] = "researcher"
        st.session_state["role"] = "researcher"
    if "selected_subject_key" not in st.session_state:
        st.session_state["selected_subject_key"] = "n74_demo_rapid_progressor"
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Patient Search"


init_session()
client = get_client()
model_info = client.get_model_info()


# ============================================================================
# Sidebar Navigation & System Status
# ============================================================================

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    pages = [
        "Login",
        "Patient Search",
        "Patient Overview",
        "Prediction",
        "Explainability",
        "Dataset Audit",
    ]

    selected_page = st.radio(
        "Go to page:",
        pages,
        index=pages.index(st.session_state["current_page"])
        if st.session_state["current_page"] in pages
        else 1,
    )
    st.session_state["current_page"] = selected_page

    st.markdown("---")
    st.markdown("### 🔐 User Session")
    if st.session_state.get("authenticated"):
        st.success(f"👤 **{st.session_state.get('username')}** ({st.session_state.get('role')})")
        if st.button("Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.session_state["role"] = None
            client.logout()
            st.rerun()
    else:
        st.warning("Not signed in")
        if st.button("Sign In →", use_container_width=True):
            st.session_state["current_page"] = "Login"
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Engine Status")
    if model_info.is_demo_mode:
        st.info("🟢 **Mode**: Synthetic Demo Engine\n\n(Reproducible Longitudinal In Silico Simulator)")
    else:
        st.success("🟢 **Mode**: PyTorch GRU Checkpoint Active")
    st.caption(f"Input features: {model_info.imaging_feature_count} MRI + {model_info.clinical_feature_count} Clinical")


# Render common banner
render_header()


# ============================================================================
# Page 1: Login
# ============================================================================

if selected_page == "Login":
    st.subheader("🔑 Platform Sign-In")
    st.markdown("Authenticate with role-based credentials to access longitudinal patient records.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        with st.form("login_form"):
            default_user = "researcher" if model_info.is_demo_mode else ""
            default_pass = "researcher123" if model_info.is_demo_mode else ""
            username_input = st.text_input("Username", value=default_user)
            password_input = st.text_input("Password", type="password", value=default_pass)
            submit_btn = st.form_submit_button("Sign In", use_container_width=True)

            if submit_btn:
                if client.login(username_input, password_input):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username_input
                    st.session_state["role"] = client.current_user["role"] if client.current_user else "researcher"
                    st.success("Authentication successful! Redirecting to Patient Search...")
                    st.session_state["current_page"] = "Patient Search"
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid username or password.")

    with col2:
        if model_info.is_demo_mode:
            st.markdown("##### 🧪 Synthetic Demo Mode")
            st.caption("Quick sign-in buttons enabled for simulated cohort evaluation:")

            if st.button("👤 Sign In as Researcher", use_container_width=True):
                if client.login("researcher", "researcher123"):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = "researcher"
                    st.session_state["role"] = "researcher"
                    st.session_state["current_page"] = "Patient Search"
                    st.rerun()

            if st.button("🩺 Sign In as Clinician", use_container_width=True):
                if client.login("clinician", "clinician123"):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = "clinician"
                    st.session_state["role"] = "clinician"
                    st.session_state["current_page"] = "Patient Search"
                    st.rerun()

            from nacc_progression.platform.database import get_user_by_username
            if get_user_by_username("admin") is not None:
                if st.button("🛡️ Sign In as Administrator", use_container_width=True):
                    if client.login("admin", "admin123"):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = "admin"
                        st.session_state["role"] = "admin"
                        st.session_state["current_page"] = "Dataset Audit"
                        st.rerun()
        else:
            st.markdown("##### 🔒 Private Research Mode Active")
            st.info(
                "Synthetic demo accounts are disabled in private research mode.\n\n"
                "Please authenticate using your institutionally provisioned credentials. "
                "Contact your system administrator if you require account registration."
            )


# ============================================================================
# Page 2: Patient Search
# ============================================================================

elif selected_page == "Patient Search":
    st.subheader("🔍 Cohort Directory & Patient Search")
    st.markdown("Explore longitudinal cohort records preprocessed under the research protocol specification.")

    if model_info.is_demo_mode:
        st.warning(
            "🧪 **SYNTHETIC DEMO COHORT ACTIVE**: The profiles listed below are simulated demonstration trajectories "
            "generated *in silico*. Fictional data is utilized to validate interactive exploration workflows without "
            "accessing restricted clinical repositories.",
            icon="🧪",
        )

    render_disclaimer_notice()

    # Search & Filters
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search_query = st.text_input("Search by Subject Key", placeholder="e.g. n74_demo or progressor")
    with f2:
        label_opt = st.selectbox("Trajectory Label", ["All", "MCI Progressor (1)", "Stable MCI (0)"])
        label_val = 1 if "Progressor" in label_opt else (0 if "Stable" in label_opt else None)
    with f3:
        min_v = st.slider("Min Visits", min_value=1, max_value=3, value=1)

    patients = client.search_patients(query=search_query, label=label_val, min_visits=min_v)

    if not patients:
        st.warning("No patients found matching the selected filters.")
    else:
        table_rows = []
        for p in patients:
            mri_status = (
                "✅ Quantitative features available"
                if p.has_mri
                else "⚠️ Quantitative features missing or masked"
            )
            row_data = {
                "Subject Key": p.subject_key,
                "Visits": p.visit_count,
                "Cohort Label": "Progressor (AD)" if p.label == 1 else "Stable MCI",
                "Est. Age": p.age_at_anchor,
                "Latest MMSE": p.latest_mmse,
                "CDR Global": p.latest_cdr_global,
                "MRI-derived Features": mri_status,
            }
            if model_info.is_demo_mode:
                row_data["Data Origin"] = "🧪 Synthetic Demo"
            table_rows.append(row_data)

        df_patients = pd.DataFrame(table_rows)
        st.dataframe(df_patients, use_container_width=True, hide_index=True)

        st.caption(
            "ℹ️ **MRI-derived Features**: Indicates whether quantitative regional anatomical measurements "
            "(e.g., FreeSurfer volumetric ROIs) are present for the anchor visit. The platform uses tabular "
            "derived features rather than raw DICOM/NIfTI neuroimaging scans."
        )

        st.markdown("#### 🎯 Select Patient for Detailed Examination")
        patient_keys = [p.subject_key for p in patients]
        chosen_key = st.selectbox(
            "Choose a patient to inspect:",
            patient_keys,
            index=patient_keys.index(st.session_state["selected_subject_key"])
            if st.session_state["selected_subject_key"] in patient_keys
            else 0,
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("View Patient Overview →", use_container_width=True, type="primary"):
                st.session_state["selected_subject_key"] = chosen_key
                st.session_state["current_page"] = "Patient Overview"
                st.rerun()
        with b2:
            if st.button("Run Prediction Directly →", use_container_width=True):
                st.session_state["selected_subject_key"] = chosen_key
                st.session_state["current_page"] = "Prediction"
                st.rerun()


# ============================================================================
# Page 3: Patient Overview
# ============================================================================

elif selected_page == "Patient Overview":
    curr_key = st.session_state.get("selected_subject_key", "n74_demo_rapid_progressor")
    demo_badge = " [🧪 SYNTHETIC DEMO]" if model_info.is_demo_mode else ""
    st.subheader(f"👤 Patient Overview: `{curr_key}`{demo_badge}")

    if model_info.is_demo_mode:
        render_synthetic_notice(curr_key)

    detail = client.get_patient_detail(curr_key)
    if detail is None:
        st.error(f"Patient `{curr_key}` not found in the active registry.")
    else:
        s = detail.summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Longitudinal Visits", s.visit_count)
        m2.metric("True Research Label", "Progressor (1)" if s.label == 1 else "Stable (0)")
        m3.metric("Anchor MMSE Score", s.latest_mmse)
        m4.metric("Anchor CDR Global", s.latest_cdr_global)

        st.markdown("---")
        render_patient_timeline(detail)

        st.markdown("---")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🚀 Run AD Progression Prediction →", use_container_width=True, type="primary"):
                st.session_state["current_page"] = "Prediction"
                st.rerun()
        with col_act2:
            if st.button("📊 Inspect Multimodal Explainability →", use_container_width=True):
                st.session_state["current_page"] = "Explainability"
                st.rerun()


# ============================================================================
# Page 4: Prediction
# ============================================================================

elif selected_page == "Prediction":
    curr_key = st.session_state.get("selected_subject_key", "n74_demo_rapid_progressor")
    demo_badge = " [🧪 SYNTHETIC DEMO]" if model_info.is_demo_mode else ""
    st.subheader(f"⚡ Progression Prediction: `{curr_key}`{demo_badge}")

    if model_info.is_demo_mode:
        render_synthetic_notice(curr_key)

    detail = client.get_patient_detail(curr_key)
    if detail is None:
        st.error("Please select a patient from Patient Search first.")
    else:
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.markdown("##### ⚙️ Decision Settings")
            threshold = st.slider(
                "Classification Decision Threshold",
                min_value=0.1,
                max_value=0.9,
                value=float(model_info.recommended_threshold),
                step=0.05,
                help="Threshold calibrated from validation balanced accuracy.",
            )
            run_btn = st.button("Compute Progression Risk", type="primary", use_container_width=True)

        pred_req = PredictionRequest(
            subject_key=curr_key,
            visits=detail.visits,
            imaging_feature_names=detail.imaging_feature_names,
            threshold=threshold,
        )

        # Trigger or show prediction
        if run_btn or "last_prediction" not in st.session_state:
            with st.spinner("Executing temporal inference across visit sequence..."):
                pred_res = client.predict(pred_req)
                st.session_state["last_prediction"] = pred_res

        pred_res = st.session_state["last_prediction"]

        with p_col2:
            render_prediction_badge(pred_res)

        st.markdown("---")
        render_trajectory_chart(pred_res)

        if model_info.is_demo_mode:
            st.caption(
                "⚠️ **Fictional Demonstration Data Notice**: Predictions and progression curves are generated "
                "from simulated demonstration data for interface evaluation and must not be used for clinical prognosis."
            )

        st.markdown("---")
        if st.button("🔍 Explain This Prediction with Integrated Gradients →", use_container_width=True):
            st.session_state["current_page"] = "Explainability"
            st.rerun()


# ============================================================================
# Page 5: Explainability
# ============================================================================

elif selected_page == "Explainability":
    curr_key = st.session_state.get("selected_subject_key", "n74_demo_rapid_progressor")
    demo_badge = " [🧪 SYNTHETIC DEMO]" if model_info.is_demo_mode else ""
    st.subheader(f"🧠 Algorithmic Interpretability (XAI): `{curr_key}`{demo_badge}")

    if model_info.is_demo_mode:
        render_synthetic_notice(curr_key)

    detail = client.get_patient_detail(curr_key)
    if detail is None:
        st.error("Please select a valid patient first.")
    else:
        st.markdown(
            "Decomposes model predictions into regional anatomical MRI features and longitudinal clinical markers "
            "using **Trapezoidal Integrated Gradients** and **Leave-One-Visit-Out Sensitivity Analysis**."
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            ig_steps = st.slider("Integrated Gradients Steps", min_value=16, max_value=64, value=32, step=16)
        with c2:
            include_abl = st.checkbox("Include Leave-One-Visit-Out", value=True)
        with c3:
            compute_xai = st.button("Run Attribution Engine", type="primary", use_container_width=True)

        exp_req = ExplanationRequest(
            subject_key=curr_key,
            visits=detail.visits,
            imaging_feature_names=detail.imaging_feature_names,
            steps=ig_steps,
            include_visit_ablation=include_abl,
        )

        if compute_xai or "last_explanation" not in st.session_state:
            with st.spinner("Calculating gradient attributions and regional sensitivity..."):
                st.session_state["last_explanation"] = client.explain(exp_req)

        render_xai_dashboard(st.session_state["last_explanation"])

        if model_info.is_demo_mode:
            st.caption(
                "⚠️ **Fictional Demonstration Data Notice**: Attributions and feature importances above are computed from "
                "simulated demonstration profiles for algorithmic verification and do not reflect anatomical findings in real patients."
            )


# ============================================================================
# Page 6: Dataset Audit
# ============================================================================

elif selected_page == "Dataset Audit":
    st.subheader("📊 Dataset Audit & Research Governance")
    st.markdown("Official audit metrics and data protection posture under **NACC Freeze 74 baseline**.")
    st.info(
        "📌 **Dataset Provenance & Pending Version Decision**: The metrics and partition splits shown below reflect "
        "the experimental preflight audit executed on NACC Freeze 74 (75 candidate participants). "
        "The project technical specification indicates Freeze 75 as the intended dataset release; "
        "formal migration and pipeline recalibration will occur once the dataset version decision is finalized.",
        icon="ℹ️",
    )

    audit = client.get_audit()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Cohort Size", audit.cohort_counts["eligible_participants"])
    a2.metric("Progression Rate", f"{audit.cohort_counts['progression_rate'] * 100:.1f}%")
    a3.metric("Approved MRI Features", audit.cohort_counts["mri_features_approved"])
    a4.metric("Retained Train Features", audit.cohort_counts["mri_features_retained"])

    st.markdown("#### ⚖️ Participant Partition Breakdown (Seed 42)")
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Training Set (70%)", f"{audit.split_counts['train']} subjects")
    s_col2.metric("Validation Set (15%)", f"{audit.split_counts['validation']} subjects")
    s_col3.metric("Reserved Test Set (15%)", f"{audit.split_counts['test']} subjects")

    st.markdown("#### 🎯 Model Evaluation Benchmark (Exploratory Test Arm)")
    if audit.evaluation_metrics:
        m_df = pd.DataFrame([
            {"Metric": "AUROC", "Score": f"{audit.evaluation_metrics['auroc']:.3f}"},
            {"Metric": "AUPRC", "Score": f"{audit.evaluation_metrics['auprc']:.3f}"},
            {"Metric": "Balanced Accuracy", "Score": f"{audit.evaluation_metrics['balanced_accuracy']:.3f}"},
            {"Metric": "Sensitivity", "Score": f"{audit.evaluation_metrics['sensitivity']:.3f}"},
            {"Metric": "Specificity", "Score": f"{audit.evaluation_metrics['specificity']:.3f}"},
            {"Metric": "Brier Score Loss", "Score": f"{audit.evaluation_metrics['brier_score']:.3f}"},
        ])
        st.dataframe(m_df, use_container_width=True, hide_index=True)

    st.markdown("#### ⚖️ Research Governance & Compliance Safeguards")
    st.warning(audit.data_protection_warning)
    st.markdown(
        """
        - **Pseudonymization**: Identifiers are replaced using keyed HMAC-SHA256 (`subject_key`). Raw participant IDs and site codes never cross the network.
        - **Training-Only Preprocessing**: Imputation parameters and scaling factors are fit strictly on available training visits.
        - **Audit Logging**: Prediction inquiries are logged with timestamps in an internal SQLite repository for clinical accountability.
        - **Non-Diagnostic Research Boundary**: This system is strictly an investigational tool for algorithmic research, not a certified clinical medical device.
        """
    )
