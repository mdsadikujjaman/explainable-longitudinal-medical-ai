"""Reusable UI widgets, charting components, and visual cards for Streamlit."""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None

from nacc_progression.platform.schemas import (
    ExplanationResponse,
    PatientDetailResponse,
    PredictionResponse,
)

__all__ = [
    "render_header",
    "render_disclaimer_notice",
    "render_synthetic_notice",
    "render_prediction_badge",
    "render_trajectory_chart",
    "render_patient_timeline",
    "render_xai_dashboard",
]


def render_header() -> None:
    """Render top header with system branding and research disclaimer."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
                    padding: 1.25rem 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;
                    border-left: 5px solid #3b82f6;">
            <h2 style="color: #f8fafc; margin: 0 0 0.25rem 0; font-family: sans-serif;">
                🧠 Explainable Longitudinal Medical AI
            </h2>
            <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">
                Longitudinal Multimodal Progression Modeling & Model Interpretability
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_notice() -> None:
    """Display regulatory research compliance banner."""
    st.info(
        "🔬 **Investigational Research Prototype**: This platform is designed strictly for research inquiry "
        "and algorithmic interpretability. It does not constitute a certified diagnostic tool and should not be used "
        "as the sole basis for clinical treatment decisions.",
        icon="ℹ️",
    )


def render_synthetic_notice(subject_key: str | None = None) -> None:
    """Display prominent synthetic simulation badge and research boundary notice."""
    key_text = f" for `{subject_key}`" if subject_key else ""
    st.warning(
        f"🧪 **SYNTHETIC DEMO RECORD{key_text}**\n\n"
        "This patient trajectory, imaging measurements, clinical scores, and risk projections are generated entirely "
        "by an *in silico* synthetic simulator for interface demonstration. They do not represent real human participants "
        "or actual clinical observations from any private medical registry.",
        icon="🧪",
    )


def render_prediction_badge(prediction: PredictionResponse) -> None:
    """Render progression risk card with dynamic color status."""
    prob = prediction.probability
    threshold = prediction.threshold

    if prob >= threshold:
        status_label = "HIGH RISK OF PROGRESSION (AD)"
        box_color = "#ef4444"
        bg_color = "rgba(239, 68, 68, 0.12)"
    elif prob >= 0.35:
        status_label = "MODERATE / BORDERLINE PROGRESSION RISK"
        box_color = "#f59e0b"
        bg_color = "rgba(245, 158, 11, 0.12)"
    else:
        status_label = "LOW RISK / STABLE MCI TRAJECTORY"
        box_color = "#10b981"
        bg_color = "rgba(16, 185, 129, 0.12)"

    st.markdown(
        f"""
        <div style="padding: 1.25rem; border-radius: 8px; border: 1.5px solid {box_color};
                    background-color: {bg_color}; margin-bottom: 1rem;">
            <div style="font-size: 0.85rem; font-weight: 600; color: {box_color}; text-transform: uppercase;">
                Model Assessment Outcome
            </div>
            <div style="display: flex; align-items: baseline; gap: 1rem; margin-top: 0.4rem;">
                <span style="font-size: 2.2rem; font-weight: 800; color: #f8fafc;">
                    {prob * 100:.1f}%
                </span>
                <span style="font-size: 1.1rem; font-weight: 600; color: {box_color};">
                    {status_label}
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">
                Decision Threshold: <b>{threshold:.2f}</b> | Predicted Class: <b>{prediction.predicted_label}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trajectory_chart(prediction: PredictionResponse) -> None:
    """Render longitudinal risk trajectory across visit prefixes using Altair."""
    st.markdown("#### 📈 Predicted 24-Month Progression Probability by Visit Prefix")
    st.caption(
        "Prefix predictions estimate the probability of progression to Alzheimer's dementia within "
        "24 months after the anchor visit, showing how the risk assessment updates as cumulative "
        "longitudinal observations accumulate across successive visits."
    )

    visits = [f"Visit {i + 1}" for i in range(len(prediction.prefix_probabilities))]
    scores = [round(float(p) * 100.0, 1) for p in prediction.prefix_probabilities]
    final_prob_pct = prediction.probability * 100.0

    df = pd.DataFrame({
        "Visit Prefix": visits,
        "Progression Probability (%)": scores,
    })

    if alt is not None:
        base = alt.Chart(df).encode(
            x=alt.X(
                "Visit Prefix:N",
                sort=visits,
                title="Accumulated Visit History (Prefix)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Progression Probability (%):Q",
                title="Progression Probability (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
        )
        line = base.mark_line(color="#3b82f6", strokeWidth=3)
        points = base.mark_point(color="#1d4ed8", size=90, filled=True).encode(
            tooltip=[
                alt.Tooltip("Visit Prefix:N", title="Visit Prefix"),
                alt.Tooltip("Progression Probability (%):Q", format=".1f", title="Probability (%)"),
            ]
        )
        labels = base.mark_text(
            dy=-12,
            color="#f8fafc",
            fontSize=12,
            fontWeight="bold",
        ).encode(
            text=alt.Text("Progression Probability (%):Q", format=".1f"),
        )
        chart = (line + points + labels).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(
            df.set_index("Visit Prefix")["Progression Probability (%)"],
            use_container_width=True,
        )

    st.markdown(
        f"**Final Sequence Progression Probability**: `{final_prob_pct:.1f}%` &nbsp;|&nbsp; "
        f"**Decision Threshold**: `{prediction.threshold:.2f}` &nbsp;|&nbsp; "
        f"**Classification**: `{'Progressor (1)' if prediction.predicted_label == 1 else 'Stable MCI (0)'}`"
    )


def render_patient_timeline(patient: PatientDetailResponse) -> None:
    """Render tabular longitudinal visit breakdown."""
    st.markdown("#### 📋 Clinical & Imaging Visit Sequence")
    st.caption(
        "ℹ️ **MRI Modality Notice**: The system evaluates quantitative MRI-derived regional measurements "
        "(such as FreeSurfer volumetric ROIs) rather than raw DICOM/NIfTI neuroimaging scans."
    )
    rows: list[dict[str, Any]] = []

    cum_time = 0.0
    for idx, v in enumerate(patient.visits):
        cum_time += v.delta_time_years
        c = v.clinical_values
        # Estimated original units
        age = round(c[0] * 7.0 + 72.0, 1)
        sex = "Female" if c[1] > 0.5 else "Male"
        cdr_glob = round(c[3], 1)
        cdr_sb = round(c[4] * 2.0 + 2.0, 1)
        mmse = round(c[5] * 3.5 + 26.0, 1)
        mri_stat = (
            "Quantitative features missing or masked"
            if all(v.imaging_missing_mask)
            else "Quantitative features available"
        )

        rows.append({
            "Visit": f"Visit {idx + 1}",
            "Elapsed Time (Yrs)": f"{cum_time:.2f}",
            "Est. Age": age,
            "Sex": sex,
            "CDR Global": cdr_glob,
            "CDR-SB": cdr_sb,
            "MMSE": mmse,
            "MRI-derived Features": mri_stat,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_xai_dashboard(explanation: ExplanationResponse) -> None:
    """Display comprehensive multimodal explainability charts."""
    st.markdown("### 🔍 Model Interpretability & Attribution Analysis")
    st.caption("Trapezoidal Integrated Gradients attributions relative to standardized neutral baseline.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🧠 Top Regional MRI Biomarkers")
        top_mri = explanation.mri_feature_importance[:8]
        if top_mri:
            df_mri = pd.DataFrame({
                "Anatomical Region": [item.feature_name for item in top_mri],
                "Mean Attribution Importance": [item.importance for item in top_mri],
            }).set_index("Anatomical Region")
            st.bar_chart(df_mri, horizontal=True)

    with col2:
        st.markdown("##### 🩺 Clinical Variable Contributions")
        clin_items = explanation.clinical_feature_importance
        df_clin = pd.DataFrame({
            "Clinical Marker": [item.feature_name.upper() for item in clin_items],
            "Attribution Importance": [item.importance for item in clin_items],
        }).set_index("Clinical Marker")
        st.bar_chart(df_clin, horizontal=True)

    # Modality and Visit Importance
    st.markdown("##### ⏱️ Temporal & Modality Importance")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        df_modal = pd.DataFrame({
            "Modality": list(explanation.modality_importance.keys()),
            "Importance": list(explanation.modality_importance.values()),
        }).set_index("Modality")
        st.bar_chart(df_modal)

    with m_col2:
        df_visit = pd.DataFrame({
            "Visit": [f"Visit {i + 1}" for i in range(len(explanation.visit_importance))],
            "Attribution Weight": explanation.visit_importance,
        }).set_index("Visit")
        st.bar_chart(df_visit)

    # Leave-One-Visit-Out Ablations
    if explanation.visit_ablation:
        st.markdown("##### 🧪 Leave-One-Visit-Out Sensitivity Analysis")
        st.caption("Evaluates score changes when removing each visit sequentially (bridging elapsed time).")
        ablation_rows = []
        for ab in explanation.visit_ablation:
            ablation_rows.append({
                "Removed Visit": f"Visit {ab.removed_visit_index + 1}",
                "Probability Without Visit": f"{ab.probability_without_visit * 100:.1f}%",
                "Signed Δ": f"{ab.signed_probability_change * 100:+.1f}%",
                "Absolute Sensitivity Impact": f"{ab.absolute_probability_change * 100:.2f}%",
            })
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)
