"""Synthetic cohort generator and demo fixtures for Workstream 5.

Generates reproducible, clinically plausible longitudinal trajectories
without touching any restricted NACC data or private identities.
"""

from __future__ import annotations

import numpy as np

from nacc_progression.contracts import CLINICAL_FEATURE_COUNT, PatientSequence
from nacc_progression.platform.schemas import CLINICAL_ORDER, PatientSummary

# Standard 12 representative FreeSurfer/NACC mixed-protocol MRI regions for demo display
DEMO_MRI_FEATURES = (
    "LHIPPO",
    "RHIPPO",
    "LENT",
    "RENT",
    "TEMPCOR",
    "RTEMPCOR",
    "LATVENT",
    "WMHVOL",
    "FRCORT",
    "OCCCORT",
    "PARCORT",
    "CEREGR",
)


def make_demo_patient(
    subject_key: str,
    label: int,
    visit_count: int = 3,
    mri_features: tuple[str, ...] = DEMO_MRI_FEATURES,
    seed: int | None = None,
) -> PatientSequence:
    """Generate a single clinically coherent synthetic longitudinal sequence.

    Progressors (label=1) exhibit worsening cognition (MMSE drop, CDR-SB rise)
    and progressive hippocampal/entorhinal volume loss.
    """
    if seed is None:
        seed = sum(ord(c) for c in subject_key)
    rng = np.random.default_rng(seed)

    feature_count = len(mri_features)
    visit_count = max(1, min(visit_count, 3))

    # Base patient demographics
    age_baseline = float(rng.uniform(65.0, 85.0))
    sex = float(rng.choice([0.0, 1.0]))  # 0: M, 1: F
    education = float(rng.choice([12.0, 14.0, 16.0, 18.0]))

    clinical_rows: list[list[float]] = []
    clinical_masks: list[list[bool]] = []
    mri_rows: list[list[float]] = []
    mri_masks: list[list[bool]] = []
    delta_times: list[list[float]] = []

    # Initial states
    mmse = 27.5 if label == 0 else 25.5
    cdr_sb = 1.0 if label == 0 else 2.5
    hippo_base = 0.5 if label == 0 else -0.8  # standardized units

    for visit_idx in range(visit_count):
        interval = 0.0 if visit_idx == 0 else float(rng.uniform(0.8, 1.2))
        delta_times.append([interval])

        # Progressive decline trajectory for positive cases
        if visit_idx > 0:
            if label == 1:
                mmse -= float(rng.uniform(1.5, 3.5))
                cdr_sb += float(rng.uniform(0.8, 2.0))
                hippo_base -= float(rng.uniform(0.4, 0.8))
            else:
                mmse += float(rng.uniform(-0.5, 0.5))
                cdr_sb += float(rng.uniform(-0.2, 0.4))
                hippo_base -= float(rng.uniform(0.05, 0.15))

        cdr_global = 0.5 if cdr_sb < 4.0 else 1.0

        # Normalized clinical row in CLINICAL_ORDER: ('age', 'sex', 'education', 'cdr_global', 'cdr_sum_boxes', 'mmse')
        c_vals = [
            (age_baseline + visit_idx * 1.0 - 72.0) / 7.0,  # standardized age
            sex,
            (education - 15.0) / 3.0,
            cdr_global,
            (cdr_sb - 2.0) / 2.0,
            (mmse - 26.0) / 3.5,
        ]
        clinical_rows.append(c_vals)
        clinical_masks.append([False] * CLINICAL_FEATURE_COUNT)

        # MRI features: first 4 correlate with hippocampus/entorhinal atrophy
        mri_vals = np.zeros(feature_count, dtype=np.float32)
        mri_vals[0] = hippo_base + rng.normal(0.0, 0.1)  # LHIPPO
        mri_vals[1] = hippo_base + rng.normal(0.0, 0.1)  # RHIPPO
        mri_vals[2] = hippo_base * 0.9 + rng.normal(0.0, 0.15)  # LENT
        mri_vals[3] = hippo_base * 0.9 + rng.normal(0.0, 0.15)  # RENT
        mri_vals[4:] = rng.normal(0.0, 0.5, size=feature_count - 4)

        # Historical visit MRI availability (first visit can occasionally be missing)
        mri_missing = (visit_idx == 0 and rng.random() < 0.25)
        if mri_missing:
            mri_rows.append([0.0] * feature_count)
            mri_masks.append([True] * feature_count)
        else:
            mri_rows.append(mri_vals.tolist())
            mri_masks.append([False] * feature_count)

    return PatientSequence(
        subject_key=subject_key,
        label=label,
        imaging_values=np.array(mri_rows, dtype=np.float32),
        imaging_missing_mask=np.array(mri_masks, dtype=np.bool_),
        clinical_values=np.array(clinical_rows, dtype=np.float32),
        clinical_missing_mask=np.array(clinical_masks, dtype=np.bool_),
        delta_time=np.array(delta_times, dtype=np.float32),
        imaging_feature_names=mri_features,
    )


def create_demo_cohort(n_patients: int = 15) -> list[PatientSequence]:
    """Generate a diverse synthetic demonstration cohort."""
    cohort: list[PatientSequence] = []
    # Curated showcase patients with representative trajectories
    cohort.append(make_demo_patient("n74_demo_rapid_progressor", label=1, visit_count=3, seed=101))
    cohort.append(make_demo_patient("n74_demo_stable_mci", label=0, visit_count=3, seed=102))
    cohort.append(make_demo_patient("n74_demo_slow_progressor", label=1, visit_count=2, seed=103))
    cohort.append(make_demo_patient("n74_demo_non_progressor_sparse_mri", label=0, visit_count=2, seed=104))
    cohort.append(make_demo_patient("n74_demo_borderline_risk", label=1, visit_count=3, seed=105))

    for i in range(len(cohort), n_patients):
        label = 1 if i % 3 == 0 else 0
        v_count = 2 if i % 2 == 0 else 3
        key = f"n74_synthetic_p{i:03d}"
        cohort.append(make_demo_patient(key, label=label, visit_count=v_count, seed=200 + i))

    return cohort


def cohort_to_summaries(cohort: list[PatientSequence]) -> list[PatientSummary]:
    """Convert patient sequences to table summaries."""
    summaries: list[PatientSummary] = []
    for seq in cohort:
        has_anchor_mri = not bool(seq.imaging_missing_mask[-1].all())
        # Denormalize approx MMSE and CDR from preprocessed features
        latest_c = seq.clinical_values[-1]
        mmse_est = round(float(latest_c[5] * 3.5 + 26.0), 1)
        age_est = round(float(latest_c[0] * 7.0 + 72.0), 1)
        cdr_est = round(float(latest_c[3]), 1)

        summaries.append(
            PatientSummary(
                subject_key=seq.subject_key,
                visit_count=seq.visit_count,
                label=seq.label,
                has_mri=has_anchor_mri,
                split="demo",
                age_at_anchor=age_est,
                latest_mmse=mmse_est,
                latest_cdr_global=cdr_est,
            )
        )
    return summaries
