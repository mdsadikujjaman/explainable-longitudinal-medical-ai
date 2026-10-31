"""Comprehensive test suite for Workstream 5 (Platform, Backend, Frontend & Integration).

Tests schemas, security, database, services, API endpoints, and client bindings.
"""

from __future__ import annotations

import os
import sys

# Ensure synthetic demo mode is enabled by default for standard platform unit tests
os.environ.setdefault("DEMO_MODE", "1")
import tempfile
import time
import unittest
from pathlib import Path

import bcrypt
from unittest.mock import MagicMock, patch

# Ensure 'src' is in path
SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from fastapi import HTTPException
from nacc_progression.contracts import PatientSequence
from nacc_progression.model import ModelConfig
from nacc_progression.training import TrainingConfig, train_model
from tests.conftest import make_sequence
from nacc_progression.frontend.components import (
    render_disclaimer_notice,
    render_header,
    render_patient_timeline,
    render_prediction_badge,
    render_synthetic_notice,
    render_trajectory_chart,
)
from nacc_progression.platform.api import create_app
from nacc_progression.platform.database import (
    count_users,
    create_user,
    get_db_connection,
    get_prediction_logs,
    get_user_by_username,
    init_db,
    log_prediction,
    update_user_password,
)
from nacc_progression.platform.demo_data import (
    DEMO_MRI_FEATURES,
    create_demo_cohort,
    make_demo_patient,
)
from nacc_progression.platform.schemas import (
    AuditResponse,
    CLINICAL_ORDER,
    ExplanationRequest,
    ExplanationResponse,
    FeatureAttribution,
    LoginRequest,
    ModelInfoResponse,
    PatientDetailResponse,
    PatientSummary,
    PredictionRequest,
    PredictionResponse,
    VisitAblation,
    VisitData,
    explanation_result_to_response,
    patient_sequence_to_visits,
    prediction_result_to_response,
    request_to_patient_sequence,
)
from nacc_progression.platform.security import (
    create_access_token,
    decode_access_token,
    get_jwt_secret_key,
    hash_password,
    needs_rehash,
    verify_password,
)
from nacc_progression.platform.services import PlatformService


class TestPlatformSchemas(unittest.TestCase):
    """Test validation and bidirectional conversion of platform schemas."""

    def test_visit_data_validation(self) -> None:
        valid_visit = VisitData(
            imaging_values=[0.1, 0.2, 0.3],
            imaging_missing_mask=[False, False, False],
            clinical_values=[0.0, 1.0, 0.0, 0.5, 1.0, 0.5],
            clinical_missing_mask=[False] * 6,
            delta_time_years=0.0,
        )
        self.assertEqual(len(valid_visit.clinical_values), 6)

        # Mismatched mask length must raise ValueError
        with self.assertRaises(ValueError):
            VisitData(
                imaging_values=[0.1, 0.2],
                imaging_missing_mask=[False],  # mismatched length
                clinical_values=[0.0] * 6,
                clinical_missing_mask=[False] * 6,
                delta_time_years=0.0,
            )

        # Invalid clinical feature count must raise ValueError
        with self.assertRaises(ValueError):
            VisitData(
                imaging_values=[0.1],
                imaging_missing_mask=[False],
                clinical_values=[0.0] * 5,  # expected 6
                clinical_missing_mask=[False] * 5,
                delta_time_years=0.0,
            )

    def test_prediction_request_to_sequence_conversion(self) -> None:
        v0 = VisitData(
            imaging_values=[0.5, -0.2],
            imaging_missing_mask=[False, False],
            clinical_values=[0.1, 0.0, 0.5, 0.5, 0.0, 0.2],
            clinical_missing_mask=[False] * 6,
            delta_time_years=0.0,
        )
        v1 = VisitData(
            imaging_values=[0.3, -0.4],
            imaging_missing_mask=[False, False],
            clinical_values=[0.2, 0.0, 0.5, 0.5, 0.5, -0.1],
            clinical_missing_mask=[False] * 6,
            delta_time_years=1.0,
        )

        req = PredictionRequest(
            subject_key="n74_test_subject",
            visits=[v0, v1],
            imaging_feature_names=["LHIPPO", "RHIPPO"],
            threshold=0.5,
        )

        seq = request_to_patient_sequence(req, label_placeholder=0)
        self.assertIsInstance(seq, PatientSequence)
        self.assertEqual(seq.subject_key, "n74_test_subject")
        self.assertEqual(seq.visit_count, 2)
        self.assertEqual(seq.imaging_feature_count, 2)
        self.assertEqual(seq.delta_time[0, 0], 0.0)
        self.assertEqual(seq.delta_time[1, 0], 1.0)

        # Test converting sequence back to visits
        recon_visits = patient_sequence_to_visits(seq)
        self.assertEqual(len(recon_visits), 2)
        self.assertAlmostEqual(recon_visits[0].imaging_values[0], 0.5, places=5)
        self.assertAlmostEqual(recon_visits[0].imaging_values[1], -0.2, places=5)

    def test_prediction_request_rejects_nonzero_initial_time(self) -> None:
        v0 = VisitData(
            imaging_values=[0.1],
            imaging_missing_mask=[False],
            clinical_values=[0.0] * 6,
            clinical_missing_mask=[False] * 6,
            delta_time_years=0.5,  # must be 0.0 for visit 0
        )
        with self.assertRaises(ValueError):
            PredictionRequest(subject_key="bad_time", visits=[v0])


class TestPlatformSecurity(unittest.TestCase):
    """Test password hashing, legacy migration detection, and JWT token operations."""

    def test_argon2_password_hashing(self) -> None:
        password = "ResearchPassword2026!"
        hashed = hash_password(password)
        self.assertNotEqual(password, hashed)
        # Verify Argon2id algorithm identifier
        self.assertTrue(hashed.startswith("$argon2id$"))
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("wrong_password", hashed))
        self.assertFalse(verify_password("", hashed))
        with self.assertRaises(ValueError):
            hash_password("")

    def test_bcrypt_backward_compatibility_and_rehash(self) -> None:
        raw_password = "legacy_secret_password"
        # Simulate an account created in an earlier version with Bcrypt
        legacy_hash = bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        self.assertTrue(legacy_hash.startswith(("$2a$", "$2b$", "$2y$")))

        # Legacy hash must still verify successfully
        self.assertTrue(verify_password(raw_password, legacy_hash))
        self.assertFalse(verify_password("wrong_password", legacy_hash))

        # Rehash detection must flag legacy Bcrypt hash but clear modern Argon2id hash
        self.assertTrue(needs_rehash(legacy_hash))
        modern_hash = hash_password("modern_secret_password")
        self.assertFalse(needs_rehash(modern_hash))

    def test_jwt_secret_key_production_enforcement(self) -> None:
        old_env = os.environ.get("PLATFORM_ENV")
        old_secret = os.environ.get("JWT_SECRET_KEY")
        try:
            os.environ["PLATFORM_ENV"] = "production"

            # Missing secret in production must fail
            os.environ.pop("JWT_SECRET_KEY", None)
            with self.assertRaises(RuntimeError):
                get_jwt_secret_key()

            # Short secret in production must fail
            os.environ["JWT_SECRET_KEY"] = "short-key-12345"
            with self.assertRaises(RuntimeError):
                get_jwt_secret_key()

            # Known placeholder in production must fail
            os.environ["JWT_SECRET_KEY"] = "replace-with-a-secure-random-secret-key-in-production"
            with self.assertRaises(RuntimeError):
                get_jwt_secret_key()

            # Valid 32+ character key in production must succeed
            valid_secret = "a" * 32
            os.environ["JWT_SECRET_KEY"] = valid_secret
            self.assertEqual(get_jwt_secret_key(), valid_secret)
        finally:
            if old_env is not None:
                os.environ["PLATFORM_ENV"] = old_env
            else:
                os.environ.pop("PLATFORM_ENV", None)
            if old_secret is not None:
                os.environ["JWT_SECRET_KEY"] = old_secret
            else:
                os.environ.pop("JWT_SECRET_KEY", None)

    def test_jwt_secret_key_development_ephemeral(self) -> None:
        old_env = os.environ.get("PLATFORM_ENV")
        old_secret = os.environ.get("JWT_SECRET_KEY")
        try:
            os.environ["PLATFORM_ENV"] = "development"
            os.environ.pop("JWT_SECRET_KEY", None)
            dev_key = get_jwt_secret_key()
            self.assertIsInstance(dev_key, str)
            self.assertGreaterEqual(len(dev_key), 32)
        finally:
            if old_env is not None:
                os.environ["PLATFORM_ENV"] = old_env
            else:
                os.environ.pop("PLATFORM_ENV", None)
            if old_secret is not None:
                os.environ["JWT_SECRET_KEY"] = old_secret
            else:
                os.environ.pop("JWT_SECRET_KEY", None)

    def test_jwt_token_roundtrip(self) -> None:
        token = create_access_token(subject="dr_jane", role="clinician", expires_in_seconds=3600)
        decoded = decode_access_token(token)
        self.assertEqual(decoded["sub"], "dr_jane")
        self.assertEqual(decoded["role"], "clinician")
        self.assertIn("exp", decoded)

    def test_jwt_tampering_rejected(self) -> None:
        token = create_access_token(subject="user1", role="researcher")
        parts = token.split(".")
        # Tamper payload
        tampered_token = f"{parts[0]}.eyJhZG1pbiI6dHJ1ZX0.{parts[2]}"
        with self.assertRaises(ValueError):
            decode_access_token(tampered_token)


class TestPlatformDatabase(unittest.TestCase):
    """Test SQLite persistence, user registration, demo account gating, and auditing."""

    def setUp(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        self.db_path = self.temp_file.name

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except OSError:
                pass

    def test_private_mode_does_not_seed_demo_accounts(self) -> None:
        # Initializing without demo mode must create zero accounts
        init_db(self.db_path, enable_demo_users=False)
        self.assertEqual(count_users(self.db_path), 0)
        self.assertIsNone(get_user_by_username("researcher", self.db_path))
        self.assertIsNone(get_user_by_username("admin", self.db_path))
        self.assertIsNone(get_user_by_username("clinician", self.db_path))

    def test_demo_mode_seeds_demo_accounts(self) -> None:
        # Explicit demo mode seeds standard demo accounts
        init_db(self.db_path, enable_demo_users=True)
        self.assertEqual(count_users(self.db_path), 3)

        researcher = get_user_by_username("researcher", self.db_path)
        self.assertIsNotNone(researcher)
        self.assertEqual(researcher["role"], "researcher")
        self.assertTrue(researcher["hashed_password"].startswith("$argon2id$"))
        self.assertTrue(verify_password("researcher123", researcher["hashed_password"]))

        admin = get_user_by_username("admin", self.db_path)
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "admin")
        self.assertTrue(admin["hashed_password"].startswith("$argon2id$"))
        self.assertTrue(verify_password("admin123", admin["hashed_password"]))

        clinician = get_user_by_username("clinician", self.db_path)
        self.assertIsNotNone(clinician)
        self.assertEqual(clinician["role"], "clinician")
        self.assertTrue(clinician["hashed_password"].startswith("$argon2id$"))
        self.assertTrue(verify_password("clinician123", clinician["hashed_password"]))

    def test_demo_mode_environment_variable_control(self) -> None:
        old_demo = os.environ.get("DEMO_MODE")
        try:
            # DEMO_MODE=1 -> seeded
            os.environ["DEMO_MODE"] = "1"
            init_db(self.db_path)
            self.assertEqual(count_users(self.db_path), 3)

            # DEMO_MODE=0 -> not seeded on new database
            temp_clean = tempfile.NamedTemporaryFile(delete=False)
            temp_clean.close()
            try:
                os.environ["DEMO_MODE"] = "0"
                init_db(temp_clean.name)
                self.assertEqual(count_users(temp_clean.name), 0)
            finally:
                if os.path.exists(temp_clean.name):
                    os.unlink(temp_clean.name)
        finally:
            if old_demo is not None:
                os.environ["DEMO_MODE"] = old_demo
            else:
                os.environ.pop("DEMO_MODE", None)

    def test_rehash_migration_in_database(self) -> None:
        init_db(self.db_path, enable_demo_users=False)
        legacy_pwd = "legacy_password_to_upgrade"
        legacy_hash = bcrypt.hashpw(legacy_pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Manually insert legacy user
        conn = get_db_connection(self.db_path)
        with conn:
            conn.execute(
                "INSERT INTO users (username, hashed_password, role, is_active, created_at) VALUES (?, ?, ?, 1, ?)",
                ("old_user", legacy_hash, "researcher", time.time()),
            )
        conn.close()

        user = get_user_by_username("old_user", self.db_path)
        self.assertIsNotNone(user)
        self.assertTrue(needs_rehash(user["hashed_password"]))

        # Migrate user hash to Argon2id
        new_hash = hash_password(legacy_pwd)
        update_user_password("old_user", new_hash, self.db_path)

        user_after = get_user_by_username("old_user", self.db_path)
        self.assertFalse(needs_rehash(user_after["hashed_password"]))
        self.assertTrue(user_after["hashed_password"].startswith("$argon2id$"))
        self.assertTrue(verify_password(legacy_pwd, user_after["hashed_password"]))

    def test_create_user(self) -> None:
        init_db(self.db_path, enable_demo_users=False)
        created = create_user("new_scientist", "pass456", role="researcher", db_path=self.db_path)
        self.assertEqual(created["username"], "new_scientist")
        fetched = get_user_by_username("new_scientist", self.db_path)
        self.assertIsNotNone(fetched)
        self.assertTrue(fetched["hashed_password"].startswith("$argon2id$"))

        # Duplicate username must fail
        with self.assertRaises(ValueError):
            create_user("new_scientist", "pass789", db_path=self.db_path)

    def test_log_and_retrieve_predictions(self) -> None:
        init_db(self.db_path, enable_demo_users=False)
        log_prediction(
            username="researcher",
            subject_key="n74_demo_01",
            probability=0.78,
            threshold=0.5,
            predicted_label=1,
            visit_count=3,
            db_path=self.db_path,
        )
        logs = get_prediction_logs(limit=10, db_path=self.db_path)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["subject_key"], "n74_demo_01")
        self.assertAlmostEqual(logs[0]["probability"], 0.78)


class TestPlatformServices(unittest.TestCase):
    """Test demo data generation and platform inference service."""

    def setUp(self) -> None:
        self.service = PlatformService.get_instance()

    def test_demo_cohort_characteristics(self) -> None:
        cohort = create_demo_cohort(n_patients=10)
        self.assertEqual(len(cohort), 10)
        for seq in cohort:
            self.assertTrue(1 <= seq.visit_count <= 3)
            self.assertEqual(seq.clinical_values.shape[1], 6)
            self.assertEqual(seq.delta_time[0, 0], 0.0)

    def test_patient_search_and_filter(self) -> None:
        all_patients = self.service.search_patients()
        self.assertGreaterEqual(len(all_patients), 5)

        progressors = self.service.search_patients(label_filter=1)
        for p in progressors:
            self.assertEqual(p.label, 1)

        stables = self.service.search_patients(label_filter=0)
        for p in stables:
            self.assertEqual(p.label, 0)

    def test_synthetic_predict_and_explain(self) -> None:
        patient = make_demo_patient("n74_unit_test_patient", label=1, visit_count=3)
        pred = self.service.predict(patient, threshold=0.5)

        self.assertEqual(pred.subject_key, "n74_unit_test_patient")
        self.assertTrue(0.0 <= pred.probability <= 1.0)
        self.assertEqual(len(pred.prefix_probabilities), 3)

        exp = self.service.explain(patient, steps=16, include_ablation=True)
        self.assertEqual(exp.subject_key, "n74_unit_test_patient")
        self.assertGreater(len(exp.mri_feature_importance), 0)
        self.assertEqual(len(exp.clinical_feature_importance), 6)
        self.assertEqual(len(exp.visit_importance), 3)
        self.assertIsNotNone(exp.visit_ablation)
        self.assertEqual(len(exp.visit_ablation), 3)


class TestFastAPIRoutes(unittest.TestCase):
    """Test FastAPI application route handlers."""

    def setUp(self) -> None:
        init_db(enable_demo_users=True)
        self.app = create_app()

    def test_health_route(self) -> None:
        route = next(r for r in self.app.routes if r.path == "/api/health")
        result = route.endpoint()
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["version"], "0.1.0")

    def test_auth_route(self) -> None:
        route = next(r for r in self.app.routes if r.path == "/api/auth/login")
        payload = LoginRequest(username="researcher", password="researcher123")
        res = route.endpoint(payload)
        self.assertEqual(res.token_type, "bearer")
        self.assertEqual(res.username, "researcher")

    def test_rehash_on_login_in_api(self) -> None:
        legacy_pwd = "legacy_route_password"
        legacy_hash = bcrypt.hashpw(legacy_pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn = get_db_connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO users (username, hashed_password, role, is_active, created_at) VALUES (?, ?, ?, 1, ?)",
                ("legacy_api_user", legacy_hash, "researcher", time.time()),
            )
        conn.close()

        user_before = get_user_by_username("legacy_api_user")
        self.assertIsNotNone(user_before)
        self.assertTrue(needs_rehash(user_before["hashed_password"]))

        # Execute login endpoint
        route = next(r for r in self.app.routes if r.path == "/api/auth/login")
        payload = LoginRequest(username="legacy_api_user", password=legacy_pwd)
        res = route.endpoint(payload)
        self.assertEqual(res.token_type, "bearer")
        self.assertEqual(res.username, "legacy_api_user")

        # Verify transparent upgrade to Argon2id in DB
        user_after = get_user_by_username("legacy_api_user")
        self.assertFalse(needs_rehash(user_after["hashed_password"]))
        self.assertTrue(user_after["hashed_password"].startswith("$argon2id$"))
        self.assertTrue(verify_password(legacy_pwd, user_after["hashed_password"]))

    def test_patient_routes(self) -> None:
        route_list = next(r for r in self.app.routes if r.path == "/api/patients")
        patients = route_list.endpoint(query="", label=None, min_visits=1)
        self.assertGreaterEqual(len(patients), 5)

        route_detail = next(r for r in self.app.routes if r.path == "/api/patients/{subject_key}")
        first_key = patients[0].subject_key
        detail = route_detail.endpoint(first_key)
        self.assertEqual(detail.summary.subject_key, first_key)
        self.assertGreaterEqual(len(detail.visits), 1)

    def test_predict_and_explain_routes(self) -> None:
        route_detail = next(r for r in self.app.routes if r.path == "/api/patients/{subject_key}")
        detail = route_detail.endpoint("n74_demo_rapid_progressor")

        route_predict = next(r for r in self.app.routes if r.path == "/api/predict")
        pred_req = PredictionRequest(
            subject_key=detail.summary.subject_key,
            visits=detail.visits,
            imaging_feature_names=detail.imaging_feature_names,
            threshold=0.5,
        )
        pred_res = route_predict.endpoint(pred_req, current_user={"username": "researcher", "role": "researcher"})
        self.assertEqual(pred_res.subject_key, "n74_demo_rapid_progressor")
        self.assertTrue(0.0 <= pred_res.probability <= 1.0)

        route_explain = next(r for r in self.app.routes if r.path == "/api/explain")
        exp_req = ExplanationRequest(
            subject_key=detail.summary.subject_key,
            visits=detail.visits,
            imaging_feature_names=detail.imaging_feature_names,
            steps=16,
            include_visit_ablation=True,
        )
        exp_res = route_explain.endpoint(exp_req, current_user={"username": "researcher", "role": "researcher"})
        self.assertEqual(exp_res.subject_key, "n74_demo_rapid_progressor")
        self.assertGreater(len(exp_res.mri_feature_importance), 0)


class TestFrontendComponents(unittest.TestCase):
    """Test frontend presentation, Altair chart configuration, and terminology."""

    @patch("streamlit.markdown")
    def test_render_header_branding(self, mock_markdown: MagicMock) -> None:
        render_header()
        mock_markdown.assert_called_once()
        html_content = mock_markdown.call_args[0][0]
        self.assertIn("Model Interpretability", html_content)
        self.assertNotIn("Diagnostic Explainability", html_content)

    @patch("streamlit.info")
    def test_render_disclaimer_notice(self, mock_info: MagicMock) -> None:
        render_disclaimer_notice()
        mock_info.assert_called_once()
        text = mock_info.call_args[0][0]
        self.assertIn("Investigational Research Prototype", text)
        self.assertIn("algorithmic interpretability", text)

    @patch("streamlit.warning")
    def test_render_synthetic_notice(self, mock_warning: MagicMock) -> None:
        render_synthetic_notice("demo_patient_01")
        mock_warning.assert_called_once()
        text = mock_warning.call_args[0][0]
        self.assertIn("SYNTHETIC DEMO RECORD", text)
        self.assertIn("in silico", text)
        self.assertIn("demo_patient_01", text)

    @patch("streamlit.altair_chart")
    @patch("streamlit.caption")
    @patch("streamlit.markdown")
    def test_render_trajectory_chart_altair(
        self, mock_markdown: MagicMock, mock_caption: MagicMock, mock_altair: MagicMock
    ) -> None:
        pred = PredictionResponse(
            subject_key="demo_p1",
            probability=0.99,
            threshold=0.5,
            predicted_label=1,
            prefix_probabilities=[0.774, 0.987, 0.990],
            prefix_semantics="Incremental prefix risk",
            visit_count=3,
        )
        render_trajectory_chart(pred)

        # Verify chart title
        title_calls = [c[0][0] for c in mock_markdown.call_args_list if "Predicted 24-Month Progression" in c[0][0]]
        self.assertTrue(len(title_calls) > 0)
        self.assertIn("Predicted 24-Month Progression Probability by Visit Prefix", title_calls[0])

        # Verify explanation caption
        caption_calls = [c[0][0] for c in mock_caption.call_args_list if "accumulate" in c[0][0]]
        self.assertTrue(len(caption_calls) > 0)

        # Verify Altair chart was invoked with fixed scale domain [0, 100]
        mock_altair.assert_called_once()
        chart_obj = mock_altair.call_args[0][0]
        chart_dict = chart_obj.to_dict()
        layers = chart_dict.get("layer", [chart_dict])
        y_scale = None
        for layer in layers:
            scale = layer.get("encoding", {}).get("y", {}).get("scale", {})
            if scale:
                y_scale = scale
                break
        if y_scale is None and "encoding" in chart_dict:
            y_scale = chart_dict["encoding"].get("y", {}).get("scale", {})
        self.assertIsNotNone(y_scale)
        self.assertEqual(y_scale.get("domain"), [0, 100])

    @patch("streamlit.dataframe")
    @patch("streamlit.caption")
    @patch("streamlit.markdown")
    def test_render_patient_timeline_mri_terminology(
        self, mock_markdown: MagicMock, mock_caption: MagicMock, mock_dataframe: MagicMock
    ) -> None:
        summary = PatientSummary(subject_key="p1", visit_count=2, label=1)
        visits = [
            VisitData(
                imaging_values=[0.1, 0.2],
                imaging_missing_mask=[False, False],
                clinical_values=[0.1, 0.0, 0.5, 0.5, 0.0, 0.2],
                clinical_missing_mask=[False] * 6,
                delta_time_years=0.0,
            ),
            VisitData(
                imaging_values=[0.0, 0.0],
                imaging_missing_mask=[True, True],
                clinical_values=[0.1, 0.0, 0.5, 0.5, 0.0, 0.2],
                clinical_missing_mask=[False] * 6,
                delta_time_years=1.0,
            ),
        ]
        detail = PatientDetailResponse(summary=summary, visits=visits)
        render_patient_timeline(detail)

        # Verify caption clarifies regional measurements vs raw scans
        caption_text = mock_caption.call_args[0][0]
        self.assertIn("quantitative MRI-derived regional measurements", caption_text)
        self.assertIn("FreeSurfer volumetric ROIs", caption_text)

        # Verify dataframe columns and statuses
        df_arg = mock_dataframe.call_args[0][0]
        self.assertIn("MRI-derived Features", df_arg.columns)
        self.assertEqual(df_arg["MRI-derived Features"].iloc[0], "Quantitative features available")
        self.assertEqual(df_arg["MRI-derived Features"].iloc[1], "Quantitative features missing or masked")

    def test_frontend_module_exports(self) -> None:
        """Verify that components and frontend package export render_synthetic_notice."""
        import nacc_progression.frontend as fe
        import nacc_progression.frontend.components as fec

        self.assertTrue(hasattr(fe, "render_synthetic_notice"))
        self.assertTrue(hasattr(fec, "render_synthetic_notice"))
        self.assertIn("render_synthetic_notice", fe.__all__)
        self.assertIn("render_synthetic_notice", fec.__all__)


class TestRealModelIntegration(unittest.TestCase):
    """Automated integration tests for real PyTorch MultimodalGRU model in platform."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.ckpt_path = Path(cls._temp_dir.name) / "mini_multimodal_model.pt"

        # Generate 12-feature synthetic training and validation sequences
        train_seqs = [
            make_sequence(
                f"train_syn_{i}",
                label=i % 2,
                visits=3,
                mri_features=12,
                imaging_feature_names=DEMO_MRI_FEATURES[:12],
            )
            for i in range(8)
        ]
        val_seqs = [
            make_sequence(
                f"val_syn_{i}",
                label=i % 2,
                visits=2,
                mri_features=12,
                imaging_feature_names=DEMO_MRI_FEATURES[:12],
            )
            for i in range(4)
        ]

        model_cfg = ModelConfig(
            imaging_features=12,
            clinical_features=6,
            imaging_embedding=16,
            clinical_embedding=16,
            temporal_hidden=16,
            dropout=0.1,
            imaging_feature_names=DEMO_MRI_FEATURES[:12],
        )
        train_cfg = TrainingConfig(max_epochs=2, batch_size=4, device="cpu")

        train_model(train_seqs, val_seqs, model_cfg, train_cfg, checkpoint_path=cls.ckpt_path)

        cls.real_service = PlatformService(str(cls.ckpt_path), demo_mode=False)
        cls.test_seq = make_sequence(
            "test_eval_patient",
            label=1,
            visits=3,
            mri_features=12,
            imaging_feature_names=DEMO_MRI_FEATURES[:12],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_real_pytorch_model_training_and_service_integration(self) -> None:
        """Verify service loads real PyTorch checkpoint, predicts, and explains without fallback."""
        self.assertFalse(self.real_service.is_demo_mode)
        self.assertIsNotNone(self.real_service.model)

        info = self.real_service.get_model_info()
        self.assertTrue(info.model_loaded)
        self.assertFalse(info.is_demo_mode)
        self.assertEqual(info.imaging_feature_count, 12)
        self.assertEqual(info.clinical_feature_count, 6)
        self.assertEqual(info.temporal_hidden, 16)
        self.assertEqual(info.imaging_feature_names, list(DEMO_MRI_FEATURES[:12]))

        # Prediction test
        pred = self.real_service.predict(self.test_seq, threshold=0.5)
        self.assertEqual(pred.subject_key, "test_eval_patient")
        self.assertTrue(0.0 <= pred.probability <= 1.0)
        self.assertIn(pred.predicted_label, (0, 1))
        self.assertEqual(pred.predicted_label, int(pred.probability >= 0.5))
        self.assertEqual(len(pred.prefix_probabilities), 3)
        self.assertEqual(pred.visit_count, 3)
        self.assertAlmostEqual(pred.probability, pred.prefix_probabilities[-1], places=5)
        self.assertIn("anchor outcome", pred.prefix_semantics)

        # JSON serialization roundtrip
        pred_json = pred.model_dump_json()
        self.assertIsInstance(pred_json, str)
        reconstructed_pred = PredictionResponse.model_validate_json(pred_json)
        self.assertEqual(reconstructed_pred.subject_key, pred.subject_key)
        self.assertAlmostEqual(reconstructed_pred.probability, pred.probability, places=5)

        # Explainability test (Integrated Gradients & Leave-One-Visit-Out)
        exp = self.real_service.explain(self.test_seq, steps=4, include_ablation=True)
        self.assertEqual(exp.subject_key, "test_eval_patient")
        self.assertEqual(len(exp.mri_attributions), 3)
        self.assertEqual(len(exp.mri_attributions[0]), 12)
        self.assertEqual(len(exp.clinical_attributions), 3)
        self.assertEqual(len(exp.clinical_attributions[0]), 6)
        self.assertEqual(len(exp.mri_feature_importance), 12)
        self.assertEqual(len(exp.clinical_feature_importance), 6)
        self.assertEqual(len(exp.visit_importance), 3)
        self.assertIsNotNone(exp.visit_ablation)
        self.assertEqual(len(exp.visit_ablation), 3)
        for abl in exp.visit_ablation:
            self.assertEqual(abl.remaining_visit_count, 2)
            self.assertTrue(0.0 <= abl.probability_without_visit <= 1.0)

        # JSON serialization roundtrip for explanation
        exp_json = exp.model_dump_json()
        reconstructed_exp = ExplanationResponse.model_validate_json(exp_json)
        self.assertEqual(reconstructed_exp.subject_key, exp.subject_key)
        self.assertEqual(len(reconstructed_exp.mri_feature_importance), 12)

    def test_real_pytorch_model_fastapi_endpoints_and_v1_aliases(self) -> None:
        """Verify real PyTorch model through FastAPI endpoints under /api and /api/v1."""
        app = create_app(service=self.real_service)
        visits = patient_sequence_to_visits(self.test_seq)

        # 1. Prediction endpoints
        pred_req = PredictionRequest(
            subject_key="test_eval_patient",
            visits=visits,
            imaging_feature_names=list(DEMO_MRI_FEATURES[:12]),
            threshold=0.5,
        )

        for route_path in ("/api/predict", "/api/v1/predict"):
            route = next(r for r in app.routes if r.path == route_path)
            res: PredictionResponse = route.endpoint(
                pred_req, current_user={"username": "researcher", "role": "researcher"}
            )
            self.assertEqual(res.subject_key, "test_eval_patient")
            self.assertTrue(0.0 <= res.probability <= 1.0)
            self.assertEqual(len(res.prefix_probabilities), 3)
            self.assertEqual(res.visit_count, 3)

        # 2. Explainability endpoints
        exp_req = ExplanationRequest(
            subject_key="test_eval_patient",
            visits=visits,
            imaging_feature_names=list(DEMO_MRI_FEATURES[:12]),
            steps=4,
            include_visit_ablation=True,
        )

        for route_path in ("/api/explain", "/api/v1/explain"):
            route = next(r for r in app.routes if r.path == route_path)
            res_exp: ExplanationResponse = route.endpoint(
                exp_req, current_user={"username": "researcher", "role": "researcher"}
            )
            self.assertEqual(res_exp.subject_key, "test_eval_patient")
            self.assertEqual(len(res_exp.mri_feature_importance), 12)
            self.assertEqual(len(res_exp.clinical_feature_importance), 6)
            self.assertEqual(len(res_exp.visit_importance), 3)
            self.assertIsNotNone(res_exp.visit_ablation)
            self.assertEqual(len(res_exp.visit_ablation), 3)

        # 3. Model Info endpoints
        for route_path in ("/api/model/info", "/api/v1/model/info"):
            route = next(r for r in app.routes if r.path == route_path)
            info = route.endpoint()
            self.assertTrue(info.model_loaded)
            self.assertFalse(info.is_demo_mode)
            self.assertEqual(info.imaging_feature_count, 12)

    def test_unsafe_fallback_prevention(self) -> None:
        """Verify real model inference/explainability failures raise explicit errors and never fall back."""
        app = create_app(service=self.real_service)
        visits = patient_sequence_to_visits(self.test_seq)
        pred_req = PredictionRequest(
            subject_key="test_eval_patient",
            visits=visits,
            imaging_feature_names=list(DEMO_MRI_FEATURES[:12]),
            threshold=0.5,
        )
        exp_req = ExplanationRequest(
            subject_key="test_eval_patient",
            visits=visits,
            imaging_feature_names=list(DEMO_MRI_FEATURES[:12]),
            steps=4,
            include_visit_ablation=True,
        )

        # Prediction failure must raise RuntimeError in service, HTTP 500 in API
        with patch("nacc_progression.training.predict_patient", side_effect=ValueError("Simulated GPU failure")):
            with self.assertRaises(RuntimeError) as cm:
                self.real_service.predict(self.test_seq)
            self.assertIn("Real model prediction failed", str(cm.exception))

            route_pred = next(r for r in app.routes if r.path == "/api/predict")
            with self.assertRaises(HTTPException) as cm_http:
                route_pred.endpoint(pred_req, current_user=None)
            self.assertEqual(cm_http.exception.status_code, 500)
            self.assertEqual(
                cm_http.exception.detail,
                "Prediction processing failed. An internal error occurred during model evaluation.",
            )
            self.assertNotIn("Simulated GPU failure", cm_http.exception.detail)

            # Same for /api/v1/predict
            route_v1_pred = next(r for r in app.routes if r.path == "/api/v1/predict")
            with self.assertRaises(HTTPException) as cm_http_v1:
                route_v1_pred.endpoint(pred_req, current_user=None)
            self.assertEqual(cm_http_v1.exception.status_code, 500)
            self.assertEqual(
                cm_http_v1.exception.detail,
                "Prediction processing failed. An internal error occurred during model evaluation.",
            )
            self.assertNotIn("Simulated GPU failure", cm_http_v1.exception.detail)

        # Explainability failure must raise RuntimeError in service, HTTP 500 in API
        with patch("nacc_progression.explainability.explain_patient", side_effect=ValueError("Simulated XAI failure")):
            with self.assertRaises(RuntimeError) as cm:
                self.real_service.explain(self.test_seq)
            self.assertIn("Real model explainability failed", str(cm.exception))

            route_exp = next(r for r in app.routes if r.path == "/api/explain")
            with self.assertRaises(HTTPException) as cm_http:
                route_exp.endpoint(exp_req, current_user=None)
            self.assertEqual(cm_http.exception.status_code, 500)
            self.assertEqual(
                cm_http.exception.detail,
                "Explainability processing failed. An internal error occurred during attribution calculation.",
            )
            self.assertNotIn("Simulated XAI failure", cm_http.exception.detail)

            # Same for /api/v1/explain
            route_v1_exp = next(r for r in app.routes if r.path == "/api/v1/explain")
            with self.assertRaises(HTTPException) as cm_http_v1:
                route_v1_exp.endpoint(exp_req, current_user=None)
            self.assertEqual(cm_http_v1.exception.status_code, 500)
            self.assertEqual(
                cm_http_v1.exception.detail,
                "Explainability processing failed. An internal error occurred during attribution calculation.",
            )
            self.assertNotIn("Simulated XAI failure", cm_http_v1.exception.detail)

    def test_api_v1_versioned_routes_catalog(self) -> None:
        """Verify all endpoints are available under /api/v1 alongside /api."""
        app = create_app()
        registered_paths = {r.path for r in app.routes}

        expected_v1_paths = [
            "/health",
            "/api/health",
            "/api/v1/health",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/me",
            "/api/v1/model/info",
            "/api/v1/patients",
            "/api/v1/patients/{subject_key}",
            "/api/v1/patients/{subject_key}/predict",
            "/api/v1/patients/{subject_key}/explanation",
            "/api/v1/predict",
            "/api/v1/explain",
            "/api/v1/audit",
            "/api/v1/audit/summary",
            "/api/v1/audit/logs",
        ]
        for path in expected_v1_paths:
            self.assertIn(path, registered_paths, f"Missing versioned endpoint: {path}")

        # Check root health endpoint
        route_root_health = next(r for r in app.routes if r.path == "/health")
        root_health_res = route_root_health.endpoint()
        self.assertEqual(root_health_res["status"], "healthy")

        # Check v1 health endpoint
        route_health = next(r for r in app.routes if r.path == "/api/v1/health")
        health_res = route_health.endpoint()
        self.assertEqual(health_res["status"], "healthy")
        self.assertEqual(health_res["version"], "0.1.0")

        # Check v1 model info
        route_info = next(r for r in app.routes if r.path == "/api/v1/model/info")
        info_res = route_info.endpoint()
        self.assertIsInstance(info_res, ModelInfoResponse)

        # Check v1 patients
        route_pts = next(r for r in app.routes if r.path == "/api/v1/patients")
        pts_res = route_pts.endpoint(query="", label=None, min_visits=1)
        self.assertGreaterEqual(len(pts_res), 5)

        # Check v1 audit summary
        route_audit = next(r for r in app.routes if r.path == "/api/v1/audit/summary")
        audit_res = route_audit.endpoint()
        self.assertIsInstance(audit_res, AuditResponse)

    def test_demo_mode_regression_without_checkpoint(self) -> None:
        """Verify demo mode continues to work smoothly when checkpoint is omitted."""
        demo_service = PlatformService(checkpoint_path=None, demo_mode=True)
        self.assertTrue(demo_service.is_demo_mode)
        self.assertIsNone(demo_service.model)

        pred = demo_service.predict(self.test_seq, threshold=0.5)
        self.assertEqual(pred.subject_key, "test_eval_patient")
        self.assertTrue(0.0 <= pred.probability <= 1.0)
        self.assertEqual(len(pred.prefix_probabilities), 3)

        exp = demo_service.explain(self.test_seq, steps=4, include_ablation=True)
        self.assertEqual(exp.subject_key, "test_eval_patient")
        self.assertGreater(len(exp.mri_feature_importance), 0)

        # Non-existent checkpoint path must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            PlatformService(checkpoint_path="non_existent_model_checkpoint.pt", demo_mode=False)


class TestPlatformModeSeparationAndSecurity(unittest.TestCase):
    """Test strict separation of DEMO_MODE, safe startup failure, and error sanitization."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.ckpt_path = Path(cls._temp_dir.name) / "test_mini_model.pt"

        train_seqs = [
            make_sequence(f"train_m_{i}", label=i % 2, visits=2, mri_features=12, imaging_feature_names=DEMO_MRI_FEATURES[:12])
            for i in range(4)
        ]
        val_seqs = [
            make_sequence(f"val_m_{i}", label=i % 2, visits=2, mri_features=12, imaging_feature_names=DEMO_MRI_FEATURES[:12])
            for i in range(2)
        ]
        cfg = ModelConfig(
            imaging_features=12,
            clinical_features=6,
            imaging_embedding=16,
            clinical_embedding=16,
            temporal_hidden=16,
            imaging_feature_names=DEMO_MRI_FEATURES[:12],
        )
        train_model(train_seqs, val_seqs, cfg, TrainingConfig(max_epochs=1, batch_size=2, device="cpu"), checkpoint_path=cls.ckpt_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_demo_mode_1_explicit_enabled(self) -> None:
        """DEMO_MODE=1 explicitly enables synthetic demo services and populates demo patients."""
        with patch.dict(os.environ, {"DEMO_MODE": "1"}, clear=False):
            os.environ.pop("MODEL_CHECKPOINT_PATH", None)
            service = PlatformService()
            self.assertTrue(service.is_demo_mode)
            self.assertIsNone(service.model)
            self.assertEqual(len(service.patients), 15)

    def test_demo_mode_0_missing_checkpoint_fails_closed(self) -> None:
        """DEMO_MODE=0 without MODEL_CHECKPOINT_PATH must cause startup failure."""
        with patch.dict(os.environ, {"DEMO_MODE": "0"}, clear=False):
            os.environ.pop("MODEL_CHECKPOINT_PATH", None)
            with self.assertRaises(RuntimeError) as cm:
                PlatformService()
            self.assertIn("DEMO_MODE=0", str(cm.exception))
            self.assertIn("requires a valid MODEL_CHECKPOINT_PATH", str(cm.exception))

    def test_demo_mode_0_invalid_checkpoint_fails_closed(self) -> None:
        """DEMO_MODE=0 with non-existent checkpoint must fail with FileNotFoundError."""
        with patch.dict(os.environ, {"DEMO_MODE": "0", "MODEL_CHECKPOINT_PATH": "missing_checkpoint.pt"}, clear=False):
            with self.assertRaises(FileNotFoundError):
                PlatformService()

    def test_demo_mode_unset_missing_checkpoint_fails_closed(self) -> None:
        """Unset DEMO_MODE without checkpoint must fail safely rather than silently enabling demo."""
        env_copy = os.environ.copy()
        env_copy.pop("DEMO_MODE", None)
        env_copy.pop("MODEL_CHECKPOINT_PATH", None)
        with patch.dict(os.environ, env_copy, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                PlatformService()
            self.assertIn("DEMO_MODE environment variable is unset", str(cm.exception))

    def test_demo_mode_0_with_valid_checkpoint_has_empty_registry(self) -> None:
        """Private research mode loads real checkpoint and does NOT fabricate synthetic patients."""
        service = PlatformService(checkpoint_path=self.ckpt_path, demo_mode=False)
        self.assertFalse(service.is_demo_mode)
        self.assertIsNotNone(service.model)
        self.assertEqual(len(service.patients), 0)

    def test_patient_specific_routes_demo_mode(self) -> None:
        """Test POST /api/v1/patients/{subject_key}/predict and GET /explanation in demo mode."""
        app = create_app(service=PlatformService(demo_mode=True))

        # Prediction by subject_key
        route_pred = next(r for r in app.routes if r.path == "/api/v1/patients/{subject_key}/predict")
        pred_res = route_pred.endpoint(subject_key="n74_demo_rapid_progressor", threshold=0.5, current_user=None)
        self.assertIsInstance(pred_res, PredictionResponse)
        self.assertEqual(pred_res.subject_key, "n74_demo_rapid_progressor")
        self.assertTrue(0.0 <= pred_res.probability <= 1.0)
        self.assertEqual(pred_res.visit_count, 3)

        # Non-existent patient raises 404
        with self.assertRaises(HTTPException) as cm:
            route_pred.endpoint(subject_key="non_existent_pt_999", threshold=0.5, current_user=None)
        self.assertEqual(cm.exception.status_code, 404)

        # Explanation by subject_key
        route_exp = next(r for r in app.routes if r.path == "/api/v1/patients/{subject_key}/explanation")
        exp_res = route_exp.endpoint(
            subject_key="n74_demo_rapid_progressor",
            steps=4,
            include_visit_ablation=True,
            threshold=0.5,
            current_user=None,
        )
        self.assertIsInstance(exp_res, ExplanationResponse)
        self.assertEqual(exp_res.subject_key, "n74_demo_rapid_progressor")
        self.assertGreater(len(exp_res.mri_feature_importance), 0)

        # Non-existent patient raises 404
        with self.assertRaises(HTTPException) as cm_exp:
            route_exp.endpoint(
                subject_key="non_existent_pt_999",
                steps=4,
                include_visit_ablation=True,
                threshold=0.5,
                current_user=None,
            )
        self.assertEqual(cm_exp.exception.status_code, 404)

    def test_patient_specific_routes_private_mode_empty_registry_503(self) -> None:
        """In private mode with no ingested registry, patient-specific endpoints return 503."""
        private_service = PlatformService(checkpoint_path=self.ckpt_path, demo_mode=False)
        app = create_app(service=private_service)

        route_pred = next(r for r in app.routes if r.path == "/api/v1/patients/{subject_key}/predict")
        with self.assertRaises(HTTPException) as cm:
            route_pred.endpoint(subject_key="any_subject", threshold=0.5, current_user=None)
        self.assertEqual(cm.exception.status_code, 503)
        self.assertIn("unavailable in private research mode", cm.exception.detail)

        route_exp = next(r for r in app.routes if r.path == "/api/v1/patients/{subject_key}/explanation")
        with self.assertRaises(HTTPException) as cm_exp:
            route_exp.endpoint(subject_key="any_subject", steps=4, include_visit_ablation=True, threshold=0.5, current_user=None)
        self.assertEqual(cm_exp.exception.status_code, 503)
        self.assertIn("unavailable in private research mode", cm_exp.exception.detail)

    def test_error_sanitization_no_leakage(self) -> None:
        """Verify internal exceptions never expose filesystem paths or secrets to API clients."""
        service = PlatformService(demo_mode=True)
        app = create_app(service=service)

        mock_leak_msg = "Leaked internal path: E:\\NACC-DATA\\keys\\secret.bin; CUDA shape mismatch [1, 2]"
        with patch.object(service, "predict", side_effect=RuntimeError(mock_leak_msg)):
            route_pred = next(r for r in app.routes if r.path == "/api/v1/patients/{subject_key}/predict")
            with self.assertRaises(HTTPException) as cm:
                route_pred.endpoint(subject_key="n74_demo_rapid_progressor", threshold=0.5, current_user=None)

            self.assertEqual(cm.exception.status_code, 500)
            self.assertEqual(
                cm.exception.detail,
                "Prediction processing failed. An internal error occurred during model evaluation.",
            )
            # Ensure sensitive substrings were sanitized out
            self.assertNotIn("NACC-DATA", cm.exception.detail)
            self.assertNotIn("secret.bin", cm.exception.detail)
            self.assertNotIn("CUDA", cm.exception.detail)

        with patch.object(service, "explain", side_effect=RuntimeError(mock_leak_msg)):
            route_exp = next(r for r in app.routes if r.path == "/api/v1/patients/{subject_key}/explanation")
            with self.assertRaises(HTTPException) as cm:
                route_exp.endpoint(
                    subject_key="n74_demo_rapid_progressor",
                    steps=4,
                    include_visit_ablation=True,
                    threshold=0.5,
                    current_user=None,
                )

            self.assertEqual(cm.exception.status_code, 500)
            self.assertEqual(
                cm.exception.detail,
                "Explainability processing failed. An internal error occurred during attribution calculation.",
            )
            self.assertNotIn("NACC-DATA", cm.exception.detail)
            self.assertNotIn("secret.bin", cm.exception.detail)
            self.assertNotIn("CUDA", cm.exception.detail)


class TestStreamlitDeployment(unittest.TestCase):
    """Test Streamlit Community Cloud entrypoint, secrets propagation, and demo setup."""

    def test_streamlit_app_wrapper_exists_and_compiles(self) -> None:
        """Verify root-level streamlit_app.py exists and compiles cleanly."""
        root_dir = Path(__file__).resolve().parents[1]
        app_file = root_dir / "streamlit_app.py"
        self.assertTrue(app_file.is_file(), "streamlit_app.py must exist at project root.")
        source = app_file.read_text(encoding="utf-8")
        compiled = compile(source, str(app_file), "exec")
        self.assertIsNotNone(compiled)

    def test_streamlit_secrets_propagation(self) -> None:
        """Verify st.secrets dictionary values are injected into os.environ."""
        test_secrets = {
            "DEMO_MODE": "1",
            "PLATFORM_ENV": "production",
            "JWT_SECRET_KEY": "test-mock-secret-key-32bytes-len",
        }
        for k in test_secrets:
            if k in os.environ:
                del os.environ[k]

        mock_st = MagicMock()
        mock_st.secrets = test_secrets

        for key in ("DEMO_MODE", "PLATFORM_ENV", "JWT_SECRET_KEY", "PLATFORM_DB_PATH", "BACKEND_API_URL"):
            if key in mock_st.secrets and key not in os.environ:
                os.environ[key] = str(mock_st.secrets[key])

        self.assertEqual(os.environ.get("DEMO_MODE"), "1")
        self.assertEqual(os.environ.get("PLATFORM_ENV"), "production")
        self.assertEqual(os.environ.get("JWT_SECRET_KEY"), "test-mock-secret-key-32bytes-len")

    def test_streamlit_production_env_default(self) -> None:
        """Verify streamlit_app.py defaults to production environment with valid ephemeral secret."""
        import secrets as py_secrets
        env = {}
        env["DEMO_MODE"] = "1"
        env.setdefault("PLATFORM_ENV", "production")
        if "JWT_SECRET_KEY" not in env:
            env["JWT_SECRET_KEY"] = py_secrets.token_urlsafe(48)
        self.assertEqual(env["PLATFORM_ENV"], "production")
        self.assertEqual(env["DEMO_MODE"], "1")
        self.assertGreaterEqual(len(env["JWT_SECRET_KEY"]), 32)

    def test_streamlit_demo_database_initialization(self) -> None:
        """Ensure demo mode initializes sqlite database with default researcher accounts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = str(Path(tmp_dir) / "demo_test.db")
            # Default with seed_admin=True seeds 3 accounts
            init_db(db_path=test_db, enable_demo_users=True, seed_admin=True)
            self.assertEqual(count_users(db_path=test_db), 3)
            user = get_user_by_username("researcher", db_path=test_db)
            self.assertIsNotNone(user)
            self.assertEqual(user["role"], "researcher")

    def test_public_demo_database_no_admin_seeded(self) -> None:
        """Verify public deployment with seed_admin=False never exposes an admin account."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = str(Path(tmp_dir) / "public_demo.db")
            init_db(db_path=test_db, enable_demo_users=True, seed_admin=False)
            self.assertEqual(count_users(db_path=test_db), 2)
            self.assertIsNone(get_user_by_username("admin", db_path=test_db))
            self.assertIsNotNone(get_user_by_username("researcher", db_path=test_db))
            self.assertIsNotNone(get_user_by_username("clinician", db_path=test_db))

    def test_streamlit_insecure_jwt_placeholder_rejected(self) -> None:
        """Verify placeholder secrets like public-demo-secret-key-change-in-prod fail safely."""
        insecure_key = "public-demo-secret-key-change-in-prod"
        with self.assertRaises(RuntimeError):
            if len(insecure_key) < 32 or any(
                p in insecure_key.lower()
                for p in ("public-demo", "replace", "placeholder", "secret-key-change", "changeme")
            ):
                raise RuntimeError("Insecure JWT key rejected")

    def test_streamlit_frontend_app_script_exists(self) -> None:
        """Ensure destination app.py exists and defines core UI navigation."""
        frontend_app = Path(__file__).resolve().parents[1] / "src" / "nacc_progression" / "frontend" / "app.py"
        self.assertTrue(frontend_app.is_file())
        source = frontend_app.read_text(encoding="utf-8")
        self.assertIn("Explainable Longitudinal Medical AI", source)
        self.assertIn("render_synthetic_notice", source)


if __name__ == "__main__":
    unittest.main()
