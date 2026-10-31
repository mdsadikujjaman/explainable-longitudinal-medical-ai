"""Streamlit Community Cloud deployment entrypoint for Explainable Longitudinal Medical AI.

Ensures execution in synthetic demonstration mode with automatic secrets propagation,
database initialization, and direct in-process inference fallback.
"""

from __future__ import annotations

import os
import runpy
import secrets
import sys
from pathlib import Path

# 1. Ensure local 'src' directory is in sys.path before any project imports
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. Propagate and validate Streamlit secrets (st.secrets) if present.
# Streamlit Community Cloud injects dashboard secrets into st.secrets, but does NOT
# automatically expose them as OS environment variables (os.environ).
try:
    import streamlit as st

    if hasattr(st, "secrets"):
        # Check for user-configured JWT secret in st.secrets
        if "JWT_SECRET_KEY" in st.secrets:
            secret_candidate = str(st.secrets["JWT_SECRET_KEY"]).strip()
            # Reject weak, placeholder, or predictable keys
            if len(secret_candidate) < 32 or any(
                p in secret_candidate.lower()
                for p in (
                    "public-demo",
                    "replace",
                    "placeholder",
                    "secret-key-change",
                    "changeme",
                    "secret",
                )
            ):
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: The JWT_SECRET_KEY in st.secrets "
                    "is insecure, predictable, or shorter than 32 characters. "
                    "Configure a cryptographically random secret of at least 32 characters."
                )
            os.environ["JWT_SECRET_KEY"] = secret_candidate

        # Propagate other allowed configuration settings
        for key in ("PLATFORM_ENV", "PLATFORM_DB_PATH", "BACKEND_API_URL"):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
except Exception as err:
    if isinstance(err, RuntimeError):
        raise

# 3. Environment configuration:
# Unconditionally enforce DEMO_MODE=1 for public community deployment regardless of any conflicting secret.
os.environ["DEMO_MODE"] = "1"
# Enforce production-grade security standards for public Streamlit Community Cloud deployment.
os.environ.setdefault("PLATFORM_ENV", "production")

# If no JWT secret was provided in st.secrets or os.environ, generate a cryptographically strong
# ephemeral random secret (384-bit CSPRNG, 64 URL-safe characters >= 32 minimum) for this container process lifetime.
# This satisfies production-mode validation while preventing hardcoded secrets and ensuring unguessable session tokens.
if "JWT_SECRET_KEY" not in os.environ:
    os.environ["JWT_SECRET_KEY"] = secrets.token_urlsafe(48)

# 4. Initialize SQLite demo database tables with restricted visitor accounts (no admin account)
from nacc_progression.platform.database import init_db

init_db(enable_demo_users=True, seed_admin=False)

# 5. Execute frontend application in the active Streamlit runtime context
APP_PATH = SRC_DIR / "nacc_progression" / "frontend" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
