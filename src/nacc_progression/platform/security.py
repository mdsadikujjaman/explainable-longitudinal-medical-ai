"""Authentication, password hashing, and token handling for the platform layer.

Implements Argon2 password hashing (RFC 9106) with backward-compatible Bcrypt
verification and rehash-on-login migration, along with strict environment-based
JWT secret key management.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Argon2id password hasher with RFC 9106 recommended parameters
_argon2_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MiB
    parallelism=1,
    hash_len=32,
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 3600 * 8  # 8-hour session for research workflows
MIN_SECRET_LENGTH = 32

INSECURE_PLACEHOLDERS = {
    "",
    "replace_with_secure_random_secret_at_least_32_chars",
    "replace_with_secure_random_secret_of_at_least_32_characters",
    "replace-with-a-secure-random-secret-key-in-production",
    "nacc-progression-dev-secret-key-at-least-32-chars-long",
    "nacc-progression-production-secret-key-32chars",
    "public-demo-secret-key-change-in-prod",
    "secret",
    "changeme",
}

# Process-level ephemeral secret used only as safe development/demo fallback when unset
_EPHEMERAL_DEV_SECRET: str | None = None

security_scheme = HTTPBearer(auto_error=False)


def get_jwt_secret_key() -> str:
    """Retrieve and validate the JWT signing secret from environment configuration.

    In production mode (PLATFORM_ENV=production), raises RuntimeError if the secret
    is missing, too short (<32 chars), or matches an insecure placeholder.
    In development/demo mode, generates a secure process-lifetime ephemeral key if unset.
    """
    global _EPHEMERAL_DEV_SECRET

    env_mode = os.environ.get("PLATFORM_ENV", "development").strip().lower()
    raw_secret = os.environ.get("JWT_SECRET_KEY")

    if env_mode == "production":
        if not raw_secret or len(raw_secret.strip()) < MIN_SECRET_LENGTH:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: Production mode requires a secure "
                f"JWT_SECRET_KEY of at least {MIN_SECRET_LENGTH} characters set in environment variables."
            )
        normalized = raw_secret.strip().lower()
        if normalized in INSECURE_PLACEHOLDERS or "replace" in normalized or "placeholder" in normalized:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: Production JWT_SECRET_KEY cannot be a known placeholder."
            )
        return raw_secret.strip()

    # Development / Demo / Test mode fallback
    if raw_secret and len(raw_secret.strip()) >= 16:
        normalized = raw_secret.strip().lower()
        if normalized not in INSECURE_PLACEHOLDERS and "replace" not in normalized:
            return raw_secret.strip()

    if _EPHEMERAL_DEV_SECRET is None:
        _EPHEMERAL_DEV_SECRET = secrets.token_urlsafe(48)
        logger.warning(
            "JWT_SECRET_KEY not set in environment. Generated secure process-lifetime ephemeral key for development."
        )

    return _EPHEMERAL_DEV_SECRET


def hash_password(password: str) -> str:
    """Hash plaintext password using Argon2id."""
    if not password:
        raise ValueError("Password cannot be empty")
    return _argon2_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against Argon2id hash or legacy Bcrypt hash."""
    if not plain_password or not hashed_password:
        return False

    # Check for legacy Bcrypt hash prefixes ($2a$, $2b$, $2y$)
    if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("ascii"))
        except Exception:
            return False

    # Standard Argon2 verification
    try:
        return _argon2_hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:
        return False


def needs_rehash(hashed_password: str) -> bool:
    """Determine if a stored hash should be upgraded on successful login.

    Returns True for legacy Bcrypt hashes or outdated Argon2 parameters.
    """
    if not hashed_password:
        return True
    if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
        return True
    try:
        return _argon2_hasher.check_needs_rehash(hashed_password)
    except Exception:
        return True


def _b64url_encode(data: bytes) -> str:
    """Encode bytes into base64url string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    """Decode base64url string with padding restoration."""
    rem = len(encoded) % 4
    if rem > 0:
        encoded += "=" * (4 - rem)
    return base64.urlsafe_b64decode(encoded.encode("ascii"))


def create_access_token(
    subject: str,
    role: str = "researcher",
    expires_in_seconds: int = ACCESS_TOKEN_EXPIRE_SECONDS,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a standard HS256 JWT access token."""
    import hashlib
    import hmac

    secret = get_jwt_secret_key()
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    if extra_claims:
        payload.update(extra_claims)

    encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    encoded_signature = _b64url_encode(signature)

    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode HS256 JWT access token."""
    import hashlib
    import hmac

    secret = get_jwt_secret_key()
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format")

    encoded_header, encoded_payload, encoded_signature = parts
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    expected_signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_signature = _b64url_decode(encoded_signature)

    if not hmac.compare_digest(expected_signature, actual_signature):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(encoded_payload).decode("utf-8"))
    now = int(time.time())
    if "exp" in payload and payload["exp"] < now:
        raise ValueError("Token has expired")

    return payload


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> dict[str, Any]:
    """FastAPI dependency: extracts and verifies JWT bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {err}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "username": payload["sub"],
        "role": payload.get("role", "researcher"),
    }
