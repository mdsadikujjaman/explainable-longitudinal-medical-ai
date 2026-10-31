"""SQLite user database and audit log repository for the platform layer."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

from nacc_progression.platform.security import hash_password

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get(
    "PLATFORM_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "platform_data.db"),
)


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite database connection with row factory."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(
    db_path: str = DEFAULT_DB_PATH,
    enable_demo_users: bool | None = None,
    seed_admin: bool = True,
) -> None:
    """Initialize database tables and optionally seed demo accounts.

    Security rule: Default demo accounts are seeded ONLY when explicit synthetic
    demo mode is enabled (DEMO_MODE=1 or enable_demo_users=True). In production
    or private research mode, no default credentials are created.
    When seed_admin=False, only the restricted demo accounts (researcher, clinician)
    are seeded to avoid exposing administrative access in public deployments.
    """
    if enable_demo_users is None:
        raw_demo_flag = os.environ.get("DEMO_MODE", "0").strip().lower()
        enable_demo_users = raw_demo_flag in ("1", "true", "yes")

    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'researcher',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                probability REAL NOT NULL,
                threshold REAL NOT NULL,
                predicted_label INTEGER NOT NULL,
                visit_count INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                created_at REAL NOT NULL
            );
        """)

        # Seed demonstration accounts ONLY when synthetic demo mode is enabled
        cursor = conn.execute("SELECT COUNT(*) as count FROM users")
        if cursor.fetchone()["count"] == 0 and enable_demo_users:
            now = time.time()
            demo_users = [
                ("researcher", "researcher123", "researcher"),
                ("clinician", "clinician123", "clinician"),
            ]
            if seed_admin:
                demo_users.insert(0, ("admin", "admin123", "admin"))
            for username, plain_pw, role in demo_users:
                conn.execute(
                    """
                    INSERT INTO users (username, hashed_password, role, is_active, created_at)
                    VALUES (?, ?, ?, 1, ?)
                    """,
                    (username, hash_password(plain_pw), role, now),
                )
            logger.info("DEMO_MODE active: Seeded default demonstration accounts. Do NOT use in production.")
        elif cursor.fetchone() is None:
            logger.info("Private research mode active: No default user accounts seeded.")

        # In public demo mode with seed_admin=False, purge any legacy admin account
        if enable_demo_users and not seed_admin:
            conn.execute("DELETE FROM users WHERE username = 'admin'")

    conn.close()


def count_users(db_path: str = DEFAULT_DB_PATH) -> int:
    """Return total number of registered users."""
    conn = get_db_connection(db_path)
    cursor = conn.execute("SELECT COUNT(*) as count FROM users")
    count = int(cursor.fetchone()["count"])
    conn.close()
    return count


def get_user_by_username(username: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieve user dictionary by username."""
    conn = get_db_connection(db_path)
    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def create_user(
    username: str,
    plain_password: str,
    role: str = "researcher",
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Register a new user account with an Argon2id hashed password."""
    conn = get_db_connection(db_path)
    now = time.time()
    hashed = hash_password(plain_password)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, hashed_password, role, is_active, created_at)
                VALUES (?, ?, ?, 1, ?)
                """,
                (username, hashed, role, now),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"User '{username}' already exists")

    conn.close()
    return {
        "id": user_id,
        "username": username,
        "role": role,
        "is_active": True,
        "created_at": now,
    }


def update_user_password(
    username: str,
    new_hashed_password: str,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Update stored password hash (used for seamless rehash-on-login to Argon2)."""
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute(
            "UPDATE users SET hashed_password = ? WHERE username = ?",
            (new_hashed_password, username),
        )
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def log_prediction(
    username: str,
    subject_key: str,
    probability: float,
    threshold: float,
    predicted_label: int,
    visit_count: int,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Record an inference event in the audit log."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO prediction_logs
            (username, subject_key, probability, threshold, predicted_label, visit_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, subject_key, probability, threshold, predicted_label, visit_count, time.time()),
        )
    conn.close()


def get_prediction_logs(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Fetch recent prediction audit logs."""
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT * FROM prediction_logs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows
