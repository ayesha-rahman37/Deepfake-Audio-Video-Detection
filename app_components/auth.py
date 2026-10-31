
from pathlib import Path
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "deepfake_intelligence.db"

PBKDF2_ITERATIONS = 310_000


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def init_db():
    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fake_score REAL NOT NULL,
                real_score REAL NOT NULL,
                duration_seconds REAL,
                device TEXT,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        # Automatically update an older database.
        # Existing users and detection records are preserved.

        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(detection_history)"
            ).fetchall()
        }

        if "duration_seconds" not in existing_columns:
            connection.execute("""
                ALTER TABLE detection_history
                ADD COLUMN duration_seconds REAL
            """)

        if "device" not in existing_columns:
            connection.execute("""
                ALTER TABLE detection_history
                ADD COLUMN device TEXT
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_user
            ON detection_history(user_id, id DESC)
        """)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS,
    ).hex()

    return digest, salt


def verify_password(password, stored_hash, salt):
    calculated, _ = hash_password(password, salt)

    return hmac.compare_digest(
        calculated,
        stored_hash,
    )


def create_user(full_name, email, password):
    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name:
        return False, "Please enter your full name."

    if len(full_name) > 100:
        return False, "Name must be 100 characters or less."

    if (
        not email
        or "@" not in email
        or "." not in email.rsplit("@", 1)[-1]
    ):
        return False, "Please enter a valid email address."

    if len(email) > 254:
        return False, "Email address is too long."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash, salt = hash_password(password)

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    full_name,
                    email,
                    password_hash,
                    salt,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    full_name,
                    email,
                    password_hash,
                    salt,
                    created_at,
                ),
            )

        return True, "Your account has been created successfully."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate_user(email, password):
    email = email.strip().lower()

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT
                id,
                full_name,
                email,
                password_hash,
                salt,
                created_at
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if user is None:
        return None

    if not verify_password(
        password,
        user["password_hash"],
        user["salt"],
    ):
        return None

    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
        "created_at": user["created_at"],
    }


def get_user_by_id(user_id):
    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT id, full_name, email, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

    return dict(user) if user else None


def save_detection(
    user_id,
    filename,
    prediction,
    fake_score,
    real_score,
    result,
):
    created_at = datetime.now(timezone.utc).isoformat()

    duration = result.get("duration_seconds")
    device = result.get("device")

    result_json = json.dumps(result, default=str)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO detection_history (
                user_id,
                filename,
                prediction,
                fake_score,
                real_score,
                duration_seconds,
                device,
                result_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                prediction,
                float(fake_score),
                float(real_score),
                duration,
                str(device) if device is not None else None,
                result_json,
                created_at,
            ),
        )


def get_user_history(user_id, limit=100):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                prediction,
                fake_score,
                real_score,
                duration_seconds,
                device,
                created_at
            FROM detection_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(int(limit), 500))),
        ).fetchall()

    return [dict(row) for row in rows]


def get_detection_by_id(user_id, detection_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM detection_history
            WHERE id = ? AND user_id = ?
            """,
            (detection_id, user_id),
        ).fetchone()

    if row is None:
        return None

    item = dict(row)

    try:
        item["result"] = json.loads(item.pop("result_json"))
    except (ValueError, TypeError):
        item["result"] = {}

    return item
