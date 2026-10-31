
import pytest

from app_components.auth import (
    init_db,
    create_user,
    authenticate_user,
    get_user_history,
    save_detection,
)


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    import app_components.auth as auth

    database = tmp_path / "test_deepfake.db"

    monkeypatch.setattr(auth, "DATABASE_PATH", database)
    monkeypatch.setattr(auth, "DATA_DIR", tmp_path)

    init_db()
    return auth


def test_register_user(isolated_database):
    success, message = isolated_database.create_user(
        "Test User",
        "test@example.com",
        "StrongPass123",
    )

    assert success
    assert "successfully" in message.lower()


def test_duplicate_email_is_rejected(isolated_database):
    isolated_database.create_user(
        "Test User",
        "test@example.com",
        "StrongPass123",
    )

    success, message = isolated_database.create_user(
        "Another User",
        "TEST@example.com",
        "AnotherPass123",
    )

    assert not success
    assert "already exists" in message.lower()


def test_short_password_is_rejected(isolated_database):
    success, message = isolated_database.create_user(
        "Test User",
        "test@example.com",
        "short",
    )

    assert not success
    assert "8 characters" in message.lower()


def test_login_success(isolated_database):
    isolated_database.create_user(
        "Test User",
        "test@example.com",
        "StrongPass123",
    )

    user = isolated_database.authenticate_user(
        "test@example.com",
        "StrongPass123",
    )

    assert user is not None
    assert user["email"] == "test@example.com"


def test_login_wrong_password(isolated_database):
    isolated_database.create_user(
        "Test User",
        "test@example.com",
        "StrongPass123",
    )

    user = isolated_database.authenticate_user(
        "test@example.com",
        "WrongPass123",
    )

    assert user is None


def test_detection_history_is_user_specific(isolated_database):
    isolated_database.create_user(
        "User One",
        "one@example.com",
        "StrongPass123",
    )

    isolated_database.create_user(
        "User Two",
        "two@example.com",
        "StrongPass123",
    )

    user_one = isolated_database.authenticate_user(
        "one@example.com",
        "StrongPass123",
    )

    user_two = isolated_database.authenticate_user(
        "two@example.com",
        "StrongPass123",
    )

    isolated_database.save_detection(
        user_id=user_one["id"],
        filename="sample.mp4",
        prediction="FAKE",
        fake_score=75.0,
        real_score=25.0,
        result={
            "prediction": "FAKE",
            "fake_score": 75.0,
            "real_score": 25.0,
            "duration_seconds": 10.0,
            "device": "cpu",
            "windows": [],
        },
    )

    one_history = isolated_database.get_user_history(user_one["id"])
    two_history = isolated_database.get_user_history(user_two["id"])

    assert len(one_history) == 1
    assert len(two_history) == 0
    assert one_history[0]["prediction"] == "FAKE"
