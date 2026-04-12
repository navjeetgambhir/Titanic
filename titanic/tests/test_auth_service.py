"""Tests for authentication: POST /auth/signup, POST /auth/login, UI routes."""
import pytest
from fastapi.testclient import TestClient


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _signup(client: TestClient, username="testuser", email="test@example.com",
            password="securepass123"):
    return client.post("/auth/signup", json={
        "username": username,
        "email":    email,
        "password": password,
    })


def _login(client: TestClient, email="test@example.com", password="securepass123"):
    return client.post("/auth/login", json={"email": email, "password": password})


# ─── /auth/signup ─────────────────────────────────────────────────────────────

class TestSignup:

    def test_signup_returns_201(self, client):
        res = _signup(client, username="alice", email="alice@example.com")
        assert res.status_code == 201

    def test_signup_response_schema(self, client):
        res = _signup(client, username="bob", email="bob@example.com")
        body = res.json()
        assert "access_token" in body
        assert "token_type" in body
        assert "user_id" in body
        assert "username" in body
        assert body["token_type"] == "bearer"

    def test_signup_token_is_string(self, client):
        res = _signup(client, username="carol", email="carol@example.com")
        assert isinstance(res.json()["access_token"], str)
        assert len(res.json()["access_token"]) > 20

    def test_duplicate_username_returns_409(self, client):
        _signup(client, username="dupeuser", email="dupe1@example.com")
        res = _signup(client, username="dupeuser", email="dupe2@example.com")
        assert res.status_code == 409

    def test_duplicate_email_returns_409(self, client):
        _signup(client, username="email1user", email="shared@example.com")
        res = _signup(client, username="email2user", email="shared@example.com")
        assert res.status_code == 409

    def test_short_password_returns_422(self, client):
        res = _signup(client, username="shortpw", email="shortpw@example.com",
                      password="abc")
        assert res.status_code == 422

    def test_invalid_email_returns_422(self, client):
        res = client.post("/auth/signup", json={
            "username": "bademail",
            "email":    "not-an-email",
            "password": "securepass123",
        })
        assert res.status_code == 422


# ─── /auth/login ──────────────────────────────────────────────────────────────

class TestLogin:

    def test_login_returns_200(self, client):
        _signup(client, username="loginuser", email="loginuser@example.com")
        res = _login(client, email="loginuser@example.com")
        assert res.status_code == 200

    def test_login_returns_token(self, client):
        _signup(client, username="tokenuser", email="tokenuser@example.com")
        body = _login(client, email="tokenuser@example.com").json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_wrong_password_returns_401(self, client):
        _signup(client, username="wrongpw", email="wrongpw@example.com")
        res = _login(client, email="wrongpw@example.com", password="wrongpassword")
        assert res.status_code == 401

    def test_unknown_email_returns_401(self, client):
        res = _login(client, email="nobody@example.com")
        assert res.status_code == 401

    def test_token_is_valid_jwt(self, client):
        import jwt as pyjwt
        from core.auth import JWT_SECRET, JWT_ALGORITHM
        _signup(client, username="jwtcheck", email="jwtcheck@example.com")
        token = _login(client, email="jwtcheck@example.com").json()["access_token"]
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        assert "sub" in payload
        assert "username" in payload
        assert "exp" in payload

    def test_login_username_matches_signup(self, client):
        _signup(client, username="matchme", email="matchme@example.com")
        body = _login(client, email="matchme@example.com").json()
        assert body["username"] == "matchme"


# ─── UI auth routes ───────────────────────────────────────────────────────────

class TestAuthUI:

    def test_login_page_returns_200(self, client):
        res = client.get("/ui/login")
        assert res.status_code == 200
        assert b"Sign in" in res.content

    def test_signup_page_returns_200(self, client):
        res = client.get("/ui/signup")
        assert res.status_code == 200
        assert b"Create" in res.content

    def test_ui_signup_sets_cookie(self, client):
        res = client.post("/ui/signup", data={
            "username":         "cookieuser",
            "email":            "cookieuser@example.com",
            "password":         "securepass123",
            "confirm_password": "securepass123",
        }, follow_redirects=False)
        assert res.status_code == 200
        assert "access_token" in res.cookies

    def test_ui_login_sets_cookie(self, client):
        _signup(client, username="cookielogin", email="cookielogin@example.com")
        res = client.post("/ui/login", data={
            "email":    "cookielogin@example.com",
            "password": "securepass123",
        }, follow_redirects=False)
        assert res.status_code == 200
        assert "access_token" in res.cookies

    def test_ui_login_bad_credentials_returns_401(self, client):
        res = client.post("/ui/login", data={
            "email":    "ghost@example.com",
            "password": "wrongpassword",
        })
        assert res.status_code == 401
        assert b"Invalid" in res.content

    def test_ui_signup_password_mismatch_returns_422(self, client):
        res = client.post("/ui/signup", data={
            "username":         "mismatch",
            "email":            "mismatch@example.com",
            "password":         "securepass123",
            "confirm_password": "different456",
        })
        assert res.status_code == 422
        assert b"match" in res.content

    def test_ui_logout_clears_cookie(self, client):
        res = client.get("/ui/logout", follow_redirects=False)
        assert res.status_code == 303
        # Cookie should be cleared (set with empty value or deleted)
        assert "access_token" not in res.cookies or res.cookies["access_token"] == ""
