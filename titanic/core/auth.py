"""
Authentication utilities:
  - Argon2 password hashing / verification
  - Fernet (AES-128-CBC) email encryption + HMAC for lookups
  - JWT creation / decoding
  - FastAPI dependency for protected routes
"""
import hmac as _hmac
import hashlib
import os
from datetime import datetime, timedelta, UTC

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from cryptography.fernet import Fernet
from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ─── Argon2 ──────────────────────────────────────────────────────────────────

_ph = PasswordHasher()


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, plain)
    except (VerifyMismatchError, VerificationError):
        return False


# ─── Email encryption (Fernet / AES-128-CBC) ─────────────────────────────────

def _load_fernet() -> Fernet:
    raw = os.getenv("FERNET_KEY", "")
    if raw:
        key = raw.encode() if isinstance(raw, str) else raw
    else:
        key = Fernet.generate_key()
    return Fernet(key)

_fernet = _load_fernet()


def encrypt_email(email: str) -> str:
    """Encrypt email for storage — reversible with the same FERNET_KEY."""
    return _fernet.encrypt(email.lower().strip().encode()).decode()


def decrypt_email(token: str) -> str:
    """Decrypt a stored email token back to plaintext."""
    return _fernet.decrypt(token.encode()).decode()


# ─── Email HMAC (deterministic lookup without decrypting every row) ───────────

_HMAC_SECRET = os.getenv("HMAC_SECRET", "change-me-in-production").encode()


def email_hmac(email: str) -> str:
    """Deterministic HMAC-SHA256 of the normalised email — used for uniqueness checks."""
    return _hmac.new(_HMAC_SECRET, email.lower().strip().encode(), hashlib.sha256).hexdigest()


# ─── JWT ─────────────────────────────────────────────────────────────────────

JWT_SECRET       = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))


def create_access_token(user_id: int, username: str) -> str:
    payload = {
        "sub":      str(user_id),
        "username": username,
        "exp":      datetime.now(UTC) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat":      datetime.now(UTC),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ─── FastAPI dependency ───────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    access_token: str | None = Cookie(default=None),
) -> dict:
    """Accepts JWT from Authorization: Bearer header OR 'access_token' cookie."""
    raw = (credentials.credentials if credentials else None) or access_token
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(raw)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid token")


def get_optional_user(
    access_token: str | None = Cookie(default=None),
) -> dict | None:
    """Like get_current_user but returns None instead of raising for unauthenticated requests."""
    if not access_token:
        return None
    try:
        return decode_access_token(access_token)
    except Exception:
        return None
