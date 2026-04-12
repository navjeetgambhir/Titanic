"""JSON API for authentication: POST /auth/signup, POST /auth/login."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

from core.auth import (
    create_access_token,
    decrypt_email,
    email_hmac,
    encrypt_email,
    hash_password,
    verify_password,
)
from core.database import get_db
from core.db_models import User
from core.logger import get_logger

log = get_logger("auth_service")

router = APIRouter(prefix="/auth", tags=["Auth"])


# ─── Schemas ─────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str  = Field(..., min_length=3, max_length=50)
    email:    EmailStr
    password: str  = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username may only contain letters, numbers, _ and -")
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      int
    username:     str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_user_by_email(db: Session, email: str) -> User | None:
    h = email_hmac(email)
    return db.query(User).filter(User.email_hash == h).first()


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a new account")
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if _get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=body.username,
        email_encrypted=encrypt_email(body.email),
        email_hash=email_hmac(body.email),
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log.info("New user registered: id=%d username=%s", user.id, user.username)
    token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse, summary="Log in and get a JWT")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = _get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    log.info("User logged in: id=%d username=%s", user.id, user.username)
    token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)
