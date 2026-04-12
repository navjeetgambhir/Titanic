from datetime import datetime, UTC
from sqlalchemy import Integer, Float, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base


class User(Base):
    __tablename__ = "user"

    id:              Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    username:        Mapped[str]      = mapped_column(String(50),  nullable=False, unique=True)
    # email stored as Fernet ciphertext; email_hash (HMAC) used for uniqueness lookups
    email_encrypted: Mapped[str]      = mapped_column(String(600), nullable=False)
    email_hash:      Mapped[str]      = mapped_column(String(64),  nullable=False, unique=True)
    password_hash:   Mapped[str]      = mapped_column(String(300), nullable=False)
    created_at:      Mapped[datetime] = mapped_column(DateTime,    default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username}>"


class SurvivalPrediction(Base):
    __tablename__ = "survival_prediction"

    id:                   Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    sex:                  Mapped[str]      = mapped_column(String(6),  nullable=False)
    age:                  Mapped[float]    = mapped_column(Float,      nullable=False)
    fare:                 Mapped[float]    = mapped_column(Float,      nullable=False)
    survived:             Mapped[int]      = mapped_column(Integer,    nullable=False)
    survival_probability: Mapped[float]    = mapped_column(Float,      nullable=False)
    model_used:           Mapped[str]      = mapped_column(String(100), nullable=False)
    source:               Mapped[str]      = mapped_column(String(10),  nullable=False, default="api")
    created_at:           Mapped[datetime] = mapped_column(DateTime,   default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return (
            f"<SurvivalPrediction id={self.id} sex={self.sex} age={self.age} "
            f"fare={self.fare} survived={self.survived} prob={self.survival_probability}>"
        )
