"""User registration/authentication logic."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.exceptions import CredentialsException, EmailAlreadyRegisteredException
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User


def register_user(db: Session, *, full_name: str, email: str, password: str) -> User:
    existing = db.query(User).filter(User.email == email.lower()).first()
    if existing:
        raise EmailAlreadyRegisteredException()

    user = User(email=email.lower(), full_name=full_name, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, *, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user or not verify_password(password, user.hashed_password):
        raise CredentialsException("Incorrect email or password")
    return user


def issue_token_for(user: User) -> str:
    return create_access_token(subject=user.id)
