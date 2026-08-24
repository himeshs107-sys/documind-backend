"""Shared FastAPI dependencies: DB session + current-user auth."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.exceptions import CredentialsException
from app.core.security import decode_access_token
from app.database import get_db
from app.models.user import User

# tokenUrl is only used to populate OpenAPI docs' "Authorize" button; the
# frontend calls /api/auth/login directly and sends the token as a Bearer header.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    if not token:
        raise CredentialsException("Not authenticated")

    user_id = decode_access_token(token)
    if not user_id:
        raise CredentialsException()

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise CredentialsException("User no longer exists")
    return user
