from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserOut
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    user = auth_service.register_user(
        db, full_name=payload.fullName, email=payload.email, password=payload.password
    )
    token = auth_service.issue_token_for(user)
    return AuthResponse(user=UserOut.from_model(user), token=token)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, email=payload.email, password=payload.password)
    token = auth_service.issue_token_for(user)
    return AuthResponse(user=UserOut.from_model(user), token=token)


@router.post("/logout")
def logout():
    # JWTs are stateless, so there's nothing to invalidate server-side here.
    # If you add refresh tokens or a blocklist, revoke them in this handler.
    return {"ok": True}


@router.get("/me", response_model=UserOut)
def me(current_user=Depends(get_current_user)):
    return UserOut.from_model(current_user)
