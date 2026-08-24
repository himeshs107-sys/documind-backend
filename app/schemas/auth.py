"""
Field names here are camelCase on purpose, not snake_case — they mirror the
JSON the DocuMind frontend's services/authApi.js already sends and expects,
so no translation layer is needed on either side.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserOut(BaseModel):
    id: str
    name: str
    email: str

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, user) -> "UserOut":
        return cls(id=user.id, name=user.full_name or user.email.split("@")[0], email=user.email)


class RegisterRequest(BaseModel):
    fullName: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user: UserOut
    token: str
