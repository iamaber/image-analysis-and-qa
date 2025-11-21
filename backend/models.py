from pydantic import BaseModel, EmailStr, validator
from datetime import datetime
from typing import Optional


class UserBase(BaseModel):
    email: EmailStr
    name: str


class UserCreate(UserBase):
    password: str

    @validator("password")
    def validate_password(cls, v):
        if len(v) > 72:
            raise ValueError("Password cannot be longer than 72 characters")
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters long")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 1:
            raise ValueError("Password is required")
        return v


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    email: Optional[str] = None


class PasswordReset(BaseModel):
    email: EmailStr


class PasswordUpdate(BaseModel):
    token: str
    new_password: str

    @validator("new_password")
    def validate_password(cls, v):
        if len(v) > 72:
            raise ValueError("Password cannot be longer than 72 characters")
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters long")
        return v
