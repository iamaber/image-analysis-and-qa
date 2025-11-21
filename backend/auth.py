import os
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
import bcrypt
from dotenv import load_dotenv

load_dotenv()


class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", "fallback-secret-key")
        self.algorithm = os.getenv("ALGORITHM", "HS256")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )

    def get_password_hash(self, password: str) -> str:
        # Ensure password doesn't exceed bcrypt's 72-byte limit
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Password cannot be longer than 72 bytes")
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str):
        payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        return payload
