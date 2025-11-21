from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Optional

from .models import UserCreate, UserLogin, UserResponse, Token
from .database import get_db, get_user_by_email, User
from .auth import AuthManager

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Initialize auth manager
auth_manager = AuthManager()

# Security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return auth_manager.verify_password(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Get hash of a password."""
    return auth_manager.get_password_hash(password)


def authenticate_user(db: Session, email: str, password: str):
    """Authenticate user with email and password."""
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not verify_password(password, str(user.hashed_password)):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create access token."""
    return auth_manager.create_access_token(data, expires_delta)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = auth_manager.verify_token(credentials.credentials)
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    user = get_user_by_email(db, email)
    if user is None:
        raise credentials_exception
    return user


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if user already exists
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Hash password and create user
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, name=user.name, hashed_password=hashed_password)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return access token."""
    user = authenticate_user(db, user_credentials.email, user_credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes in seconds
    }


@router.post("/verify", response_model=UserResponse)
async def verify_token(current_user: User = Depends(get_current_user)):
    """Verify JWT token and return user info."""
    return current_user


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """Refresh access token."""
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": current_user.email}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes in seconds
    }


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return current_user


@router.post("/logout")
async def logout():
    """Logout user (client-side token removal)."""
    return {"message": "Successfully logged out"}


@router.get("/health")
async def auth_health_check():
    """Health check endpoint for authentication service."""
    return {"status": "healthy", "service": "authentication"}
