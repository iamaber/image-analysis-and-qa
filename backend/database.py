from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
import sqlite3


# Database configuration
# Use /app/data directory in container, current directory otherwise
db_dir = os.getenv("DB_DIR", ".")
DATABASE_URL = f"sqlite:///{db_dir}/users.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Only needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def recreate_tables():
    """Drop and recreate all tables to fix schema issues."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def check_and_update_schema():
    """Check if the database schema matches the model and update if needed."""
    # Check if the database file exists
    db_dir = os.getenv("DB_DIR", ".")
    db_path = f"{db_dir}/users.db"
    if not os.path.exists(db_path):
        print("Creating new database...")
        create_tables()
        return

    try:
        # Try to query the users table to see if it has the correct schema
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Get the schema of the users table
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        connection.close()

        # Check if all required columns exist
        required_columns = {
            "id",
            "email",
            "name",
            "hashed_password",
            "is_active",
            "created_at",
            "updated_at",
        }
        existing_columns = {col[1] for col in columns}  # col[1] is the column name

        if not required_columns.issubset(existing_columns):
            print("Database schema mismatch detected. Recreating tables...")
            recreate_tables()
        else:
            print("Database schema is up to date.")

    except Exception as e:
        print(f"Error checking database schema: {e}")
        print("Recreating tables...")
        recreate_tables()


def create_tables():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper functions
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, user):
    db_user = User(
        email=user.email, name=user.name, hashed_password=user.hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    # We'll need to implement password verification here
    # This will be handled by the auth.py file
    return user
