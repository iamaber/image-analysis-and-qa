#!/usr/bin/env python3
"""
Script to fix database schema issues
"""

import os
import sys


def fix_database():
    """Fix the database by removing the existing file and creating a new one."""

    db_path = "./users.db"

    # Check if database file exists
    if os.path.exists(db_path):
        print(f"🔄 Removing existing database: {db_path}")
        os.remove(db_path)
        print("✅ Database file removed")

    # Import and create new database
    try:
        from backend.database import create_tables

        print("🏗️ Creating new database with correct schema...")
        create_tables()
        print("✅ Database created successfully with all required columns:")
        print("   - id")
        print("   - email")
        print("   - name")
        print("   - hashed_password")
        print("   - is_active")
        print("   - created_at")
        print("   - updated_at")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🔧 Database Schema Fix Tool")
    print("=" * 40)

    if fix_database():
        print("\n🎉 Database schema fixed successfully!")
        print("You can now run your authentication system.")
    else:
        print("\n💥 Failed to fix database schema")
        sys.exit(1)
