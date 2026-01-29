#!/usr/bin/env python3
"""
Database seeder for creating test users.

Usage:
    python scripts/seed_users.py
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database.connection import AsyncSessionLocal


async def seed_user():
    """Seed a single test user using raw SQL."""
    
    user_id = str(uuid.uuid4())
    email = "mainamanasseh02@gmail.com"
    name = "Manasseh Maina"
    google_id = "seed_placeholder_" + str(uuid.uuid4())[:8]  # Placeholder until OAuth login
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # Use timezone-aware then strip for PG
    
    async with AsyncSessionLocal() as session:
        # Check if user already exists
        result = await session.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email}
        )
        existing = result.fetchone()
        
        if existing:
            print(f"✅ User already exists: {email}")
            return
        
        # Insert new user with google_id placeholder
        await session.execute(
            text("""
                INSERT INTO users (id, google_id, email, name, created_at, updated_at)
                VALUES (:id, :google_id, :email, :name, :created_at, :updated_at)
            """),
            {
                "id": user_id,
                "google_id": google_id,
                "email": email,
                "name": name,
                "created_at": now,
                "updated_at": now,
            }
        )
        await session.commit()
        
        print(f"✅ Created user: {name} ({email})")


async def main():
    print("\n🌱 DANI Engine - User Seeder")
    print("=" * 40)
    
    await seed_user()
    
    print("\n✨ Seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
