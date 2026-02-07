#!/usr/bin/env python3
"""
Get a working authentication token for testing.

Usage:
    python get_auth_token.py
"""
import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.tokens import create_access_token
import asyncpg


async def main():
    """Get or create a test user and generate a token."""

    # Connect directly to database
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password='postgres',
        database='dani_db'
    )

    try:
        # Get any user from database
        user = await conn.fetchrow('SELECT id, email, name FROM users LIMIT 1')

        if not user:
            print("❌ No users found in database!")
            print("\nCreate a user first by running:")
            print("  INSERT INTO users (id, email, google_id, name) VALUES")
            print("  ('00000000-0000-0000-0000-000000000001', 'test@example.com', 'dev-user-123', 'Test User');")
            return

        user_id = user['id']
        user_email = user['email']
        user_name = user['name'] or 'Unknown'

        # Generate access token
        token = create_access_token(user_id, user_email)

        print("✅ Authentication Token Generated!")
        print("\n" + "="*80)
        print("USER INFO:")
        print(f"  ID: {user_id}")
        print(f"  Email: {user_email}")
        print(f"  Name: {user_name}")
        print("\n" + "="*80)
        print("ACCESS TOKEN:")
        print(f"\n{token}\n")
        print("="*80)
        print("\n📋 USAGE:")
        print("\n1. cURL:")
        print(f'   curl -H "Authorization: Bearer {token}" \\')
        print('        http://localhost:8000/api/v1/auth/me')

        print("\n2. Swagger UI:")
        print("   - Open: http://localhost:8000/docs")
        print("   - Click 'Authorize' button")
        print(f"   - Paste: {token}")
        print("   - Click 'Authorize'")

        print("\n3. Save to file:")
        with open(".auth_token", "w") as f:
            f.write(token)
        print("   ✅ Saved to .auth_token file")

        print("\n4. Test it:")
        print(f'   curl -H "Authorization: Bearer {token}" \\')
        print('        http://localhost:8000/api/v1/documents')

        print("\n" + "="*80)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
