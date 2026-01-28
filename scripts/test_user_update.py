import asyncio
import sys
import os
import logging

# Configure logging to suppress SQLAlchemy output
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('app').setLevel(logging.WARNING)

# Add the parent directory to the path
sys.path.append(os.getcwd())

from app.database.connection import AsyncSessionLocal
from app.services.user_service import UserService
from app.database.models import User
from app.repositories.user_repository import UserRepository, utc_now

async def test_update_user():
    print("Starting test...")
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        user_service = UserService(session)

        # Create a test user
        email = f"test_{int(utc_now().timestamp())}@example.com"
        name = "Original Name"
        
        print(f"Creating test user: {email}")
        user = await user_repo.create(
            email=email,
            name=name,
            google_id=f"test_google_id_{int(utc_now().timestamp())}",
            last_login_at=utc_now()
        )
        user_id = str(user.id)
        print(f"User created with ID: {user_id}, Name: {user.name}")

        try:
            # Update the user
            new_name = "Updated Name"
            print(f"Updating user name to: {new_name}")
            
            # Simulate what the API handler does
            updated_user = await user_service.update_user(user_id, name=new_name)

            if updated_user.name == new_name:
                print("SUCCESS: User name updated correctly in returned object.")
            else:
                print(f"FAILURE: User name is {updated_user.name}, expected {new_name}")
                
            # Verify in DB
            refreshed_user = await user_repo.get_by_id(user_id)
            if refreshed_user.name == new_name:
                 print("SUCCESS: User name updated correctly in Database.")
            else:
                 print(f"FAILURE: Database has {refreshed_user.name}, expected {new_name}")

        finally:
            # Clean up
            await user_repo.hard_delete(user_id)
            print("Test user deleted.")

if __name__ == "__main__":
    asyncio.run(test_update_user())
