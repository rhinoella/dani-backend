
import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

async def create_types():
    """Create missing enum types in the database."""
    print(f"Connecting to database...")
    
    engine = create_async_engine(
        str(settings.DATABASE_URL),
        echo=True,
    )
    
    async with engine.begin() as conn:
        print("Checking for document_type enum...")
        try:
            await conn.execute(text("CREATE TYPE document_type AS ENUM ('PDF', 'DOCX', 'TXT');"))
            print("✅ Created document_type enum.")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️ document_type enum already exists.")
            else:
                print(f"❌ Failed to create document_type: {e}")
        
        print("Checking for document_status enum...")
        try:
            await conn.execute(text("CREATE TYPE document_status AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');"))
            print("✅ Created document_status enum.")
        except Exception as e:
            if "already exists" in str(e):
                print("ℹ️ document_status enum already exists.")
            else:
                print(f"❌ Failed to create document_status: {e}")
                
        # Also check if we need to alter the table to use these types if columns were created as varchar
        # This is a bit risky blindly, but usually the error is "type does not exist" on INSERT
        # which implies the column expects the type but the type definition is missing.

    await engine.dispose()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(create_types())
