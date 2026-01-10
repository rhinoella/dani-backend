
import asyncio
import sys
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text, inspect

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

async def check_schema():
    print("Checking database schema...")
    engine = create_async_engine(str(settings.DATABASE_URL))
    
    async with engine.connect() as conn:
        # Check column types in information_schema
        result = await conn.execute(text("""
            SELECT column_name, udt_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' 
            AND column_name IN ('file_type', 'status');
        """))
        
        rows = result.fetchall()
        print("\nCurrent Column Types:")
        for row in rows:
            print(f"  {row[0]}: {row[1]}")
            
        print("-" * 30)
        
        # Check if old types exist
        result = await conn.execute(text("""
            SELECT typname FROM pg_type WHERE typname IN ('documenttype', 'documentstatus', 'document_type', 'document_status');
        """))
        
        print("\nExisting Enum Types:")
        for row in result.fetchall():
            print(f"  {row[0]}")

    await engine.dispose()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_schema())
