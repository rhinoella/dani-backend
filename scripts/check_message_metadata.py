"""
Check message metadata in the database to verify tool_result data is stored correctly.
"""
import asyncio
import os
import json
from dotenv import load_dotenv

load_dotenv()

async def check_message_metadata():
    """Check messages that have tool_result in their metadata."""
    
    # Build DATABASE_URL from individual env vars
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_DATABASE', 'dani')
    db_user = os.getenv('DB_USERNAME', 'dani')
    db_pass = os.getenv('DB_PASSWORD', '')
    
    database_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    try:
        import asyncpg
    except ImportError:
        print("Installing asyncpg...")
        os.system("pip install asyncpg")
        import asyncpg
    
    print(f"Connecting to database at {db_host}:{db_port}/{db_name}...")
    
    conn = await asyncpg.connect(database_url)
    
    try:
        # Query messages that have tool_result in metadata
        print("\n=== Messages with tool_result in metadata ===\n")
        
        rows = await conn.fetch("""
            SELECT 
                m.id,
                m.conversation_id,
                m.role,
                SUBSTRING(m.content, 1, 100) as content_preview,
                m.metadata,
                m.created_at
            FROM messages m
            WHERE m.metadata::text LIKE '%tool_result%'
            ORDER BY m.created_at DESC
            LIMIT 20
        """)
        
        if not rows:
            print("No messages found with tool_result in metadata.")
            
            # Check if there are any messages with metadata at all
            metadata_rows = await conn.fetch("""
                SELECT COUNT(*) as count FROM messages WHERE metadata IS NOT NULL AND metadata::text != 'null'
            """)
            print(f"\nTotal messages with non-null metadata: {metadata_rows[0]['count']}")
            
            # Sample some metadata to see structure
            sample_rows = await conn.fetch("""
                SELECT 
                    m.id,
                    m.role,
                    m.metadata,
                    m.created_at
                FROM messages m
                WHERE m.metadata IS NOT NULL AND m.metadata::text != 'null'
                ORDER BY m.created_at DESC
                LIMIT 5
            """)
            
            if sample_rows:
                print("\n=== Sample metadata from recent messages ===\n")
                for row in sample_rows:
                    print(f"Message ID: {row['id']}")
                    print(f"Role: {row['role']}")
                    print(f"Created: {row['created_at']}")
                    metadata = row['metadata']
                    if metadata:
                        print(f"Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'not a dict'}")
                        print(f"Metadata preview: {json.dumps(metadata, indent=2, default=str)[:500]}")
                    print("-" * 50)
            return
        
        print(f"Found {len(rows)} messages with tool_result\n")
        
        for row in rows:
            print(f"Message ID: {row['id']}")
            print(f"Conversation ID: {row['conversation_id']}")
            print(f"Role: {row['role']}")
            print(f"Content preview: {row['content_preview']}...")
            print(f"Created: {row['created_at']}")
            
            metadata = row['metadata']
            if metadata:
                # Handle both string and dict metadata
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        print(f"  Failed to parse metadata as JSON")
                        print("-" * 60)
                        continue
                
                tool_result = metadata.get('tool_result', {})
                tool_name = metadata.get('tool_name', 'N/A')
                
                print(f"Tool name: {tool_name}")
                print(f"Tool result keys: {list(tool_result.keys()) if tool_result else 'None'}")
                
                if tool_result:
                    has_image = 'image' in tool_result and tool_result['image']
                    has_image_url = 'image_url' in tool_result and tool_result['image_url']
                    has_s3_key = 's3_key' in tool_result and tool_result['s3_key']
                    
                    print(f"  - Has base64 image: {has_image}")
                    print(f"  - Has image_url: {has_image_url}")
                    print(f"  - Has s3_key: {has_s3_key}")
                    
                    if has_s3_key:
                        print(f"  - s3_key: {tool_result['s3_key']}")
                    if has_image_url:
                        print(f"  - image_url (first 100 chars): {tool_result['image_url'][:100]}...")
                    
                    # Check for structured_data
                    if 'structured_data' in tool_result:
                        sd = tool_result['structured_data']
                        print(f"  - Headline: {sd.get('headline', 'N/A')}")
            
            print("-" * 60)
        
        # Also check the infographics table directly
        print("\n=== Recent infographics from infographics table ===\n")
        
        infographic_rows = await conn.fetch("""
            SELECT 
                id,
                s3_key,
                status,
                created_at,
                SUBSTRING(headline, 1, 50) as headline_preview
            FROM infographics
            WHERE s3_key IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        for row in infographic_rows:
            print(f"Infographic ID: {row['id']}")
            print(f"S3 Key: {row['s3_key']}")
            print(f"Status: {row['status']}")
            print(f"Headline: {row['headline_preview']}")
            print(f"Created: {row['created_at']}")
            print("-" * 40)
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_message_metadata())
