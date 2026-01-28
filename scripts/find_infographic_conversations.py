#!/usr/bin/env python3
"""Find conversations that have infographic messages for testing."""

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import asyncpg


async def main():
    conn = await asyncpg.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5433)),
        user=os.getenv('DB_USERNAME'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_DATABASE')
    )
    
    # Get conversations that have infographic messages
    rows = await conn.fetch('''
        SELECT m.conversation_id, c.title, m.content, m.metadata, m.created_at
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE m.metadata::text LIKE '%tool_result%'
          AND m.metadata::text LIKE '%infographic%'
        ORDER BY m.created_at DESC
        LIMIT 5
    ''')
    
    print('Conversations with infographic messages:')
    print('=' * 60)
    for row in rows:
        print(f"  Conversation ID: {row['conversation_id']}")
        print(f"  Title: {row['title']}")
        print(f"  Content preview: {row['content'][:80]}...")
        print()
    
    print()
    print("To test in browser, go to:")
    if rows:
        print(f"  http://localhost:3000/chat?conversation={rows[0]['conversation_id']}")
    
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
