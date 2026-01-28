#!/usr/bin/env python3
"""
Check infographics stored in the database and compare with S3.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.config import Config


async def check_infographics_db():
    """Check infographic records in the database."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text
    
    # Build database URL from separate env vars
    db_connection = os.getenv('DB_CONNECTION', 'postgresql+asyncpg')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_database = os.getenv('DB_DATABASE', 'dani')
    db_username = os.getenv('DB_USERNAME', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    
    db_url = f"{db_connection}://{db_username}:{db_password}@{db_host}:{db_port}/{db_database}"
    
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Get total count
        count_result = await session.execute(text('''
            SELECT COUNT(*) FROM infographics WHERE deleted_at IS NULL
        '''))
        total_count = count_result.scalar()
        
        # Get count with s3_key
        s3_count_result = await session.execute(text('''
            SELECT COUNT(*) FROM infographics WHERE deleted_at IS NULL AND s3_key IS NOT NULL
        '''))
        s3_count = s3_count_result.scalar()
        
        # Get infographic records
        result = await session.execute(text('''
            SELECT id, created_at, s3_key, s3_bucket, image_url, headline, status
            FROM infographics 
            WHERE deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT 20
        '''))
        
        rows = result.fetchall()
        
    await engine.dispose()
    
    return total_count, s3_count, rows


def check_s3_infographics():
    """Check infographic objects in S3."""
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', '')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'dani-documents')
    endpoint_url = os.getenv('S3_ENDPOINT_URL', None)
    
    config = Config(
        signature_version='s3v4',
        retries={'max_attempts': 3, 'mode': 'standard'},
    )
    
    client_kwargs = {
        'service_name': 's3',
        'region_name': aws_region,
        'config': config,
    }
    
    if aws_access_key and aws_secret_key:
        client_kwargs['aws_access_key_id'] = aws_access_key
        client_kwargs['aws_secret_access_key'] = aws_secret_key
    
    if endpoint_url:
        client_kwargs['endpoint_url'] = endpoint_url
    
    s3_client = boto3.client(**client_kwargs)
    
    # List all objects and filter for infographics
    infographic_objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix='documents/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if 'infographic' in key.lower() or key.endswith('.png'):
                infographic_objects.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                })
    
    return bucket_name, infographic_objects


async def main():
    print("=" * 80)
    print("INFOGRAPHIC STORAGE AUDIT")
    print("=" * 80)
    
    # Check database
    print("\n[DATABASE CHECK]")
    print("-" * 40)
    total_count, s3_count, rows = await check_infographics_db()
    
    print(f"Total infographics in DB: {total_count}")
    print(f"Infographics with S3 key: {s3_count}")
    print(f"Infographics without S3 key: {total_count - s3_count}")
    
    if rows:
        print(f"\nRecent infographics (showing {len(rows)}):")
        for row in rows:
            id_, created_at, s3_key, s3_bucket, image_url, headline, status = row
            headline_preview = (headline[:40] + '...') if headline and len(headline) > 40 else headline
            print(f"\n  ID: {id_}")
            print(f"    Created: {created_at}")
            print(f"    Status: {status}")
            print(f"    Headline: {headline_preview}")
            print(f"    S3 Key: {s3_key or '❌ NOT SET'}")
            print(f"    S3 Bucket: {s3_bucket or '❌ NOT SET'}")
            if image_url:
                print(f"    Image URL: {image_url[:60]}...")
            else:
                print(f"    Image URL: ❌ NOT SET")
    
    # Check S3
    print("\n" + "=" * 80)
    print("[S3 CHECK]")
    print("-" * 40)
    
    bucket_name, s3_objects = check_s3_infographics()
    
    print(f"Bucket: {bucket_name}")
    print(f"Infographic-related objects found: {len(s3_objects)}")
    
    if s3_objects:
        print("\nObjects in S3:")
        for obj in s3_objects[:20]:
            print(f"  - {obj['key']} ({obj['size']} bytes, {obj['last_modified']})")
        if len(s3_objects) > 20:
            print(f"  ... and {len(s3_objects) - 20} more")
    
    # Analysis
    print("\n" + "=" * 80)
    print("[ANALYSIS]")
    print("-" * 40)
    
    if total_count > 0 and s3_count == 0:
        print("⚠️  ISSUE: Infographics exist in DB but none have S3 keys!")
        print("   This suggests images are not being uploaded to S3.")
    elif s3_count < total_count:
        print(f"⚠️  ISSUE: {total_count - s3_count} infographics are missing S3 keys.")
    elif s3_count == total_count and total_count > 0:
        print("✅ All infographics have S3 keys assigned.")
    
    # Check if DB keys match S3 objects
    db_keys = set()
    for row in rows:
        if row[2]:  # s3_key
            db_keys.add(row[2])
    
    s3_keys = set(obj['key'] for obj in s3_objects)
    
    missing_in_s3 = db_keys - s3_keys
    if missing_in_s3:
        print(f"\n⚠️  DB references S3 keys not found in S3: {len(missing_in_s3)}")
        for key in list(missing_in_s3)[:5]:
            print(f"   - {key}")
    
    orphaned_in_s3 = s3_keys - db_keys
    if orphaned_in_s3 and len(rows) == total_count:  # Only if we checked all records
        print(f"\n⚠️  S3 objects not referenced in DB: {len(orphaned_in_s3)}")
        for key in list(orphaned_in_s3)[:5]:
            print(f"   - {key}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
