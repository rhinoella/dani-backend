#!/usr/bin/env python
"""Test S3 connection and operations with multiple files."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

# Project root for file paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

s3 = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

bucket = settings.S3_BUCKET_NAME

print(f'Testing connection to bucket: {bucket}')
print(f'Region: {settings.AWS_REGION}')
print(f'Access Key: {settings.AWS_ACCESS_KEY_ID[:10]}...')

# Files to upload
test_files = [
    ('README.md', 'text/markdown'),
    ('requirements.txt', 'text/plain'),
    ('pyproject.toml', 'text/plain'),
    ('Makefile', 'text/plain'),
    ('alembic.ini', 'text/plain'),
]

uploaded_keys = []

try:
    s3.head_bucket(Bucket=bucket)
    print('✅ Bucket exists and is accessible!\n')
    
    # Upload multiple files
    print('=' * 50)
    print('📤 UPLOADING MULTIPLE FILES')
    print('=' * 50)
    
    for filename, content_type in test_files:
        file_path = os.path.join(project_root, filename)
        
        if not os.path.exists(file_path):
            print(f'⚠️  Skipping {filename} (not found)')
            continue
        
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_size_kb = len(file_content) / 1024
        test_key = f'documents/test-batch/{filename}'
        
        s3.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=file_content,
            ContentType=content_type
        )
        
        uploaded_keys.append(test_key)
        print(f'✅ Uploaded: {filename} ({file_size_kb:.1f} KB) → {test_key}')
    
    print(f'\n📊 Total files uploaded: {len(uploaded_keys)}')
    
    # List uploaded files
    print('\n' + '=' * 50)
    print('📋 LISTING FILES IN S3')
    print('=' * 50)
    
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='documents/test-batch/'
    )
    
    total_size = 0
    for obj in response.get('Contents', []):
        size_kb = obj['Size'] / 1024
        total_size += obj['Size']
        print(f"   {obj['Key']} ({size_kb:.1f} KB)")
    
    print(f'\n📊 Total size: {total_size / 1024:.1f} KB')
    
    # Download and verify one file
    print('\n' + '=' * 50)
    print('📥 DOWNLOAD VERIFICATION')
    print('=' * 50)
    
    if uploaded_keys:
        verify_key = uploaded_keys[0]
        response = s3.get_object(Bucket=bucket, Key=verify_key)
        downloaded = response['Body'].read()
        print(f'✅ Downloaded {verify_key}: {len(downloaded)} bytes')
    
    # Generate presigned URLs for all files
    print('\n' + '=' * 50)
    print('🔗 PRESIGNED URLs (valid for 1 hour)')
    print('=' * 50)
    
    for key in uploaded_keys[:3]:  # Show first 3 URLs
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600
        )
        filename = key.split('/')[-1]
        print(f'   {filename}: {url[:60]}...')
    
    # Delete all uploaded files
    print('\n' + '=' * 50)
    print('🗑️  DELETING ALL TEST FILES')
    print('=' * 50)
    
    for key in uploaded_keys:
        s3.delete_object(Bucket=bucket, Key=key)
        print(f'✅ Deleted: {key}')
    
    # Verify deletion
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='documents/test-batch/'
    )
    remaining = response.get('Contents', [])
    
    if not remaining:
        print(f'\n✅ All {len(uploaded_keys)} files successfully deleted!')
    else:
        print(f'\n⚠️  {len(remaining)} files still remain')
    
    print('\n' + '=' * 50)
    print('🎉 ALL S3 MULTI-FILE TESTS PASSED!')
    print('=' * 50)
    
except ClientError as e:
    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
    error_msg = e.response.get('Error', {}).get('Message', str(e))
    print(f'❌ Error [{error_code}]: {error_msg}')
except Exception as e:
    print(f'❌ Error: {e}')
