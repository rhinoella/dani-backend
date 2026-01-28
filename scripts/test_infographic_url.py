#!/usr/bin/env python3
"""Test infographic URL accessibility."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.config import Config
from dotenv import load_dotenv
import urllib.request

load_dotenv()

# S3 config
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', '')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
aws_region = os.getenv('AWS_REGION', 'us-east-1')
bucket_name = os.getenv('S3_BUCKET_NAME', 'dani-documents')

config = Config(
    signature_version='s3v4',
    s3={'addressing_style': 'virtual'}
)
s3_client = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    config=config,
    endpoint_url=f'https://s3.{aws_region}.amazonaws.com'
)

# Test with the most recent infographic
test_key = 'documents/3c041dff-af8f-499f-bd3f-6753ff7454bc/2026-01-25/6fefcf37_infographic_1769364690.png'

print('Testing infographic URL accessibility')
print('=' * 60)
print(f'S3 Key: {test_key}')
print(f'Bucket: {bucket_name}')
print()

# Check if object exists
try:
    head = s3_client.head_object(Bucket=bucket_name, Key=test_key)
    print('✅ Object exists!')
    print(f'   Size: {head["ContentLength"]} bytes')
    print(f'   Content-Type: {head["ContentType"]}')
    print(f'   Last Modified: {head["LastModified"]}')
except Exception as e:
    print(f'❌ Object not found: {e}')
    sys.exit(1)

# Generate presigned URL
print()
print('Generating presigned URL (valid for 1 hour)...')
try:
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': test_key},
        ExpiresIn=3600,
    )
    print('✅ Presigned URL generated:')
    print()
    print(url)
    print()
    
    # Test the URL with GET (read first 1KB)
    print('Testing URL accessibility...')
    req = urllib.request.Request(url)  # GET request, not HEAD
    with urllib.request.urlopen(req, timeout=10) as response:
        data = response.read(1024)  # Read just first 1KB to verify
        print(f'✅ URL is accessible! Status: {response.status}')
        print(f'   Content-Length: {response.headers.get("Content-Length")} bytes')
        print(f'   Content-Type: {response.headers.get("Content-Type")}')
        print(f'   Sample downloaded: {len(data)} bytes')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)

print()
print('=' * 60)
print('🎉 Infographic is stored correctly and accessible!')
