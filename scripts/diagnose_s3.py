#!/usr/bin/env python3
"""Check S3 bucket settings and test presigned URL."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from dotenv import load_dotenv
import urllib.request

load_dotenv()

s3 = boto3.client('s3',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

bucket = os.getenv('S3_BUCKET_NAME')
test_key = 'documents/3c041dff-af8f-499f-bd3f-6753ff7454bc/2026-01-25/6fefcf37_infographic_1769364690.png'

print('S3 Bucket Diagnostics')
print('=' * 60)
print(f'Bucket: {bucket}')
print(f'Region: {os.getenv("AWS_REGION")}')
print()

# Check public access block
print('[1] Public Access Block Settings:')
try:
    pab = s3.get_public_access_block(Bucket=bucket)
    config = pab['PublicAccessBlockConfiguration']
    print(f'    BlockPublicAcls: {config.get("BlockPublicAcls", False)}')
    print(f'    IgnorePublicAcls: {config.get("IgnorePublicAcls", False)}')
    print(f'    BlockPublicPolicy: {config.get("BlockPublicPolicy", False)}')
    print(f'    RestrictPublicBuckets: {config.get("RestrictPublicBuckets", False)}')
except Exception as e:
    print(f'    Error: {e}')

print()

# Check bucket policy
print('[2] Bucket Policy:')
try:
    policy = s3.get_bucket_policy(Bucket=bucket)
    print(f'    Policy exists: Yes')
    import json
    p = json.loads(policy['Policy'])
    print(f'    Statements: {len(p.get("Statement", []))}')
except Exception as e:
    if 'NoSuchBucketPolicy' in str(e):
        print('    No bucket policy configured')
    else:
        print(f'    Error: {e}')

print()

# Try SDK download
print('[3] SDK Direct Download Test:')
try:
    obj = s3.get_object(Bucket=bucket, Key=test_key)
    data = obj['Body'].read(1024)
    print(f'    ✅ SDK download works! Got {len(data)} bytes (sample)')
except Exception as e:
    print(f'    ❌ SDK download failed: {e}')

print()

# Test presigned URL with curl equivalent
print('[4] Presigned URL Test:')
from botocore.config import Config
config = Config(signature_version='s3v4', s3={'addressing_style': 'virtual'})
s3_presigned = boto3.client('s3',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    config=config,
    endpoint_url=f'https://s3.{os.getenv("AWS_REGION")}.amazonaws.com')

url = s3_presigned.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket, 'Key': test_key},
    ExpiresIn=3600,
)
print(f'    URL: {url[:80]}...')

# Test with requests-style (following redirects)
print('    Testing accessibility...')
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as response:
        data = response.read(1024)
        print(f'    ✅ URL accessible! Got {len(data)} bytes')
except urllib.error.HTTPError as e:
    print(f'    ❌ HTTP Error {e.code}: {e.reason}')
    # Check if it's signature issue
    if e.code == 403:
        print()
        print('    Possible causes of 403:')
        print('    - Bucket has "Block Public Access" enabled')
        print('    - IAM user lacks s3:GetObject permission')
        print('    - Signature version mismatch')
        print('    - Clock skew between client and server')
except Exception as e:
    print(f'    ❌ Error: {e}')

print()
print('=' * 60)
