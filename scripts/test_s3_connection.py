#!/usr/bin/env python3
"""
S3 Bucket Connectivity Test Script.

Tests whether the configured S3 bucket is reachable and accessible.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config


def test_s3_connection():
    """Test S3 bucket connectivity."""
    # Get settings from environment
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', '')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'dani-documents')
    endpoint_url = os.getenv('S3_ENDPOINT_URL', None)

    print('S3 Connectivity Test')
    print('=' * 50)
    print(f'Bucket Name: {bucket_name}')
    print(f'Region: {aws_region}')
    print(f'Endpoint URL: {endpoint_url or "Default AWS S3"}')
    
    if aws_access_key and len(aws_access_key) > 8:
        print(f'Access Key ID: {aws_access_key[:4]}...{aws_access_key[-4:]}')
    else:
        print('Access Key ID: (not set or too short)')
    print('=' * 50)

    # Configure S3 client
    config = Config(
        signature_version='s3v4',
        retries={'max_attempts': 3, 'mode': 'standard'},
        connect_timeout=5,
        read_timeout=10
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

    try:
        s3_client = boto3.client(**client_kwargs)

        # Test 1: Check if bucket exists and is accessible
        print('\n[Test 1] Checking bucket access...')
        s3_client.head_bucket(Bucket=bucket_name)
        print('✅ Bucket is reachable and accessible!')

        # Test 2: List objects (limited to 1)
        print('\n[Test 2] Listing objects (max 5)...')
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
        obj_count = response.get('KeyCount', 0)
        print(f'✅ Successfully listed objects. Found {obj_count} object(s) in sample.')
        
        if obj_count > 0:
            print('   Sample objects:')
            for obj in response.get('Contents', [])[:5]:
                print(f"   - {obj['Key']} ({obj['Size']} bytes)")

        # Test 3: Get bucket location
        print('\n[Test 3] Getting bucket location...')
        location = s3_client.get_bucket_location(Bucket=bucket_name)
        region = location.get('LocationConstraint') or 'us-east-1'
        print(f'✅ Bucket location: {region}')

        print('\n' + '=' * 50)
        print('🎉 All S3 connectivity tests passed!')
        print('=' * 50)
        return True

    except NoCredentialsError:
        print('\n❌ ERROR: AWS credentials not found or invalid.')
        print('   Please check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file')
        return False

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        print(f'\n❌ ERROR: {error_code} - {error_msg}')
        if error_code == '403':
            print('   Access denied. Check your IAM permissions.')
        elif error_code == '404':
            print(f'   Bucket "{bucket_name}" does not exist.')
        return False

    except EndpointConnectionError as e:
        print(f'\n❌ ERROR: Cannot connect to S3 endpoint.')
        print(f'   {str(e)}')
        return False

    except Exception as e:
        print(f'\n❌ ERROR: {type(e).__name__}: {str(e)}')
        return False


if __name__ == '__main__':
    success = test_s3_connection()
    sys.exit(0 if success else 1)
