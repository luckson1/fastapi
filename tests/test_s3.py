import pytest
import boto3
from moto import mock_aws
import os
from app.storage.s3 import S3Storage

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID_ADS"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY_ADS"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_REGION_ADS"] = "us-east-1"
    os.environ["SCREENSHOT_BUCKET"] = "test-screenshot-bucket"

@pytest.fixture
def s3_client(aws_credentials):
    """Mocked S3 client."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=os.environ["SCREENSHOT_BUCKET"])
        yield client

def test_s3_upload(s3_client):
    """
    Tests that the S3Storage class correctly uploads a file to a mock S3 bucket.
    """
    storage = S3Storage()
    dummy_bytes = b"This is a test file."
    
    # Upload the file
    object_key = storage.upload_file(file_bytes=dummy_bytes)
    
    # Verify the returned key format
    assert object_key.startswith("screenshots/")
    assert object_key.endswith(".png")
    
    # Verify the object exists in the mock bucket and has the correct content
    response = s3_client.get_object(Bucket=os.environ["SCREENSHOT_BUCKET"], Key=object_key)
    assert response["Body"].read() == dummy_bytes
    assert response["ContentType"] == "image/png" 