import boto3
import os
import uuid
from datetime import datetime, timezone

class S3Storage:
    """A wrapper for Boto3 to handle S3 operations."""
    def __init__(self, region_name=None, bucket_name=None):
        self.region_name = region_name or os.getenv("AWS_REGION_ADS")
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME_ADS")
        
        if not self.region_name or not self.bucket_name:
            raise ValueError("AWS region and S3  name must be configured.")
            
        self.s3_client = boto3.client(
            "s3",
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_ADS"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_ADS"),
        )

    def upload_file(self, file_bytes: bytes, content_type: str = "image/png") -> str:
        """
        Uploads a file-like object to S3 and returns the object key.
        
        Args:
            file_bytes: The file content in bytes.
            content_type: The MIME type of the file.
            
        Returns:
            The key of the uploaded object in S3.
        """
        now = datetime.now(timezone.utc)
        key = f"screenshots/{now.strftime('%Y/%m/%d')}/{uuid.uuid4()}.png"
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=file_bytes,
            ContentType=content_type,
        )
        
        return key 