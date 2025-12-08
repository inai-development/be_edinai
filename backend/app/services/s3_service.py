"""S3 service for uploading and managing files in AWS S3."""
import logging
import os
from typing import Optional, BinaryIO
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class S3Service:
    """Service for handling S3 uploads and file management."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        region: str,
        bucket_name: str,
    ):
        """Initialize S3 service with AWS credentials."""
        self.bucket_name = bucket_name
        self.region = region
        
        # Validate credentials
        if not access_key or not secret_key:
            logger.error("AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
            raise ValueError("AWS credentials are required but not configured")
        
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
            logger.info(f"S3 client initialized for bucket: {bucket_name} in region: {region}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        folder: str = "",
        content_type: str = "application/octet-stream",
        public: bool = False,
    ) -> dict:
        """
        Upload a file to S3.
        
        Args:
            file_content: File content as bytes
            file_name: Name of the file
            folder: S3 folder/prefix (e.g., 'pdfs', 'images', 'audio')
            content_type: MIME type of the file
            public: Whether to make the file publicly readable
            
        Returns:
            Dictionary with S3 URL and metadata
        """
        try:
            # Construct S3 key with folder prefix
            s3_key = f"{folder}/{file_name}" if folder else file_name
            
            # Prepare upload parameters
            extra_args = {
                "ContentType": content_type,
            }
            
            # For PDFs, set ContentDisposition to inline so they open in browser instead of downloading
            if content_type == "application/pdf" or file_name.endswith(".pdf"):
                extra_args["ContentDisposition"] = "inline"
            
            # Add CORS headers for cross-origin access
            extra_args["Metadata"] = {
                "Access-Control-Allow-Origin": "*",
            }
            
            # Note: ACLs are disabled on this bucket, using bucket policy for public access instead
            # if public:
            #     extra_args["ACL"] = "public-read"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                **extra_args,
            )
            
            # Generate S3 URL
            s3_url = self._generate_s3_url(s3_key)
            
            logger.info(f"File uploaded successfully: {s3_key} (public={public})")
            
            return {
                "s3_key": s3_key,
                "s3_url": s3_url,
                "bucket": self.bucket_name,
                "file_name": file_name,
                "folder": folder,
                "file_size": len(file_content),
            }
            
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file to S3: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during file upload: {str(e)}"
            )

    def upload_file_from_path(
        self,
        file_path: str,
        folder: str = "",
        content_type: str = "application/octet-stream",
        public: bool = False,
    ) -> dict:
        """
        Upload a file from local path to S3.
        
        Args:
            file_path: Local file path
            folder: S3 folder/prefix
            content_type: MIME type of the file
            public: Whether to make the file publicly readable
            
        Returns:
            Dictionary with S3 URL and metadata
        """
        try:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"File not found: {file_path}"
                )
            
            file_name = os.path.basename(file_path)
            
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            return self.upload_file(
                file_content=file_content,
                file_name=file_name,
                folder=folder,
                content_type=content_type,
                public=public,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading file from path: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file: {str(e)}"
            )

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: S3 key of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"File deleted from S3: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False

    def get_file(self, s3_key: str) -> Optional[bytes]:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 key of the file
            
        Returns:
            File content as bytes or None if not found
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"File not found in S3: {s3_key}")
                return None
            logger.error(f"Error downloading file from S3: {e}")
            return None

    def get_file_url(self, s3_key: str) -> str:
        """
        Generate a public URL for an S3 file.
        
        Args:
            s3_key: S3 key of the file
            
        Returns:
            Public URL of the file
        """
        return self._generate_s3_url(s3_key)

    def _generate_s3_url(self, s3_key: str) -> str:
        """Generate S3 URL for a given key."""
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 key of the file
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Error checking file existence: {e}")
            return False

    def get_file_metadata(self, s3_key: str) -> Optional[dict]:
        """
        Get metadata of an S3 file.
        
        Args:
            s3_key: S3 key of the file
            
        Returns:
            Dictionary with file metadata or None if not found
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                "size": response.get("ContentLength"),
                "content_type": response.get("ContentType"),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag"),
            }
        except ClientError as e:
            logger.error(f"Error getting file metadata: {e}")
            return None

    def list_files(self, folder: str = "") -> list:
        """
        List all files in a folder.
        
        Args:
            folder: S3 folder/prefix
            
        Returns:
            List of file keys in the folder
        """
        try:
            prefix = f"{folder}/" if folder else ""
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )
            
            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    if not obj["Key"].endswith("/"):  # Skip folders
                        files.append(obj["Key"])
            
            return files
        except ClientError as e:
            logger.error(f"Error listing files: {e}")
            return []
