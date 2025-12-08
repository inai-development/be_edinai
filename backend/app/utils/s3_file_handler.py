"""S3-based file handling utilities for uploads."""
from __future__ import annotations

import io
import logging
import os
import uuid
from typing import List, Optional, Set

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.services.s3_service import S3Service

logger = logging.getLogger(__name__)

# File type configurations
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_PDF_TYPES = {"application/pdf"}

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".svg", ".webp"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/svg+xml", "image/webp"}

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
}

DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for S3

MIMETYPE_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
}

# S3 folder structure
S3_FOLDERS = {
    "pdf": "pdfs",
    "image": "images",
    "audio": "audio",
    "lecture": "lectures",
    "merged": "merged_lectures",
}


def get_s3_service(settings) -> S3Service:
    """Get S3 service instance from settings."""
    return S3Service(
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key,
        region=settings.aws_region,
        bucket_name=settings.aws_s3_bucket_name,
    )


def validate_file(
    file: UploadFile,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_types: Optional[Set[str]] = None,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> bool:
    """Validate uploaded file."""
    extensions = allowed_extensions or ALLOWED_IMAGE_EXTENSIONS
    content_types = allowed_types or ALLOWED_IMAGE_TYPES

    if hasattr(file, "size") and file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum allowed size is {max_size // (1024 * 1024)}MB",
        )

    if file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Allowed extensions: {', '.join(sorted(extensions))}",
            )

    if file.content_type not in content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(sorted(content_types))}",
        )

    return True


async def upload_pdf_to_s3(
    file: UploadFile,
    s3_service: S3Service,
    subfolder: str = "",
) -> dict:
    """
    Upload PDF file to S3.
    
    Args:
        file: Uploaded file
        s3_service: S3 service instance
        subfolder: Additional subfolder under pdfs/
        
    Returns:
        Dictionary with S3 URL and metadata
    """
    validate_file(
        file,
        allowed_extensions=ALLOWED_PDF_EXTENSIONS,
        allowed_types=ALLOWED_PDF_TYPES,
        max_size=DEFAULT_MAX_FILE_SIZE,
    )

    try:
        content = await file.read()
        
        # Generate unique filename
        if file.filename:
            file_ext = os.path.splitext(file.filename)[1]
        else:
            file_ext = ".pdf"
        
        saved_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Determine S3 folder
        s3_folder = S3_FOLDERS["pdf"]
        if subfolder:
            s3_folder = f"{s3_folder}/{subfolder}"
        
        # Upload to S3
        result = s3_service.upload_file(
            file_content=content,
            file_name=saved_filename,
            folder=s3_folder,
            content_type=file.content_type or "application/pdf",
            public=True,
        )
        
        logger.info(f"PDF uploaded to S3: {result['s3_key']}")
        
        return {
            "filename": file.filename,
            "saved_filename": saved_filename,
            "s3_key": result["s3_key"],
            "s3_url": result["s3_url"],
            "file_size": len(content),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {str(e)}")


async def upload_image_to_s3(
    file: UploadFile,
    s3_service: S3Service,
    subfolder: str = "",
) -> dict:
    """
    Upload image file to S3.
    
    Args:
        file: Uploaded file
        s3_service: S3 service instance
        subfolder: Additional subfolder under images/
        
    Returns:
        Dictionary with S3 URL and metadata
    """
    validate_file(
        file,
        allowed_extensions=ALLOWED_IMAGE_EXTENSIONS,
        allowed_types=ALLOWED_IMAGE_TYPES,
        max_size=DEFAULT_MAX_FILE_SIZE,
    )

    try:
        content = await file.read()
        
        # Validate image
        if file.content_type and file.content_type.startswith("image/") and file.content_type != "image/svg+xml":
            try:
                image = Image.open(io.BytesIO(content))
                image.verify()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Generate unique filename
        if file.filename:
            file_ext = os.path.splitext(file.filename)[1]
        else:
            file_ext = ".jpg"
        
        saved_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Determine S3 folder
        s3_folder = S3_FOLDERS["image"]
        if subfolder:
            s3_folder = f"{s3_folder}/{subfolder}"
        
        # Upload to S3
        result = s3_service.upload_file(
            file_content=content,
            file_name=saved_filename,
            folder=s3_folder,
            content_type=file.content_type or "image/jpeg",
            public=True,  # Images are publicly readable
        )
        
        logger.info(f"Image uploaded to S3: {result['s3_key']}")
        
        return {
            "filename": file.filename,
            "saved_filename": saved_filename,
            "s3_key": result["s3_key"],
            "s3_url": result["s3_url"],
            "file_size": len(content),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")


async def upload_audio_to_s3(
    file_content: bytes,
    file_name: str,
    s3_service: S3Service,
    content_type: str = "audio/mpeg",
    subfolder: str = "",
) -> dict:
    """
    Upload audio file to S3.
    
    Args:
        file_content: Audio file content as bytes
        file_name: Name of the audio file
        s3_service: S3 service instance
        content_type: MIME type of the audio
        subfolder: Additional subfolder under audio/
        
    Returns:
        Dictionary with S3 URL and metadata
    """
    try:
        # Determine S3 folder
        s3_folder = S3_FOLDERS["audio"]
        if subfolder:
            s3_folder = f"{s3_folder}/{subfolder}"
        
        # Upload to S3
        result = s3_service.upload_file(
            file_content=file_content,
            file_name=file_name,
            folder=s3_folder,
            content_type=content_type,
            public=True,  # Audio is publicly readable
        )
        
        logger.info(f"Audio uploaded to S3: {result['s3_key']}")
        
        return {
            "filename": file_name,
            "s3_key": result["s3_key"],
            "s3_url": result["s3_url"],
            "file_size": len(file_content),
        }

    except Exception as e:
        logger.error(f"Error uploading audio to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {str(e)}")


async def upload_lecture_json_to_s3(
    json_content: str,
    file_name: str,
    s3_service: S3Service,
    subfolder: str = "",
) -> dict:
    """
    Upload lecture JSON file to S3.
    
    Args:
        json_content: JSON content as string
        file_name: Name of the JSON file
        s3_service: S3 service instance
        subfolder: Additional subfolder under lectures/
        
    Returns:
        Dictionary with S3 URL and metadata
    """
    try:
        # Determine S3 folder
        s3_folder = S3_FOLDERS["lecture"]
        if subfolder:
            s3_folder = f"{s3_folder}/{subfolder}"
        
        # Convert string to bytes
        file_content = json_content.encode("utf-8")
        
        # Upload to S3
        result = s3_service.upload_file(
            file_content=file_content,
            file_name=file_name,
            folder=s3_folder,
            content_type="application/json",
            public=True,
        )
        
        logger.info(f"Lecture JSON uploaded to S3: {result['s3_key']}")
        
        return {
            "filename": file_name,
            "s3_key": result["s3_key"],
            "s3_url": result["s3_url"],
            "file_size": len(file_content),
        }

    except Exception as e:
        logger.error(f"Error uploading lecture JSON to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save lecture: {str(e)}")


def delete_file_from_s3(s3_key: str, s3_service: S3Service) -> bool:
    """
    Delete a file from S3.
    
    Args:
        s3_key: S3 key of the file
        s3_service: S3 service instance
        
    Returns:
        True if successful, False otherwise
    """
    return s3_service.delete_file(s3_key)
