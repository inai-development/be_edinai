"""Superadministration Portal routes - Login endpoint only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Header,
)
from jose import jwt, JWTError

from ..config import settings, get_settings
from ..repository.chapter_material_repository import create_chapter_material
from ..schemas.auth_schema import LoginRequest as AuthLoginRequest
from ..schemas.response import ResponseBase
from ..utils.session_store import valid_tokens
from ..utils.file_handler import ALLOWED_PDF_TYPES
from ..utils.s3_file_handler import upload_pdf_to_s3, get_s3_service


router = APIRouter(prefix="/superadministration", tags=["Superadministration Portal"])

# Superadministration credentials
SUPERADMIN_EMAIL = "admin@inai.edu"
SUPERADMIN_PASSWORD = "superadmin123"


def _verify_superadmin_authorization(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid superadmin token",
        )

    if payload.get("role") != "superadmin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadministration access required",
        )

    session_id = payload.get("session_id")
    if not session_id or valid_tokens.get("superadmin") != session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Superadministration session expired",
        )


@router.post("/portal/login", response_model=ResponseBase)
async def superadmin_portal_login(payload: AuthLoginRequest) -> ResponseBase:
    """Superadministration portal login endpoint with hardcoded credentials."""
    if payload.email != SUPERADMIN_EMAIL or payload.password != SUPERADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid superadministration credentials",
        )

    # Create tokens for superadmin
    session_id = f"superadmin_{int(datetime.now(timezone.utc).timestamp())}"
    
    access_token_data = {
        "sub": "superadmin",
        "user_type": "superadmin",
        "role": "superadmin",
        "id": "superadmin",
        "session_id": session_id,
    }
    
    expire_delta = timedelta(minutes=settings.access_token_expire_minutes)
    access_payload = access_token_data.copy()
    access_payload.update({"exp": datetime.now(timezone.utc) + expire_delta, "type": "access"})
    access_token = jwt.encode(access_payload, settings.secret_key, algorithm=settings.algorithm)
    
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    refresh_payload = {
        "sub": "superadmin",
        "exp": expire,
        "type": "refresh",
        "session_id": session_id,
    }
    refresh_token = jwt.encode(refresh_payload, settings.secret_key, algorithm=settings.algorithm)
    
    valid_tokens["superadmin"] = session_id
    
    return ResponseBase(
        status=True,
        message="🎉 Superadministration Portal Login Successful",
        data={
            "user_type": "superadmin",
            "role": "superadmin",
            "access_token": access_token,
            "token": access_token,
            "refresh_token": refresh_token,
        },
    )


@router.post("/portal/upload-chapter", response_model=ResponseBase)
async def superadmin_upload_chapter_material(
    std: str = Form(...),
    subject: str = Form(...),
    chapter_number: str = Form(...),
    sem: str = Form(default=""),
    board: str = Form(default=""),
    chapter_title: str | None = Form(default=None),
    pdf_file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> ResponseBase:
    """Upload a chapter PDF once and distribute it to every admin account."""

    _verify_superadmin_authorization(authorization)

    if pdf_file.content_type not in ALLOWED_PDF_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    try:
        s3_service = get_s3_service(get_settings())
    except Exception as exc:  # pragma: no cover - configuration errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 configuration error: {exc}",
        ) from exc

    try:
        file_info = await upload_pdf_to_s3(
            pdf_file,
            s3_service,
            subfolder="superadministration",
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - upload failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload PDF: {exc}",
        ) from exc

    material = create_chapter_material(
        admin_id=0,
        std=std,
        subject=subject,
        sem=sem,
        board=board,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        file_info={**file_info, "is_global": True},
    )

    return ResponseBase(
        status=True,
        message="Chapter PDF uploaded for all admins",
        data={"material": material},
    )


@router.get("/portal/status", response_model=ResponseBase)
async def superadmin_portal_status(authorization: str | None = Header(default=None)) -> ResponseBase:
    """Lightweight heartbeat for the superadministration portal."""

    _verify_superadmin_authorization(authorization)

    return ResponseBase(
        status=True,
        message="Superadministration portal is operational",
        data={
            "portal_name": "Superadministration Portal",
            "version": "2.0",
        },
    )


@router.post("/portal/logout", response_model=ResponseBase)
async def superadmin_portal_logout(authorization: str | None = Header(default=None)) -> ResponseBase:
    """Invalidate the current superadministration session."""

    _verify_superadmin_authorization(authorization)
    valid_tokens.pop("superadmin", None)

    return ResponseBase(
        status=True,
        message="Superadministration logout successful",
        data={},
    )
