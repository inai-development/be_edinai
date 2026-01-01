"""Auth helpers for the modular example backend."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from jose import jwt, JWTError

from ..config import settings

ALGORITHM = "HS256"


def create_access_token(data: Dict[str, Any], expires_minutes: int | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes or settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)

def decode_token(token: str) -> Dict[str, Any] | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def create_refresh_token() -> str:
    """Create a secure refresh token."""
    return secrets.token_urlsafe(32)

def get_token_expiry(token: str) -> datetime | None:
    """Get the expiration time of a JWT token."""
    payload = decode_token(token)
    if payload and "exp" in payload:
        return datetime.fromtimestamp(payload["exp"], timezone.utc)
    return None
