"""Blacklisted token repository for managing revoked access tokens."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models.blacklisted_token import BlacklistedToken


def blacklist_token(db: Session, token: str, expires_at: datetime) -> None:
    """Add a token to the blacklist."""
    blacklisted_token = BlacklistedToken(
        token=token,
        expires_at=expires_at,
        blacklisted_at=datetime.now(timezone.utc)
    )
    
    db.add(blacklisted_token)
    db.commit()


def is_token_blacklisted(db: Session, token: str) -> bool:
    """Check if a token is blacklisted."""
    blacklisted = db.query(BlacklistedToken).filter(
        BlacklistedToken.token == token,
        BlacklistedToken.expires_at > datetime.now(timezone.utc)
    ).first()
    
    return blacklisted is not None


def cleanup_expired_blacklisted_tokens(db: Session) -> int:
    """Remove expired blacklisted tokens from database."""
    count = db.query(BlacklistedToken).filter(
        BlacklistedToken.expires_at <= datetime.now(timezone.utc)
    ).delete()
    
    db.commit()
    return count