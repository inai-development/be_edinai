from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

try:  # pragma: no cover
    import redis.asyncio as redis_async  # type: ignore
except ImportError:  # pragma: no cover
    redis_async = None  # type: ignore

logger = logging.getLogger(__name__)


class RequestTooLargeError(RuntimeError):
    """Raised when the incoming request body exceeds the configured limit."""


@dataclass
class RateLimitRule:
    name: str
    methods: Sequence[str]
    path: str
    requests: int
    window_seconds: int
    block_seconds: int
    match_type: str = "exact"  # exact | prefix | contains | all
    bucket: Optional[str] = None

    def matches(self, request_path: str, method: str) -> bool:
        method = method.upper()
        if self.methods and "*" not in self.methods and method not in self.methods:
            return False

        path = request_path.lower()
        target = self.path.lower()
        match_type = self.match_type

        if match_type == "exact":
            return path == target
        if match_type == "prefix":
            return path.startswith(target.rstrip("/"))
        if match_type == "contains":
            return target in path
        if match_type == "all":
            return True
        logger.warning("Unknown rate limit match_type=%s for rule=%s", match_type, self.name)
        return False

    @property
    def bucket_name(self) -> str:
        return self.bucket or self.name or self.path or "default"


@dataclass
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_timestamp: int
    retry_after: int
    rule: Optional[RateLimitRule]


class RedisRateLimitBackend:
    """Redis-backed counter for rate limiting."""

    def __init__(
        self,
        *,
        url: Optional[str],
        host: str,
        port: int,
        db: int,
        password: Optional[str],
        ssl: bool,
        key_prefix: str,
    ) -> None:
        self._url = url
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._ssl = ssl
        self._key_prefix = key_prefix.rstrip(":") or "rl"
        self._client: Optional["redis_async.Redis"] = None
        self._lock = asyncio.Lock()
        self._warned = False

    async def _ensure_client(self) -> "redis_async.Redis":  # type: ignore[override]
        if redis_async is None:  # pragma: no cover - dependency missing guard
            raise RuntimeError("redis package is not installed")
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def _create_client(self) -> "redis_async.Redis":  # type: ignore[override]
        if self._url:
            return redis_async.from_url(  # type: ignore[return-value]
                self._url,
                encoding=None,
                decode_responses=False,
            )
        return redis_async.Redis(  # type: ignore[return-value]
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._password,
            ssl=self._ssl or None,
            encoding=None,
            decode_responses=False,
        )

    async def evaluate(
        self,
        *,
        identifier: str,
        rule: RateLimitRule,
        method: str,
    ) -> RateLimitDecision:
        client = await self._ensure_client()
        bucket = rule.bucket_name
        method = method.upper()
        key = f"{self._key_prefix}:{bucket}:{method}:{identifier}"
        block_key = f"{key}:blocked"
        now = int(time.time())

        try:
            blocked_ttl = await client.ttl(block_key)
        except Exception as exc:  # pragma: no cover - transient connection failure
            self._log_backend_error(exc)
            return RateLimitDecision(True, rule.requests, rule.requests, now, 0, rule)

        if blocked_ttl and blocked_ttl > 0:
            reset_ts = now + int(blocked_ttl)
            return RateLimitDecision(False, rule.requests, 0, reset_ts, int(blocked_ttl), rule)

        if rule.requests <= 0:
            return RateLimitDecision(True, rule.requests, rule.requests, now, 0, rule)

        try:
            current = await client.incr(key)
            if current == 1:
                await client.expire(key, rule.window_seconds)
            ttl = await client.ttl(key)
            if ttl is None or ttl < 0:
                ttl = rule.window_seconds
                await client.expire(key, ttl)
        except Exception as exc:  # pragma: no cover - transient connection failure
            self._log_backend_error(exc)
            return RateLimitDecision(True, rule.requests, rule.requests, now, 0, rule)

        if current > rule.requests:
            block_seconds = max(rule.block_seconds, 1)
            try:
                await client.set(block_key, b"1", ex=block_seconds)
            except Exception as exc:  # pragma: no cover
                self._log_backend_error(exc)
            retry_after = block_seconds
            return RateLimitDecision(False, rule.requests, 0, now + retry_after, retry_after, rule)

        remaining = max(rule.requests - current, 0)
        reset_ts = now + max(ttl, 0)
        return RateLimitDecision(True, rule.requests, remaining, reset_ts, max(ttl, 0), rule)

    def _log_backend_error(self, exc: Exception) -> None:
        if not self._warned:
            logger.warning("Rate limiter Redis backend error: %s", exc)
            self._warned = True


class RateLimiter:
    def __init__(
        self,
        *,
        enabled: bool,
        rules: Sequence[RateLimitRule],
        default_rule: RateLimitRule,
        backend: Optional[RedisRateLimitBackend],
        whitelist_ips: Iterable[str],
        exempt_paths: Iterable[str],
    ) -> None:
        self.enabled = enabled and backend is not None
        self.rules = tuple(rules)
        self.default_rule = default_rule
        self.backend = backend
        self.whitelist = {ip.strip().lower() for ip in whitelist_ips if ip.strip()}
        self.exempt_paths = tuple(self._normalize_path(p) for p in exempt_paths)
        self._warned_disabled = False

    async def check_request(self, request: Request) -> RateLimitDecision:
        if not self.enabled or request.scope.get("type") != "http":
            return RateLimitDecision(True, 0, 0, int(time.time()), 0, None)

        path = self._normalize_path(request.url.path)
        if self._is_exempt_path(path):
            return RateLimitDecision(True, 0, 0, int(time.time()), 0, None)

        identifier = self._identify(request)
        if identifier.lower() in self.whitelist:
            return RateLimitDecision(True, 0, 0, int(time.time()), 0, None)

        rule = self._select_rule(path, request.method)
        if rule is None or rule.requests <= 0:
            return RateLimitDecision(True, 0, 0, int(time.time()), 0, None)

        if not self.backend:
            self._warn_backend_missing()
            return RateLimitDecision(True, rule.requests, rule.requests, int(time.time()), 0, rule)

        return await self.backend.evaluate(identifier=identifier, rule=rule, method=request.method)

    def _warn_backend_missing(self) -> None:
        if not self._warned_disabled:
            logger.warning("Rate limiting enabled but Redis backend is unavailable; allowing all requests.")
            self._warned_disabled = True

    def _normalize_path(self, path: str) -> str:
        return path.strip() or "/"

    def _is_exempt_path(self, path: str) -> bool:
        normalized = path.lower()
        for exempt in self.exempt_paths:
            if not exempt:
                continue
            if normalized == exempt:
                return True
            if exempt.endswith("/*"):
                prefix = exempt[:-2]
                if normalized.startswith(prefix):
                    return True
            elif normalized.startswith(exempt.rstrip("/")):
                return True
        return False

    def _select_rule(self, path: str, method: str) -> RateLimitRule:
        for rule in self.rules:
            if rule.matches(path, method):
                return rule
        return self.default_rule

    def _identify(self, request: Request) -> str:
        headers = request.headers
        forwarded_for = headers.get("x-forwarded-for")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()
            if ip:
                return ip
        cf_ip = headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
        real_ip = headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        client_host = getattr(request.client, "host", None)
        return client_host or "anonymous"


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, rate_limiter: RateLimiter) -> None:
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        decision = await self.rate_limiter.check_request(request)
        if not decision.allowed and decision.rule:
            retry_after = max(int(decision.retry_after), 1)
            detail = {
                "detail": "Rate limit exceeded",
                "rule": decision.rule.name,
                "retry_after": retry_after,
            }
            headers = {
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(decision.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Rule": decision.rule.name,
            }
            return JSONResponse(status_code=429, content=detail, headers=headers)

        try:
            response = await call_next(request)
        except RequestTooLargeError:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})

        if decision.rule and decision.limit > 0:
            response.headers.setdefault("X-RateLimit-Limit", str(decision.limit))
            response.headers.setdefault("X-RateLimit-Remaining", str(max(decision.remaining, 0)))
            response.headers.setdefault("X-RateLimit-Reset", str(decision.reset_timestamp))
            response.headers.setdefault("X-RateLimit-Rule", decision.rule.name)
        return response


class RequestFilterMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        blocked_user_agents: Sequence[str],
        max_body_bytes: int,
    ) -> None:
        super().__init__(app)
        self.blocked_agents = tuple(ua.lower() for ua in blocked_user_agents if ua)
        self.max_body_bytes = max_body_bytes if max_body_bytes and max_body_bytes > 0 else 0

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.scope.get("type") != "http":
            return await call_next(request)

        if self.blocked_agents:
            user_agent = (request.headers.get("user-agent") or "").lower()
            if user_agent and any(agent in user_agent for agent in self.blocked_agents):
                logger.warning("Blocked request due to user-agent: %s", user_agent)
                return JSONResponse(status_code=400, content={"detail": "Blocked user agent"})

        limited_request = request
        if self.max_body_bytes:
            limited_request = self._wrap_request_body(request)

        return await call_next(limited_request)

    def _wrap_request_body(self, request: Request) -> Request:
        received = 0
        original_receive = request._receive  # type: ignore[attr-defined]

        async def receive() -> dict:
            nonlocal received
            message = await original_receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                received += len(body)
                if self.max_body_bytes and received > self.max_body_bytes:
                    raise RequestTooLargeError()
            return message

        request._receive = receive  # type: ignore[attr-defined]
        return request


def build_rate_limiter(settings) -> RateLimiter:
    backend: Optional[RedisRateLimitBackend] = None
    if settings.rate_limit_enabled and redis_async is not None:
        backend = RedisRateLimitBackend(
            url=getattr(settings, "redis_url", None),
            host=getattr(settings, "redis_host", "localhost"),
            port=int(getattr(settings, "redis_port", 6379)),
            db=int(getattr(settings, "redis_db", 0)),
            password=getattr(settings, "redis_password", None),
            ssl=bool(getattr(settings, "redis_ssl", False)),
            key_prefix=settings.rate_limit_key_prefix,
        )
    elif settings.rate_limit_enabled:
        logger.warning("Rate limiting enabled but redis package is not installed; allowing all requests.")

    rules = [
        RateLimitRule(
            name="login",
            methods=("POST",),
            path="/auth/login",
            requests=settings.rate_limit_login_requests,
            window_seconds=settings.rate_limit_login_window_seconds,
            block_seconds=settings.rate_limit_login_block_seconds,
        ),
        RateLimitRule(
            name="otp",
            methods=("POST", "GET"),
            path="/otp",
            match_type="contains",
            requests=settings.rate_limit_otp_requests,
            window_seconds=settings.rate_limit_otp_window_seconds,
            block_seconds=settings.rate_limit_otp_block_seconds,
        ),
        RateLimitRule(
            name="health",
            methods=("GET",),
            path="/health",
            requests=settings.rate_limit_health_requests,
            window_seconds=settings.rate_limit_health_window_seconds,
            block_seconds=settings.rate_limit_health_block_seconds,
        ),
        RateLimitRule(
            name="metrics",
            methods=("GET",),
            path="/metrics",
            requests=settings.rate_limit_metrics_requests,
            window_seconds=settings.rate_limit_metrics_window_seconds,
            block_seconds=settings.rate_limit_metrics_block_seconds,
        ),
    ]

    default_rule = RateLimitRule(
        name="default",
        methods=("*",),
        path="*",
        match_type="all",
        requests=settings.rate_limit_default_requests,
        window_seconds=settings.rate_limit_default_window_seconds,
        block_seconds=settings.rate_limit_default_block_seconds,
    )

    return RateLimiter(
        enabled=settings.rate_limit_enabled,
        rules=rules,
        default_rule=default_rule,
        backend=backend,
        whitelist_ips=settings.rate_limit_whitelist_ips,
        exempt_paths=settings.rate_limit_exempt_paths,
    )


__all__ = [
    "RateLimiter",
    "RateLimiterMiddleware",
    "RateLimitRule",
    "RequestFilterMiddleware",
    "build_rate_limiter",
]
