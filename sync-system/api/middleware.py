import time
import traceback
from typing import Callable
from datetime import datetime

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.models import ErrorResponse
from src.utils.logging import get_logger

logger = get_logger("api.middleware")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            # Validation errors
            logger.warning(f"Validation error: {e}")
            error_response = ErrorResponse(
                error="ValidationError",
                message=str(e),
                details={"type": "ValueError"},
                timestamp=datetime.now()
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=error_response.dict()
            )

        except TimeoutError as e:
            # Timeout errors
            logger.error(f"Timeout error: {e}")
            error_response = ErrorResponse(
                error="TimeoutError",
                message="Request timed out. The collaboration took too long to complete.",
                details={"type": "TimeoutError"},
                timestamp=datetime.now()
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=error_response.dict()
            )

        except Exception as e:
            # All other errors
            logger.error(f"Unhandled error: {e}", exc_info=True)
            error_response = ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred. Please try again.",
                details={
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc() if logger.level <= 10 else None
                },
                timestamp=datetime.now()
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.dict()
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware

    Logs all incoming requests and their processing time.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(
            f"[{request.method}] {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Compute duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"[{request.method}] {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.3f}s"
        )

        # Add custom headers
        response.headers["X-Process-Time"] = f"{duration:.3f}"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware

    Tracks requests per IP and enforces rate limits.
    Note: For production, use Redis-based rate limiting.
    """

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts = {}  # In-memory store (use Redis in prod)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for health check
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        # Get current time window
        current_time = int(time.time())
        window_start = current_time - (current_time % self.window_seconds)

        # Initialize or get request count
        key = f"{client_ip}:{window_start}"
        if key not in self.request_counts:
            # Clean old windows
            old_keys = [
                k for k in self.request_counts.keys()
                if int(k.split(':')[1]) < window_start - self.window_seconds
            ]
            for old_key in old_keys:
                del self.request_counts[old_key]

            self.request_counts[key] = 0

        # Check rate limit
        if self.request_counts[key] >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            error_response = ErrorResponse(
                error="RateLimitExceeded",
                message=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
                details={
                    "max_requests": self.max_requests,
                    "window_seconds": self.window_seconds,
                    "retry_after": self.window_seconds
                },
                timestamp=datetime.now()
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response.dict(),
                headers={"Retry-After": str(self.window_seconds)}
            )

        # Increment counter
        self.request_counts[key] += 1

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.max_requests - self.request_counts[key])
        )
        response.headers["X-RateLimit-Reset"] = str(window_start + self.window_seconds)

        return response


def setup_cors(app):
    """
    Setup CORS middleware

    Allows cross-origin requests from specified origins.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React dev server
            "http://localhost:8000",  # FastAPI dev server
            "https://yourdomain.com",  # Production domain
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    logger.info("CORS middleware configured")


def setup_middleware(app):
    """
    Setup all middleware for the application

    Args:
        app: FastAPI application instance
    """
    # CORS (must be first)
    setup_cors(app)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Rate limiting (10 requests per minute for now - adjust as needed)
    app.add_middleware(RateLimitMiddleware, max_requests=10, window_seconds=60)

    # Error handling (should be last)
    app.add_middleware(ErrorHandlingMiddleware)

    logger.info("All middleware configured successfully")
