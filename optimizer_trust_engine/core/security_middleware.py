"""
Security Middleware for GPU Optimizer Platform
Implements security best practices for FastAPI applications
"""

import time
import hashlib
import secrets
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SecurityHeaders(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        return response

class RateLimiter(BaseHTTPMiddleware):
    """Implement rate limiting for API endpoints"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.window_start = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.window_start > 60:
            self.request_counts = {}
            self.window_start = current_time
        
        # Check rate limit
        if client_ip in self.request_counts:
            if self.request_counts[client_ip] >= self.requests_per_minute:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"}
                )
            self.request_counts[client_ip] += 1
        else:
            self.request_counts[client_ip] = 1
        
        response = await call_next(request)
        return response

class RequestValidator(BaseHTTPMiddleware):
    """Validate and sanitize incoming requests"""
    
    def __init__(self, app, max_content_length: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_content_length = max_content_length
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request too large"}
            )
        
        # Log request for audit
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
        
        response = await call_next(request)
        return response

class JWTAuthMiddleware:
    """JWT authentication middleware"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.bearer = HTTPBearer()
    
    def create_token(self, user_id: str, expiration_minutes: int = 30) -> str:
        """Create a JWT token"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(minutes=expiration_minutes),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID for tracking
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    async def __call__(self, request: Request) -> Optional[dict]:
        """Middleware callable for FastAPI dependency injection"""
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        token = authorization.replace("Bearer ", "")
        payload = self.verify_token(token)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return payload

class InputSanitizer:
    """Sanitize user inputs to prevent injection attacks"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Truncate to max length
        value = value[:max_length]
        
        # Remove control characters
        import re
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        return value
    
    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize SQL identifiers (table names, column names)"""
        # Only allow alphanumeric and underscore
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

class AuditLogger:
    """Audit logging for security events"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.logger = logging.getLogger("security_audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_authentication(self, user_id: str, success: bool, ip: str):
        """Log authentication attempts"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"AUTH {status}: user={user_id}, ip={ip}")
    
    def log_authorization(self, user_id: str, resource: str, action: str, allowed: bool):
        """Log authorization decisions"""
        status = "ALLOWED" if allowed else "DENIED"
        self.logger.info(f"AUTHZ {status}: user={user_id}, resource={resource}, action={action}")
    
    def log_data_access(self, user_id: str, data_type: str, operation: str):
        """Log data access events"""
        self.logger.info(f"DATA ACCESS: user={user_id}, type={data_type}, operation={operation}")
    
    def log_security_event(self, event_type: str, details: dict):
        """Log general security events"""
        self.logger.warning(f"SECURITY EVENT: type={event_type}, details={details}")

# Export middleware classes
__all__ = [
    'SecurityHeaders',
    'RateLimiter',
    'RequestValidator',
    'JWTAuthMiddleware',
    'InputSanitizer',
    'AuditLogger'
]
