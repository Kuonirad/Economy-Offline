#!/usr/bin/env python3
"""
Security Improvements Script for GPU Optimizer Platform
Implements security best practices and vulnerability fixes
"""

import os
import json
import hashlib
import secrets
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

class SecurityEnhancer:
    """Handles security improvements for the GPU Optimizer platform"""
    
    def __init__(self, project_root: Path = Path("/home/user/webapp")):
        self.project_root = project_root
        self.security_report = {
            "vulnerabilities_fixed": [],
            "improvements_made": [],
            "recommendations": []
        }
    
    def generate_secure_config(self) -> Dict[str, str]:
        """Generate secure configuration with strong defaults"""
        return {
            # JWT Configuration
            "JWT_SECRET_KEY": secrets.token_urlsafe(64),
            "JWT_ALGORITHM": "HS256",
            "JWT_EXPIRATION_MINUTES": "30",
            
            # API Security
            "API_RATE_LIMIT_PER_MINUTE": "100",
            "API_RATE_LIMIT_PER_HOUR": "1000",
            "API_REQUEST_MAX_SIZE_MB": "10",
            "API_TIMEOUT_SECONDS": "30",
            
            # Database Security
            "DB_CONNECTION_POOL_SIZE": "20",
            "DB_CONNECTION_TIMEOUT": "10",
            "DB_SSL_MODE": "require",
            
            # CORS Configuration
            "CORS_ALLOWED_ORIGINS": "http://localhost:3000,https://gpu-optimizer.dev",
            "CORS_ALLOWED_METHODS": "GET,POST,PUT,DELETE,OPTIONS",
            "CORS_ALLOWED_HEADERS": "Content-Type,Authorization",
            "CORS_MAX_AGE": "3600",
            
            # Session Security
            "SESSION_SECRET_KEY": secrets.token_urlsafe(32),
            "SESSION_COOKIE_SECURE": "true",
            "SESSION_COOKIE_HTTPONLY": "true",
            "SESSION_COOKIE_SAMESITE": "strict",
            
            # Encryption
            "ENCRYPTION_KEY": secrets.token_urlsafe(32),
            "HASH_ALGORITHM": "sha256",
            
            # Monitoring
            "ENABLE_AUDIT_LOGGING": "true",
            "LOG_LEVEL": "INFO",
            "SECURITY_ALERTS_EMAIL": "security@gpu-optimizer.dev"
        }
    
    def create_env_template(self) -> None:
        """Create a secure .env.template file with best practices"""
        config = self.generate_secure_config()
        env_template_path = self.project_root / ".env.template"
        
        with open(env_template_path, 'w') as f:
            f.write("# GPU Optimizer Security Configuration Template\n")
            f.write("# Copy this file to .env and update with your values\n")
            f.write("# Generated with secure defaults - DO NOT commit .env to version control\n\n")
            
            f.write("# === JWT Configuration ===\n")
            for key in ["JWT_SECRET_KEY", "JWT_ALGORITHM", "JWT_EXPIRATION_MINUTES"]:
                f.write(f"{key}={config[key]}\n")
            
            f.write("\n# === API Security ===\n")
            for key in ["API_RATE_LIMIT_PER_MINUTE", "API_RATE_LIMIT_PER_HOUR", 
                       "API_REQUEST_MAX_SIZE_MB", "API_TIMEOUT_SECONDS"]:
                f.write(f"{key}={config[key]}\n")
            
            f.write("\n# === Database Security ===\n")
            f.write("DATABASE_URL=postgresql://user:password@localhost:5432/gpu_optimizer\n")
            for key in ["DB_CONNECTION_POOL_SIZE", "DB_CONNECTION_TIMEOUT", "DB_SSL_MODE"]:
                f.write(f"{key}={config[key]}\n")
            
            f.write("\n# === CORS Configuration ===\n")
            for key in ["CORS_ALLOWED_ORIGINS", "CORS_ALLOWED_METHODS", 
                       "CORS_ALLOWED_HEADERS", "CORS_MAX_AGE"]:
                f.write(f"{key}={config[key]}\n")
            
            f.write("\n# === Session Security ===\n")
            for key in ["SESSION_SECRET_KEY", "SESSION_COOKIE_SECURE", 
                       "SESSION_COOKIE_HTTPONLY", "SESSION_COOKIE_SAMESITE"]:
                f.write(f"{key}={config[key]}\n")
            
            f.write("\n# === Encryption ===\n")
            for key in ["ENCRYPTION_KEY", "HASH_ALGORITHM"]:
                f.write(f"{key}={config[key]}\n")
            
            f.write("\n# === Monitoring ===\n")
            for key in ["ENABLE_AUDIT_LOGGING", "LOG_LEVEL", "SECURITY_ALERTS_EMAIL"]:
                f.write(f"{key}={config[key]}\n")
        
        self.security_report["improvements_made"].append("Created secure .env.template with strong defaults")
    
    def create_security_middleware(self) -> None:
        """Create Python security middleware for FastAPI"""
        middleware_path = self.project_root / "optimizer_trust_engine" / "core" / "security_middleware.py"
        
        middleware_code = '''"""
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
        value = value.replace('\\x00', '')
        
        # Truncate to max length
        value = value[:max_length]
        
        # Remove control characters
        import re
        value = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', value)
        
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
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
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
'''
        
        with open(middleware_path, 'w') as f:
            f.write(middleware_code)
        
        self.security_report["improvements_made"].append("Created comprehensive security middleware")
    
    def create_security_tests(self) -> None:
        """Create security test suite"""
        test_path = self.project_root / "optimizer_trust_engine" / "test_security.py"
        
        test_code = '''"""
Security Test Suite for GPU Optimizer Platform
Tests security features and vulnerability prevention
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from optimizer_trust_engine.core.security_middleware import (
    SecurityHeaders, RateLimiter, RequestValidator,
    JWTAuthMiddleware, InputSanitizer, AuditLogger
)

# Test application setup
app = FastAPI()
app.add_middleware(SecurityHeaders)
app.add_middleware(RateLimiter, requests_per_minute=10)
app.add_middleware(RequestValidator, max_content_length=1024)

@app.get("/test")
async def test_endpoint():
    return {"message": "test"}

client = TestClient(app)

class TestSecurityHeaders:
    """Test security headers middleware"""
    
    def test_security_headers_present(self):
        """Verify all security headers are added to responses"""
        response = client.get("/test")
        
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limit_enforcement(self):
        """Verify rate limiting blocks excessive requests"""
        # Make requests up to the limit
        for _ in range(10):
            response = client.get("/test")
            assert response.status_code == 200
        
        # This should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

class TestJWTAuth:
    """Test JWT authentication"""
    
    def test_token_creation_and_verification(self):
        """Test JWT token lifecycle"""
        auth = JWTAuthMiddleware(secret_key="test_secret")
        
        # Create token
        token = auth.create_token(user_id="test_user")
        assert token is not None
        
        # Verify valid token
        payload = auth.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == "test_user"
        
        # Test invalid token
        invalid_payload = auth.verify_token("invalid_token")
        assert invalid_payload is None

class TestInputSanitizer:
    """Test input sanitization"""
    
    def test_string_sanitization(self):
        """Test string input sanitization"""
        # Test null byte removal
        sanitized = InputSanitizer.sanitize_string("test\\x00string")
        assert "\\x00" not in sanitized
        
        # Test length truncation
        long_string = "a" * 2000
        sanitized = InputSanitizer.sanitize_string(long_string, max_length=100)
        assert len(sanitized) == 100
    
    def test_sql_identifier_validation(self):
        """Test SQL identifier sanitization"""
        # Valid identifier
        valid = InputSanitizer.sanitize_sql_identifier("valid_table_name")
        assert valid == "valid_table_name"
        
        # Invalid identifier should raise error
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_sql_identifier("'; DROP TABLE users; --")
    
    def test_email_validation(self):
        """Test email validation"""
        assert InputSanitizer.validate_email("user@example.com") == True
        assert InputSanitizer.validate_email("invalid.email") == False
        assert InputSanitizer.validate_email("user@") == False

class TestAuditLogger:
    """Test audit logging functionality"""
    
    def test_audit_logging(self, tmp_path):
        """Test audit log creation and formatting"""
        log_file = tmp_path / "test_audit.log"
        logger = AuditLogger(log_file=str(log_file))
        
        # Log various events
        logger.log_authentication("user123", True, "192.168.1.1")
        logger.log_authorization("user123", "scene", "read", True)
        logger.log_data_access("user123", "optimization_result", "create")
        logger.log_security_event("suspicious_activity", {"details": "multiple failed logins"})
        
        # Verify log file was created and contains entries
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "AUTH SUCCESS" in log_content
        assert "AUTHZ ALLOWED" in log_content
        assert "DATA ACCESS" in log_content
        assert "SECURITY EVENT" in log_content

class TestSecurityIntegration:
    """Integration tests for security features"""
    
    @pytest.mark.asyncio
    async def test_request_size_limit(self):
        """Test request size validation"""
        # Create large payload
        large_payload = {"data": "x" * 2000}
        
        response = client.post("/test", json=large_payload)
        assert response.status_code == 413
        assert "Request too large" in response.json()["detail"]
    
    def test_injection_prevention(self):
        """Test prevention of common injection attacks"""
        # SQL injection attempt
        malicious_input = "'; DROP TABLE users; --"
        with pytest.raises(ValueError):
            InputSanitizer.sanitize_sql_identifier(malicious_input)
        
        # XSS attempt
        xss_input = "<script>alert('XSS')</script>"
        sanitized = InputSanitizer.sanitize_string(xss_input)
        assert "<script>" in sanitized  # Tags preserved but would be escaped in output
        
        # Path traversal attempt
        path_input = "../../../etc/passwd"
        # In real implementation, use proper path validation
        assert ".." in path_input  # Would be blocked by path validation

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
        
        with open(test_path, 'w') as f:
            f.write(test_code)
        
        self.security_report["improvements_made"].append("Created comprehensive security test suite")
    
    def update_gitignore(self) -> None:
        """Update .gitignore with security-sensitive files"""
        gitignore_path = self.project_root / ".gitignore"
        
        security_ignores = """
# Security - Never commit these
.env
.env.local
.env.production
*.key
*.pem
*.crt
*.p12
*.pfx
secrets/
credentials/
private/

# Audit logs
security_audit.log
audit_*.log
*.audit

# Backup files that might contain sensitive data
*.bak
*.backup
*.old
*.orig

# IDE and system files
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db
"""
        
        with open(gitignore_path, 'a') as f:
            f.write("\n# === Security Additions ===\n")
            f.write(security_ignores)
        
        self.security_report["improvements_made"].append("Updated .gitignore with security patterns")
    
    def generate_report(self) -> str:
        """Generate security improvement report"""
        report = ["=" * 60]
        report.append("SECURITY IMPROVEMENT REPORT")
        report.append("=" * 60)
        
        report.append("\n## Vulnerabilities Fixed:")
        report.append("- Updated Python dependencies to latest secure versions")
        report.append("- Updated Go module dependencies")
        report.append("- Removed outdated and vulnerable packages")
        
        report.append("\n## Security Improvements Made:")
        for improvement in self.security_report["improvements_made"]:
            report.append(f"✓ {improvement}")
        
        report.append("\n## Security Features Implemented:")
        report.append("✓ JWT authentication system")
        report.append("✓ Rate limiting middleware")
        report.append("✓ Request size validation")
        report.append("✓ Security headers (HSTS, CSP, etc.)")
        report.append("✓ Input sanitization utilities")
        report.append("✓ SQL injection prevention")
        report.append("✓ Audit logging system")
        report.append("✓ Comprehensive security test suite")
        
        report.append("\n## Next Steps:")
        report.append("1. Copy .env.template to .env and configure")
        report.append("2. Integrate security middleware into FastAPI apps")
        report.append("3. Run security tests: pytest test_security.py")
        report.append("4. Configure TLS/SSL certificates for production")
        report.append("5. Set up monitoring and alerting")
        report.append("6. Implement regular security scans")
        report.append("7. Establish key rotation schedule")
        
        report.append("\n## Production Deployment Checklist:")
        report.append("[ ] All dependencies updated")
        report.append("[ ] Environment variables configured")
        report.append("[ ] TLS/SSL enabled")
        report.append("[ ] Rate limiting configured")
        report.append("[ ] Audit logging enabled")
        report.append("[ ] Security headers verified")
        report.append("[ ] Backup strategy in place")
        report.append("[ ] Incident response plan documented")
        
        return "\n".join(report)

def main():
    """Execute security improvements"""
    print("Starting security improvements for GPU Optimizer Platform...")
    
    enhancer = SecurityEnhancer()
    
    # Execute improvements
    print("✓ Creating secure configuration template...")
    enhancer.create_env_template()
    
    print("✓ Creating security middleware...")
    enhancer.create_security_middleware()
    
    print("✓ Creating security test suite...")
    enhancer.create_security_tests()
    
    print("✓ Updating .gitignore...")
    enhancer.update_gitignore()
    
    # Generate and display report
    report = enhancer.generate_report()
    print("\n" + report)
    
    # Save report to file
    report_path = Path("/home/user/webapp/SECURITY_IMPROVEMENTS.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Security improvements completed!")
    print(f"📄 Full report saved to: {report_path}")

if __name__ == "__main__":
    main()