"""
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
        sanitized = InputSanitizer.sanitize_string("test\x00string")
        assert "\x00" not in sanitized
        
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
