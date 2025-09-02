"""
Custom Exception Classes for Optimizer Trust Engine
====================================================

Comprehensive error handling with proper exception hierarchy.
"""

from typing import Optional, Dict, Any


class OptimizerException(Exception):
    """Base exception for all optimizer-related errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code or "OPT_ERROR"
        self.details = details or {}
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.error_code,
            "details": self.details
        }


class SceneAnalysisError(OptimizerException):
    """Raised when scene analysis fails"""
    
    def __init__(self, message: str, scene_id: Optional[str] = None):
        super().__init__(
            message, 
            error_code="SCENE_ANALYSIS_ERROR",
            details={"scene_id": scene_id}
        )


class PipelineRoutingError(OptimizerException):
    """Raised when pipeline routing fails"""
    
    def __init__(self, message: str, pipeline: Optional[str] = None):
        super().__init__(
            message,
            error_code="PIPELINE_ROUTING_ERROR", 
            details={"pipeline": pipeline}
        )


class VerificationError(OptimizerException):
    """Base exception for verification-related errors"""
    
    def __init__(self, message: str, verification_id: Optional[str] = None):
        super().__init__(
            message,
            error_code="VERIFICATION_ERROR",
            details={"verification_id": verification_id}
        )


class ConsensusFailureError(VerificationError):
    """Raised when consensus cannot be reached"""
    
    def __init__(self, message: str, consensus_score: float, required_score: float):
        super().__init__(message)
        self.error_code = "CONSENSUS_FAILURE"
        self.details.update({
            "consensus_score": consensus_score,
            "required_score": required_score
        })


class QualityThresholdError(VerificationError):
    """Raised when quality thresholds are not met"""
    
    def __init__(self, message: str, metrics: Dict[str, float], thresholds: Dict[str, float]):
        super().__init__(message)
        self.error_code = "QUALITY_THRESHOLD_ERROR"
        self.details.update({
            "metrics": metrics,
            "thresholds": thresholds
        })


class SchedulingError(OptimizerException):
    """Raised when job scheduling fails"""
    
    def __init__(self, message: str, job_id: Optional[str] = None):
        super().__init__(
            message,
            error_code="SCHEDULING_ERROR",
            details={"job_id": job_id}
        )


class ResourceAllocationError(OptimizerException):
    """Raised when resource allocation fails"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(
            message,
            error_code="RESOURCE_ALLOCATION_ERROR",
            details={"resource_type": resource_type}
        )


class ValidationError(OptimizerException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": str(value)}
        )


class ConfigurationError(OptimizerException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )