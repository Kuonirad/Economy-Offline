"""
Data models for Trust Engine Verifier Service
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Pydantic models for API

class WorkSubmission(BaseModel):
    """Work submission for verification"""
    job_id: str
    shard_id: str
    node_id: str
    redundancy_id: int = 1
    is_canary: bool = False
    result_data: Dict[str, Any]
    
class QualityMetrics(BaseModel):
    """Quality metrics from verification"""
    ssim: float = Field(..., ge=0, le=1, description="Structural Similarity Index")
    psnr: float = Field(..., ge=0, description="Peak Signal-to-Noise Ratio in dB")
    lpips: Optional[float] = Field(None, ge=0, le=1, description="Learned Perceptual Image Patch Similarity")
    
# Dataclasses for internal use

@dataclass
class VerificationResult:
    """Result of verification process"""
    job_id: str
    shard_id: str
    node_id: str
    quality_metrics: Optional[QualityMetrics]
    passed: bool
    consensus_achieved: bool
    processing_time_ms: float
    
@dataclass
class ConsensusResult:
    """Result of consensus checking"""
    achieved: bool
    agreement_rate: float
    participating_nodes: int
    required_nodes: int
    aggregated_metrics: Optional[QualityMetrics] = None