"""
Data Models for Optimizer Trust Engine
=======================================

Properly validated data models with type safety.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import hashlib
import json
from datetime import datetime

from .exceptions import ValidationError


class SceneComplexity(Enum):
    """Scene complexity classification"""
    SIMPLE = "simple"        # < 100K triangles
    MODERATE = "moderate"    # 100K-1M triangles  
    COMPLEX = "complex"      # 1M-10M triangles
    EXTREME = "extreme"      # > 10M triangles


class OptimizationProfile(Enum):
    """Optimization profiles for different use cases"""
    QUALITY = "quality"          # Maximum quality
    BALANCED = "balanced"        # Balance quality/cost
    ECONOMY = "economy"          # Minimum cost
    REALTIME = "realtime"        # Real-time rendering
    ARCHVIZ = "archviz"          # Architectural visualization
    GAMING = "gaming"            # Game asset optimization
    AI_TRAINING = "ai_training"  # Synthetic data generation


class PipelineType(Enum):
    """Available optimization pipelines"""
    BAKING = "baking"
    THREEJS = "3dgs"
    HYBRID = "hybrid"


class VerificationStatus(Enum):
    """Status of verification process"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DISPUTED = "disputed"
    CONSENSUS_REACHED = "consensus_reached"


class QualityMetric(Enum):
    """Quality metrics for verification"""
    SSIM = "ssim"
    PSNR = "psnr"
    LPIPS = "lpips"
    VMAF = "vmaf"
    FLIP = "flip"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Types of GPU resources"""
    CONSUMER = "consumer"
    DATACENTER = "datacenter"
    CLOUD = "cloud"
    EDGE = "edge"
    DISTRIBUTED = "distributed"


class PricingTier(Enum):
    """Pricing tiers for processing"""
    PEAK = "peak"
    STANDARD = "standard"
    OFF_PEAK = "off_peak"
    ECONOMY = "economy"
    SPOT = "spot"


@dataclass
class BaseModel:
    """Base class for all data models"""
    
    def validate(self) -> None:
        """Validate the model instance"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert model to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from dictionary"""
        return cls(**data)


@dataclass
class SceneMetadata(BaseModel):
    """Metadata for scene analysis"""
    scene_id: str
    source_application: str
    polygon_count: int
    texture_count: int
    material_count: int
    light_count: int
    animation_frames: int
    resolution: Tuple[int, int]
    file_size_mb: float
    complexity: SceneComplexity
    has_transparency: bool = False
    has_volumetrics: bool = False
    has_dynamic_elements: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation"""
        self.validate()
    
    def validate(self) -> None:
        """Validate scene metadata"""
        if not self.scene_id:
            raise ValidationError("Scene ID cannot be empty", field="scene_id")
        
        if self.polygon_count < 0:
            raise ValidationError("Polygon count cannot be negative", 
                                field="polygon_count", value=self.polygon_count)
        
        if self.file_size_mb < 0:
            raise ValidationError("File size cannot be negative",
                                field="file_size_mb", value=self.file_size_mb)
        
        if len(self.resolution) != 2:
            raise ValidationError("Resolution must be a tuple of (width, height)",
                                field="resolution", value=self.resolution)
        
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValidationError("Resolution dimensions must be positive",
                                field="resolution", value=self.resolution)
    
    def calculate_hash(self) -> str:
        """Calculate unique hash for scene state"""
        data = f"{self.scene_id}{self.polygon_count}{self.material_count}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_complexity_score(self) -> float:
        """Calculate numerical complexity score (0.0 to 1.0)"""
        scores = {
            SceneComplexity.SIMPLE: 0.25,
            SceneComplexity.MODERATE: 0.5,
            SceneComplexity.COMPLEX: 0.75,
            SceneComplexity.EXTREME: 1.0
        }
        return scores.get(self.complexity, 0.5)


@dataclass
class OptimizationRequest(BaseModel):
    """Request for scene optimization"""
    scene_metadata: SceneMetadata
    profile: OptimizationProfile
    target_quality: float  # 0.0 to 1.0
    max_processing_time: Optional[float] = None  # seconds
    max_cost: Optional[float] = None  # USD
    priority: int = 5  # 1-10
    custom_params: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: "")
    
    def __post_init__(self):
        """Generate request ID if not provided"""
        if not self.request_id:
            self.request_id = self._generate_request_id()
        self.validate()
    
    def validate(self) -> None:
        """Validate optimization request"""
        if not 0.0 <= self.target_quality <= 1.0:
            raise ValidationError("Target quality must be between 0.0 and 1.0",
                                field="target_quality", value=self.target_quality)
        
        if not 1 <= self.priority <= 10:
            raise ValidationError("Priority must be between 1 and 10",
                                field="priority", value=self.priority)
        
        if self.max_processing_time is not None and self.max_processing_time <= 0:
            raise ValidationError("Max processing time must be positive",
                                field="max_processing_time", value=self.max_processing_time)
        
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValidationError("Max cost must be positive",
                                field="max_cost", value=self.max_cost)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        data = f"{self.scene_metadata.scene_id}{time.time()}{self.priority}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


@dataclass
class OptimizationResult(BaseModel):
    """Result of optimization process"""
    request_id: str
    scene_id: str
    success: bool
    optimization_path: str
    processing_time: float  # seconds
    estimated_cost: float  # USD
    quality_metrics: Dict[str, float]
    file_size_reduction: float  # percentage
    performance_gain: float  # percentage
    artifacts_path: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate optimization result"""
        if self.processing_time < 0:
            raise ValidationError("Processing time cannot be negative",
                                field="processing_time", value=self.processing_time)
        
        if self.estimated_cost < 0:
            raise ValidationError("Estimated cost cannot be negative",
                                field="estimated_cost", value=self.estimated_cost)
        
        if not 0 <= self.file_size_reduction <= 100:
            raise ValidationError("File size reduction must be between 0 and 100",
                                field="file_size_reduction", value=self.file_size_reduction)
    
    def get_efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        # Weighted combination of metrics
        cost_weight = 0.3
        time_weight = 0.2
        quality_weight = 0.3
        size_weight = 0.2
        
        # Normalize metrics
        cost_score = max(0, 1 - (self.estimated_cost / 10))  # Assume $10 is max
        time_score = max(0, 1 - (self.processing_time / 3600))  # Assume 1 hour is max
        quality_score = sum(self.quality_metrics.values()) / len(self.quality_metrics) if self.quality_metrics else 0
        size_score = self.file_size_reduction / 100
        
        return (cost_score * cost_weight + 
                time_score * time_weight + 
                quality_score * quality_weight + 
                size_score * size_weight)


@dataclass
class QualityThresholds(BaseModel):
    """Quality thresholds for verification"""
    ssim_min: float = 0.98
    psnr_min: float = 35.0
    lpips_max: float = 0.05
    vmaf_min: float = 90.0
    flip_max: float = 0.1
    custom_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate quality thresholds"""
        if not 0 <= self.ssim_min <= 1:
            raise ValidationError("SSIM threshold must be between 0 and 1",
                                field="ssim_min", value=self.ssim_min)
        
        if self.psnr_min < 0:
            raise ValidationError("PSNR threshold cannot be negative",
                                field="psnr_min", value=self.psnr_min)
        
        if not 0 <= self.lpips_max <= 1:
            raise ValidationError("LPIPS threshold must be between 0 and 1",
                                field="lpips_max", value=self.lpips_max)
        
        if not 0 <= self.vmaf_min <= 100:
            raise ValidationError("VMAF threshold must be between 0 and 100",
                                field="vmaf_min", value=self.vmaf_min)
    
    def check_metrics(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if metrics meet thresholds"""
        passed = True
        failures = []
        
        if metrics.get("ssim", 0) < self.ssim_min:
            passed = False
            failures.append(f"SSIM {metrics.get('ssim', 0):.3f} < {self.ssim_min}")
        
        if metrics.get("psnr", 0) < self.psnr_min:
            passed = False
            failures.append(f"PSNR {metrics.get('psnr', 0):.1f} < {self.psnr_min}")
        
        if metrics.get("lpips", 1) > self.lpips_max:
            passed = False
            failures.append(f"LPIPS {metrics.get('lpips', 1):.3f} > {self.lpips_max}")
        
        if metrics.get("vmaf", 0) < self.vmaf_min:
            passed = False
            failures.append(f"VMAF {metrics.get('vmaf', 0):.1f} < {self.vmaf_min}")
        
        if metrics.get("flip", 1) > self.flip_max:
            passed = False
            failures.append(f"FLIP {metrics.get('flip', 1):.3f} > {self.flip_max}")
        
        # Check custom thresholds
        for metric, threshold in self.custom_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                passed = False
                failures.append(f"{metric} {metrics[metric]:.3f} < {threshold}")
        
        return passed, failures


@dataclass
class VerificationNode(BaseModel):
    """Represents a verification node in the network"""
    node_id: str
    reputation_score: float = 1.0
    specialization: List[str] = field(default_factory=list)
    compute_capability: float = 1.0
    location: str = "global"
    is_trusted: bool = True
    verification_count: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate verification node"""
        if not self.node_id:
            raise ValidationError("Node ID cannot be empty", field="node_id")
        
        if not 0 <= self.reputation_score <= 1:
            raise ValidationError("Reputation score must be between 0 and 1",
                                field="reputation_score", value=self.reputation_score)
        
        if self.compute_capability <= 0:
            raise ValidationError("Compute capability must be positive",
                                field="compute_capability", value=self.compute_capability)
    
    def update_reputation(self, success: bool, consensus_agreement: float = 1.0) -> None:
        """Update node reputation based on verification outcome"""
        self.verification_count += 1
        
        if success:
            self.successful_verifications += 1
            # Increase reputation based on consensus agreement
            self.reputation_score = min(1.0, self.reputation_score + 0.01 * consensus_agreement)
        else:
            self.failed_verifications += 1
            # Decrease reputation for failures
            self.reputation_score = max(0.0, self.reputation_score - 0.05)
        
        # Apply trust penalty for consistently bad behavior
        if self.verification_count > 10:
            success_rate = self.successful_verifications / self.verification_count
            if success_rate < 0.5:
                self.is_trusted = False
        
        self.last_active = time.time()
    
    def get_weight(self) -> float:
        """Get node weight for consensus calculation"""
        if not self.is_trusted:
            return 0.0
        
        # Weight based on reputation and compute capability
        base_weight = self.reputation_score * self.compute_capability
        
        # Boost for specialized nodes
        specialization_boost = 1.0 + (0.1 * len(self.specialization))
        
        return base_weight * specialization_boost
    
    def get_success_rate(self) -> float:
        """Get verification success rate"""
        if self.verification_count == 0:
            return 0.0
        return self.successful_verifications / self.verification_count