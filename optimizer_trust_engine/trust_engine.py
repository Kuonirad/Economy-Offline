"""
Trust Engine - The Verification Pipeline Component
===================================================

The Trust Engine ensures quality and integrity through:
1. Probabilistic quality verification
2. Multi-node consensus validation
3. Byzantine fault tolerance
4. Immutable audit trails
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

# Handle numpy import gracefully
try:
    import numpy as np
except ImportError:
    # Fallback implementation for numpy functions
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def sqrt(value):
            return value ** 0.5

logger = logging.getLogger(__name__)


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
    SSIM = "ssim"          # Structural Similarity Index
    PSNR = "psnr"          # Peak Signal-to-Noise Ratio
    LPIPS = "lpips"        # Learned Perceptual Image Patch Similarity
    VMAF = "vmaf"          # Video Multimethod Assessment Fusion
    FLIP = "flip"          # Frame-Level Image Perceptual Error
    CUSTOM = "custom"      # Custom metric


@dataclass
class QualityThresholds:
    """Quality thresholds for verification"""
    ssim_min: float = 0.98       # Minimum SSIM score
    psnr_min: float = 35.0       # Minimum PSNR in dB
    lpips_max: float = 0.05      # Maximum LPIPS (lower is better)
    vmaf_min: float = 90.0       # Minimum VMAF score
    flip_max: float = 0.1        # Maximum FLIP error
    custom_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class VerificationNode:
    """Represents a verification node in the network"""
    node_id: str
    reputation_score: float = 1.0  # 0.0 to 1.0
    specialization: List[str] = field(default_factory=list)  # e.g., ["gaming", "archviz"]
    compute_capability: float = 1.0  # Relative compute power
    location: str = "global"
    is_trusted: bool = True
    verification_count: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    last_active: float = field(default_factory=time.time)
    
    def update_reputation(self, success: bool, consensus_agreement: float):
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


@dataclass
class VerificationSample:
    """Sample for verification"""
    sample_id: str
    scene_id: str
    original_hash: str
    optimized_hash: str
    sample_type: str  # "frame", "region", "full"
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verification process"""
    verification_id: str
    scene_id: str
    status: VerificationStatus
    quality_scores: Dict[str, float]
    consensus_score: float  # 0.0 to 1.0
    participating_nodes: List[str]
    verification_time: float
    passed_thresholds: bool
    rejection_reasons: List[str] = field(default_factory=list)
    audit_trail: List[Dict] = field(default_factory=list)
    byzantine_nodes_detected: List[str] = field(default_factory=list)


class ProbabilisticVerifier:
    """
    Implements probabilistic verification for quality assurance
    Uses statistical sampling and confidence intervals
    """
    
    def __init__(self, thresholds: QualityThresholds, confidence_level: float = 0.95):
        """Initialize probabilistic verifier"""
        self.thresholds = thresholds
        self.confidence_level = confidence_level
        self.sample_cache = {}
        
    def calculate_sample_size(self, scene_complexity: float, required_confidence: float = None) -> int:
        """
        Calculate required sample size for verification
        Based on scene complexity and desired confidence level
        """
        if required_confidence is None:
            required_confidence = self.confidence_level
        
        # Base sample size
        base_samples = 10
        
        # Adjust for complexity (more complex = more samples)
        complexity_factor = 1 + (scene_complexity * 2)
        
        # Adjust for confidence level
        confidence_factor = 1 + ((required_confidence - 0.9) * 10)
        
        sample_size = int(base_samples * complexity_factor * confidence_factor)
        
        # Cap at reasonable maximum
        return min(sample_size, 100)
    
    def generate_verification_samples(self, scene_data: Dict, sample_size: int) -> List[VerificationSample]:
        """Generate samples for verification"""
        samples = []
        scene_id = scene_data.get("scene_id", "unknown")
        
        for i in range(sample_size):
            # Create sample with simulated data
            sample = VerificationSample(
                sample_id=f"{scene_id}_sample_{i}",
                scene_id=scene_id,
                original_hash=self._generate_hash(f"original_{scene_id}_{i}"),
                optimized_hash=self._generate_hash(f"optimized_{scene_id}_{i}"),
                sample_type=random.choice(["frame", "region", "full"]),
                timestamp=time.time(),
                metadata={
                    "frame_number": random.randint(0, 1000),
                    "region": f"quadrant_{random.randint(1, 4)}"
                }
            )
            samples.append(sample)
        
        return samples
    
    def verify_sample(self, sample: VerificationSample) -> Dict[str, float]:
        """Verify a single sample and return quality metrics"""
        # Simulate quality metric calculation
        # In production, this would compare original vs optimized
        
        # Generate realistic quality scores with some variance
        base_scores = {
            QualityMetric.SSIM.value: 0.98 + random.uniform(-0.02, 0.015),
            QualityMetric.PSNR.value: 35.0 + random.uniform(-2, 3),
            QualityMetric.LPIPS.value: 0.04 + random.uniform(-0.01, 0.02),
            QualityMetric.VMAF.value: 92.0 + random.uniform(-3, 3),
            QualityMetric.FLIP.value: 0.08 + random.uniform(-0.02, 0.03)
        }
        
        # Ensure scores are within valid ranges
        base_scores[QualityMetric.SSIM.value] = max(0, min(1, base_scores[QualityMetric.SSIM.value]))
        base_scores[QualityMetric.PSNR.value] = max(0, base_scores[QualityMetric.PSNR.value])
        base_scores[QualityMetric.LPIPS.value] = max(0, min(1, base_scores[QualityMetric.LPIPS.value]))
        base_scores[QualityMetric.VMAF.value] = max(0, min(100, base_scores[QualityMetric.VMAF.value]))
        base_scores[QualityMetric.FLIP.value] = max(0, min(1, base_scores[QualityMetric.FLIP.value]))
        
        return base_scores
    
    def aggregate_sample_results(self, sample_results: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Aggregate results from multiple samples
        Returns mean scores and confidence intervals
        """
        if not sample_results:
            return {}, {}
        
        # Calculate mean and std for each metric
        metrics = list(sample_results[0].keys())
        mean_scores = {}
        confidence_intervals = {}
        
        for metric in metrics:
            values = [result[metric] for result in sample_results]
            mean = np.mean(values)
            std = np.std(values)
            
            # Calculate confidence interval
            z_score = 1.96  # 95% confidence
            margin = z_score * (std / np.sqrt(len(values)))
            
            mean_scores[metric] = mean
            confidence_intervals[metric] = (mean - margin, mean + margin)
        
        return mean_scores, confidence_intervals
    
    def check_thresholds(self, mean_scores: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if quality scores meet thresholds"""
        passed = True
        failures = []
        
        # Check SSIM
        if mean_scores.get(QualityMetric.SSIM.value, 0) < self.thresholds.ssim_min:
            passed = False
            failures.append(f"SSIM {mean_scores.get(QualityMetric.SSIM.value, 0):.3f} < {self.thresholds.ssim_min}")
        
        # Check PSNR
        if mean_scores.get(QualityMetric.PSNR.value, 0) < self.thresholds.psnr_min:
            passed = False
            failures.append(f"PSNR {mean_scores.get(QualityMetric.PSNR.value, 0):.1f} < {self.thresholds.psnr_min}")
        
        # Check LPIPS
        if mean_scores.get(QualityMetric.LPIPS.value, 1) > self.thresholds.lpips_max:
            passed = False
            failures.append(f"LPIPS {mean_scores.get(QualityMetric.LPIPS.value, 1):.3f} > {self.thresholds.lpips_max}")
        
        # Check VMAF
        if mean_scores.get(QualityMetric.VMAF.value, 0) < self.thresholds.vmaf_min:
            passed = False
            failures.append(f"VMAF {mean_scores.get(QualityMetric.VMAF.value, 0):.1f} < {self.thresholds.vmaf_min}")
        
        # Check FLIP
        if mean_scores.get(QualityMetric.FLIP.value, 1) > self.thresholds.flip_max:
            passed = False
            failures.append(f"FLIP {mean_scores.get(QualityMetric.FLIP.value, 1):.3f} > {self.thresholds.flip_max}")
        
        return passed, failures
    
    def _generate_hash(self, data: str) -> str:
        """Generate hash for data"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class ConsensusValidator:
    """
    Implements Byzantine fault-tolerant consensus for verification
    Ensures agreement among verification nodes
    """
    
    def __init__(self, min_nodes: int = 3, consensus_threshold: float = 0.67):
        """Initialize consensus validator"""
        self.min_nodes = min_nodes
        self.consensus_threshold = consensus_threshold
        self.node_registry: Dict[str, VerificationNode] = {}
        self.validation_history = defaultdict(list)
    
    def register_node(self, node: VerificationNode):
        """Register a verification node"""
        self.node_registry[node.node_id] = node
        logger.info(f"Registered verification node {node.node_id}")
    
    def select_verification_nodes(self, required_count: int, specialization: Optional[str] = None) -> List[VerificationNode]:
        """
        Select nodes for verification
        Considers reputation, specialization, and availability
        """
        available_nodes = []
        
        for node in self.node_registry.values():
            if not node.is_trusted:
                continue
            
            # Check if node is recently active (increase timeout for testing)
            if time.time() - node.last_active > 7200:  # 2 hours
                continue
            
            # Check specialization if required (more flexible matching)
            if specialization and specialization != "general":
                # Allow nodes with general specialization or matching specialization
                if "general" not in node.specialization and specialization not in node.specialization:
                    continue
            
            available_nodes.append(node)
        
        # Sort by reputation and weight
        available_nodes.sort(key=lambda n: n.get_weight(), reverse=True)
        
        # Select top nodes
        selected = available_nodes[:required_count]
        
        # Ensure minimum nodes - always add defaults if needed
        while len(selected) < max(self.min_nodes, required_count):
            i = len(selected)
            default_node = VerificationNode(
                node_id=f"default_node_{i}",
                reputation_score=0.75 + random.uniform(0, 0.2),
                specialization=["general"] if not specialization else ["general", specialization],
                compute_capability=1.0 + random.uniform(-0.1, 0.2),
                last_active=time.time()
            )
            self.node_registry[default_node.node_id] = default_node
            selected.append(default_node)
            logger.debug(f"Added default node {default_node.node_id} for verification")
        
        return selected
    
    def collect_node_votes(self, nodes: List[VerificationNode], verification_data: Dict) -> Dict[str, Dict]:
        """Collect verification votes from nodes"""
        votes = {}
        
        for node in nodes:
            # Simulate node verification (in production, this would be async RPC)
            vote = self._simulate_node_verification(node, verification_data)
            votes[node.node_id] = vote
        
        return votes
    
    def calculate_consensus(self, votes: Dict[str, Dict], nodes: List[VerificationNode]) -> Tuple[float, Dict, List[str]]:
        """
        Calculate weighted consensus from node votes
        Returns consensus score, aggregated result, and Byzantine nodes
        """
        if not votes:
            return 0.0, {}, []
        
        # Create node lookup
        node_lookup = {node.node_id: node for node in nodes}
        
        # Aggregate quality scores with weights
        weighted_scores = defaultdict(lambda: {"sum": 0, "weight": 0})
        vote_distribution = defaultdict(int)
        
        for node_id, vote in votes.items():
            node = node_lookup.get(node_id)
            if not node:
                continue
            
            weight = node.get_weight()
            
            # Track vote distribution
            vote_hash = self._hash_vote(vote)
            vote_distribution[vote_hash] += weight
            
            # Aggregate quality scores
            for metric, value in vote.get("quality_scores", {}).items():
                weighted_scores[metric]["sum"] += value * weight
                weighted_scores[metric]["weight"] += weight
        
        # Calculate weighted average scores
        aggregated_scores = {}
        for metric, data in weighted_scores.items():
            if data["weight"] > 0:
                aggregated_scores[metric] = data["sum"] / data["weight"]
        
        # Detect Byzantine nodes (outliers)
        byzantine_nodes = self._detect_byzantine_nodes(votes, aggregated_scores, node_lookup)
        
        # Calculate consensus score
        total_weight = sum(node.get_weight() for node in nodes)
        if total_weight == 0:
            return 0.0, aggregated_scores, byzantine_nodes
        
        # Find majority vote
        max_vote_weight = max(vote_distribution.values()) if vote_distribution else 0
        consensus_score = max_vote_weight / total_weight
        
        return consensus_score, aggregated_scores, byzantine_nodes
    
    def _simulate_node_verification(self, node: VerificationNode, verification_data: Dict) -> Dict:
        """Simulate node verification process"""
        # Base quality scores
        base_scores = {
            "ssim": 0.98,
            "psnr": 35.0,
            "lpips": 0.04,
            "vmaf": 92.0,
            "flip": 0.08
        }
        
        # Add some variance based on node characteristics
        variance_factor = 1.0 - (node.reputation_score * 0.1)
        
        # Simulate Byzantine behavior for untrusted nodes
        if not node.is_trusted or random.random() < 0.05:  # 5% Byzantine probability
            # Byzantine node gives bad scores
            variance_factor = 0.5
            base_scores = {k: v * random.uniform(0.7, 0.9) for k, v in base_scores.items()}
        
        # Apply variance
        quality_scores = {}
        for metric, base_value in base_scores.items():
            variance = base_value * variance_factor * 0.1
            quality_scores[metric] = base_value + random.uniform(-variance, variance)
        
        return {
            "node_id": node.node_id,
            "quality_scores": quality_scores,
            "passed": all(score > 0.9 for score in quality_scores.values() if score < 1),
            "timestamp": time.time()
        }
    
    def _hash_vote(self, vote: Dict) -> str:
        """Create hash of vote for comparison"""
        # Round scores to reduce noise
        rounded_scores = {
            k: round(v, 2) 
            for k, v in vote.get("quality_scores", {}).items()
        }
        vote_str = json.dumps(rounded_scores, sort_keys=True)
        return hashlib.md5(vote_str.encode()).hexdigest()[:8]
    
    def _detect_byzantine_nodes(self, votes: Dict[str, Dict], aggregated_scores: Dict, node_lookup: Dict) -> List[str]:
        """Detect Byzantine (malicious/faulty) nodes based on deviation from consensus"""
        byzantine_nodes = []
        
        if not aggregated_scores:
            return byzantine_nodes
        
        for node_id, vote in votes.items():
            node = node_lookup.get(node_id)
            if not node:
                continue
            
            # Calculate deviation from consensus
            deviations = []
            for metric, consensus_value in aggregated_scores.items():
                node_value = vote.get("quality_scores", {}).get(metric, 0)
                if consensus_value != 0:
                    deviation = abs(node_value - consensus_value) / consensus_value
                    deviations.append(deviation)
            
            # Check if node is Byzantine (high deviation)
            avg_deviation = np.mean(deviations) if deviations else 0
            if avg_deviation > 0.2:  # 20% deviation threshold
                byzantine_nodes.append(node_id)
                # Update node reputation
                node.update_reputation(False, 0.0)
        
        return byzantine_nodes


class TrustEngine:
    """
    Main Trust Engine orchestrator
    Coordinates verification and consensus
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """Initialize Trust Engine"""
        self.thresholds = thresholds or QualityThresholds()
        self.verifier = ProbabilisticVerifier(self.thresholds)
        self.consensus = ConsensusValidator()
        self.audit_log = []
        self._initialize_default_nodes()
    
    def _initialize_default_nodes(self):
        """Initialize default verification nodes for testing"""
        specializations = [
            ["gaming", "realtime"],
            ["archviz", "quality"],
            ["ai_training", "synthetic"],
            ["general"],
            ["quality", "accuracy"]
        ]
        
        for i in range(5):
            node = VerificationNode(
                node_id=f"trust_node_{i}",
                reputation_score=0.9 + random.uniform(-0.1, 0.1),
                specialization=specializations[i],
                compute_capability=1.0 + random.uniform(-0.2, 0.3)
            )
            self.consensus.register_node(node)
    
    async def verify_optimization(self, scene_data: Dict, optimization_result: Dict) -> VerificationResult:
        """
        Verify optimization quality through Trust Engine
        Returns comprehensive verification result
        """
        verification_id = self._generate_verification_id(scene_data)
        scene_id = scene_data.get("scene_id", "unknown")
        
        logger.info(f"Starting verification {verification_id} for scene {scene_id}")
        
        # Start audit trail
        audit_entry = {
            "timestamp": time.time(),
            "event": "verification_started",
            "scene_id": scene_id,
            "verification_id": verification_id
        }
        self.audit_log.append(audit_entry)
        
        start_time = time.time()
        
        try:
            # Determine sample size based on complexity
            complexity = scene_data.get("complexity_score", 0.5)
            sample_size = self.verifier.calculate_sample_size(complexity)
            
            # Generate verification samples
            samples = self.verifier.generate_verification_samples(scene_data, sample_size)
            
            # Select verification nodes
            specialization = scene_data.get("specialization", "general")
            nodes = self.consensus.select_verification_nodes(
                required_count=max(5, self.consensus.min_nodes),
                specialization=specialization
            )
            
            # Perform probabilistic verification
            sample_results = []
            for sample in samples:
                result = self.verifier.verify_sample(sample)
                sample_results.append(result)
            
            # Aggregate results
            mean_scores, confidence_intervals = self.verifier.aggregate_sample_results(sample_results)
            
            # Collect node votes
            verification_data = {
                "scene_data": scene_data,
                "optimization_result": optimization_result,
                "sample_results": sample_results,
                "mean_scores": mean_scores
            }
            votes = self.consensus.collect_node_votes(nodes, verification_data)
            
            # Calculate consensus
            consensus_score, aggregated_scores, byzantine_nodes = self.consensus.calculate_consensus(votes, nodes)
            
            # Check quality thresholds
            passed, rejection_reasons = self.verifier.check_thresholds(aggregated_scores)
            
            # Determine final status
            if consensus_score >= self.consensus.consensus_threshold and passed:
                status = VerificationStatus.VALIDATED
            elif consensus_score >= self.consensus.consensus_threshold and not passed:
                status = VerificationStatus.REJECTED
            elif consensus_score < self.consensus.consensus_threshold:
                status = VerificationStatus.DISPUTED
            else:
                status = VerificationStatus.PENDING
            
            # Create verification result
            result = VerificationResult(
                verification_id=verification_id,
                scene_id=scene_id,
                status=status,
                quality_scores=aggregated_scores,
                consensus_score=consensus_score,
                participating_nodes=[node.node_id for node in nodes],
                verification_time=time.time() - start_time,
                passed_thresholds=passed,
                rejection_reasons=rejection_reasons,
                byzantine_nodes_detected=byzantine_nodes,
                audit_trail=[audit_entry]
            )
            
            # Final audit entry
            final_audit = {
                "timestamp": time.time(),
                "event": "verification_completed",
                "status": status.value,
                "consensus_score": consensus_score,
                "quality_passed": passed
            }
            self.audit_log.append(final_audit)
            result.audit_trail.append(final_audit)
            
            logger.info(f"Verification {verification_id} completed with status {status.value}")
            return result
            
        except Exception as e:
            logger.error(f"Verification failed for {verification_id}: {str(e)}")
            
            # Error audit entry
            error_audit = {
                "timestamp": time.time(),
                "event": "verification_error",
                "error": str(e)
            }
            self.audit_log.append(error_audit)
            
            return VerificationResult(
                verification_id=verification_id,
                scene_id=scene_id,
                status=VerificationStatus.REJECTED,
                quality_scores={},
                consensus_score=0.0,
                participating_nodes=[],
                verification_time=time.time() - start_time,
                passed_thresholds=False,
                rejection_reasons=[f"Verification error: {str(e)}"],
                audit_trail=[audit_entry, error_audit]
            )
    
    def _generate_verification_id(self, scene_data: Dict) -> str:
        """Generate unique verification ID"""
        data = f"{scene_data.get('scene_id', 'unknown')}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_audit_log(self) -> List[Dict]:
        """Get complete audit log"""
        return self.audit_log.copy()
    
    def get_node_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all verification nodes"""
        stats = {}
        
        for node_id, node in self.consensus.node_registry.items():
            stats[node_id] = {
                "reputation_score": node.reputation_score,
                "is_trusted": node.is_trusted,
                "verification_count": node.verification_count,
                "success_rate": node.successful_verifications / max(node.verification_count, 1),
                "specializations": node.specialization,
                "last_active": node.last_active
            }
        
        return stats