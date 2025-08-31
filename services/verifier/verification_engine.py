"""
Verification Engine for Trust Engine
Implements statistical verification with GPU acceleration
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import torch
import numpy as np
from PIL import Image
import cv2

from metrics_calculator import MetricsCalculator
from models import VerificationResult, QualityMetrics, ConsensusResult

logger = logging.getLogger(__name__)

@dataclass
class VerificationConfig:
    """Configuration for verification engine"""
    ssim_threshold: float = 0.98
    psnr_threshold: float = 35.0
    lpips_threshold: float = 0.05
    consensus_threshold: float = 0.66  # 2/3 agreement
    redundancy_factor: int = 2
    canary_rate: float = 0.1
    
class VerificationEngine:
    """Main verification engine with GPU acceleration"""
    
    def __init__(self, device: torch.device, config: Optional[VerificationConfig] = None):
        self.device = device
        self.config = config or VerificationConfig()
        self.metrics_calculator = MetricsCalculator(device)
        self.verification_cache = {}
        self.statistics = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "total_ssim": 0,
            "total_psnr": 0,
            "total_time_ms": 0,
            "consensus_achieved": 0,
            "consensus_attempts": 0
        }
        
        logger.info(f"Verification engine initialized on {device}")
        logger.info(f"Config: SSIM={self.config.ssim_threshold}, "
                   f"PSNR={self.config.psnr_threshold}, "
                   f"LPIPS={self.config.lpips_threshold}")
    
    async def verify(
        self,
        job_id: str,
        shard_id: str,
        node_id: str,
        result_data: Dict,
        is_canary: bool = False
    ) -> VerificationResult:
        """Verify a work submission"""
        
        start_time = time.time()
        self.statistics["total"] += 1
        
        try:
            # Extract result data
            output_url = result_data.get("outputUrl")
            reference_url = result_data.get("referenceUrl")
            
            if not output_url:
                raise ValueError("No output URL provided")
            
            # Load images (in production, these would be fetched from URLs)
            output_image = await self._load_image(output_url)
            reference_image = await self._load_image(reference_url) if reference_url else None
            
            # Calculate quality metrics
            quality_metrics = None
            if reference_image is not None:
                quality_metrics = await self._calculate_metrics(output_image, reference_image)
                
                # Update statistics
                self.statistics["total_ssim"] += quality_metrics.ssim
                self.statistics["total_psnr"] += quality_metrics.psnr
            
            # Check if metrics meet thresholds
            passed = self._check_thresholds(quality_metrics) if quality_metrics else True
            
            # Store result for consensus checking
            cache_key = f"{job_id}:{shard_id}:{node_id}"
            self.verification_cache[cache_key] = {
                "quality_metrics": quality_metrics,
                "passed": passed,
                "timestamp": time.time(),
                "is_canary": is_canary
            }
            
            # Check for consensus if this is a redundant computation
            consensus_achieved = False
            if not is_canary:
                consensus_result = await self.check_consensus(job_id, shard_id)
                consensus_achieved = consensus_result.achieved
            
            if passed:
                self.statistics["successful"] += 1
            else:
                self.statistics["failed"] += 1
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            self.statistics["total_time_ms"] += processing_time_ms
            
            logger.info(f"Verification completed for {job_id}:{shard_id} "
                       f"(passed={passed}, consensus={consensus_achieved}, "
                       f"time={processing_time_ms:.2f}ms)")
            
            return VerificationResult(
                job_id=job_id,
                shard_id=shard_id,
                node_id=node_id,
                quality_metrics=quality_metrics,
                passed=passed,
                consensus_achieved=consensus_achieved,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Verification failed for {job_id}:{shard_id}: {e}")
            self.statistics["failed"] += 1
            raise
    
    async def check_consensus(self, job_id: str, shard_id: str) -> ConsensusResult:
        """Check if consensus has been achieved for a shard"""
        
        self.statistics["consensus_attempts"] += 1
        
        # Find all verifications for this shard
        shard_results = []
        for key, value in self.verification_cache.items():
            if key.startswith(f"{job_id}:{shard_id}:"):
                shard_results.append(value)
        
        if len(shard_results) < self.config.redundancy_factor:
            # Not enough results yet
            return ConsensusResult(
                achieved=False,
                agreement_rate=0,
                participating_nodes=len(shard_results),
                required_nodes=self.config.redundancy_factor
            )
        
        # Calculate agreement
        passed_count = sum(1 for r in shard_results if r["passed"])
        agreement_rate = passed_count / len(shard_results)
        
        # Check if consensus threshold is met
        consensus_achieved = agreement_rate >= self.config.consensus_threshold
        
        if consensus_achieved:
            self.statistics["consensus_achieved"] += 1
        
        # Aggregate metrics
        aggregated_metrics = None
        metrics_list = [r["quality_metrics"] for r in shard_results 
                       if r["quality_metrics"] is not None]
        
        if metrics_list:
            aggregated_metrics = QualityMetrics(
                ssim=np.mean([m.ssim for m in metrics_list]),
                psnr=np.mean([m.psnr for m in metrics_list]),
                lpips=np.mean([m.lpips for m in metrics_list]) if metrics_list[0].lpips else None
            )
        
        logger.info(f"Consensus check for {job_id}:{shard_id}: "
                   f"achieved={consensus_achieved}, "
                   f"agreement={agreement_rate:.2%}, "
                   f"nodes={len(shard_results)}")
        
        return ConsensusResult(
            achieved=consensus_achieved,
            agreement_rate=agreement_rate,
            participating_nodes=len(shard_results),
            required_nodes=self.config.redundancy_factor,
            aggregated_metrics=aggregated_metrics
        )
    
    async def _load_image(self, url: str) -> torch.Tensor:
        """Load image from URL or file path"""
        # In production, this would fetch from actual URL
        # For M0, we'll create a dummy image
        
        # Simulate async image loading
        await asyncio.sleep(0.01)
        
        # Create dummy image for testing
        image = torch.rand(3, 1080, 1920).to(self.device)
        return image
    
    async def _calculate_metrics(
        self,
        output: torch.Tensor,
        reference: torch.Tensor
    ) -> QualityMetrics:
        """Calculate quality metrics between output and reference"""
        
        # Ensure images are the same size
        if output.shape != reference.shape:
            # Resize to match reference
            output = torch.nn.functional.interpolate(
                output.unsqueeze(0),
                size=reference.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Calculate metrics
        ssim = await self.metrics_calculator.calculate_ssim(output, reference)
        psnr = await self.metrics_calculator.calculate_psnr(output, reference)
        lpips = await self.metrics_calculator.calculate_lpips(output, reference)
        
        return QualityMetrics(
            ssim=float(ssim),
            psnr=float(psnr),
            lpips=float(lpips) if lpips is not None else None
        )
    
    def _check_thresholds(self, metrics: QualityMetrics) -> bool:
        """Check if metrics meet quality thresholds"""
        
        if metrics.ssim < self.config.ssim_threshold:
            logger.warning(f"SSIM {metrics.ssim:.4f} below threshold {self.config.ssim_threshold}")
            return False
        
        if metrics.psnr < self.config.psnr_threshold:
            logger.warning(f"PSNR {metrics.psnr:.2f} below threshold {self.config.psnr_threshold}")
            return False
        
        if metrics.lpips is not None and metrics.lpips > self.config.lpips_threshold:
            logger.warning(f"LPIPS {metrics.lpips:.4f} above threshold {self.config.lpips_threshold}")
            return False
        
        return True
    
    async def get_statistics(self) -> Dict:
        """Get verification statistics"""
        
        total = self.statistics["total"]
        if total == 0:
            return self.statistics
        
        return {
            "total": total,
            "successful": self.statistics["successful"],
            "failed": self.statistics["failed"],
            "success_rate": self.statistics["successful"] / total,
            "avg_ssim": self.statistics["total_ssim"] / total if total > 0 else 0,
            "avg_psnr": self.statistics["total_psnr"] / total if total > 0 else 0,
            "avg_time_ms": self.statistics["total_time_ms"] / total if total > 0 else 0,
            "consensus_rate": (self.statistics["consensus_achieved"] / 
                             self.statistics["consensus_attempts"] 
                             if self.statistics["consensus_attempts"] > 0 else 0)
        }
    
    def clear_cache(self, max_age_seconds: int = 3600):
        """Clear old entries from verification cache"""
        
        current_time = time.time()
        keys_to_remove = []
        
        for key, value in self.verification_cache.items():
            if current_time - value["timestamp"] > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.verification_cache[key]
        
        if keys_to_remove:
            logger.info(f"Cleared {len(keys_to_remove)} old cache entries")