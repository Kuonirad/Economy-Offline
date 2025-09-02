"""
Optimizer and Trust Engine Framework
=====================================

This module implements the core architecture for the GPU Optimization System:
- Optimizer: The Authoring Plugin that optimizes rendering workflows
- Trust Engine: The Verification Pipeline ensuring quality and integrity
- Economy Offline: Time-shifted processing for cost optimization
"""

from .optimizer import OptimizerCore, SceneAnalyzer, OptimizationRouter
from .trust_engine import TrustEngine, ProbabilisticVerifier, ConsensusValidator
from .economy_offline import EconomyOfflineBeach, TimeShiftedPipeline, CostOptimizer

__version__ = "2.1.0"
__all__ = [
    "OptimizerCore",
    "SceneAnalyzer", 
    "OptimizationRouter",
    "TrustEngine",
    "ProbabilisticVerifier",
    "ConsensusValidator",
    "EconomyOfflineBeach",
    "TimeShiftedPipeline",
    "CostOptimizer"
]