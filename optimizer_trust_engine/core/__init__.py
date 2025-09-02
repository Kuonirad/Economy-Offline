"""
Optimizer Trust Engine Core Module
===================================

Production-grade implementation with complete error handling,
type safety, and validation.

Copyright 2024 - Apache License 2.0
"""

__version__ = "2.1.0"
__author__ = "GPU Optimization Team"

from typing import TYPE_CHECKING

# Lazy imports for better performance
if TYPE_CHECKING:
    from .optimizer_core import OptimizerCore
    from .trust_engine_core import TrustEngineCore
    from .economy_offline_core import EconomyOfflineCore
    from .models import *
    from .exceptions import *
    from .utils import *

__all__ = [
    "OptimizerCore",
    "TrustEngineCore", 
    "EconomyOfflineCore",
]

def get_version() -> str:
    """Get the current version of the package"""
    return __version__