"""
Trust Engine Verifier Service
GPU-accelerated perceptual metrics computation for quality verification
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import redis.asyncio as redis

from verification_engine import VerificationEngine
from metrics_calculator import MetricsCalculator
from models import WorkSubmission, VerificationResult, QualityMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()
verification_counter = Counter(
    'verifications_total', 
    'Total number of verifications',
    ['status'],
    registry=registry
)
verification_duration = Histogram(
    'verification_duration_seconds',
    'Verification processing time',
    registry=registry
)
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'Current GPU utilization',
    registry=registry
)

# Global instances
verification_engine: Optional[VerificationEngine] = None
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global verification_engine, redis_client
    
    logger.info("Starting Trust Engine Verifier Service")
    
    # Initialize GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available, using CPU (performance will be degraded)")
    
    # Initialize verification engine
    verification_engine = VerificationEngine(device=device)
    
    # Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    
    try:
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down Trust Engine Verifier Service")
    if redis_client:
        await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Trust Engine Verifier",
    description="GPU-accelerated verification service for WorldShare MVP",
    version="0.1.0-m0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class VerificationRequest(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    shard_id: str = Field(..., description="Shard identifier")
    node_id: str = Field(..., description="Node that processed the work")
    redundancy_id: int = Field(1, description="Which redundancy copy")
    is_canary: bool = Field(False, description="Is this a canary validation")
    result_data: Dict = Field(..., description="Processing results")
    
class VerificationResponse(BaseModel):
    verification_id: str
    status: str
    quality_metrics: Optional[QualityMetrics] = None
    consensus_achieved: bool = False
    message: str

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_info = None
    
    if gpu_available:
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
        }
    
    return {
        "status": "healthy",
        "service": "trust-engine-verifier",
        "version": "0.1.0-m0",
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "redis_connected": redis_client is not None
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if verification_engine is None:
        raise HTTPException(status_code=503, detail="Verification engine not initialized")
    
    return {"status": "ready"}

@app.post("/verify", response_model=VerificationResponse)
async def verify_work(
    request: VerificationRequest,
    background_tasks: BackgroundTasks
):
    """Submit work for verification"""
    
    if verification_engine is None:
        raise HTTPException(status_code=503, detail="Verification engine not available")
    
    verification_id = f"{request.job_id}-{request.shard_id}-{request.redundancy_id}"
    
    try:
        # Start verification timer
        with verification_duration.time():
            # Perform verification
            result = await verification_engine.verify(
                job_id=request.job_id,
                shard_id=request.shard_id,
                node_id=request.node_id,
                result_data=request.result_data,
                is_canary=request.is_canary
            )
        
        # Update metrics
        verification_counter.labels(status="success").inc()
        
        # Cache result if Redis available
        if redis_client:
            background_tasks.add_task(
                cache_verification_result,
                verification_id,
                result
            )
        
        return VerificationResponse(
            verification_id=verification_id,
            status="completed",
            quality_metrics=result.quality_metrics,
            consensus_achieved=result.consensus_achieved,
            message="Verification completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        verification_counter.labels(status="failed").inc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )

@app.get("/verify/{verification_id}")
async def get_verification_status(verification_id: str):
    """Get verification status by ID"""
    
    if redis_client:
        # Try to get from cache
        cached = await redis_client.get(f"verification:{verification_id}")
        if cached:
            import json
            return json.loads(cached)
    
    return {
        "verification_id": verification_id,
        "status": "not_found",
        "message": "Verification not found in cache"
    }

@app.post("/consensus")
async def check_consensus(job_id: str, shard_id: str):
    """Check if consensus has been achieved for a shard"""
    
    if verification_engine is None:
        raise HTTPException(status_code=503, detail="Verification engine not available")
    
    try:
        consensus = await verification_engine.check_consensus(job_id, shard_id)
        
        return {
            "job_id": job_id,
            "shard_id": shard_id,
            "consensus_achieved": consensus.achieved,
            "agreement_rate": consensus.agreement_rate,
            "participating_nodes": consensus.participating_nodes,
            "quality_metrics": consensus.aggregated_metrics
        }
        
    except Exception as e:
        logger.error(f"Consensus check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Consensus check failed: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    
    # Update GPU utilization if available
    if torch.cuda.is_available():
        gpu_util = torch.cuda.utilization()
        gpu_utilization.set(gpu_util)
    
    return generate_latest(registry)

@app.get("/stats")
async def get_statistics():
    """Get verification statistics"""
    
    if verification_engine is None:
        return {"error": "Verification engine not available"}
    
    stats = await verification_engine.get_statistics()
    
    return {
        "total_verifications": stats.get("total", 0),
        "successful_verifications": stats.get("successful", 0),
        "failed_verifications": stats.get("failed", 0),
        "average_ssim": stats.get("avg_ssim", 0),
        "average_psnr": stats.get("avg_psnr", 0),
        "average_processing_time_ms": stats.get("avg_time_ms", 0),
        "consensus_rate": stats.get("consensus_rate", 0)
    }

# Helper functions

async def cache_verification_result(verification_id: str, result: VerificationResult):
    """Cache verification result in Redis"""
    try:
        import json
        cache_key = f"verification:{verification_id}"
        cache_value = json.dumps({
            "verification_id": verification_id,
            "status": "completed",
            "quality_metrics": {
                "ssim": result.quality_metrics.ssim,
                "psnr": result.quality_metrics.psnr,
                "lpips": result.quality_metrics.lpips
            } if result.quality_metrics else None,
            "consensus_achieved": result.consensus_achieved
        })
        
        # Cache for 1 hour
        await redis_client.setex(cache_key, 3600, cache_value)
        
    except Exception as e:
        logger.error(f"Failed to cache verification result: {e}")

if __name__ == "__main__":
    # Run the service
    port = int(os.getenv("PORT", 8081))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )