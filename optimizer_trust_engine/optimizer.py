"""
Optimizer Core - The Authoring Plugin Component
================================================

The Optimizer serves as the intelligent orchestration layer that:
1. Analyzes incoming scenes from Unity/Blender
2. Routes optimization through appropriate pipelines
3. Manages resource allocation and scheduling
4. Provides real-time feedback to content creators
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationProfile(Enum):
    """Optimization profiles for different use cases"""
    QUALITY = "quality"          # Maximum quality, cost secondary
    BALANCED = "balanced"         # Balance between quality and cost
    ECONOMY = "economy"          # Minimum cost, acceptable quality
    REALTIME = "realtime"        # Optimized for real-time rendering
    ARCHVIZ = "archviz"          # Architectural visualization
    GAMING = "gaming"            # Game asset optimization
    AI_TRAINING = "ai_training"  # Synthetic data generation


class SceneComplexity(Enum):
    """Scene complexity classification"""
    SIMPLE = "simple"        # < 100K triangles, basic materials
    MODERATE = "moderate"    # 100K-1M triangles, PBR materials
    COMPLEX = "complex"      # 1M-10M triangles, complex shaders
    EXTREME = "extreme"      # > 10M triangles, advanced features


@dataclass
class SceneMetadata:
    """Metadata for scene analysis"""
    scene_id: str
    source_application: str  # Unity, Blender, Maya, etc.
    polygon_count: int
    texture_count: int
    material_count: int
    light_count: int
    animation_frames: int
    resolution: Tuple[int, int]
    file_size_mb: float
    complexity: SceneComplexity
    has_transparency: bool
    has_volumetrics: bool
    has_dynamic_elements: bool
    timestamp: float = field(default_factory=time.time)
    
    def calculate_hash(self) -> str:
        """Calculate unique hash for scene state"""
        data = f"{self.scene_id}{self.polygon_count}{self.material_count}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class OptimizationRequest:
    """Request for scene optimization"""
    scene_metadata: SceneMetadata
    profile: OptimizationProfile
    target_quality: float  # 0.0 to 1.0
    max_processing_time: Optional[float] = None  # seconds
    max_cost: Optional[float] = None  # USD
    priority: int = 5  # 1-10, higher is more important
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    request_id: str
    scene_id: str
    success: bool
    optimization_path: str  # baking or 3dgs
    processing_time: float
    estimated_cost: float
    quality_metrics: Dict[str, float]
    file_size_reduction: float  # percentage
    performance_gain: float  # percentage
    artifacts_path: Optional[str] = None
    error_message: Optional[str] = None


class SceneAnalyzer:
    """
    Intelligent scene analysis for optimization routing
    Uses ML-based classification to determine optimal processing path
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize scene analyzer with optional pre-trained model"""
        self.model_path = model_path
        self.analysis_cache = {}
        self.feature_extractors = self._initialize_extractors()
        
    def _initialize_extractors(self) -> Dict:
        """Initialize feature extraction functions"""
        return {
            "geometric": self._extract_geometric_features,
            "material": self._extract_material_features,
            "lighting": self._extract_lighting_features,
            "dynamic": self._extract_dynamic_features
        }
    
    def analyze_scene(self, metadata: SceneMetadata) -> Dict[str, Any]:
        """
        Perform comprehensive scene analysis
        Returns feature vectors and recommendations
        """
        # Check cache first
        scene_hash = metadata.calculate_hash()
        if scene_hash in self.analysis_cache:
            logger.info(f"Using cached analysis for scene {metadata.scene_id}")
            return self.analysis_cache[scene_hash]
        
        analysis = {
            "scene_id": metadata.scene_id,
            "complexity_score": self._calculate_complexity_score(metadata),
            "features": {},
            "recommendations": {},
            "optimization_hints": []
        }
        
        # Extract features
        for feature_type, extractor in self.feature_extractors.items():
            analysis["features"][feature_type] = extractor(metadata)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(
            metadata, analysis["features"]
        )
        
        # Generate optimization hints
        analysis["optimization_hints"] = self._generate_hints(metadata)
        
        # Cache the result
        self.analysis_cache[scene_hash] = analysis
        
        logger.info(f"Scene analysis complete for {metadata.scene_id}")
        return analysis
    
    def _calculate_complexity_score(self, metadata: SceneMetadata) -> float:
        """Calculate overall scene complexity score (0.0 to 1.0)"""
        scores = []
        
        # Geometry complexity
        poly_score = min(metadata.polygon_count / 10_000_000, 1.0)
        scores.append(poly_score * 0.3)
        
        # Material complexity
        mat_score = min(metadata.material_count / 1000, 1.0)
        scores.append(mat_score * 0.2)
        
        # Lighting complexity
        light_score = min(metadata.light_count / 100, 1.0)
        scores.append(light_score * 0.15)
        
        # Special features
        if metadata.has_transparency:
            scores.append(0.1)
        if metadata.has_volumetrics:
            scores.append(0.15)
        if metadata.has_dynamic_elements:
            scores.append(0.1)
        
        return min(sum(scores), 1.0)
    
    def _extract_geometric_features(self, metadata: SceneMetadata) -> Dict:
        """Extract geometric features from scene"""
        return {
            "polygon_density": metadata.polygon_count / (metadata.resolution[0] * metadata.resolution[1]),
            "geometric_complexity": metadata.complexity.value,
            "estimated_draw_calls": self._estimate_draw_calls(metadata),
            "lod_potential": self._calculate_lod_potential(metadata)
        }
    
    def _extract_material_features(self, metadata: SceneMetadata) -> Dict:
        """Extract material-related features"""
        return {
            "material_diversity": metadata.material_count / max(metadata.polygon_count / 1000, 1),
            "texture_memory_estimate": metadata.texture_count * 4,  # MB estimate
            "shader_complexity": self._estimate_shader_complexity(metadata),
            "pbr_usage": 0.8 if metadata.source_application in ["Unity", "Blender"] else 0.5
        }
    
    def _extract_lighting_features(self, metadata: SceneMetadata) -> Dict:
        """Extract lighting-related features"""
        return {
            "light_density": metadata.light_count / max(metadata.polygon_count / 10000, 1),
            "shadow_complexity": metadata.light_count * 0.3,
            "gi_potential": 0.9 if metadata.complexity in [SceneComplexity.SIMPLE, SceneComplexity.MODERATE] else 0.4,
            "baking_suitability": self._calculate_baking_suitability(metadata)
        }
    
    def _extract_dynamic_features(self, metadata: SceneMetadata) -> Dict:
        """Extract features related to dynamic elements"""
        return {
            "animation_complexity": metadata.animation_frames / 1000 if metadata.animation_frames > 0 else 0,
            "dynamic_ratio": 1.0 if metadata.has_dynamic_elements else 0.0,
            "temporal_coherence": 0.7 if metadata.animation_frames > 0 else 1.0,
            "realtime_requirements": 0.8 if metadata.has_dynamic_elements else 0.2
        }
    
    def _estimate_draw_calls(self, metadata: SceneMetadata) -> int:
        """Estimate number of draw calls"""
        base_calls = metadata.material_count * 2
        if metadata.has_transparency:
            base_calls *= 1.5
        return int(base_calls)
    
    def _calculate_lod_potential(self, metadata: SceneMetadata) -> float:
        """Calculate potential for LOD optimization"""
        if metadata.polygon_count < 100_000:
            return 0.2
        elif metadata.polygon_count < 1_000_000:
            return 0.6
        else:
            return 0.9
    
    def _estimate_shader_complexity(self, metadata: SceneMetadata) -> float:
        """Estimate shader complexity based on features"""
        complexity = 0.3  # Base complexity
        if metadata.has_transparency:
            complexity += 0.2
        if metadata.has_volumetrics:
            complexity += 0.3
        if metadata.material_count > 50:
            complexity += 0.2
        return min(complexity, 1.0)
    
    def _calculate_baking_suitability(self, metadata: SceneMetadata) -> float:
        """Calculate how suitable the scene is for light baking"""
        suitability = 0.5
        
        # Static scenes are better for baking
        if not metadata.has_dynamic_elements:
            suitability += 0.3
        
        # Moderate complexity is ideal
        if metadata.complexity in [SceneComplexity.SIMPLE, SceneComplexity.MODERATE]:
            suitability += 0.2
        
        # Too many lights might not be suitable
        if metadata.light_count > 50:
            suitability -= 0.2
        
        return max(0, min(suitability, 1.0))
    
    def _generate_recommendations(self, metadata: SceneMetadata, features: Dict) -> Dict:
        """Generate optimization recommendations based on analysis"""
        recommendations = {
            "primary_pipeline": None,
            "secondary_pipeline": None,
            "optimization_settings": {},
            "estimated_savings": {}
        }
        
        # Determine primary pipeline
        baking_score = features["lighting"]["baking_suitability"]
        dynamic_score = features["dynamic"]["dynamic_ratio"]
        complexity_score = self._calculate_complexity_score(metadata)
        
        if baking_score > 0.7 and dynamic_score < 0.3:
            recommendations["primary_pipeline"] = "baking"
            recommendations["secondary_pipeline"] = "3dgs" if complexity_score > 0.6 else None
        else:
            recommendations["primary_pipeline"] = "3dgs"
            recommendations["secondary_pipeline"] = "baking" if baking_score > 0.5 else None
        
        # Generate optimization settings
        recommendations["optimization_settings"] = {
            "target_polygon_reduction": min(0.7, 1.0 - complexity_score),
            "texture_compression": "BC7" if metadata.has_transparency else "BC1",
            "lod_levels": 3 if features["geometric"]["lod_potential"] > 0.5 else 1,
            "shadow_resolution": 2048 if metadata.light_count > 10 else 1024,
            "gi_quality": "high" if baking_score > 0.7 else "medium"
        }
        
        # Estimate savings
        recommendations["estimated_savings"] = {
            "processing_time_reduction": f"{int(40 + complexity_score * 30)}%",
            "cost_reduction": f"{int(50 + baking_score * 25)}%",
            "file_size_reduction": f"{int(30 + features['geometric']['lod_potential'] * 40)}%",
            "render_time_improvement": f"{int(25 + (1.0 - dynamic_score) * 45)}%"
        }
        
        return recommendations
    
    def _generate_hints(self, metadata: SceneMetadata) -> List[str]:
        """Generate optimization hints for the user"""
        hints = []
        
        if metadata.polygon_count > 5_000_000:
            hints.append("Consider using mesh decimation for distant objects")
        
        if metadata.material_count > 100:
            hints.append("Material atlas packing could reduce draw calls significantly")
        
        if metadata.has_transparency and metadata.has_volumetrics:
            hints.append("Combination of transparency and volumetrics will impact performance")
        
        if metadata.light_count > 20 and not metadata.has_dynamic_elements:
            hints.append("Static lighting could benefit from full light baking")
        
        if metadata.texture_count > 50:
            hints.append("Texture streaming could improve memory usage")
        
        return hints


class OptimizationRouter:
    """
    Routes optimization requests to appropriate pipelines
    Manages resource allocation and scheduling
    """
    
    def __init__(self, analyzer: SceneAnalyzer):
        """Initialize router with scene analyzer"""
        self.analyzer = analyzer
        self.active_jobs = {}
        self.job_queue = asyncio.Queue()
        self.pipeline_handlers = {
            "baking": self._handle_baking_pipeline,
            "3dgs": self._handle_3dgs_pipeline,
            "hybrid": self._handle_hybrid_pipeline
        }
    
    async def route_optimization(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Route optimization request to appropriate pipeline
        Returns optimization result
        """
        # Analyze the scene
        analysis = self.analyzer.analyze_scene(request.scene_metadata)
        
        # Determine routing based on analysis and profile
        pipeline = self._determine_pipeline(analysis, request.profile)
        
        # Create job
        job_id = self._create_job_id(request)
        job = {
            "id": job_id,
            "request": request,
            "analysis": analysis,
            "pipeline": pipeline,
            "status": "queued",
            "created_at": time.time()
        }
        
        # Add to active jobs
        self.active_jobs[job_id] = job
        
        # Queue for processing
        await self.job_queue.put(job)
        
        # Process job
        result = await self._process_job(job)
        
        # Clean up
        del self.active_jobs[job_id]
        
        return result
    
    def _determine_pipeline(self, analysis: Dict, profile: OptimizationProfile) -> str:
        """Determine which pipeline to use based on analysis and profile"""
        recommendations = analysis["recommendations"]
        
        # Profile-based overrides
        if profile == OptimizationProfile.ECONOMY:
            # Economy profile prefers baking for cost efficiency
            if recommendations["primary_pipeline"] == "baking":
                return "baking"
            elif analysis["features"]["lighting"]["baking_suitability"] > 0.4:
                return "baking"
            else:
                return "3dgs"
        
        elif profile == OptimizationProfile.QUALITY:
            # Quality profile may use hybrid approach
            if recommendations["secondary_pipeline"] is not None:
                return "hybrid"
            else:
                return recommendations["primary_pipeline"]
        
        elif profile == OptimizationProfile.REALTIME:
            # Realtime profile prefers 3DGS for dynamic content
            if analysis["features"]["dynamic"]["dynamic_ratio"] > 0.5:
                return "3dgs"
            else:
                return recommendations["primary_pipeline"]
        
        else:
            # Default to primary recommendation
            return recommendations["primary_pipeline"]
    
    def _create_job_id(self, request: OptimizationRequest) -> str:
        """Create unique job ID"""
        data = f"{request.scene_metadata.scene_id}{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    async def _process_job(self, job: Dict) -> OptimizationResult:
        """Process optimization job through selected pipeline"""
        job["status"] = "processing"
        start_time = time.time()
        
        try:
            # Get appropriate handler
            handler = self.pipeline_handlers.get(job["pipeline"])
            if not handler:
                raise ValueError(f"Unknown pipeline: {job['pipeline']}")
            
            # Process through pipeline
            result = await handler(job)
            
            # Update result with timing
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Job {job['id']} failed: {str(e)}")
            return OptimizationResult(
                request_id=job["id"],
                scene_id=job["request"].scene_metadata.scene_id,
                success=False,
                optimization_path=job["pipeline"],
                processing_time=time.time() - start_time,
                estimated_cost=0.0,
                quality_metrics={},
                file_size_reduction=0.0,
                performance_gain=0.0,
                error_message=str(e)
            )
    
    async def _handle_baking_pipeline(self, job: Dict) -> OptimizationResult:
        """Handle optimization through baking pipeline"""
        logger.info(f"Processing job {job['id']} through baking pipeline")
        
        # Simulate baking process
        await asyncio.sleep(2)  # Placeholder for actual processing
        
        return OptimizationResult(
            request_id=job["id"],
            scene_id=job["request"].scene_metadata.scene_id,
            success=True,
            optimization_path="baking",
            processing_time=0.0,  # Will be updated
            estimated_cost=0.6,  # USD per scene
            quality_metrics={
                "ssim": 0.982,
                "psnr": 34.7,
                "lpips": 0.041
            },
            file_size_reduction=65.0,
            performance_gain=75.0,
            artifacts_path=f"/output/baked/{job['id']}"
        )
    
    async def _handle_3dgs_pipeline(self, job: Dict) -> OptimizationResult:
        """Handle optimization through 3D Gaussian Splatting pipeline"""
        logger.info(f"Processing job {job['id']} through 3DGS pipeline")
        
        # Simulate 3DGS process
        await asyncio.sleep(3)  # Placeholder for actual processing
        
        return OptimizationResult(
            request_id=job["id"],
            scene_id=job["request"].scene_metadata.scene_id,
            success=True,
            optimization_path="3dgs",
            processing_time=0.0,  # Will be updated
            estimated_cost=0.8,  # USD per scene
            quality_metrics={
                "ssim": 0.991,
                "psnr": 36.2,
                "lpips": 0.028
            },
            file_size_reduction=70.0,
            performance_gain=85.0,
            artifacts_path=f"/output/3dgs/{job['id']}"
        )
    
    async def _handle_hybrid_pipeline(self, job: Dict) -> OptimizationResult:
        """Handle optimization through hybrid pipeline (both baking and 3DGS)"""
        logger.info(f"Processing job {job['id']} through hybrid pipeline")
        
        # Process through both pipelines
        baking_result = await self._handle_baking_pipeline(job)
        gs_result = await self._handle_3dgs_pipeline(job)
        
        # Combine results (best of both)
        return OptimizationResult(
            request_id=job["id"],
            scene_id=job["request"].scene_metadata.scene_id,
            success=True,
            optimization_path="hybrid",
            processing_time=0.0,  # Will be updated
            estimated_cost=1.2,  # Combined cost
            quality_metrics={
                "ssim": max(baking_result.quality_metrics["ssim"], gs_result.quality_metrics["ssim"]),
                "psnr": max(baking_result.quality_metrics["psnr"], gs_result.quality_metrics["psnr"]),
                "lpips": min(baking_result.quality_metrics["lpips"], gs_result.quality_metrics["lpips"])
            },
            file_size_reduction=max(baking_result.file_size_reduction, gs_result.file_size_reduction),
            performance_gain=max(baking_result.performance_gain, gs_result.performance_gain),
            artifacts_path=f"/output/hybrid/{job['id']}"
        )


class OptimizerCore:
    """
    Main Optimizer orchestrator
    Coordinates all optimization components
    """
    
    def __init__(self):
        """Initialize the Optimizer core"""
        self.analyzer = SceneAnalyzer()
        self.router = OptimizationRouter(self.analyzer)
        self.metrics = {
            "total_scenes_processed": 0,
            "total_cost_saved": 0.0,
            "average_quality_score": 0.0,
            "total_processing_time": 0.0
        }
    
    async def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Main optimization entry point
        Processes scene through complete optimization pipeline
        """
        logger.info(f"Starting optimization for scene {request.scene_metadata.scene_id}")
        
        # Route through optimization pipeline
        result = await self.router.route_optimization(request)
        
        # Update metrics
        self._update_metrics(result)
        
        logger.info(f"Optimization complete for scene {request.scene_metadata.scene_id}")
        return result
    
    def _update_metrics(self, result: OptimizationResult):
        """Update internal metrics based on result"""
        self.metrics["total_scenes_processed"] += 1
        
        if result.success:
            # Calculate cost saved (assuming $2.4 baseline)
            baseline_cost = 2.4
            saved = baseline_cost - result.estimated_cost
            self.metrics["total_cost_saved"] += saved
            
            # Update average quality
            quality_score = np.mean(list(result.quality_metrics.values()))
            current_avg = self.metrics["average_quality_score"]
            n = self.metrics["total_scenes_processed"]
            self.metrics["average_quality_score"] = (current_avg * (n-1) + quality_score) / n
            
            # Update processing time
            self.metrics["total_processing_time"] += result.processing_time
    
    def get_metrics(self) -> Dict:
        """Get current optimizer metrics"""
        return self.metrics.copy()
    
    async def batch_optimize(self, requests: List[OptimizationRequest]) -> List[OptimizationResult]:
        """Optimize multiple scenes in batch"""
        logger.info(f"Starting batch optimization for {len(requests)} scenes")
        
        # Process concurrently with controlled parallelism
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.optimize(request))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Batch optimization complete for {len(requests)} scenes")
        return results