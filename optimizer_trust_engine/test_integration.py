"""
Integration Tests for Optimizer and Trust Engine Framework
===========================================================

Demonstrates the complete workflow:
1. Scene submission to Optimizer
2. Intelligent routing through pipelines
3. Verification via Trust Engine
4. Cost optimization through Economy Offline beach
"""

import asyncio
import json
import logging
import time
from typing import Dict, List

from optimizer import (
    OptimizerCore,
    SceneMetadata,
    SceneComplexity,
    OptimizationProfile,
    OptimizationRequest
)
from trust_engine import TrustEngine, QualityThresholds
from economy_offline import EconomyOfflineBeach

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """Complete integration test suite for the framework"""
    
    def __init__(self):
        """Initialize test suite components"""
        self.optimizer = OptimizerCore()
        self.trust_engine = TrustEngine()
        self.economy_beach = EconomyOfflineBeach()
        self.test_results = []
    
    async def test_simple_scene_optimization(self):
        """Test optimization of a simple scene"""
        logger.info("=== Testing Simple Scene Optimization ===")
        
        # Create simple scene metadata
        scene_metadata = SceneMetadata(
            scene_id="test_simple_001",
            source_application="Unity",
            polygon_count=50000,
            texture_count=10,
            material_count=5,
            light_count=3,
            animation_frames=0,
            resolution=(1920, 1080),
            file_size_mb=100.0,
            complexity=SceneComplexity.SIMPLE,
            has_transparency=False,
            has_volumetrics=False,
            has_dynamic_elements=False
        )
        
        # Create optimization request
        request = OptimizationRequest(
            scene_metadata=scene_metadata,
            profile=OptimizationProfile.ECONOMY,
            target_quality=0.95,
            max_processing_time=300.0,
            max_cost=1.0,
            priority=7
        )
        
        # Process through optimizer
        result = await self.optimizer.optimize(request)
        
        # Log results
        logger.info(f"Optimization Result: {result.success}")
        logger.info(f"Path: {result.optimization_path}")
        logger.info(f"Cost: ${result.estimated_cost:.2f}")
        logger.info(f"Quality: {result.quality_metrics}")
        
        self.test_results.append({
            "test": "simple_scene",
            "success": result.success,
            "metrics": result.quality_metrics
        })
        
        return result
    
    async def test_complex_scene_with_verification(self):
        """Test complex scene with full verification pipeline"""
        logger.info("=== Testing Complex Scene with Verification ===")
        
        # Create complex scene metadata
        scene_metadata = SceneMetadata(
            scene_id="test_complex_001",
            source_application="Blender",
            polygon_count=5000000,
            texture_count=150,
            material_count=80,
            light_count=25,
            animation_frames=240,
            resolution=(3840, 2160),
            file_size_mb=2500.0,
            complexity=SceneComplexity.COMPLEX,
            has_transparency=True,
            has_volumetrics=True,
            has_dynamic_elements=True
        )
        
        # Create optimization request with quality focus
        request = OptimizationRequest(
            scene_metadata=scene_metadata,
            profile=OptimizationProfile.QUALITY,
            target_quality=0.99,
            max_processing_time=1800.0,
            max_cost=5.0,
            priority=9
        )
        
        # Process through optimizer
        optimization_result = await self.optimizer.optimize(request)
        
        # Prepare scene data for verification
        scene_data = {
            "scene_id": scene_metadata.scene_id,
            "complexity_score": 0.8,
            "specialization": "quality"
        }
        
        # Verify through Trust Engine
        verification_result = await self.trust_engine.verify_optimization(
            scene_data,
            {
                "optimization_path": optimization_result.optimization_path,
                "quality_metrics": optimization_result.quality_metrics
            }
        )
        
        # Log verification results
        logger.info(f"Verification Status: {verification_result.status.value}")
        logger.info(f"Consensus Score: {verification_result.consensus_score:.2f}")
        logger.info(f"Quality Passed: {verification_result.passed_thresholds}")
        
        if verification_result.byzantine_nodes_detected:
            logger.warning(f"Byzantine nodes detected: {verification_result.byzantine_nodes_detected}")
        
        self.test_results.append({
            "test": "complex_scene_verification",
            "optimization_success": optimization_result.success,
            "verification_status": verification_result.status.value,
            "consensus_score": verification_result.consensus_score
        })
        
        return optimization_result, verification_result
    
    async def test_economy_offline_processing(self):
        """Test Economy Offline beach processing"""
        logger.info("=== Testing Economy Offline Processing ===")
        
        # Create scene data for economy processing
        scene_data = {
            "scene_id": "test_economy_001",
            "complexity": "moderate",
            "resolution": (2560, 1440),
            "has_volumetrics": False,
            "has_transparency": True,
            "animation_frames": 120,
            "priority": 5,
            "quality": 0.9,
            "max_budget": 2.0
        }
        
        # Process through Economy Offline beach
        result = await self.economy_beach.process_scene(scene_data, "economy")
        
        # Log economy results
        logger.info(f"Job Status: {result['status']}")
        if result['status'] == 'completed':
            cost_analysis = result['cost_analysis']
            logger.info(f"Baseline Cost: ${cost_analysis['baseline_cost']:.2f}")
            logger.info(f"Actual Cost: ${cost_analysis['actual_cost']:.2f}")
            logger.info(f"Savings: ${cost_analysis['savings']:.2f} ({cost_analysis['savings_percentage']:.1f}%)")
            logger.info(f"Carbon Footprint: {cost_analysis['carbon_footprint_kg']:.2f} kg CO2")
            
            # Log recommendations
            logger.info("Optimization Recommendations:")
            for rec in result['optimization_recommendations']:
                logger.info(f"  - {rec}")
        
        self.test_results.append({
            "test": "economy_offline",
            "status": result['status'],
            "savings_percentage": result.get('cost_analysis', {}).get('savings_percentage', 0)
        })
        
        return result
    
    async def test_batch_optimization(self):
        """Test batch optimization of multiple scenes"""
        logger.info("=== Testing Batch Optimization ===")
        
        # Create multiple scenes
        scenes = []
        for i in range(3):
            scene_metadata = SceneMetadata(
                scene_id=f"test_batch_{i:03d}",
                source_application="Unity",
                polygon_count=100000 * (i + 1),
                texture_count=20 * (i + 1),
                material_count=10 * (i + 1),
                light_count=5 * (i + 1),
                animation_frames=0,
                resolution=(1920, 1080),
                file_size_mb=200.0 * (i + 1),
                complexity=SceneComplexity.MODERATE,
                has_transparency=i % 2 == 0,
                has_volumetrics=False,
                has_dynamic_elements=i == 2
            )
            
            request = OptimizationRequest(
                scene_metadata=scene_metadata,
                profile=OptimizationProfile.BALANCED,
                target_quality=0.95,
                max_processing_time=600.0,
                max_cost=2.0,
                priority=5 + i
            )
            scenes.append(request)
        
        # Process batch
        results = await self.optimizer.batch_optimize(scenes)
        
        # Log batch results
        logger.info(f"Processed {len(results)} scenes in batch")
        for i, result in enumerate(results):
            logger.info(f"  Scene {i}: Success={result.success}, Cost=${result.estimated_cost:.2f}")
        
        self.test_results.append({
            "test": "batch_optimization",
            "scenes_processed": len(results),
            "success_rate": sum(1 for r in results if r.success) / len(results)
        })
        
        return results
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("=== Testing End-to-End Workflow ===")
        
        # Step 1: Scene Analysis
        scene_metadata = SceneMetadata(
            scene_id="test_e2e_001",
            source_application="Blender",
            polygon_count=2000000,
            texture_count=75,
            material_count=40,
            light_count=15,
            animation_frames=60,
            resolution=(2560, 1440),
            file_size_mb=1200.0,
            complexity=SceneComplexity.MODERATE,
            has_transparency=True,
            has_volumetrics=False,
            has_dynamic_elements=True
        )
        
        # Step 2: Optimization Request
        request = OptimizationRequest(
            scene_metadata=scene_metadata,
            profile=OptimizationProfile.ECONOMY,
            target_quality=0.92,
            max_processing_time=900.0,
            max_cost=1.5,
            priority=8
        )
        
        # Step 3: Optimize
        optimization_result = await self.optimizer.optimize(request)
        logger.info(f"Step 3 - Optimization: {optimization_result.success}")
        
        # Step 4: Verify
        scene_data = {
            "scene_id": scene_metadata.scene_id,
            "complexity_score": 0.6,
            "specialization": "general"
        }
        
        verification_result = await self.trust_engine.verify_optimization(
            scene_data,
            {"quality_metrics": optimization_result.quality_metrics}
        )
        logger.info(f"Step 4 - Verification: {verification_result.status.value}")
        
        # Step 5: Economy Processing
        economy_data = {
            "scene_id": scene_metadata.scene_id,
            "complexity": "moderate",
            "resolution": scene_metadata.resolution,
            "has_volumetrics": scene_metadata.has_volumetrics,
            "has_transparency": scene_metadata.has_transparency,
            "animation_frames": scene_metadata.animation_frames,
            "priority": request.priority,
            "quality": request.target_quality,
            "max_budget": request.max_cost
        }
        
        economy_result = await self.economy_beach.process_scene(economy_data, "economy")
        logger.info(f"Step 5 - Economy Processing: {economy_result['status']}")
        
        # Final metrics
        logger.info("\n=== End-to-End Workflow Summary ===")
        logger.info(f"Scene ID: {scene_metadata.scene_id}")
        logger.info(f"Optimization Path: {optimization_result.optimization_path}")
        logger.info(f"Quality Achieved: SSIM={optimization_result.quality_metrics.get('ssim', 0):.3f}")
        logger.info(f"Verification Consensus: {verification_result.consensus_score:.2f}")
        logger.info(f"Total Cost: ${economy_result.get('cost_analysis', {}).get('actual_cost', 0):.2f}")
        logger.info(f"Cost Savings: {economy_result.get('cost_analysis', {}).get('savings_percentage', 0):.1f}%")
        
        self.test_results.append({
            "test": "end_to_end",
            "all_steps_completed": True,
            "final_cost": economy_result.get('cost_analysis', {}).get('actual_cost', 0)
        })
        
        return {
            "optimization": optimization_result,
            "verification": verification_result,
            "economy": economy_result
        }
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("\n" + "="*60)
        logger.info("Starting Integration Test Suite")
        logger.info("="*60 + "\n")
        
        # Run tests
        await self.test_simple_scene_optimization()
        await asyncio.sleep(1)
        
        await self.test_complex_scene_with_verification()
        await asyncio.sleep(1)
        
        await self.test_economy_offline_processing()
        await asyncio.sleep(1)
        
        await self.test_batch_optimization()
        await asyncio.sleep(1)
        
        await self.test_end_to_end_workflow()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        logger.info("\n" + "="*60)
        logger.info("Test Suite Summary")
        logger.info("="*60)
        
        for result in self.test_results:
            logger.info(f"\nTest: {result['test']}")
            for key, value in result.items():
                if key != 'test':
                    logger.info(f"  {key}: {value}")
        
        # Get final metrics
        optimizer_metrics = self.optimizer.get_metrics()
        beach_status = self.economy_beach.get_beach_status()
        node_stats = self.trust_engine.get_node_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("System Metrics")
        logger.info("="*60)
        
        logger.info("\nOptimizer Metrics:")
        for key, value in optimizer_metrics.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nEconomy Beach Status:")
        logger.info(f"  Active Jobs: {beach_status['active_jobs']}")
        logger.info(f"  Completed Jobs: {beach_status['completed_jobs']}")
        logger.info(f"  Total Cost Saved: ${beach_status['metrics']['total_cost_saved']:.2f}")
        
        logger.info("\nTrust Engine Nodes:")
        logger.info(f"  Total Nodes: {len(node_stats)}")
        trusted_nodes = sum(1 for stats in node_stats.values() if stats['is_trusted'])
        logger.info(f"  Trusted Nodes: {trusted_nodes}")
        
        logger.info("\n" + "="*60)
        logger.info("Integration Tests Complete!")
        logger.info("="*60)


async def main():
    """Main test runner"""
    test_suite = IntegrationTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())