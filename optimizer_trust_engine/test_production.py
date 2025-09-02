#!/usr/bin/env python3
"""
Production Integration Tests
=============================

Comprehensive test suite to verify all components work correctly.
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# Add core module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

try:
    from core.optimizer_core import OptimizerCore, SceneAnalyzer
    from core.trust_engine_core import TrustEngineCore, ProbabilisticVerifier, ConsensusValidator
    from core.models import (
        SceneMetadata, OptimizationRequest, OptimizationResult,
        SceneComplexity, OptimizationProfile, QualityThresholds,
        VerificationNode
    )
    from core.exceptions import OptimizerException, ValidationError
    from core.utils import timer, generate_unique_id
    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class ProductionTestSuite:
    """Production-grade test suite"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_true(self, condition: bool, message: str = ""):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            print(f"  ✅ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  ❌ {message}")
    
    def assert_equals(self, actual: Any, expected: Any, message: str = ""):
        """Assert values are equal"""
        if actual == expected:
            self.passed += 1
            print(f"  ✅ {message}")
        else:
            self.failed += 1
            self.errors.append(f"{message}: {actual} != {expected}")
            print(f"  ❌ {message}: {actual} != {expected}")
    
    def assert_in_range(self, value: float, min_val: float, max_val: float, message: str = ""):
        """Assert value is in range"""
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"  ✅ {message}: {value:.3f} in [{min_val}, {max_val}]")
        else:
            self.failed += 1
            self.errors.append(f"{message}: {value} not in [{min_val}, {max_val}]")
            print(f"  ❌ {message}: {value} not in [{min_val}, {max_val}]")
    
    async def test_scene_analyzer(self):
        """Test SceneAnalyzer functionality"""
        print("\n📋 Testing SceneAnalyzer")
        
        analyzer = SceneAnalyzer()
        
        # Test scene metadata creation
        metadata = SceneMetadata(
            scene_id="test_001",
            source_application="Unity",
            polygon_count=1000000,
            texture_count=50,
            material_count=25,
            light_count=10,
            animation_frames=0,
            resolution=(1920, 1080),
            file_size_mb=500.0,
            complexity=SceneComplexity.MODERATE
        )
        
        # Test analysis
        analysis = analyzer.analyze(metadata)
        
        self.assert_true("scene_id" in analysis, "Analysis contains scene_id")
        self.assert_true("complexity_score" in analysis, "Analysis contains complexity_score")
        self.assert_in_range(analysis["complexity_score"], 0.0, 1.0, "Complexity score")
        self.assert_true("features" in analysis, "Analysis contains features")
        self.assert_true("recommendations" in analysis, "Analysis contains recommendations")
        
        # Test caching
        analysis2 = analyzer.analyze(metadata)
        self.assert_equals(analysis["scene_id"], analysis2["scene_id"], "Cache returns same result")
    
    async def test_optimizer_core(self):
        """Test OptimizerCore functionality"""
        print("\n📋 Testing OptimizerCore")
        
        optimizer = OptimizerCore()
        
        # Create test request
        metadata = SceneMetadata(
            scene_id="test_002",
            source_application="Blender",
            polygon_count=2000000,
            texture_count=75,
            material_count=40,
            light_count=15,
            animation_frames=60,
            resolution=(2560, 1440),
            file_size_mb=1200.0,
            complexity=SceneComplexity.COMPLEX,
            has_transparency=True
        )
        
        request = OptimizationRequest(
            scene_metadata=metadata,
            profile=OptimizationProfile.BALANCED,
            target_quality=0.95,
            max_cost=3.0,
            priority=7
        )
        
        # Test optimization
        with timer("Optimization"):
            result = await optimizer.optimize(request)
        
        self.assert_true(result.success, "Optimization successful")
        self.assert_true(result.optimization_path in ["baking", "3dgs", "hybrid"], "Valid pipeline")
        self.assert_in_range(result.estimated_cost, 0.0, 3.0, "Cost within budget")
        self.assert_true(len(result.quality_metrics) > 0, "Quality metrics present")
        
        # Test quality metrics
        if "ssim" in result.quality_metrics:
            self.assert_in_range(result.quality_metrics["ssim"], 0.9, 1.0, "SSIM score")
        if "psnr" in result.quality_metrics:
            self.assert_in_range(result.quality_metrics["psnr"], 30.0, 50.0, "PSNR score")
        
        # Test metrics
        metrics = optimizer.get_metrics()
        self.assert_true(metrics["total_scenes_processed"] > 0, "Scenes processed tracked")
    
    async def test_trust_engine(self):
        """Test TrustEngine functionality"""
        print("\n📋 Testing TrustEngine")
        
        trust_engine = TrustEngineCore()
        
        # Test verification
        scene_data = {
            "scene_id": "test_003",
            "complexity_score": 0.6,
            "specialization": "general"
        }
        
        optimization_result = {
            "quality_metrics": {
                "ssim": 0.99,
                "psnr": 36.5,
                "lpips": 0.03,
                "vmaf": 94.0,
                "flip": 0.06
            }
        }
        
        with timer("Verification"):
            result = await trust_engine.verify_optimization(scene_data, optimization_result)
        
        self.assert_true(result.verification_id != "", "Verification ID generated")
        self.assert_true(result.status is not None, "Status determined")
        self.assert_in_range(result.consensus_score, 0.0, 1.0, "Consensus score")
        self.assert_true(len(result.participating_nodes) >= 3, "Minimum nodes participated")
        self.assert_in_range(result.confidence_level, 0.0, 1.0, "Confidence level")
        
        # Test node statistics
        stats = trust_engine.get_node_statistics()
        self.assert_true(len(stats) >= 3, "Node statistics available")
    
    async def test_probabilistic_verifier(self):
        """Test ProbabilisticVerifier functionality"""
        print("\n📋 Testing ProbabilisticVerifier")
        
        thresholds = QualityThresholds()
        verifier = ProbabilisticVerifier(thresholds)
        
        # Test sample size calculation
        size_simple = verifier.calculate_sample_size(0.2)
        size_complex = verifier.calculate_sample_size(0.8)
        
        self.assert_true(size_simple < size_complex, "Complex scenes need more samples")
        self.assert_in_range(size_simple, 10, 100, "Simple scene sample size")
        self.assert_in_range(size_complex, 10, 100, "Complex scene sample size")
        
        # Test sample generation
        samples = verifier.generate_samples({"scene_id": "test"}, 20)
        self.assert_equals(len(samples), 20, "Correct number of samples")
        
        # Test sample verification
        for sample in samples[:5]:
            scores = verifier.verify_sample(sample, {"base_quality": 0.95})
            self.assert_true("ssim" in scores, "SSIM in scores")
            self.assert_true("psnr" in scores, "PSNR in scores")
    
    async def test_consensus_validator(self):
        """Test ConsensusValidator functionality"""
        print("\n📋 Testing ConsensusValidator")
        
        validator = ConsensusValidator()
        
        # Test node registration
        node = VerificationNode(
            node_id="test_node",
            reputation_score=0.9,
            specialization=["general"],
            compute_capability=1.2
        )
        validator.register_node(node)
        
        # Test node selection
        nodes = validator.select_nodes(5, "general")
        self.assert_true(len(nodes) >= 5, "Sufficient nodes selected")
        
        # Test vote collection
        votes = validator.collect_votes(nodes, {"mean_scores": {"ssim": 0.98}})
        self.assert_equals(len(votes), len(nodes), "All nodes voted")
        
        # Test consensus calculation
        consensus_score, aggregated, byzantine, confidence = validator.calculate_consensus(votes, nodes)
        self.assert_in_range(consensus_score, 0.0, 1.0, "Consensus score")
        self.assert_in_range(confidence, 0.0, 1.0, "Confidence level")
    
    async def test_validation_errors(self):
        """Test input validation"""
        print("\n📋 Testing Input Validation")
        
        # Test invalid scene metadata
        try:
            metadata = SceneMetadata(
                scene_id="",  # Invalid: empty
                source_application="Unity",
                polygon_count=-100,  # Invalid: negative
                texture_count=10,
                material_count=5,
                light_count=3,
                animation_frames=0,
                resolution=(0, 0),  # Invalid: zero dimensions
                file_size_mb=-10,  # Invalid: negative
                complexity=SceneComplexity.SIMPLE
            )
            self.assert_true(False, "Should have raised ValidationError")
        except ValidationError:
            self.assert_true(True, "ValidationError raised for invalid metadata")
        
        # Test invalid optimization request
        try:
            request = OptimizationRequest(
                scene_metadata=None,  # Invalid: None
                profile=OptimizationProfile.BALANCED,
                target_quality=1.5,  # Invalid: > 1.0
                priority=15  # Invalid: > 10
            )
            self.assert_true(False, "Should have raised ValidationError")
        except (ValidationError, AttributeError):
            self.assert_true(True, "ValidationError raised for invalid request")
    
    async def test_batch_processing(self):
        """Test batch optimization"""
        print("\n📋 Testing Batch Processing")
        
        optimizer = OptimizerCore()
        
        # Create multiple requests
        requests = []
        for i in range(3):
            metadata = SceneMetadata(
                scene_id=f"batch_{i}",
                source_application="Unity",
                polygon_count=100000 * (i + 1),
                texture_count=10 * (i + 1),
                material_count=5 * (i + 1),
                light_count=3,
                animation_frames=0,
                resolution=(1920, 1080),
                file_size_mb=100.0 * (i + 1),
                complexity=SceneComplexity.SIMPLE
            )
            
            request = OptimizationRequest(
                scene_metadata=metadata,
                profile=OptimizationProfile.ECONOMY,
                target_quality=0.9,
                priority=5
            )
            requests.append(request)
        
        # Process batch
        with timer("Batch optimization"):
            results = await optimizer.batch_optimize(requests)
        
        self.assert_equals(len(results), 3, "All requests processed")
        
        for result in results:
            self.assert_true(result.success or result.error_message, "Result has status")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("🧪 PRODUCTION INTEGRATION TESTS")
        print("="*60)
        
        start_time = time.time()
        
        # Run test suites
        await self.test_scene_analyzer()
        await self.test_optimizer_core()
        await self.test_trust_engine()
        await self.test_probabilistic_verifier()
        await self.test_consensus_validator()
        await self.test_validation_errors()
        await self.test_batch_processing()
        
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        if self.failed > 0:
            print("\n❌ FAILURES:")
            for error in self.errors:
                print(f"  - {error}")
        
        success_rate = (self.passed / (self.passed + self.failed)) * 100 if (self.passed + self.failed) > 0 else 0
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n✅ ALL TESTS PASSED - PRODUCTION READY!")
        else:
            print("\n⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        
        return 0 if success_rate == 100 else 1


async def main():
    """Main test runner"""
    test_suite = ProductionTestSuite()
    exit_code = await test_suite.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    print("Starting production integration tests...")
    asyncio.run(main())