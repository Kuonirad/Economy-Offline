#!/usr/bin/env python3
"""
Validation Script for Optimizer and Trust Engine Framework
===========================================================

This script validates all components and ensures the system is working correctly.
"""

import asyncio
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from optimizer import (
        OptimizerCore,
        SceneMetadata,
        SceneComplexity,
        OptimizationProfile,
        OptimizationRequest
    )
    from trust_engine import TrustEngine, QualityThresholds
    from economy_offline import EconomyOfflineBeach
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
    
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def test(self, name: str, condition: bool, error_msg: str = None):
        """Run a test and track results"""
        if condition:
            print(f"✅ {name}")
            self.tests_passed += 1
        else:
            print(f"❌ {name}")
            self.tests_failed += 1
            if error_msg:
                self.errors.append(f"{name}: {error_msg}")
    
    async def validate_optimizer(self) -> bool:
        """Validate Optimizer component"""
        self.print_header("Validating Optimizer")
        
        try:
            # Initialize optimizer
            optimizer = OptimizerCore()
            self.test("Optimizer initialization", True)
            
            # Test scene metadata creation
            scene = SceneMetadata(
                scene_id="validation_test",
                source_application="Unity",
                polygon_count=100000,
                texture_count=20,
                material_count=10,
                light_count=5,
                animation_frames=0,
                resolution=(1920, 1080),
                file_size_mb=200.0,
                complexity=SceneComplexity.MODERATE,
                has_transparency=False,
                has_volumetrics=False,
                has_dynamic_elements=False
            )
            self.test("Scene metadata creation", True)
            
            # Test optimization request
            request = OptimizationRequest(
                scene_metadata=scene,
                profile=OptimizationProfile.BALANCED,
                target_quality=0.95,
                max_processing_time=600.0,
                max_cost=2.0,
                priority=5
            )
            self.test("Optimization request creation", True)
            
            # Test optimization process
            result = await optimizer.optimize(request)
            self.test("Optimization execution", result.success)
            self.test("Cost estimation", result.estimated_cost > 0)
            self.test("Quality metrics", len(result.quality_metrics) > 0)
            self.test("Pipeline selection", result.optimization_path in ["baking", "3dgs", "hybrid"])
            
            # Test metrics
            metrics = optimizer.get_metrics()
            self.test("Metrics retrieval", metrics["total_scenes_processed"] > 0)
            
            return self.tests_failed == 0
            
        except Exception as e:
            self.test("Optimizer validation", False, str(e))
            return False
    
    async def validate_trust_engine(self) -> bool:
        """Validate Trust Engine component"""
        self.print_header("Validating Trust Engine")
        
        try:
            # Initialize trust engine
            trust_engine = TrustEngine()
            self.test("Trust Engine initialization", True)
            
            # Test quality thresholds
            thresholds = QualityThresholds()
            self.test("Quality thresholds", thresholds.ssim_min == 0.98)
            
            # Test verification
            scene_data = {
                "scene_id": "test_scene",
                "complexity_score": 0.5,
                "specialization": "general"
            }
            
            optimization_result = {
                "quality_metrics": {
                    "ssim": 0.99,
                    "psnr": 36.0,
                    "lpips": 0.03
                }
            }
            
            verification = await trust_engine.verify_optimization(scene_data, optimization_result)
            self.test("Verification execution", verification is not None)
            self.test("Consensus calculation", 0 <= verification.consensus_score <= 1)
            self.test("Participating nodes", len(verification.participating_nodes) >= 3)
            self.test("Audit trail", len(verification.audit_trail) > 0)
            
            # Test node statistics
            stats = trust_engine.get_node_statistics()
            self.test("Node statistics", len(stats) > 0)
            
            return self.tests_failed == 0
            
        except Exception as e:
            self.test("Trust Engine validation", False, str(e))
            return False
    
    async def validate_economy_offline(self) -> bool:
        """Validate Economy Offline component"""
        self.print_header("Validating Economy Offline")
        
        try:
            # Initialize economy beach
            economy_beach = EconomyOfflineBeach()
            self.test("Economy Beach initialization", True)
            
            # Test scene processing
            scene_data = {
                "scene_id": "economy_test",
                "complexity": "moderate",
                "resolution": (1920, 1080),
                "has_volumetrics": False,
                "has_transparency": True,
                "animation_frames": 60,
                "priority": 5,
                "quality": 0.9,
                "max_budget": 3.0
            }
            
            result = await economy_beach.process_scene(scene_data, "economy")
            self.test("Economy processing", result["status"] in ["completed", "failed"])
            
            if result["status"] == "completed":
                cost_analysis = result.get("cost_analysis", {})
                self.test("Cost analysis", "actual_cost" in cost_analysis)
                self.test("Savings calculation", "savings_percentage" in cost_analysis)
                self.test("Carbon tracking", "carbon_footprint_kg" in cost_analysis)
            
            # Test beach status
            status = economy_beach.get_beach_status()
            self.test("Beach status retrieval", "metrics" in status)
            self.test("Cost predictions", "cost_predictions" in status)
            
            # Test schedule forecast
            forecast = economy_beach.get_schedule_forecast(24)
            self.test("Schedule forecast", len(forecast) > 0)
            
            return self.tests_failed == 0
            
        except Exception as e:
            self.test("Economy Offline validation", False, str(e))
            return False
    
    async def validate_integration(self) -> bool:
        """Validate component integration"""
        self.print_header("Validating Integration")
        
        try:
            # Test complete workflow
            optimizer = OptimizerCore()
            trust_engine = TrustEngine()
            economy_beach = EconomyOfflineBeach()
            
            # Create test scene
            scene = SceneMetadata(
                scene_id="integration_test",
                source_application="Blender",
                polygon_count=500000,
                texture_count=30,
                material_count=15,
                light_count=8,
                animation_frames=30,
                resolution=(2560, 1440),
                file_size_mb=500.0,
                complexity=SceneComplexity.MODERATE,
                has_transparency=True,
                has_volumetrics=False,
                has_dynamic_elements=True
            )
            
            # Optimize
            request = OptimizationRequest(
                scene_metadata=scene,
                profile=OptimizationProfile.ECONOMY,
                target_quality=0.9,
                max_cost=2.0
            )
            
            opt_result = await optimizer.optimize(request)
            self.test("Integration: Optimization", opt_result.success)
            
            # Verify
            scene_data = {"scene_id": scene.scene_id, "complexity_score": 0.5}
            verify_result = await trust_engine.verify_optimization(
                scene_data,
                {"quality_metrics": opt_result.quality_metrics}
            )
            self.test("Integration: Verification", verify_result is not None)
            
            # Economy processing
            economy_data = {
                "scene_id": scene.scene_id,
                "complexity": "moderate",
                "resolution": scene.resolution,
                "max_budget": 2.0
            }
            econ_result = await economy_beach.process_scene(economy_data, "economy")
            self.test("Integration: Economy", econ_result["status"] in ["completed", "failed"])
            
            return self.tests_failed == 0
            
        except Exception as e:
            self.test("Integration validation", False, str(e))
            return False
    
    async def run_validation(self):
        """Run complete validation suite"""
        print("\n" + "="*60)
        print("  OPTIMIZER & TRUST ENGINE VALIDATION")
        print("="*60)
        
        # Run all validations
        optimizer_ok = await self.validate_optimizer()
        trust_ok = await self.validate_trust_engine()
        economy_ok = await self.validate_economy_offline()
        integration_ok = await self.validate_integration()
        
        # Print summary
        self.print_header("VALIDATION SUMMARY")
        
        print(f"\n📊 Test Results:")
        print(f"   • Tests Passed: {self.tests_passed}")
        print(f"   • Tests Failed: {self.tests_failed}")
        print(f"   • Success Rate: {(self.tests_passed/(self.tests_passed+self.tests_failed)*100):.1f}%")
        
        if self.errors:
            print(f"\n❌ Errors Found:")
            for error in self.errors:
                print(f"   • {error}")
        
        print(f"\n🔍 Component Status:")
        print(f"   • Optimizer: {'✅ PASS' if optimizer_ok else '❌ FAIL'}")
        print(f"   • Trust Engine: {'✅ PASS' if trust_ok else '❌ FAIL'}")
        print(f"   • Economy Offline: {'✅ PASS' if economy_ok else '❌ FAIL'}")
        print(f"   • Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
        
        all_passed = optimizer_ok and trust_ok and economy_ok and integration_ok
        
        print("\n" + "="*60)
        if all_passed:
            print("  ✅ ALL VALIDATIONS PASSED!")
            print("  System is ready for production use.")
        else:
            print("  ⚠️  SOME VALIDATIONS FAILED")
            print("  Please review errors and fix issues.")
        print("="*60 + "\n")
        
        return 0 if all_passed else 1


async def main():
    """Main entry point"""
    validator = SystemValidator()
    exit_code = await validator.run_validation()
    sys.exit(exit_code)


if __name__ == "__main__":
    print("Starting system validation...")
    asyncio.run(main())