#!/usr/bin/env python3
"""
Interactive Demo of the Optimizer and Trust Engine Framework
=============================================================

This demo showcases the complete workflow from scene submission
through optimization, verification, and cost analysis.
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict

from optimizer import (
    OptimizerCore,
    SceneMetadata,
    SceneComplexity,
    OptimizationProfile,
    OptimizationRequest
)
from trust_engine import TrustEngine
from economy_offline import EconomyOfflineBeach


class InteractiveDemo:
    """Interactive demonstration of the framework"""
    
    def __init__(self):
        self.optimizer = OptimizerCore()
        self.trust_engine = TrustEngine()
        self.economy_beach = EconomyOfflineBeach()
    
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n▶ {title}")
        print("-" * 40)
    
    async def demo_workflow(self):
        """Run the complete demo workflow"""
        self.print_header("OPTIMIZER & TRUST ENGINE FRAMEWORK DEMO")
        print("\nWelcome to the GPU Optimization System demonstration!")
        print("This demo will walk you through the complete optimization pipeline.\n")
        
        # Step 1: Scene Creation
        self.print_section("Step 1: Scene Analysis & Classification")
        
        scene_metadata = SceneMetadata(
            scene_id="demo_scene_2024",
            source_application="Blender",
            polygon_count=1500000,
            texture_count=65,
            material_count=35,
            light_count=12,
            animation_frames=120,
            resolution=(2560, 1440),
            file_size_mb=850.0,
            complexity=SceneComplexity.MODERATE,
            has_transparency=True,
            has_volumetrics=False,
            has_dynamic_elements=True
        )
        
        print(f"📊 Scene: {scene_metadata.scene_id}")
        print(f"   • Polygons: {scene_metadata.polygon_count:,}")
        print(f"   • Materials: {scene_metadata.material_count}")
        print(f"   • Lights: {scene_metadata.light_count}")
        print(f"   • Resolution: {scene_metadata.resolution[0]}x{scene_metadata.resolution[1]}")
        print(f"   • Complexity: {scene_metadata.complexity.value}")
        
        await asyncio.sleep(1)
        
        # Step 2: Optimization
        self.print_section("Step 2: Intelligent Optimization Routing")
        
        print("🎯 Available optimization profiles:")
        print("   1. QUALITY   - Maximum visual fidelity")
        print("   2. BALANCED  - Optimal quality/cost ratio")
        print("   3. ECONOMY   - Minimum cost processing")
        
        profile = OptimizationProfile.BALANCED
        print(f"\n✓ Selected profile: {profile.value}")
        
        request = OptimizationRequest(
            scene_metadata=scene_metadata,
            profile=profile,
            target_quality=0.95,
            max_processing_time=600.0,
            max_cost=2.0,
            priority=8
        )
        
        print("\n🔄 Processing optimization request...")
        optimization_result = await self.optimizer.optimize(request)
        
        print(f"\n✅ Optimization Complete!")
        print(f"   • Pipeline: {optimization_result.optimization_path}")
        print(f"   • Success: {optimization_result.success}")
        print(f"   • Estimated Cost: ${optimization_result.estimated_cost:.2f}")
        print(f"   • Quality Metrics:")
        for metric, value in optimization_result.quality_metrics.items():
            print(f"     - {metric.upper()}: {value:.3f}")
        print(f"   • File Size Reduction: {optimization_result.file_size_reduction:.1f}%")
        print(f"   • Performance Gain: {optimization_result.performance_gain:.1f}%")
        
        await asyncio.sleep(1)
        
        # Step 3: Trust Engine Verification
        self.print_section("Step 3: Trust Engine Verification")
        
        print("🔐 Initiating multi-node verification...")
        print("   • Selecting verification nodes")
        print("   • Performing probabilistic sampling")
        print("   • Calculating consensus")
        
        scene_data = {
            "scene_id": scene_metadata.scene_id,
            "complexity_score": 0.6,
            "specialization": "general"
        }
        
        verification_result = await self.trust_engine.verify_optimization(
            scene_data,
            {
                "optimization_path": optimization_result.optimization_path,
                "quality_metrics": optimization_result.quality_metrics
            }
        )
        
        print(f"\n✅ Verification Complete!")
        print(f"   • Status: {verification_result.status.value}")
        print(f"   • Consensus Score: {verification_result.consensus_score:.2%}")
        print(f"   • Participating Nodes: {len(verification_result.participating_nodes)}")
        print(f"   • Quality Threshold: {'PASSED ✓' if verification_result.passed_thresholds else 'FAILED ✗'}")
        
        if verification_result.byzantine_nodes_detected:
            print(f"   ⚠️  Byzantine nodes detected: {len(verification_result.byzantine_nodes_detected)}")
        
        await asyncio.sleep(1)
        
        # Step 4: Economy Offline Processing
        self.print_section("Step 4: Economy Offline Beach Processing")
        
        print("💰 Analyzing cost optimization opportunities...")
        print("   • Checking global resource availability")
        print("   • Calculating time-shift benefits")
        print("   • Scheduling for off-peak processing")
        
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
        
        cost_analysis = economy_result.get('cost_analysis', {
            'baseline_cost': 2.4,
            'actual_cost': 0.8,
            'savings': 1.6,
            'savings_percentage': 66.7,
            'carbon_footprint_kg': 0.4
        })
        
        if economy_result['status'] == 'completed':
            print(f"\n✅ Economy Processing Complete!")
            print(f"   • Baseline Cost: ${cost_analysis['baseline_cost']:.2f}")
            print(f"   • Optimized Cost: ${cost_analysis['actual_cost']:.2f}")
            print(f"   • Savings: ${cost_analysis['savings']:.2f} ({cost_analysis['savings_percentage']:.1f}%)")
            print(f"   • Carbon Footprint: {cost_analysis['carbon_footprint_kg']:.2f} kg CO2")
            
            print("\n📋 Optimization Recommendations:")
            for i, rec in enumerate(economy_result.get('optimization_recommendations', ['Insufficient data for recommendations'])[:3], 1):
                print(f"   {i}. {rec}")
        else:
            print(f"\n⚠️ Economy Processing Status: {economy_result['status']}")
            print("   Using estimated values for demonstration")
        
        await asyncio.sleep(1)
        
        # Final Summary
        self.print_header("OPTIMIZATION SUMMARY")
        
        # Get system metrics
        optimizer_metrics = self.optimizer.get_metrics()
        beach_status = self.economy_beach.get_beach_status()
        
        print("\n📊 Session Statistics:")
        print(f"   • Scenes Processed: {optimizer_metrics['total_scenes_processed']}")
        print(f"   • Average Quality Score: {optimizer_metrics['average_quality_score']:.3f}")
        print(f"   • Total Processing Time: {optimizer_metrics['total_processing_time']:.1f}s")
        print(f"   • Total Cost Saved: ${beach_status['metrics']['total_cost_saved']:.2f}")
        
        print("\n🎯 Key Achievements:")
        if cost_analysis['savings_percentage'] > 50:
            print(f"   ⭐ Exceptional cost reduction: {cost_analysis['savings_percentage']:.1f}%")
        if verification_result.consensus_score > 0.9:
            print(f"   ⭐ High consensus validation: {verification_result.consensus_score:.2%}")
        if optimization_result.quality_metrics.get('ssim', 0) > 0.98:
            print(f"   ⭐ Superior quality maintained: SSIM {optimization_result.quality_metrics['ssim']:.3f}")
        
        print("\n💡 Next Steps:")
        print("   1. Deploy optimized assets to production")
        print("   2. Monitor real-world performance metrics")
        print("   3. Iterate based on user feedback")
        
        # Cost prediction for different scenarios
        self.print_section("Cost Predictions for Different Workloads")
        
        predictions = beach_status['cost_predictions']
        print("\n📈 Estimated costs with Economy Offline processing:")
        for complexity, pred in predictions.items():
            print(f"   • {complexity.capitalize()} scenes:")
            print(f"     - Baseline: ${pred['baseline_cost']:.2f}")
            print(f"     - Optimized: ${pred['optimized_cost']:.2f}")
            print(f"     - Savings: {pred['savings_percentage']:.1f}%")
        
        print("\n" + "="*70)
        print("  Thank you for using the Optimizer & Trust Engine Framework!")
        print("  Building the future of distributed, sustainable rendering 🌟")
        print("="*70 + "\n")


async def main():
    """Main entry point for the demo"""
    demo = InteractiveDemo()
    
    try:
        await demo.demo_workflow()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Optimizer & Trust Engine Framework Demo...")
    print("This will demonstrate the complete optimization pipeline.\n")
    asyncio.run(main())