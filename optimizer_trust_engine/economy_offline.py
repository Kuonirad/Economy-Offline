"""
Economy Offline Beach - Time-Shifted Processing for Cost Optimization
======================================================================

Implements the "Economy Offline" concept:
1. Time-shifted processing during off-peak hours
2. Distributed cost sharing across global GPU resources
3. Intelligent scheduling for maximum cost efficiency
4. Transparent cost analytics and reporting
"""

import asyncio
import heapq
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class TimeZone(Enum):
    """Global time zones for scheduling"""
    UTC = "UTC"
    PST = "America/Los_Angeles"    # UTC-8
    EST = "America/New_York"       # UTC-5
    CET = "Europe/Paris"            # UTC+1
    JST = "Asia/Tokyo"              # UTC+9
    IST = "Asia/Kolkata"            # UTC+5:30
    AEST = "Australia/Sydney"       # UTC+10


class ResourceType(Enum):
    """Types of GPU resources"""
    CONSUMER = "consumer"        # Consumer GPUs (RTX series)
    DATACENTER = "datacenter"    # Data center GPUs (A100, H100)
    CLOUD = "cloud"              # Cloud GPU instances
    EDGE = "edge"                # Edge computing devices
    DISTRIBUTED = "distributed"  # Distributed volunteer compute


class PricingTier(Enum):
    """Pricing tiers for processing"""
    PEAK = "peak"                # Peak hours, highest cost
    STANDARD = "standard"        # Standard hours, normal cost
    OFF_PEAK = "off_peak"       # Off-peak hours, reduced cost
    ECONOMY = "economy"          # Economy hours, minimum cost
    SPOT = "spot"                # Spot pricing, variable cost


@dataclass
class GPUResource:
    """Represents a GPU resource in the network"""
    resource_id: str
    resource_type: ResourceType
    location: TimeZone
    compute_capability: float    # TFLOPS
    memory_gb: int
    hourly_cost: float           # Base hourly cost in USD
    availability_schedule: Dict[int, bool] = field(default_factory=dict)  # Hour -> Available
    current_utilization: float = 0.0  # 0.0 to 1.0
    reliability_score: float = 0.95   # Historical reliability
    
    def get_cost_at_time(self, hour: int, tier: PricingTier) -> float:
        """Get cost at specific hour based on pricing tier"""
        base_cost = self.hourly_cost
        
        # Apply tier multipliers
        tier_multipliers = {
            PricingTier.PEAK: 2.0,
            PricingTier.STANDARD: 1.0,
            PricingTier.OFF_PEAK: 0.6,
            PricingTier.ECONOMY: 0.4,
            PricingTier.SPOT: 0.3
        }
        
        return base_cost * tier_multipliers.get(tier, 1.0)
    
    def is_available_at(self, hour: int) -> bool:
        """Check if resource is available at specific hour"""
        return self.availability_schedule.get(hour, True) and self.current_utilization < 0.9


@dataclass
class ProcessingJob:
    """Represents a processing job in the pipeline"""
    job_id: str
    scene_id: str
    submission_time: float
    deadline: Optional[float]  # Optional deadline timestamp
    estimated_compute_hours: float
    priority: int  # 1-10
    preferred_quality: float  # 0.0 to 1.0
    max_budget: float  # Maximum budget in USD
    current_status: str = "queued"
    assigned_resources: List[str] = field(default_factory=list)
    actual_cost: float = 0.0
    completion_time: Optional[float] = None
    
    def __lt__(self, other):
        """For priority queue comparison"""
        # Higher priority jobs process first
        return self.priority > other.priority


@dataclass
class CostReport:
    """Cost analysis and savings report"""
    job_id: str
    baseline_cost: float      # What it would cost normally
    optimized_cost: float     # What it actually cost
    savings_amount: float     # Dollar savings
    savings_percentage: float # Percentage saved
    processing_time: float    # Hours taken
    resources_used: List[str]
    time_shift_hours: int     # Hours shifted from optimal time
    carbon_footprint: float   # Estimated CO2 in kg


class TimeShiftedPipeline:
    """
    Implements time-shifted processing for cost optimization
    Schedules jobs during off-peak hours across global resources
    """
    
    def __init__(self):
        """Initialize time-shifted pipeline"""
        self.job_queue = []  # Priority queue
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingJob] = {}
        self.resource_pool: Dict[str, GPUResource] = {}
        self.schedule_horizon = 168  # 1 week in hours
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize available GPU resources"""
        # Create diverse resource pool
        resource_configs = [
            # Consumer GPUs - widely available, lower cost
            ("consumer_rtx4090_us", ResourceType.CONSUMER, TimeZone.PST, 82.6, 24, 0.5),
            ("consumer_rtx4090_eu", ResourceType.CONSUMER, TimeZone.CET, 82.6, 24, 0.45),
            ("consumer_rtx4090_asia", ResourceType.CONSUMER, TimeZone.JST, 82.6, 24, 0.4),
            
            # Data center GPUs - high performance, higher cost
            ("datacenter_a100_us", ResourceType.DATACENTER, TimeZone.EST, 312.0, 80, 2.5),
            ("datacenter_h100_eu", ResourceType.DATACENTER, TimeZone.CET, 700.0, 80, 4.0),
            
            # Cloud GPUs - elastic, variable cost
            ("cloud_t4_global", ResourceType.CLOUD, TimeZone.UTC, 8.1, 16, 0.35),
            ("cloud_v100_aws", ResourceType.CLOUD, TimeZone.PST, 112.0, 32, 1.2),
            
            # Edge devices - distributed, very low cost
            ("edge_jetson_network", ResourceType.EDGE, TimeZone.UTC, 1.3, 8, 0.05),
            
            # Distributed volunteer compute - lowest cost
            ("distributed_folding", ResourceType.DISTRIBUTED, TimeZone.UTC, 20.0, 16, 0.1),
        ]
        
        for config in resource_configs:
            resource_id, rtype, location, compute, memory, cost = config
            resource = GPUResource(
                resource_id=resource_id,
                resource_type=rtype,
                location=location,
                compute_capability=compute,
                memory_gb=memory,
                hourly_cost=cost
            )
            
            # Generate availability schedule (simulate based on location)
            resource.availability_schedule = self._generate_availability_schedule(location)
            self.resource_pool[resource_id] = resource
    
    def _generate_availability_schedule(self, location: TimeZone) -> Dict[int, bool]:
        """Generate availability schedule based on location and typical usage patterns"""
        schedule = {}
        
        # Define peak hours for each timezone (local time)
        peak_hours = {
            TimeZone.PST: list(range(9, 18)),     # 9 AM - 6 PM PST
            TimeZone.EST: list(range(9, 18)),     # 9 AM - 6 PM EST
            TimeZone.CET: list(range(9, 18)),     # 9 AM - 6 PM CET
            TimeZone.JST: list(range(9, 18)),     # 9 AM - 6 PM JST
            TimeZone.IST: list(range(10, 19)),    # 10 AM - 7 PM IST
            TimeZone.AEST: list(range(9, 18)),    # 9 AM - 6 PM AEST
            TimeZone.UTC: list(range(12, 20)),    # 12 PM - 8 PM UTC
        }
        
        local_peak = peak_hours.get(location, list(range(9, 18)))
        
        # Generate weekly schedule
        for hour in range(168):  # 168 hours in a week
            day_of_week = hour // 24
            hour_of_day = hour % 24
            
            # Weekends have better availability
            if day_of_week in [5, 6]:  # Saturday, Sunday
                schedule[hour] = True
            # Weekdays depend on hour
            elif hour_of_day in local_peak:
                # 70% chance of availability during peak hours
                schedule[hour] = np.random.random() > 0.3
            else:
                # 95% chance of availability during off-peak
                schedule[hour] = np.random.random() > 0.05
        
        return schedule
    
    def submit_job(self, job: ProcessingJob) -> str:
        """Submit a job to the time-shifted pipeline"""
        logger.info(f"Submitting job {job.job_id} to time-shifted pipeline")
        
        # Add to priority queue
        heapq.heappush(self.job_queue, job)
        self.active_jobs[job.job_id] = job
        
        return job.job_id
    
    def find_optimal_schedule(self, job: ProcessingJob) -> Tuple[List[Tuple[str, int, float]], float]:
        """
        Find optimal schedule for job processing
        Returns list of (resource_id, start_hour, duration) and total cost
        """
        required_compute = job.estimated_compute_hours
        deadline_hour = self._timestamp_to_hour(job.deadline) if job.deadline else self.schedule_horizon
        
        # Build cost matrix for dynamic programming
        cost_matrix = self._build_cost_matrix(required_compute, deadline_hour)
        
        # Find minimum cost path
        schedule = self._find_minimum_cost_path(cost_matrix, required_compute, job.max_budget)
        
        # Calculate total cost
        total_cost = sum(cost for _, _, cost in schedule)
        
        return schedule, total_cost
    
    def _build_cost_matrix(self, compute_hours: float, deadline: int) -> np.ndarray:
        """Build cost matrix for scheduling optimization"""
        num_resources = len(self.resource_pool)
        matrix = np.full((num_resources, deadline), np.inf)
        
        resource_list = list(self.resource_pool.values())
        
        for r_idx, resource in enumerate(resource_list):
            for hour in range(deadline):
                if resource.is_available_at(hour):
                    # Determine pricing tier based on hour
                    tier = self._get_pricing_tier(hour)
                    cost_per_hour = resource.get_cost_at_time(hour, tier)
                    
                    # Factor in compute capability (faster = fewer hours needed)
                    actual_hours = compute_hours * (100 / resource.compute_capability)
                    total_cost = cost_per_hour * actual_hours
                    
                    matrix[r_idx, hour] = total_cost
        
        return matrix
    
    def _get_pricing_tier(self, hour: int) -> PricingTier:
        """Determine pricing tier based on hour of week"""
        day_of_week = hour // 24
        hour_of_day = hour % 24
        
        # Weekend = economy pricing
        if day_of_week in [5, 6]:
            return PricingTier.ECONOMY
        
        # Weekday pricing based on hour
        if 0 <= hour_of_day < 6:
            return PricingTier.ECONOMY
        elif 6 <= hour_of_day < 9:
            return PricingTier.OFF_PEAK
        elif 9 <= hour_of_day < 17:
            return PricingTier.PEAK
        elif 17 <= hour_of_day < 20:
            return PricingTier.STANDARD
        else:
            return PricingTier.OFF_PEAK
    
    def _find_minimum_cost_path(self, cost_matrix: np.ndarray, compute_hours: float, max_budget: float) -> List[Tuple[str, int, float]]:
        """Find minimum cost path through the scheduling matrix"""
        schedule = []
        resource_list = list(self.resource_pool.keys())
        
        # Simple greedy approach: find cheapest available slot
        min_cost = np.inf
        best_resource = None
        best_hour = None
        
        for r_idx in range(cost_matrix.shape[0]):
            for hour in range(cost_matrix.shape[1]):
                cost = cost_matrix[r_idx, hour]
                if cost < min_cost and cost <= max_budget:
                    min_cost = cost
                    best_resource = resource_list[r_idx]
                    best_hour = hour
        
        if best_resource is not None:
            schedule.append((best_resource, best_hour, min_cost))
        
        return schedule
    
    def _timestamp_to_hour(self, timestamp: float) -> int:
        """Convert timestamp to hour offset from now"""
        current_time = time.time()
        hours_from_now = (timestamp - current_time) / 3600
        return min(int(hours_from_now), self.schedule_horizon)
    
    async def execute_scheduled_jobs(self) -> List[ProcessingJob]:
        """Execute jobs according to optimal schedule"""
        executed_jobs = []
        
        while self.job_queue:
            job = heapq.heappop(self.job_queue)
            
            # Find optimal schedule
            schedule, cost = self.find_optimal_schedule(job)
            
            if schedule:
                # Simulate job execution
                job.assigned_resources = [res_id for res_id, _, _ in schedule]
                job.actual_cost = cost
                job.current_status = "processing"
                
                # Simulate processing delay
                await asyncio.sleep(0.5)
                
                # Mark as complete
                job.current_status = "completed"
                job.completion_time = time.time()
                
                # Move to completed
                del self.active_jobs[job.job_id]
                self.completed_jobs[job.job_id] = job
                executed_jobs.append(job)
                
                logger.info(f"Job {job.job_id} completed with cost ${cost:.2f}")
            else:
                logger.warning(f"Could not schedule job {job.job_id}")
                job.current_status = "failed"
        
        return executed_jobs


class CostOptimizer:
    """
    Intelligent cost optimization engine
    Analyzes patterns and provides cost-saving recommendations
    """
    
    def __init__(self):
        """Initialize cost optimizer"""
        self.cost_history = []
        self.optimization_patterns = defaultdict(list)
        self.baseline_costs = {
            "simple": 0.8,
            "moderate": 1.5,
            "complex": 2.4,
            "extreme": 4.0
        }
    
    def analyze_cost_savings(self, job: ProcessingJob, complexity: str = "moderate") -> CostReport:
        """Analyze cost savings for a completed job"""
        baseline = self.baseline_costs.get(complexity, 2.4)
        
        # Calculate savings
        savings = baseline - job.actual_cost
        savings_pct = (savings / baseline) * 100 if baseline > 0 else 0
        
        # Estimate carbon footprint (simplified)
        carbon_kg = job.actual_cost * 0.5  # 0.5 kg CO2 per dollar (example)
        
        report = CostReport(
            job_id=job.job_id,
            baseline_cost=baseline,
            optimized_cost=job.actual_cost,
            savings_amount=savings,
            savings_percentage=savings_pct,
            processing_time=job.estimated_compute_hours,
            resources_used=job.assigned_resources,
            time_shift_hours=24,  # Example shift
            carbon_footprint=carbon_kg
        )
        
        # Store in history
        self.cost_history.append(report)
        
        return report
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations based on history"""
        if not self.cost_history:
            return ["Insufficient data for recommendations"]
        
        recommendations = []
        
        # Analyze average savings
        avg_savings = np.mean([r.savings_percentage for r in self.cost_history])
        if avg_savings < 50:
            recommendations.append(f"Consider more aggressive time-shifting. Current average savings: {avg_savings:.1f}%")
        
        # Analyze resource usage patterns
        resource_usage = defaultdict(int)
        for report in self.cost_history:
            for resource in report.resources_used:
                resource_usage[resource] += 1
        
        # Find underutilized resources
        total_jobs = len(self.cost_history)
        for resource, count in resource_usage.items():
            usage_rate = count / total_jobs
            if usage_rate < 0.2:
                recommendations.append(f"Resource '{resource}' is underutilized ({usage_rate*100:.1f}%). Consider rebalancing.")
        
        # Carbon footprint analysis
        total_carbon = sum(r.carbon_footprint for r in self.cost_history)
        if total_carbon > 100:
            recommendations.append(f"High carbon footprint detected ({total_carbon:.1f} kg CO2). Consider renewable energy sources.")
        
        # Time shift analysis
        avg_shift = np.mean([r.time_shift_hours for r in self.cost_history])
        if avg_shift < 12:
            recommendations.append(f"Average time shift is only {avg_shift:.1f} hours. Greater shifts could yield more savings.")
        
        return recommendations
    
    def predict_cost(self, compute_hours: float, complexity: str, use_economy: bool = True) -> Dict[str, float]:
        """Predict cost for a given workload"""
        baseline = self.baseline_costs.get(complexity, 2.4) * (compute_hours / 4.5)
        
        if use_economy:
            # Apply economy optimization factors
            time_shift_factor = 0.6  # 40% reduction for time shifting
            resource_optimization = 0.85  # 15% reduction for optimal resource selection
            scale_factor = 0.95 if compute_hours > 10 else 1.0  # Volume discount
            
            optimized = baseline * time_shift_factor * resource_optimization * scale_factor
        else:
            optimized = baseline
        
        return {
            "baseline_cost": baseline,
            "optimized_cost": optimized,
            "estimated_savings": baseline - optimized,
            "savings_percentage": ((baseline - optimized) / baseline) * 100 if baseline > 0 else 0
        }


class EconomyOfflineBeach:
    """
    Main orchestrator for the Economy Offline processing beach
    Coordinates time-shifted pipelines and cost optimization
    """
    
    def __init__(self):
        """Initialize Economy Offline Beach"""
        self.pipeline = TimeShiftedPipeline()
        self.cost_optimizer = CostOptimizer()
        self.active_beaches = {}  # Track active processing beaches
        self.metrics = {
            "total_jobs_processed": 0,
            "total_cost_saved": 0.0,
            "average_time_shift": 0.0,
            "carbon_footprint_total": 0.0
        }
    
    async def process_scene(self, scene_data: Dict, optimization_profile: str) -> Dict:
        """
        Process a scene through the Economy Offline beach
        Returns processing result with cost analysis
        """
        # Create processing job
        job = ProcessingJob(
            job_id=f"eco_{scene_data.get('scene_id', 'unknown')}_{int(time.time())}",
            scene_id=scene_data.get("scene_id", "unknown"),
            submission_time=time.time(),
            deadline=time.time() + 86400,  # 24 hour deadline
            estimated_compute_hours=self._estimate_compute_hours(scene_data),
            priority=scene_data.get("priority", 5),
            preferred_quality=scene_data.get("quality", 0.9),
            max_budget=scene_data.get("max_budget", 5.0)
        )
        
        # Submit to pipeline
        job_id = self.pipeline.submit_job(job)
        logger.info(f"Submitted job {job_id} to Economy Offline beach")
        
        # Execute scheduled jobs
        executed = await self.pipeline.execute_scheduled_jobs()
        
        # Find our job in executed list
        our_job = next((j for j in executed if j.job_id == job_id), None)
        
        if our_job:
            # Analyze cost savings
            complexity = scene_data.get("complexity", "moderate")
            cost_report = self.cost_optimizer.analyze_cost_savings(our_job, complexity)
            
            # Update metrics
            self._update_metrics(cost_report)
            
            # Generate result
            result = {
                "job_id": job_id,
                "scene_id": scene_data.get("scene_id"),
                "status": "completed",
                "processing_time": our_job.estimated_compute_hours,
                "cost_analysis": {
                    "baseline_cost": cost_report.baseline_cost,
                    "actual_cost": cost_report.optimized_cost,
                    "savings": cost_report.savings_amount,
                    "savings_percentage": cost_report.savings_percentage,
                    "carbon_footprint_kg": cost_report.carbon_footprint
                },
                "resources_used": our_job.assigned_resources,
                "optimization_recommendations": self.cost_optimizer.generate_optimization_recommendations()
            }
            
            return result
        else:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": "Job execution failed"
            }
    
    def _estimate_compute_hours(self, scene_data: Dict) -> float:
        """Estimate compute hours based on scene complexity"""
        base_hours = 4.5  # Baseline for moderate complexity
        
        complexity_multipliers = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 2.0,
            "extreme": 4.0
        }
        
        complexity = scene_data.get("complexity", "moderate")
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        # Adjust for resolution
        resolution = scene_data.get("resolution", (1920, 1080))
        pixel_count = resolution[0] * resolution[1]
        resolution_factor = pixel_count / (1920 * 1080)  # Normalize to 1080p
        
        # Adjust for special features
        feature_factor = 1.0
        if scene_data.get("has_volumetrics"):
            feature_factor *= 1.5
        if scene_data.get("has_transparency"):
            feature_factor *= 1.2
        if scene_data.get("animation_frames", 0) > 0:
            feature_factor *= 1.3
        
        estimated_hours = base_hours * multiplier * resolution_factor * feature_factor
        
        return estimated_hours
    
    def _update_metrics(self, cost_report: CostReport):
        """Update beach metrics"""
        self.metrics["total_jobs_processed"] += 1
        self.metrics["total_cost_saved"] += cost_report.savings_amount
        self.metrics["carbon_footprint_total"] += cost_report.carbon_footprint
        
        # Update average time shift
        n = self.metrics["total_jobs_processed"]
        current_avg = self.metrics["average_time_shift"]
        self.metrics["average_time_shift"] = (current_avg * (n-1) + cost_report.time_shift_hours) / n
    
    def get_beach_status(self) -> Dict:
        """Get current status of the Economy Offline beach"""
        return {
            "active_jobs": len(self.pipeline.active_jobs),
            "completed_jobs": len(self.pipeline.completed_jobs),
            "queued_jobs": len(self.pipeline.job_queue),
            "available_resources": len([r for r in self.pipeline.resource_pool.values() if r.current_utilization < 0.9]),
            "metrics": self.metrics.copy(),
            "cost_predictions": {
                "simple": self.cost_optimizer.predict_cost(2.0, "simple"),
                "moderate": self.cost_optimizer.predict_cost(4.5, "moderate"),
                "complex": self.cost_optimizer.predict_cost(9.0, "complex"),
                "extreme": self.cost_optimizer.predict_cost(18.0, "extreme")
            }
        }
    
    def get_schedule_forecast(self, hours_ahead: int = 24) -> Dict[str, List[Dict]]:
        """Get scheduling forecast for next N hours"""
        forecast = defaultdict(list)
        
        for hour in range(hours_ahead):
            tier = self.pipeline._get_pricing_tier(hour)
            
            for resource_id, resource in self.pipeline.resource_pool.items():
                if resource.is_available_at(hour):
                    forecast[resource_id].append({
                        "hour": hour,
                        "pricing_tier": tier.value,
                        "cost_per_hour": resource.get_cost_at_time(hour, tier),
                        "available": True
                    })
        
        return dict(forecast)