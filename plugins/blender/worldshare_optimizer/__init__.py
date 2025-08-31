# WorldShare Optimizer Plugin for Blender
# Economy Offline Scene Analysis and Optimization
# ENHANCED: Added blocking execution warning for M0

bl_info = {
    "name": "WorldShare Optimizer",
    "author": "WorldShare Team",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > WorldShare",
    "description": "Distributed GPU optimization for 3D scenes with Economy Offline processing",
    "category": "Scene",
    "doc_url": "https://docs.worldshare.dev/blender-plugin",
    "tracker_url": "https://github.com/worldshare/blender-plugin/issues",
}

import bpy
from bpy.types import Panel, Operator, PropertyGroup, AddonPreferences
from bpy.props import (
    StringProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    PointerProperty
)
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from mathutils import Vector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WorldShareOptimizer")

# API Configuration
API_BASE_URL = "https://api.worldshare.dev/v1"
API_TIMEOUT = 30

# Scene Analysis Thresholds
COMPLEXITY_THRESHOLDS = {
    "simple": {"vertices": 10000, "objects": 50, "materials": 10},
    "moderate": {"vertices": 100000, "objects": 200, "materials": 50},
    "complex": {"vertices": 1000000, "objects": 1000, "materials": 200},
    "extreme": {"vertices": float('inf'), "objects": float('inf'), "materials": float('inf')}
}

class SceneAnalyzer:
    """Analyzes Blender scenes for optimization routing"""
    
    @staticmethod
    def analyze_scene(context) -> Dict:
        """Perform comprehensive scene analysis"""
        scene = context.scene
        
        # Collect scene statistics
        stats = {
            "vertex_count": 0,
            "face_count": 0,
            "object_count": 0,
            "material_count": 0,
            "texture_count": 0,
            "has_animation": False,
            "has_particles": False,
            "has_volumetrics": False,
            "has_transparency": False,
            "scene_type": "unknown",
            "complexity_level": "simple",
            "bounding_box": {"min": [0, 0, 0], "max": [0, 0, 0]}
        }
        
        # Count objects and geometry
        materials = set()
        textures = set()
        bbox_min = Vector((float('inf'), float('inf'), float('inf')))
        bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for obj in scene.objects:
            if obj.type == 'MESH':
                stats["object_count"] += 1
                mesh = obj.data
                stats["vertex_count"] += len(mesh.vertices)
                stats["face_count"] += len(mesh.polygons)
                
                # Update bounding box
                for vertex in mesh.vertices:
                    world_co = obj.matrix_world @ vertex.co
                    bbox_min.x = min(bbox_min.x, world_co.x)
                    bbox_min.y = min(bbox_min.y, world_co.y)
                    bbox_min.z = min(bbox_min.z, world_co.z)
                    bbox_max.x = max(bbox_max.x, world_co.x)
                    bbox_max.y = max(bbox_max.y, world_co.y)
                    bbox_max.z = max(bbox_max.z, world_co.z)
                
                # Collect materials
                for mat_slot in obj.material_slots:
                    if mat_slot.material:
                        materials.add(mat_slot.material.name)
                        
                        # Check for transparency
                        if mat_slot.material.use_backface_culling == False:
                            stats["has_transparency"] = True
                        
                        # Check for textures
                        if mat_slot.material.node_tree:
                            for node in mat_slot.material.node_tree.nodes:
                                if node.type == 'TEX_IMAGE':
                                    if node.image:
                                        textures.add(node.image.name)
            
            # Check for particles
            if len(obj.particle_systems) > 0:
                stats["has_particles"] = True
            
            # Check for animation
            if obj.animation_data and obj.animation_data.action:
                stats["has_animation"] = True
        
        # Check for volumetrics
        if scene.world and scene.world.node_tree:
            for node in scene.world.node_tree.nodes:
                if node.type == 'VOLUME_SCATTER' or node.type == 'VOLUME_ABSORPTION':
                    stats["has_volumetrics"] = True
        
        stats["material_count"] = len(materials)
        stats["texture_count"] = len(textures)
        stats["bounding_box"]["min"] = list(bbox_min)
        stats["bounding_box"]["max"] = list(bbox_max)
        
        # Classify scene type
        stats["scene_type"] = SceneAnalyzer._classify_scene_type(stats)
        stats["complexity_level"] = SceneAnalyzer._determine_complexity(stats)
        
        return stats
    
    @staticmethod
    def _classify_scene_type(stats: Dict) -> str:
        """Classify scene as indoor/outdoor/complex/hybrid"""
        # Simplified heuristic for M0
        bbox_size = [
            stats["bounding_box"]["max"][i] - stats["bounding_box"]["min"][i]
            for i in range(3)
        ]
        
        # Check aspect ratios
        horizontal_extent = max(bbox_size[0], bbox_size[1])
        vertical_extent = bbox_size[2]
        
        if horizontal_extent > vertical_extent * 5:
            return "outdoor"  # Wide open spaces
        elif vertical_extent > horizontal_extent * 2:
            return "indoor"  # Tall indoor spaces
        elif stats["has_particles"] or stats["has_volumetrics"]:
            return "complex"  # Effects-heavy scenes
        else:
            return "hybrid"
    
    @staticmethod
    def _determine_complexity(stats: Dict) -> str:
        """Determine scene complexity level"""
        for level, thresholds in COMPLEXITY_THRESHOLDS.items():
            if (stats["vertex_count"] <= thresholds["vertices"] and
                stats["object_count"] <= thresholds["objects"] and
                stats["material_count"] <= thresholds["materials"]):
                return level
        return "extreme"
    
    @staticmethod
    def recommend_optimization_path(stats: Dict) -> str:
        """Recommend optimization pipeline based on analysis"""
        # Heuristic-based routing (M0 simplified version)
        if stats["complexity_level"] in ["simple", "moderate"]:
            if not stats["has_animation"] and not stats["has_particles"]:
                return "baking"  # Traditional baking for simple static scenes
        
        if stats["has_volumetrics"] or stats["has_transparency"]:
            return "gaussian-splatting"  # 3DGS for complex materials
        
        if stats["scene_type"] == "outdoor" and stats["vertex_count"] > 500000:
            return "gaussian-splatting"  # 3DGS for large outdoor scenes
        
        if stats["has_animation"]:
            return "hybrid"  # Hybrid approach for animated content
        
        return "baking"  # Default to baking

class WORLDSHARE_OT_analyze_scene(Operator):
    """Analyze scene for optimization"""
    bl_idname = "worldshare.analyze_scene"
    bl_label = "Analyze Scene"
    bl_description = "Analyze current scene for optimization recommendations"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        try:
            # Perform scene analysis
            stats = SceneAnalyzer.analyze_scene(context)
            
            # Store results in scene properties
            props = context.scene.worldshare_props
            props.scene_stats = json.dumps(stats, indent=2)
            props.scene_type = stats["scene_type"]
            props.complexity_level = stats["complexity_level"]
            props.recommended_path = SceneAnalyzer.recommend_optimization_path(stats)
            
            # ENHANCEMENT: M0 blocking execution warning
            self.report({'WARNING'}, 
                       "M0: Scene analysis complete. API calls will block UI temporarily.")
            
            self.report({'INFO'}, 
                       f"Scene classified as {stats['scene_type']} "
                       f"({stats['complexity_level']} complexity). "
                       f"Recommended: {props.recommended_path}")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Analysis failed: {str(e)}")
            logger.exception("Scene analysis error")
            return {'CANCELLED'}

class WORLDSHARE_OT_submit_job(Operator):
    """Submit optimization job to WorldShare"""
    bl_idname = "worldshare.submit_job"
    bl_label = "Submit Job"
    bl_description = "Submit scene for distributed optimization"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.worldshare_props
        
        if not props.scene_stats:
            self.report({'ERROR'}, "Please analyze scene first")
            return {'CANCELLED'}
        
        try:
            stats = json.loads(props.scene_stats)
            
            # Build job manifest
            manifest = {
                "sceneId": props.scene_id or context.scene.name,
                "sceneType": props.scene_type,
                "optimizationPath": props.optimization_path,
                "verificationPolicy": {
                    "redundancyFactor": props.redundancy_factor,
                    "canaryRate": props.canary_rate,
                    "qualityTarget": {
                        "ssim": props.target_ssim,
                        "psnr": props.target_psnr,
                        "lpips": 0.05
                    }
                },
                "priority": props.job_priority,
                "metadata": stats
            }
            
            # ENHANCEMENT: Warning about blocking execution in M0
            self.report({'WARNING'}, 
                       "M0: Submitting job. This will freeze Blender temporarily. "
                       "Non-blocking execution coming in M1.")
            
            # M0: Simplified synchronous submission (blocking)
            # M1 will implement Modal Operator for non-blocking
            job_id = self._submit_job_sync(manifest, props.api_key)
            
            if job_id:
                props.job_id = job_id
                self.report({'INFO'}, f"Job submitted successfully: {job_id}")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Failed to submit job")
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Submission failed: {str(e)}")
            logger.exception("Job submission error")
            return {'CANCELLED'}
    
    def _submit_job_sync(self, manifest: Dict, api_key: str) -> Optional[str]:
        """Synchronous job submission (M0 blocking implementation)"""
        import requests
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": api_key
            }
            
            response = requests.post(
                f"{API_BASE_URL}/jobs",
                json=manifest,
                headers=headers,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 201:
                return response.json().get("jobId")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

class WorldShareProperties(PropertyGroup):
    """Plugin properties"""
    
    api_key: StringProperty(
        name="API Key",
        description="WorldShare API key",
        subtype='PASSWORD'
    )
    
    scene_id: StringProperty(
        name="Scene ID",
        description="Unique scene identifier",
        default=""
    )
    
    scene_type: StringProperty(
        name="Scene Type",
        description="Detected scene type",
        default=""
    )
    
    complexity_level: StringProperty(
        name="Complexity",
        description="Scene complexity level",
        default=""
    )
    
    scene_stats: StringProperty(
        name="Scene Statistics",
        description="JSON scene analysis data",
        default=""
    )
    
    optimization_path: EnumProperty(
        name="Optimization Path",
        description="Selected optimization pipeline",
        items=[
            ('baking', "Baking", "Traditional lightmap baking"),
            ('gaussian-splatting', "3D Gaussian Splatting", "Neural scene representation"),
            ('hybrid', "Hybrid", "Combined approach"),
            ('auto', "Auto", "Automatic selection based on analysis")
        ],
        default='auto'
    )
    
    recommended_path: StringProperty(
        name="Recommended Path",
        description="System recommended optimization path",
        default=""
    )
    
    redundancy_factor: IntProperty(
        name="Redundancy Factor",
        description="Number of redundant processing nodes",
        default=2,
        min=1,
        max=5
    )
    
    canary_rate: FloatProperty(
        name="Canary Rate",
        description="Additional validation percentage",
        default=0.1,
        min=0.0,
        max=0.5
    )
    
    target_ssim: FloatProperty(
        name="Target SSIM",
        description="Target structural similarity",
        default=0.98,
        min=0.9,
        max=1.0
    )
    
    target_psnr: FloatProperty(
        name="Target PSNR",
        description="Target peak signal-to-noise ratio (dB)",
        default=35.0,
        min=30.0,
        max=50.0
    )
    
    job_priority: IntProperty(
        name="Job Priority",
        description="Processing priority",
        default=5,
        min=1,
        max=10
    )
    
    job_id: StringProperty(
        name="Job ID",
        description="Current job identifier",
        default=""
    )

class WORLDSHARE_PT_main_panel(Panel):
    """Main UI panel"""
    bl_label = "WorldShare Optimizer"
    bl_idname = "WORLDSHARE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "WorldShare"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.worldshare_props
        
        # API Configuration
        box = layout.box()
        box.label(text="Configuration", icon='PREFERENCES')
        box.prop(props, "api_key")
        
        # Scene Analysis
        box = layout.box()
        box.label(text="Scene Analysis", icon='VIEWZOOM')
        box.operator("worldshare.analyze_scene", icon='FILE_REFRESH')
        
        if props.scene_type:
            col = box.column(align=True)
            col.label(text=f"Type: {props.scene_type}")
            col.label(text=f"Complexity: {props.complexity_level}")
            if props.recommended_path:
                col.label(text=f"Recommended: {props.recommended_path}")
        
        # Job Configuration
        box = layout.box()
        box.label(text="Optimization Settings", icon='MODIFIER')
        box.prop(props, "optimization_path")
        box.prop(props, "redundancy_factor")
        box.prop(props, "canary_rate")
        
        row = box.row(align=True)
        row.prop(props, "target_ssim")
        row.prop(props, "target_psnr")
        
        box.prop(props, "job_priority")
        
        # Job Submission
        box = layout.box()
        box.label(text="Job Management", icon='TIME')
        
        if not props.job_id:
            box.operator("worldshare.submit_job", icon='EXPORT')
        else:
            col = box.column(align=True)
            col.label(text=f"Job ID: {props.job_id[:8]}...")
            row = col.row(align=True)
            row.operator("worldshare.check_status", icon='FILE_REFRESH')
            row.operator("worldshare.cancel_job", icon='CANCEL')

# Registration
classes = [
    WorldShareProperties,
    WORLDSHARE_OT_analyze_scene,
    WORLDSHARE_OT_submit_job,
    WORLDSHARE_PT_main_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.worldshare_props = PointerProperty(type=WorldShareProperties)
    logger.info("WorldShare Optimizer plugin registered")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.worldshare_props
    logger.info("WorldShare Optimizer plugin unregistered")

if __name__ == "__main__":
    register()