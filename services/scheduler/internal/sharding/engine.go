package sharding

import (
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "math"
    
    "github.com/google/uuid"
    "github.com/sirupsen/logrus"
)

var log = logrus.New()

// ShardingStrategy defines how to split work
type ShardingStrategy string

const (
    StrategySceneBased  ShardingStrategy = "scene-based"
    StrategyTemporal    ShardingStrategy = "temporal"
    StrategyGeometric   ShardingStrategy = "geometric"
    StrategyHybrid      ShardingStrategy = "hybrid"
)

// Shard represents a unit of work
type Shard struct {
    ID               string
    JobID            string
    Index            int
    TotalShards      int
    Type             ShardType
    Data             ShardData
    EstimatedCost    float64
    EstimatedTimeMs  int64
    Priority         int
}

// ShardType defines the type of processing needed
type ShardType string

const (
    ShardTypeBaking    ShardType = "baking"
    ShardType3DGS      ShardType = "3dgs"
    ShardTypeHybrid    ShardType = "hybrid"
)

// ShardData contains the actual work specification
type ShardData struct {
    SceneSegment    SceneSegment    `json:"sceneSegment"`
    ProcessingHints ProcessingHints `json:"processingHints"`
    InputURL        string          `json:"inputUrl"`
    OutputFormat    string          `json:"outputFormat"`
}

// SceneSegment defines the portion of the scene to process
type SceneSegment struct {
    BoundingBox     BoundingBox `json:"boundingBox"`
    ObjectIDs       []string    `json:"objectIds"`
    FrameRange      FrameRange  `json:"frameRange"`
    VertexRange     VertexRange `json:"vertexRange"`
    LODLevel        int         `json:"lodLevel"`
}

// BoundingBox defines spatial boundaries
type BoundingBox struct {
    Min [3]float64 `json:"min"`
    Max [3]float64 `json:"max"`
}

// FrameRange for temporal sharding
type FrameRange struct {
    Start int `json:"start"`
    End   int `json:"end"`
}

// VertexRange for geometric sharding
type VertexRange struct {
    Start int `json:"start"`
    End   int `json:"end"`
}

// ProcessingHints provides optimization guidance
type ProcessingHints struct {
    PreferredMethod  string            `json:"preferredMethod"`
    QualityPreset    string            `json:"qualityPreset"`
    MaxMemoryMB      int               `json:"maxMemoryMb"`
    MaxProcessingMs  int64             `json:"maxProcessingMs"`
    CustomParameters map[string]string `json:"customParameters"`
}

// Engine is the sharding engine
type Engine struct {
    config Config
}

// Config holds sharding engine configuration
type Config struct {
    MaxShardSize     int
    MinShardSize     int
    Strategy         string
    FrameChunkSize   int
    SpatialDivisions int
}

// NewEngine creates a new sharding engine
func NewEngine(config Config) *Engine {
    // Set defaults
    if config.MaxShardSize == 0 {
        config.MaxShardSize = 10000 // vertices
    }
    if config.MinShardSize == 0 {
        config.MinShardSize = 100
    }
    if config.Strategy == "" {
        config.Strategy = string(StrategyHybrid)
    }
    if config.FrameChunkSize == 0 {
        config.FrameChunkSize = 30
    }
    if config.SpatialDivisions == 0 {
        config.SpatialDivisions = 8
    }
    
    return &Engine{
        config: config,
    }
}

// GenerateShards creates shards from a job manifest
func (e *Engine) GenerateShards(manifest interface{}) ([]Shard, error) {
    // Type assertion for the manifest
    manifestMap, ok := manifest.(map[string]interface{})
    if !ok {
        // Try to handle the actual JobManifest type
        // For M0, we'll create a simple sharding strategy
        return e.generateDefaultShards(manifest)
    }
    
    sceneType, _ := manifestMap["sceneType"].(string)
    optimizationPath, _ := manifestMap["optimizationPath"].(string)
    
    log.WithFields(logrus.Fields{
        "sceneType":        sceneType,
        "optimizationPath": optimizationPath,
        "strategy":         e.config.Strategy,
    }).Debug("Generating shards")
    
    switch ShardingStrategy(e.config.Strategy) {
    case StrategySceneBased:
        return e.generateSceneBasedShards(manifestMap)
    case StrategyTemporal:
        return e.generateTemporalShards(manifestMap)
    case StrategyGeometric:
        return e.generateGeometricShards(manifestMap)
    case StrategyHybrid:
        return e.generateHybridShards(manifestMap)
    default:
        return e.generateDefaultShards(manifestMap)
    }
}

// generateDefaultShards creates a simple sharding strategy for M0
func (e *Engine) generateDefaultShards(manifest interface{}) ([]Shard, error) {
    jobID := uuid.New().String()
    
    // For M0, create a fixed number of shards
    numShards := 4
    shards := make([]Shard, 0, numShards)
    
    for i := 0; i < numShards; i++ {
        shard := Shard{
            ID:          e.generateShardID(jobID, i),
            JobID:       jobID,
            Index:       i,
            TotalShards: numShards,
            Type:        ShardTypeBaking, // Default to baking for M0
            Data: ShardData{
                SceneSegment: SceneSegment{
                    BoundingBox: BoundingBox{
                        Min: [3]float64{float64(i) * 10, 0, 0},
                        Max: [3]float64{float64(i+1) * 10, 10, 10},
                    },
                    LODLevel: 0,
                },
                ProcessingHints: ProcessingHints{
                    PreferredMethod: "baking",
                    QualityPreset:   "balanced",
                    MaxMemoryMB:     4096,
                    MaxProcessingMs: 300000, // 5 minutes
                },
                OutputFormat: "optimized",
            },
            EstimatedCost:   10.0,
            EstimatedTimeMs: 120000, // 2 minutes
            Priority:        5,
        }
        shards = append(shards, shard)
    }
    
    log.WithField("shardCount", len(shards)).Info("Generated default shards")
    return shards, nil
}

// generateSceneBasedShards divides work based on scene structure
func (e *Engine) generateSceneBasedShards(manifest map[string]interface{}) ([]Shard, error) {
    metadata, _ := manifest["metadata"].(map[string]interface{})
    vertexCount := 0
    if vc, ok := metadata["vertex_count"].(float64); ok {
        vertexCount = int(vc)
    }
    
    // Calculate number of shards based on vertex count
    numShards := int(math.Ceil(float64(vertexCount) / float64(e.config.MaxShardSize)))
    if numShards < 1 {
        numShards = 1
    }
    
    jobID := uuid.New().String()
    shards := make([]Shard, 0, numShards)
    verticesPerShard := vertexCount / numShards
    
    for i := 0; i < numShards; i++ {
        startVertex := i * verticesPerShard
        endVertex := startVertex + verticesPerShard
        if i == numShards-1 {
            endVertex = vertexCount
        }
        
        shard := Shard{
            ID:          e.generateShardID(jobID, i),
            JobID:       jobID,
            Index:       i,
            TotalShards: numShards,
            Type:        e.determineShardType(manifest),
            Data: ShardData{
                SceneSegment: SceneSegment{
                    VertexRange: VertexRange{
                        Start: startVertex,
                        End:   endVertex,
                    },
                    LODLevel: 0,
                },
                ProcessingHints: e.generateProcessingHints(manifest),
                OutputFormat:    "optimized",
            },
            EstimatedCost:   e.estimateCost(verticesPerShard),
            EstimatedTimeMs: e.estimateTime(verticesPerShard),
            Priority:        e.calculatePriority(manifest),
        }
        shards = append(shards, shard)
    }
    
    return shards, nil
}

// generateTemporalShards divides work based on animation frames
func (e *Engine) generateTemporalShards(manifest map[string]interface{}) ([]Shard, error) {
    metadata, _ := manifest["metadata"].(map[string]interface{})
    hasAnimation, _ := metadata["has_animation"].(bool)
    
    if !hasAnimation {
        // Fall back to scene-based sharding if no animation
        return e.generateSceneBasedShards(manifest)
    }
    
    // For M0, assume 300 frames total
    totalFrames := 300
    framesPerShard := e.config.FrameChunkSize
    numShards := int(math.Ceil(float64(totalFrames) / float64(framesPerShard)))
    
    jobID := uuid.New().String()
    shards := make([]Shard, 0, numShards)
    
    for i := 0; i < numShards; i++ {
        startFrame := i * framesPerShard
        endFrame := startFrame + framesPerShard
        if endFrame > totalFrames {
            endFrame = totalFrames
        }
        
        shard := Shard{
            ID:          e.generateShardID(jobID, i),
            JobID:       jobID,
            Index:       i,
            TotalShards: numShards,
            Type:        ShardTypeHybrid,
            Data: ShardData{
                SceneSegment: SceneSegment{
                    FrameRange: FrameRange{
                        Start: startFrame,
                        End:   endFrame,
                    },
                },
                ProcessingHints: e.generateProcessingHints(manifest),
                OutputFormat:    "animated",
            },
            EstimatedCost:   e.estimateCost(framesPerShard * 1000), // Rough estimate
            EstimatedTimeMs: e.estimateTime(framesPerShard * 1000),
            Priority:        e.calculatePriority(manifest),
        }
        shards = append(shards, shard)
    }
    
    return shards, nil
}

// generateGeometricShards divides work based on spatial regions
func (e *Engine) generateGeometricShards(manifest map[string]interface{}) ([]Shard, error) {
    metadata, _ := manifest["metadata"].(map[string]interface{})
    bbox, _ := metadata["bounding_box"].(map[string]interface{})
    
    // Get bounding box dimensions
    min, _ := bbox["min"].([]interface{})
    max, _ := bbox["max"].([]interface{})
    
    if len(min) != 3 || len(max) != 3 {
        return e.generateDefaultShards(manifest)
    }
    
    // Calculate spatial divisions
    divisions := e.config.SpatialDivisions
    totalShards := divisions
    
    jobID := uuid.New().String()
    shards := make([]Shard, 0, totalShards)
    
    // Simple grid division for M0
    gridSize := int(math.Cbrt(float64(divisions)))
    shardIndex := 0
    
    for x := 0; x < gridSize; x++ {
        for y := 0; y < gridSize; y++ {
            for z := 0; z < gridSize && shardIndex < divisions; z++ {
                shard := e.createSpatialShard(jobID, shardIndex, totalShards, 
                                             x, y, z, gridSize, min, max, manifest)
                shards = append(shards, shard)
                shardIndex++
            }
        }
    }
    
    return shards, nil
}

// generateHybridShards combines multiple sharding strategies
func (e *Engine) generateHybridShards(manifest map[string]interface{}) ([]Shard, error) {
    metadata, _ := manifest["metadata"].(map[string]interface{})
    
    hasAnimation, _ := metadata["has_animation"].(bool)
    vertexCount := 0
    if vc, ok := metadata["vertex_count"].(float64); ok {
        vertexCount = int(vc)
    }
    
    // Decide strategy based on scene characteristics
    if hasAnimation {
        return e.generateTemporalShards(manifest)
    } else if vertexCount > 100000 {
        return e.generateGeometricShards(manifest)
    } else {
        return e.generateSceneBasedShards(manifest)
    }
}

// Helper methods

func (e *Engine) generateShardID(jobID string, index int) string {
    data := fmt.Sprintf("%s-%d-%d", jobID, index, uuid.New().ID())
    hash := sha256.Sum256([]byte(data))
    return hex.EncodeToString(hash[:])[:16]
}

func (e *Engine) determineShardType(manifest map[string]interface{}) ShardType {
    path, _ := manifest["optimizationPath"].(string)
    switch path {
    case "baking":
        return ShardTypeBaking
    case "gaussian-splatting":
        return ShardType3DGS
    default:
        return ShardTypeHybrid
    }
}

func (e *Engine) generateProcessingHints(manifest map[string]interface{}) ProcessingHints {
    return ProcessingHints{
        PreferredMethod: "auto",
        QualityPreset:   "balanced",
        MaxMemoryMB:     4096,
        MaxProcessingMs: 300000,
        CustomParameters: map[string]string{
            "compression": "optimal",
            "format":      "webgl2",
        },
    }
}

func (e *Engine) estimateCost(workSize int) float64 {
    // Simple cost model for M0
    baseCost := 1.0
    sizeFactor := float64(workSize) / 1000.0
    return baseCost + sizeFactor
}

func (e *Engine) estimateTime(workSize int) int64 {
    // Simple time estimation for M0
    baseTimeMs := int64(60000) // 1 minute base
    sizeTimeMs := int64(workSize / 100) * 1000
    return baseTimeMs + sizeTimeMs
}

func (e *Engine) calculatePriority(manifest map[string]interface{}) int {
    if priority, ok := manifest["priority"].(float64); ok {
        return int(priority)
    }
    return 5 // Default priority
}

func (e *Engine) createSpatialShard(jobID string, index, total, x, y, z, gridSize int,
                                    min, max []interface{}, manifest map[string]interface{}) Shard {
    // Calculate spatial bounds for this shard
    minX, _ := min[0].(float64)
    minY, _ := min[1].(float64)
    minZ, _ := min[2].(float64)
    maxX, _ := max[0].(float64)
    maxY, _ := max[1].(float64)
    maxZ, _ := max[2].(float64)
    
    deltaX := (maxX - minX) / float64(gridSize)
    deltaY := (maxY - minY) / float64(gridSize)
    deltaZ := (maxZ - minZ) / float64(gridSize)
    
    shardMinX := minX + float64(x)*deltaX
    shardMinY := minY + float64(y)*deltaY
    shardMinZ := minZ + float64(z)*deltaZ
    shardMaxX := shardMinX + deltaX
    shardMaxY := shardMinY + deltaY
    shardMaxZ := shardMinZ + deltaZ
    
    return Shard{
        ID:          e.generateShardID(jobID, index),
        JobID:       jobID,
        Index:       index,
        TotalShards: total,
        Type:        e.determineShardType(manifest),
        Data: ShardData{
            SceneSegment: SceneSegment{
                BoundingBox: BoundingBox{
                    Min: [3]float64{shardMinX, shardMinY, shardMinZ},
                    Max: [3]float64{shardMaxX, shardMaxY, shardMaxZ},
                },
                LODLevel: 0,
            },
            ProcessingHints: e.generateProcessingHints(manifest),
            OutputFormat:    "optimized",
        },
        EstimatedCost:   e.estimateCost(10000),
        EstimatedTimeMs: e.estimateTime(10000),
        Priority:        e.calculatePriority(manifest),
    }
}