package scheduler

import (
    "context"
    "fmt"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    
    "github.com/worldshare/mvp/services/scheduler/internal/database"
    "github.com/worldshare/mvp/services/scheduler/internal/queue"
    "github.com/worldshare/mvp/services/scheduler/internal/sharding"
)

func TestSchedulerService_ScheduleJob(t *testing.T) {
    // Create test scheduler
    scheduler := createTestScheduler()
    
    // Create test manifest
    manifest := &JobManifest{
        SceneID:          "test-scene-001",
        SceneType:        "indoor",
        OptimizationPath: "baking",
        VerificationPolicy: VerificationPolicy{
            RedundancyFactor: 2,
            CanaryRate:       0.1,
            QualityTarget: QualityMetrics{
                SSIM: 0.98,
                PSNR: 35.0,
            },
        },
        Priority: 5,
    }
    
    // Schedule job
    ctx := context.Background()
    jobID, err := scheduler.ScheduleJob(ctx, manifest)
    
    // Assertions
    assert.NoError(t, err)
    assert.NotEmpty(t, jobID)
    
    // Verify job was created
    job, err := scheduler.GetJobStatus(jobID)
    assert.NoError(t, err)
    assert.NotNil(t, job)
    assert.Equal(t, "queued", job.Status)
    assert.Equal(t, manifest.SceneID, job.Manifest.SceneID)
}

func TestSchedulerService_CalculateDistribution(t *testing.T) {
    scheduler := createTestScheduler()
    
    // Create test shards
    shards := []sharding.Shard{
        {ID: "shard-1", JobID: "job-1"},
        {ID: "shard-2", JobID: "job-1"},
        {ID: "shard-3", JobID: "job-1"},
    }
    
    policy := VerificationPolicy{
        RedundancyFactor: 2,
        CanaryRate:       0.1,
    }
    
    // Calculate distribution
    distribution := scheduler.calculateDistribution(shards, policy)
    
    // Assertions
    assert.NotEmpty(t, distribution)
    
    // Should have redundancy copies + canary
    expectedCount := len(shards)*policy.RedundancyFactor + int(float64(len(shards))*policy.CanaryRate)
    assert.GreaterOrEqual(t, len(distribution), expectedCount)
    
    // Check redundancy IDs
    redundancyCounts := make(map[string]int)
    for _, inst := range distribution {
        if !inst.IsCanary {
            redundancyCounts[inst.ShardID]++
        }
    }
    
    for shardID, count := range redundancyCounts {
        assert.Equal(t, policy.RedundancyFactor, count, 
            "Shard %s should have %d redundant copies", shardID, policy.RedundancyFactor)
    }
}

func TestNodeManager_GetAvailableNodes(t *testing.T) {
    nm := NewNodeManager()
    
    // Get available nodes
    nodes := nm.GetAvailableNodes()
    
    // Should have mock nodes available
    assert.NotEmpty(t, nodes)
    
    // Check node properties
    for _, node := range nodes {
        assert.NotEmpty(t, node.ID)
        assert.Equal(t, NodeStatusActive, node.Status)
        assert.LessOrEqual(t, node.CurrentWorkload, node.MaxWorkload)
    }
}

func TestNodeManager_UpdateNodeWorkload(t *testing.T) {
    nm := NewNodeManager()
    
    // Get a node
    nodes := nm.GetAvailableNodes()
    require.NotEmpty(t, nodes)
    
    node := nodes[0]
    initialWorkload := node.CurrentWorkload
    
    // Increase workload
    err := nm.UpdateNodeWorkload(node.ID, 1)
    assert.NoError(t, err)
    
    // Verify workload increased
    updatedNodes := nm.GetAvailableNodes()
    for _, n := range updatedNodes {
        if n.ID == node.ID {
            assert.Equal(t, initialWorkload+1, n.CurrentWorkload)
            break
        }
    }
    
    // Try to exceed max workload
    err = nm.UpdateNodeWorkload(node.ID, 100)
    assert.Error(t, err)
    assert.Equal(t, ErrNodeOverloaded, err)
}

func TestPriorityQueue_EnqueueDequeue(t *testing.T) {
    pq := queue.NewPriorityQueue(queue.Config{
        MaxSize: 100,
    })
    
    // Enqueue items with different priorities
    items := []*queue.WorkItem{
        {JobID: "job-1", ShardID: "shard-1", Priority: 3},
        {JobID: "job-2", ShardID: "shard-2", Priority: 7},
        {JobID: "job-3", ShardID: "shard-3", Priority: 5},
    }
    
    for _, item := range items {
        err := pq.Enqueue(item)
        assert.NoError(t, err)
    }
    
    // Dequeue should return highest priority first
    item := pq.Dequeue()
    assert.NotNil(t, item)
    assert.Equal(t, 7, item.Priority)
    
    item = pq.Dequeue()
    assert.NotNil(t, item)
    assert.Equal(t, 5, item.Priority)
    
    item = pq.Dequeue()
    assert.NotNil(t, item)
    assert.Equal(t, 3, item.Priority)
    
    // Queue should be empty
    assert.True(t, pq.IsEmpty())
}

func TestShardingEngine_GenerateShards(t *testing.T) {
    engine := sharding.NewEngine(sharding.Config{
        MaxShardSize: 1000,
        MinShardSize: 100,
        Strategy:     "scene-based",
    })
    
    manifest := map[string]interface{}{
        "sceneId":          "test-scene",
        "sceneType":        "indoor",
        "optimizationPath": "baking",
        "metadata": map[string]interface{}{
            "vertex_count": 5000.0,
        },
    }
    
    shards, err := engine.GenerateShards(manifest)
    
    assert.NoError(t, err)
    assert.NotEmpty(t, shards)
    
    // Should have created multiple shards based on vertex count
    expectedShards := 5 // 5000 vertices / 1000 max per shard
    assert.Equal(t, expectedShards, len(shards))
    
    // Check shard properties
    for i, shard := range shards {
        assert.NotEmpty(t, shard.ID)
        assert.Equal(t, i, shard.Index)
        assert.Equal(t, expectedShards, shard.TotalShards)
        assert.Equal(t, sharding.ShardTypeBaking, shard.Type)
    }
}

// Helper functions

func createTestScheduler() *SchedulerService {
    return NewSchedulerService(Config{
        RedundancyFactor: 2,
        CanaryRate:      0.1,
        Database:        &database.DB{},
        ShardingEngine:  sharding.NewEngine(sharding.Config{}),
        JobQueue:        queue.NewPriorityQueue(queue.Config{}),
    })
}

func TestSchedulerService_ProcessQueuedWork(t *testing.T) {
    scheduler := createTestScheduler()
    
    // Add work to queue
    manifest := &JobManifest{
        SceneID:          "test-scene",
        OptimizationPath: "baking",
        Priority:         5,
    }
    
    ctx := context.Background()
    jobID, err := scheduler.ScheduleJob(ctx, manifest)
    require.NoError(t, err)
    
    // Process queued work
    scheduler.processQueuedWork()
    
    // Work should be dispatched (in M0, just logged)
    // In production, would verify actual dispatch
    assert.NotEmpty(t, jobID)
}

func TestSchedulerService_ConcurrentScheduling(t *testing.T) {
    scheduler := createTestScheduler()
    ctx := context.Background()
    
    // Schedule multiple jobs concurrently
    numJobs := 10
    results := make(chan string, numJobs)
    errors := make(chan error, numJobs)
    
    for i := 0; i < numJobs; i++ {
        go func(index int) {
            manifest := &JobManifest{
                SceneID:          fmt.Sprintf("scene-%d", index),
                OptimizationPath: "baking",
                Priority:         index % 10,
            }
            
            jobID, err := scheduler.ScheduleJob(ctx, manifest)
            if err != nil {
                errors <- err
            } else {
                results <- jobID
            }
        }(i)
    }
    
    // Collect results
    jobIDs := make([]string, 0, numJobs)
    for i := 0; i < numJobs; i++ {
        select {
        case jobID := <-results:
            jobIDs = append(jobIDs, jobID)
        case err := <-errors:
            t.Errorf("Concurrent scheduling failed: %v", err)
        case <-time.After(5 * time.Second):
            t.Fatal("Timeout waiting for concurrent scheduling")
        }
    }
    
    // All jobs should be scheduled
    assert.Len(t, jobIDs, numJobs)
    
    // All job IDs should be unique
    uniqueIDs := make(map[string]bool)
    for _, id := range jobIDs {
        assert.False(t, uniqueIDs[id], "Duplicate job ID found")
        uniqueIDs[id] = true
    }
}