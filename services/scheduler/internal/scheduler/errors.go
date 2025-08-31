package scheduler

import "errors"

var (
    // Node-related errors
    ErrNodeNotFound   = errors.New("node not found")
    ErrNodeOverloaded = errors.New("node workload exceeded")
    ErrNodeOffline    = errors.New("node is offline")
    
    // Job-related errors
    ErrJobNotFound    = errors.New("job not found")
    ErrInvalidManifest = errors.New("invalid job manifest")
    ErrJobCancelled   = errors.New("job was cancelled")
    
    // Sharding errors
    ErrShardingFailed = errors.New("failed to generate shards")
    ErrInvalidShard   = errors.New("invalid shard configuration")
    
    // Queue errors
    ErrQueueFull      = errors.New("queue is full")
    ErrQueueEmpty     = errors.New("queue is empty")
    
    // Database errors
    ErrDatabaseConnection = errors.New("database connection failed")
    ErrDatabaseQuery      = errors.New("database query failed")
)