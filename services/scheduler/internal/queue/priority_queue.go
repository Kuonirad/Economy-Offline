package queue

import (
    "container/heap"
    "sync"
    "time"
    
    "github.com/sirupsen/logrus"
)

var log = logrus.New()

// WorkItem represents a unit of work in the queue
type WorkItem struct {
    JobID       string
    ShardID     string
    Priority    int
    Instruction interface{} // DispatchInstruction from scheduler
    EnqueuedAt  time.Time
    Attempts    int
    LastAttempt time.Time
}

// PriorityQueue implements a thread-safe priority queue
type PriorityQueue struct {
    mu              sync.RWMutex
    items           priorityHeap
    maxSize         int
    processingCount int
    stats           QueueStats
}

// QueueStats tracks queue metrics
type QueueStats struct {
    TotalEnqueued   int64
    TotalDequeued   int64
    TotalDropped    int64
    CurrentSize     int
    AverageWaitTime time.Duration
}

// Config holds queue configuration
type Config struct {
    MaxSize           int
    ProcessingThreads int
}

// priorityHeap implements heap.Interface for priority queue
type priorityHeap []*WorkItem

func (h priorityHeap) Len() int { return len(h) }

func (h priorityHeap) Less(i, j int) bool {
    // Higher priority items come first
    if h[i].Priority != h[j].Priority {
        return h[i].Priority > h[j].Priority
    }
    // For same priority, older items come first (FIFO)
    return h[i].EnqueuedAt.Before(h[j].EnqueuedAt)
}

func (h priorityHeap) Swap(i, j int) {
    h[i], h[j] = h[j], h[i]
}

func (h *priorityHeap) Push(x interface{}) {
    *h = append(*h, x.(*WorkItem))
}

func (h *priorityHeap) Pop() interface{} {
    old := *h
    n := len(old)
    item := old[n-1]
    *h = old[0 : n-1]
    return item
}

// NewPriorityQueue creates a new priority queue
func NewPriorityQueue(config Config) *PriorityQueue {
    if config.MaxSize == 0 {
        config.MaxSize = 10000
    }
    if config.ProcessingThreads == 0 {
        config.ProcessingThreads = 10
    }
    
    pq := &PriorityQueue{
        items:   make(priorityHeap, 0),
        maxSize: config.MaxSize,
    }
    
    heap.Init(&pq.items)
    
    log.WithFields(logrus.Fields{
        "maxSize":           config.MaxSize,
        "processingThreads": config.ProcessingThreads,
    }).Info("Priority queue initialized")
    
    return pq
}

// Enqueue adds a work item to the queue
func (pq *PriorityQueue) Enqueue(item *WorkItem) error {
    pq.mu.Lock()
    defer pq.mu.Unlock()
    
    if len(pq.items) >= pq.maxSize {
        pq.stats.TotalDropped++
        return ErrQueueFull
    }
    
    if item.EnqueuedAt.IsZero() {
        item.EnqueuedAt = time.Now()
    }
    
    heap.Push(&pq.items, item)
    pq.stats.TotalEnqueued++
    pq.stats.CurrentSize = len(pq.items)
    
    log.WithFields(logrus.Fields{
        "jobId":     item.JobID,
        "shardId":   item.ShardID,
        "priority":  item.Priority,
        "queueSize": pq.stats.CurrentSize,
    }).Debug("Work item enqueued")
    
    return nil
}

// Dequeue removes and returns the highest priority item
func (pq *PriorityQueue) Dequeue() *WorkItem {
    pq.mu.Lock()
    defer pq.mu.Unlock()
    
    if len(pq.items) == 0 {
        return nil
    }
    
    item := heap.Pop(&pq.items).(*WorkItem)
    
    // Update statistics
    pq.stats.TotalDequeued++
    pq.stats.CurrentSize = len(pq.items)
    
    // Update average wait time
    waitTime := time.Since(item.EnqueuedAt)
    if pq.stats.AverageWaitTime == 0 {
        pq.stats.AverageWaitTime = waitTime
    } else {
        // Exponential moving average
        alpha := 0.1
        pq.stats.AverageWaitTime = time.Duration(
            alpha*float64(waitTime) + (1-alpha)*float64(pq.stats.AverageWaitTime),
        )
    }
    
    log.WithFields(logrus.Fields{
        "jobId":    item.JobID,
        "shardId":  item.ShardID,
        "waitTime": waitTime,
    }).Debug("Work item dequeued")
    
    return item
}

// Peek returns the highest priority item without removing it
func (pq *PriorityQueue) Peek() *WorkItem {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    
    if len(pq.items) == 0 {
        return nil
    }
    
    return pq.items[0]
}

// Size returns the current queue size
func (pq *PriorityQueue) Size() int {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    return len(pq.items)
}

// IsEmpty returns true if the queue is empty
func (pq *PriorityQueue) IsEmpty() bool {
    return pq.Size() == 0
}

// IsFull returns true if the queue is at capacity
func (pq *PriorityQueue) IsFull() bool {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    return len(pq.items) >= pq.maxSize
}

// GetStats returns queue statistics
func (pq *PriorityQueue) GetStats() QueueStats {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    
    stats := pq.stats
    stats.CurrentSize = len(pq.items)
    return stats
}

// RemoveJob removes all items for a specific job
func (pq *PriorityQueue) RemoveJob(jobID string) int {
    pq.mu.Lock()
    defer pq.mu.Unlock()
    
    removed := 0
    newItems := make(priorityHeap, 0)
    
    for _, item := range pq.items {
        if item.JobID != jobID {
            newItems = append(newItems, item)
        } else {
            removed++
        }
    }
    
    pq.items = newItems
    heap.Init(&pq.items)
    pq.stats.CurrentSize = len(pq.items)
    
    if removed > 0 {
        log.WithFields(logrus.Fields{
            "jobId":        jobID,
            "itemsRemoved": removed,
        }).Info("Removed job items from queue")
    }
    
    return removed
}

// UpdatePriority updates the priority of items for a job
func (pq *PriorityQueue) UpdatePriority(jobID string, newPriority int) int {
    pq.mu.Lock()
    defer pq.mu.Unlock()
    
    updated := 0
    
    for _, item := range pq.items {
        if item.JobID == jobID {
            item.Priority = newPriority
            updated++
        }
    }
    
    if updated > 0 {
        heap.Init(&pq.items)
        log.WithFields(logrus.Fields{
            "jobId":        jobID,
            "newPriority":  newPriority,
            "itemsUpdated": updated,
        }).Info("Updated job priority")
    }
    
    return updated
}

// GetJobItems returns all items for a specific job
func (pq *PriorityQueue) GetJobItems(jobID string) []*WorkItem {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    
    items := make([]*WorkItem, 0)
    
    for _, item := range pq.items {
        if item.JobID == jobID {
            items = append(items, item)
        }
    }
    
    return items
}

// Clear removes all items from the queue
func (pq *PriorityQueue) Clear() {
    pq.mu.Lock()
    defer pq.mu.Unlock()
    
    pq.items = make(priorityHeap, 0)
    heap.Init(&pq.items)
    pq.stats.CurrentSize = 0
    
    log.Info("Queue cleared")
}

// ProcessingComplete marks an item as processed
func (pq *PriorityQueue) ProcessingComplete(shardID string) {
    pq.mu.Lock()
    defer pq.mu.Unlock()
    
    if pq.processingCount > 0 {
        pq.processingCount--
    }
}

// GetQueueDepth returns items grouped by priority
func (pq *PriorityQueue) GetQueueDepth() map[int]int {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    
    depth := make(map[int]int)
    
    for _, item := range pq.items {
        depth[item.Priority]++
    }
    
    return depth
}

// GetOldestItem returns the oldest item in the queue
func (pq *PriorityQueue) GetOldestItem() *WorkItem {
    pq.mu.RLock()
    defer pq.mu.RUnlock()
    
    if len(pq.items) == 0 {
        return nil
    }
    
    oldest := pq.items[0]
    for _, item := range pq.items {
        if item.EnqueuedAt.Before(oldest.EnqueuedAt) {
            oldest = item
        }
    }
    
    return oldest
}

// RequeueFailedItem requeues an item with exponential backoff
func (pq *PriorityQueue) RequeueFailedItem(item *WorkItem, maxAttempts int) error {
    if maxAttempts == 0 {
        maxAttempts = 3
    }
    
    if item.Attempts >= maxAttempts {
        log.WithFields(logrus.Fields{
            "jobId":   item.JobID,
            "shardId": item.ShardID,
            "attempts": item.Attempts,
        }).Warn("Max attempts reached, not requeueing")
        return ErrMaxAttemptsReached
    }
    
    // Exponential backoff
    item.Attempts++
    item.LastAttempt = time.Now()
    
    // Reduce priority for failed items
    if item.Priority > 1 {
        item.Priority--
    }
    
    return pq.Enqueue(item)
}