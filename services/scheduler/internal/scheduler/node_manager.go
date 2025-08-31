package scheduler

import (
    "sync"
    "time"
    
    "github.com/google/uuid"
    "github.com/sirupsen/logrus"
)

// NodeStatus represents the current state of a compute node
type NodeStatus string

const (
    NodeStatusActive   NodeStatus = "active"
    NodeStatusIdle     NodeStatus = "idle"
    NodeStatusOffline  NodeStatus = "offline"
    NodeStatusDraining NodeStatus = "draining"
)

// Node represents a compute node in the network
type Node struct {
    ID              string
    Status          NodeStatus
    Capabilities    NodeCapabilities
    Performance     NodePerformance
    LastHeartbeat   time.Time
    CurrentWorkload int
    MaxWorkload     int
}

// NodeCapabilities describes what a node can process
type NodeCapabilities struct {
    HasGPU            bool
    GPUModel          string
    VRAM              int // in MB
    ComputeCapability string
    SupportsBaking    bool
    Supports3DGS      bool
    MaxSceneSize      int64 // in bytes
}

// NodePerformance tracks historical performance metrics
type NodePerformance struct {
    AverageProcessingTime time.Duration
    SuccessRate           float64
    ReliabilityScore      float64
    TotalJobsProcessed    int
    TotalJobsFailed       int
}

// NodeManager manages the pool of compute nodes
type NodeManager struct {
    mu              sync.RWMutex
    nodes           map[string]*Node
    activeNodes     []*Node
    lastHealthCheck time.Time
    log             *logrus.Logger
}

// NewNodeManager creates a new node manager instance
func NewNodeManager() *NodeManager {
    nm := &NodeManager{
        nodes:       make(map[string]*Node),
        activeNodes: make([]*Node, 0),
        log:         logrus.New(),
    }
    
    // Initialize with mock nodes for M0
    nm.initializeMockNodes()
    
    return nm
}

// initializeMockNodes creates mock nodes for M0 testing
func (nm *NodeManager) initializeMockNodes() {
    mockNodes := []Node{
        {
            ID:     uuid.New().String(),
            Status: NodeStatusActive,
            Capabilities: NodeCapabilities{
                HasGPU:            true,
                GPUModel:          "NVIDIA RTX 3090",
                VRAM:              24576,
                ComputeCapability: "8.6",
                SupportsBaking:    true,
                Supports3DGS:      true,
                MaxSceneSize:      10 * 1024 * 1024 * 1024, // 10GB
            },
            Performance: NodePerformance{
                AverageProcessingTime: 5 * time.Minute,
                SuccessRate:           0.98,
                ReliabilityScore:      0.95,
                TotalJobsProcessed:    1000,
                TotalJobsFailed:       20,
            },
            LastHeartbeat:   time.Now(),
            CurrentWorkload: 0,
            MaxWorkload:     5,
        },
        {
            ID:     uuid.New().String(),
            Status: NodeStatusActive,
            Capabilities: NodeCapabilities{
                HasGPU:            true,
                GPUModel:          "NVIDIA RTX 4090",
                VRAM:              24576,
                ComputeCapability: "8.9",
                SupportsBaking:    true,
                Supports3DGS:      true,
                MaxSceneSize:      15 * 1024 * 1024 * 1024, // 15GB
            },
            Performance: NodePerformance{
                AverageProcessingTime: 4 * time.Minute,
                SuccessRate:           0.99,
                ReliabilityScore:      0.97,
                TotalJobsProcessed:    1500,
                TotalJobsFailed:       15,
            },
            LastHeartbeat:   time.Now(),
            CurrentWorkload: 1,
            MaxWorkload:     8,
        },
        {
            ID:     uuid.New().String(),
            Status: NodeStatusIdle,
            Capabilities: NodeCapabilities{
                HasGPU:            false,
                GPUModel:          "",
                VRAM:              0,
                ComputeCapability: "",
                SupportsBaking:    true,
                Supports3DGS:      false,
                MaxSceneSize:      5 * 1024 * 1024 * 1024, // 5GB
            },
            Performance: NodePerformance{
                AverageProcessingTime: 15 * time.Minute,
                SuccessRate:           0.95,
                ReliabilityScore:      0.90,
                TotalJobsProcessed:    500,
                TotalJobsFailed:       25,
            },
            LastHeartbeat:   time.Now(),
            CurrentWorkload: 0,
            MaxWorkload:     3,
        },
    }
    
    for i := range mockNodes {
        node := &mockNodes[i]
        nm.nodes[node.ID] = node
        if node.Status == NodeStatusActive {
            nm.activeNodes = append(nm.activeNodes, node)
        }
    }
    
    nm.log.WithField("nodeCount", len(nm.nodes)).Info("Initialized mock nodes")
}

// RegisterNode adds a new node to the pool
func (nm *NodeManager) RegisterNode(node *Node) error {
    nm.mu.Lock()
    defer nm.mu.Unlock()
    
    if node.ID == "" {
        node.ID = uuid.New().String()
    }
    
    node.LastHeartbeat = time.Now()
    nm.nodes[node.ID] = node
    
    if node.Status == NodeStatusActive {
        nm.activeNodes = append(nm.activeNodes, node)
    }
    
    nm.log.WithFields(logrus.Fields{
        "nodeId": node.ID,
        "hasGPU": node.Capabilities.HasGPU,
        "status": node.Status,
    }).Info("Node registered")
    
    return nil
}

// GetAvailableNodes returns nodes that can accept new work
func (nm *NodeManager) GetAvailableNodes() []*Node {
    nm.mu.RLock()
    defer nm.mu.RUnlock()
    
    available := make([]*Node, 0)
    
    for _, node := range nm.activeNodes {
        // Check if node is healthy and has capacity
        if node.Status == NodeStatusActive &&
           node.CurrentWorkload < node.MaxWorkload &&
           time.Since(node.LastHeartbeat) < 2*time.Minute {
            available = append(available, node)
        }
    }
    
    return available
}

// GetNodesByCapability returns nodes that support specific optimization paths
func (nm *NodeManager) GetNodesByCapability(requiresGPU bool, requires3DGS bool) []*Node {
    nm.mu.RLock()
    defer nm.mu.RUnlock()
    
    suitable := make([]*Node, 0)
    
    for _, node := range nm.activeNodes {
        if node.Status != NodeStatusActive {
            continue
        }
        
        if requiresGPU && !node.Capabilities.HasGPU {
            continue
        }
        
        if requires3DGS && !node.Capabilities.Supports3DGS {
            continue
        }
        
        if node.CurrentWorkload < node.MaxWorkload {
            suitable = append(suitable, node)
        }
    }
    
    return suitable
}

// UpdateNodeHeartbeat updates the last heartbeat time for a node
func (nm *NodeManager) UpdateNodeHeartbeat(nodeID string) error {
    nm.mu.Lock()
    defer nm.mu.Unlock()
    
    node, exists := nm.nodes[nodeID]
    if !exists {
        return ErrNodeNotFound
    }
    
    node.LastHeartbeat = time.Now()
    
    // Reactivate node if it was idle
    if node.Status == NodeStatusIdle {
        node.Status = NodeStatusActive
        nm.updateActiveNodes()
    }
    
    return nil
}

// UpdateNodeWorkload updates the current workload for a node
func (nm *NodeManager) UpdateNodeWorkload(nodeID string, delta int) error {
    nm.mu.Lock()
    defer nm.mu.Unlock()
    
    node, exists := nm.nodes[nodeID]
    if !exists {
        return ErrNodeNotFound
    }
    
    newWorkload := node.CurrentWorkload + delta
    if newWorkload < 0 {
        newWorkload = 0
    }
    if newWorkload > node.MaxWorkload {
        return ErrNodeOverloaded
    }
    
    node.CurrentWorkload = newWorkload
    return nil
}

// UpdateNodePerformance updates performance metrics after job completion
func (nm *NodeManager) UpdateNodePerformance(nodeID string, success bool, processingTime time.Duration) error {
    nm.mu.Lock()
    defer nm.mu.Unlock()
    
    node, exists := nm.nodes[nodeID]
    if !exists {
        return ErrNodeNotFound
    }
    
    // Update counters
    node.Performance.TotalJobsProcessed++
    if !success {
        node.Performance.TotalJobsFailed++
    }
    
    // Update success rate
    total := float64(node.Performance.TotalJobsProcessed)
    failed := float64(node.Performance.TotalJobsFailed)
    node.Performance.SuccessRate = (total - failed) / total
    
    // Update average processing time (exponential moving average)
    alpha := 0.2 // smoothing factor
    currentAvg := node.Performance.AverageProcessingTime.Seconds()
    newAvg := alpha*processingTime.Seconds() + (1-alpha)*currentAvg
    node.Performance.AverageProcessingTime = time.Duration(newAvg) * time.Second
    
    // Update reliability score (combination of success rate and consistency)
    node.Performance.ReliabilityScore = node.Performance.SuccessRate * 0.8 + 
                                        (1 - (newAvg / (newAvg + 300))) * 0.2 // consistency factor
    
    return nil
}

// CheckNodeHealth performs health checks on all nodes
func (nm *NodeManager) CheckNodeHealth() {
    nm.mu.Lock()
    defer nm.mu.Unlock()
    
    now := time.Now()
    staleThreshold := 2 * time.Minute
    
    for _, node := range nm.nodes {
        timeSinceHeartbeat := now.Sub(node.LastHeartbeat)
        
        switch {
        case timeSinceHeartbeat > 5*time.Minute:
            // Node is offline
            if node.Status != NodeStatusOffline {
                nm.log.WithField("nodeId", node.ID).Warn("Node marked offline")
                node.Status = NodeStatusOffline
            }
        case timeSinceHeartbeat > staleThreshold:
            // Node is idle
            if node.Status == NodeStatusActive {
                nm.log.WithField("nodeId", node.ID).Info("Node marked idle")
                node.Status = NodeStatusIdle
            }
        default:
            // Node is healthy
            if node.Status == NodeStatusOffline || node.Status == NodeStatusIdle {
                nm.log.WithField("nodeId", node.ID).Info("Node reactivated")
                node.Status = NodeStatusActive
            }
        }
    }
    
    nm.updateActiveNodes()
    nm.lastHealthCheck = now
}

// updateActiveNodes rebuilds the active nodes list
func (nm *NodeManager) updateActiveNodes() {
    nm.activeNodes = make([]*Node, 0)
    for _, node := range nm.nodes {
        if node.Status == NodeStatusActive {
            nm.activeNodes = append(nm.activeNodes, node)
        }
    }
}

// GetNodeMetrics returns aggregated metrics for monitoring
func (nm *NodeManager) GetNodeMetrics() map[string]interface{} {
    nm.mu.RLock()
    defer nm.mu.RUnlock()
    
    totalNodes := len(nm.nodes)
    activeCount := 0
    idleCount := 0
    offlineCount := 0
    totalCapacity := 0
    currentLoad := 0
    
    for _, node := range nm.nodes {
        switch node.Status {
        case NodeStatusActive:
            activeCount++
        case NodeStatusIdle:
            idleCount++
        case NodeStatusOffline:
            offlineCount++
        }
        
        totalCapacity += node.MaxWorkload
        currentLoad += node.CurrentWorkload
    }
    
    utilizationRate := float64(0)
    if totalCapacity > 0 {
        utilizationRate = float64(currentLoad) / float64(totalCapacity)
    }
    
    return map[string]interface{}{
        "total_nodes":      totalNodes,
        "active_nodes":     activeCount,
        "idle_nodes":       idleCount,
        "offline_nodes":    offlineCount,
        "total_capacity":   totalCapacity,
        "current_load":     currentLoad,
        "utilization_rate": utilizationRate,
        "last_health_check": nm.lastHealthCheck,
    }
}