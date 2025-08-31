package api

import (
    "encoding/json"
    "net/http"
    
    "github.com/gorilla/mux"
    "github.com/sirupsen/logrus"
    
    "github.com/worldshare/mvp/services/scheduler/internal/scheduler"
)

// Handler handles HTTP API requests
type Handler struct {
    scheduler *scheduler.SchedulerService
    log       *logrus.Logger
}

// NewHandler creates a new API handler
func NewHandler(scheduler *scheduler.SchedulerService, log *logrus.Logger) *Handler {
    return &Handler{
        scheduler: scheduler,
        log:       log,
    }
}

// SubmitJob handles job submission requests
func (h *Handler) SubmitJob(w http.ResponseWriter, r *http.Request) {
    var manifest scheduler.JobManifest
    
    if err := json.NewDecoder(r.Body).Decode(&manifest); err != nil {
        h.respondError(w, http.StatusBadRequest, "Invalid request body: "+err.Error())
        return
    }
    
    // Validate manifest
    if manifest.SceneID == "" {
        h.respondError(w, http.StatusBadRequest, "Scene ID is required")
        return
    }
    
    // Submit job to scheduler
    jobID, err := h.scheduler.ScheduleJob(r.Context(), &manifest)
    if err != nil {
        h.log.WithError(err).Error("Failed to schedule job")
        h.respondError(w, http.StatusInternalServerError, "Failed to schedule job")
        return
    }
    
    response := map[string]interface{}{
        "jobId":  jobID,
        "status": "queued",
        "message": "Job successfully queued for processing",
    }
    
    h.respondJSON(w, http.StatusCreated, response)
}

// GetJobStatus handles job status requests
func (h *Handler) GetJobStatus(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    jobID := vars["jobId"]
    
    if jobID == "" {
        h.respondError(w, http.StatusBadRequest, "Job ID is required")
        return
    }
    
    job, err := h.scheduler.GetJobStatus(jobID)
    if err != nil {
        h.respondError(w, http.StatusNotFound, "Job not found")
        return
    }
    
    // Build response
    response := map[string]interface{}{
        "jobId":     job.ID,
        "status":    job.Status,
        "sceneId":   job.Manifest.SceneID,
        "sceneType": job.Manifest.SceneType,
        "createdAt": job.CreatedAt,
        "updatedAt": job.UpdatedAt,
        "shards": map[string]interface{}{
            "total":     job.TotalShards,
            "completed": job.CompletedShards,
            "failed":    job.FailedShards,
        },
    }
    
    h.respondJSON(w, http.StatusOK, response)
}

// CancelJob handles job cancellation requests
func (h *Handler) CancelJob(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    jobID := vars["jobId"]
    
    if jobID == "" {
        h.respondError(w, http.StatusBadRequest, "Job ID is required")
        return
    }
    
    // Cancel the job
    if err := h.scheduler.CancelJob(jobID); err != nil {
        h.respondError(w, http.StatusBadRequest, err.Error())
        return
    }
    
    response := map[string]interface{}{
        "jobId":   jobID,
        "status":  "cancelled",
        "message": "Job successfully cancelled",
    }
    
    h.respondJSON(w, http.StatusOK, response)
}

// ListNodes handles node listing requests
func (h *Handler) ListNodes(w http.ResponseWriter, r *http.Request) {
    // Get nodes from scheduler's node manager
    nodeManager := h.scheduler.GetNodeManager()
    if nodeManager == nil {
        h.respondError(w, http.StatusInternalServerError, "Node manager not available")
        return
    }
    
    availableNodes := nodeManager.GetAvailableNodes()
    
    nodes := make([]map[string]interface{}, 0, len(availableNodes))
    for _, node := range availableNodes {
        nodes = append(nodes, map[string]interface{}{
            "nodeId": node.ID,
            "status": string(node.Status),
            "capabilities": map[string]interface{}{
                "hasGpu":   node.Capabilities.HasGPU,
                "gpuModel": node.Capabilities.GPUModel,
                "vram":     node.Capabilities.VRAM,
            },
            "currentWorkload": node.CurrentWorkload,
            "maxWorkload":     node.MaxWorkload,
            "performance": map[string]interface{}{
                "avgProcessingTime": node.Performance.AverageProcessingTime.Milliseconds(),
                "successRate":       node.Performance.SuccessRate,
                "reliabilityScore":  node.Performance.ReliabilityScore,
            },
        })
    }
    
    h.respondJSON(w, http.StatusOK, nodes)
}

// GetNodeStatus handles individual node status requests
func (h *Handler) GetNodeStatus(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    nodeID := vars["nodeId"]
    
    if nodeID == "" {
        h.respondError(w, http.StatusBadRequest, "Node ID is required")
        return
    }
    
    // TODO: Get actual node status from scheduler
    
    response := map[string]interface{}{
        "nodeId": nodeID,
        "status": "active",
        "lastHeartbeat": "2024-01-01T00:00:00Z",
        "performance": map[string]interface{}{
            "avgProcessingTime": 120000,
            "successRate":       0.98,
        },
    }
    
    h.respondJSON(w, http.StatusOK, response)
}

// Helper methods

func (h *Handler) respondJSON(w http.ResponseWriter, status int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    
    if err := json.NewEncoder(w).Encode(data); err != nil {
        h.log.WithError(err).Error("Failed to encode response")
    }
}

func (h *Handler) respondError(w http.ResponseWriter, status int, message string) {
    h.respondJSON(w, status, map[string]string{
        "error": message,
    })
}