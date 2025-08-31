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
            "completed": 0, // TODO: Track completed shards
            "failed":    0, // TODO: Track failed shards
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
    
    // TODO: Implement job cancellation in scheduler
    
    response := map[string]interface{}{
        "jobId":   jobID,
        "status":  "cancelled",
        "message": "Job cancellation requested",
    }
    
    h.respondJSON(w, http.StatusOK, response)
}

// ListNodes handles node listing requests
func (h *Handler) ListNodes(w http.ResponseWriter, r *http.Request) {
    // TODO: Get actual nodes from scheduler
    nodes := []map[string]interface{}{
        {
            "nodeId": "node-1",
            "status": "active",
            "capabilities": map[string]interface{}{
                "hasGpu":   true,
                "gpuModel": "NVIDIA RTX 3090",
                "vram":     24576,
            },
            "currentWorkload": 2,
            "maxWorkload":     5,
        },
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