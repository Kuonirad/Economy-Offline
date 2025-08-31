// Economy Scheduler Service - Main Entry Point
// Implements the Optimizer component for time-shifted orchestration

package main

import (
    "context"
    "fmt"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/gorilla/mux"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "github.com/sirupsen/logrus"
    "github.com/spf13/viper"
    
    "github.com/worldshare/mvp/services/scheduler/internal/api"
    "github.com/worldshare/mvp/services/scheduler/internal/scheduler"
    "github.com/worldshare/mvp/services/scheduler/internal/sharding"
    "github.com/worldshare/mvp/services/scheduler/internal/queue"
    "github.com/worldshare/mvp/services/scheduler/internal/database"
)

var (
    log = logrus.New()
    version = "0.1.0-m0"
)

func init() {
    // Configure logging
    log.SetFormatter(&logrus.JSONFormatter{})
    log.SetOutput(os.Stdout)
    
    // Load configuration
    viper.SetConfigName("scheduler")
    viper.SetConfigType("yaml")
    viper.AddConfigPath("/etc/worldshare")
    viper.AddConfigPath("./configs")
    viper.AutomaticEnv()
    
    if err := viper.ReadInConfig(); err != nil {
        log.Warnf("Config file not found, using defaults: %v", err)
    }
    
    // Set log level
    level, err := logrus.ParseLevel(viper.GetString("LOG_LEVEL"))
    if err == nil {
        log.SetLevel(level)
    }
}

func main() {
    log.Infof("Starting Economy Scheduler Service v%s", version)
    
    // Initialize database connection
    db, err := database.NewConnection(viper.GetString("database.connectionString"))
    if err != nil {
        log.Fatalf("Failed to connect to database: %v", err)
    }
    defer db.Close()
    
    // Initialize sharding engine
    shardingEngine := sharding.NewEngine(sharding.Config{
        MaxShardSize: viper.GetInt("sharding.maxShardSize"),
        MinShardSize: viper.GetInt("sharding.minShardSize"),
        Strategy:     viper.GetString("sharding.strategy"),
    })
    
    // Initialize job queue
    jobQueue := queue.NewPriorityQueue(queue.Config{
        MaxSize:           viper.GetInt("queue.maxQueueSize"),
        ProcessingThreads: viper.GetInt("queue.processingThreads"),
    })
    
    // Initialize scheduler service
    schedulerService := scheduler.NewSchedulerService(scheduler.Config{
        RedundancyFactor: viper.GetInt("distribution.redundancy"),
        CanaryRate:      viper.GetFloat64("distribution.canaryRate"),
        Database:        db,
        ShardingEngine:  shardingEngine,
        JobQueue:        jobQueue,
    })
    
    // Start background processing
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    go schedulerService.StartDispatchLoop(ctx)
    go schedulerService.StartHealthMonitor(ctx)
    
    // Initialize HTTP router
    router := mux.NewRouter()
    
    // API endpoints
    apiHandler := api.NewHandler(schedulerService, log)
    router.HandleFunc("/api/v1/jobs", apiHandler.SubmitJob).Methods("POST")
    router.HandleFunc("/api/v1/jobs/{jobId}", apiHandler.GetJobStatus).Methods("GET")
    router.HandleFunc("/api/v1/jobs/{jobId}/cancel", apiHandler.CancelJob).Methods("POST")
    router.HandleFunc("/api/v1/nodes", apiHandler.ListNodes).Methods("GET")
    router.HandleFunc("/api/v1/nodes/{nodeId}/status", apiHandler.GetNodeStatus).Methods("GET")
    
    // Health and metrics endpoints
    router.HandleFunc("/health", healthHandler).Methods("GET")
    router.HandleFunc("/ready", readyHandler).Methods("GET")
    router.Handle("/metrics", promhttp.Handler())
    
    // Configure HTTP server
    srv := &http.Server{
        Addr:         ":8080",
        Handler:      router,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }
    
    // Start metrics server
    go func() {
        metricsServer := &http.Server{
            Addr:    ":9090",
            Handler: promhttp.Handler(),
        }
        log.Info("Starting metrics server on :9090")
        if err := metricsServer.ListenAndServe(); err != nil {
            log.Errorf("Metrics server error: %v", err)
        }
    }()
    
    // Start main server
    go func() {
        log.Info("Starting HTTP server on :8080")
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("HTTP server error: %v", err)
        }
    }()
    
    // Wait for interrupt signal
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
    <-sigChan
    
    log.Info("Shutting down gracefully...")
    
    // Graceful shutdown
    shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer shutdownCancel()
    
    if err := srv.Shutdown(shutdownCtx); err != nil {
        log.Errorf("Server shutdown error: %v", err)
    }
    
    log.Info("Economy Scheduler Service stopped")
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, `{"status":"healthy","version":"%s"}`, version)
}

func readyHandler(w http.ResponseWriter, r *http.Request) {
    // TODO: Add actual readiness checks (DB connection, etc.)
    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, `{"status":"ready","version":"%s"}`, version)
}