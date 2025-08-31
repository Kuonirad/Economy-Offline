package database

import (
    "context"
    "database/sql"
    "fmt"
    "time"
    
    _ "github.com/lib/pq"
    "github.com/sirupsen/logrus"
)

var log = logrus.New()

// DB wraps the SQL database connection
type DB struct {
    *sql.DB
    config Config
}

// Config holds database configuration
type Config struct {
    Host     string
    Port     int
    User     string
    Password string
    Database string
    SSLMode  string
    MaxConns int
    Timeout  time.Duration
}

// NewConnection creates a new database connection
func NewConnection(connectionString string) (*DB, error) {
    // Parse connection string or use default config
    config := Config{
        Host:     "localhost",
        Port:     5432,
        User:     "worldshare",
        Password: "worldshare",
        Database: "worldshare_scheduler",
        SSLMode:  "disable", // For development
        MaxConns: 50,
        Timeout:  10 * time.Second,
    }
    
    if connectionString == "" {
        connectionString = fmt.Sprintf(
            "host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
            config.Host, config.Port, config.User, config.Password, config.Database, config.SSLMode,
        )
    }
    
    // Open database connection
    sqlDB, err := sql.Open("postgres", connectionString)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }
    
    // Configure connection pool
    sqlDB.SetMaxOpenConns(config.MaxConns)
    sqlDB.SetMaxIdleConns(config.MaxConns / 2)
    sqlDB.SetConnMaxLifetime(5 * time.Minute)
    
    // Test connection
    ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
    defer cancel()
    
    if err := sqlDB.PingContext(ctx); err != nil {
        sqlDB.Close()
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }
    
    db := &DB{
        DB:     sqlDB,
        config: config,
    }
    
    // Initialize schema
    if err := db.initSchema(); err != nil {
        log.WithError(err).Warn("Failed to initialize schema (expected in M0)")
    }
    
    log.Info("Database connection established")
    return db, nil
}

// initSchema creates the database schema if it doesn't exist
func (db *DB) initSchema() error {
    schema := `
    CREATE TABLE IF NOT EXISTS jobs (
        id UUID PRIMARY KEY,
        scene_id VARCHAR(255),
        scene_type VARCHAR(50),
        optimization_path VARCHAR(50),
        status VARCHAR(50),
        priority INTEGER,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
    
    CREATE TABLE IF NOT EXISTS shards (
        id VARCHAR(255) PRIMARY KEY,
        job_id UUID REFERENCES jobs(id),
        shard_index INTEGER,
        total_shards INTEGER,
        shard_type VARCHAR(50),
        shard_data JSONB,
        status VARCHAR(50),
        node_id VARCHAR(255),
        attempts INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        dispatched_at TIMESTAMP,
        completed_at TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_shards_job_id ON shards(job_id);
    CREATE INDEX IF NOT EXISTS idx_shards_status ON shards(status);
    
    CREATE TABLE IF NOT EXISTS nodes (
        id VARCHAR(255) PRIMARY KEY,
        status VARCHAR(50),
        capabilities JSONB,
        performance JSONB,
        last_heartbeat TIMESTAMP,
        current_workload INTEGER DEFAULT 0,
        max_workload INTEGER DEFAULT 5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
    CREATE INDEX IF NOT EXISTS idx_nodes_heartbeat ON nodes(last_heartbeat);
    
    CREATE TABLE IF NOT EXISTS work_results (
        id UUID PRIMARY KEY,
        job_id UUID REFERENCES jobs(id),
        shard_id VARCHAR(255) REFERENCES shards(id),
        node_id VARCHAR(255) REFERENCES nodes(id),
        result_data JSONB,
        quality_metrics JSONB,
        verification_status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_work_results_job_id ON work_results(job_id);
    `
    
    _, err := db.Exec(schema)
    return err
}

// Close closes the database connection
func (db *DB) Close() error {
    log.Info("Closing database connection")
    return db.DB.Close()
}