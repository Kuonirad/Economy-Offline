#!/bin/bash
# Development startup script for WorldShare MVP

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== WorldShare MVP Development Setup ===${NC}"
echo ""

# Check for required tools
echo -e "${YELLOW}Checking requirements...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check for GPU support (optional)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ No GPU detected. Verifier service will run in CPU mode.${NC}"
    GPU_AVAILABLE=false
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created. Please update with your configuration.${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p logs
mkdir -p data/postgres
mkdir -p data/redis
mkdir -p data/minio
mkdir -p configs/grafana/dashboards
mkdir -p configs/grafana/datasources

# Start infrastructure services
echo -e "${YELLOW}Starting infrastructure services...${NC}"
docker-compose up -d postgres redis minio

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check service health
docker-compose ps

# Initialize MinIO buckets
echo -e "${YELLOW}Initializing MinIO buckets...${NC}"
docker-compose exec -T minio mc alias set local http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
docker-compose exec -T minio mc mb local/scene-assets 2>/dev/null || true
docker-compose exec -T minio mc mb local/optimized-results 2>/dev/null || true
docker-compose exec -T minio mc mb local/verification-data 2>/dev/null || true

# Build services
echo -e "${YELLOW}Building services...${NC}"
if [ "$1" == "--build" ]; then
    docker-compose build scheduler
    if [ "$GPU_AVAILABLE" == "true" ]; then
        docker-compose build verifier
    fi
fi

# Start application services
echo -e "${YELLOW}Starting application services...${NC}"
docker-compose up -d scheduler

if [ "$GPU_AVAILABLE" == "true" ]; then
    docker-compose up -d verifier
else
    echo -e "${YELLOW}Skipping verifier service (no GPU available)${NC}"
fi

# Start monitoring services
if [ "$1" == "--monitoring" ]; then
    echo -e "${YELLOW}Starting monitoring services...${NC}"
    docker-compose up -d prometheus grafana
fi

# Show service status
echo ""
echo -e "${GREEN}=== Services Status ===${NC}"
docker-compose ps

echo ""
echo -e "${GREEN}=== Service URLs ===${NC}"
echo -e "Scheduler API:    ${GREEN}http://localhost:8080${NC}"
echo -e "Verifier API:     ${GREEN}http://localhost:8081${NC}"
echo -e "MinIO Console:    ${GREEN}http://localhost:9001${NC} (minioadmin/minioadmin)"
echo -e "PostgreSQL:       ${GREEN}localhost:5432${NC} (worldshare/worldshare)"
echo -e "Redis:            ${GREEN}localhost:6379${NC}"

if [ "$1" == "--monitoring" ]; then
    echo -e "Prometheus:       ${GREEN}http://localhost:9092${NC}"
    echo -e "Grafana:          ${GREEN}http://localhost:3000${NC} (admin/admin)"
fi

echo ""
echo -e "${GREEN}=== Useful Commands ===${NC}"
echo "View logs:        docker-compose logs -f [service]"
echo "Stop services:    docker-compose down"
echo "Clean everything: docker-compose down -v"
echo "Run tests:        make test"
echo ""
echo -e "${GREEN}Development environment ready!${NC}"