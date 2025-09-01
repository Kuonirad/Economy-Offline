#!/bin/bash
# Health check script for WorldShare MVP services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== WorldShare MVP Health Check ===${NC}"
echo ""

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $name... "
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        return 1
    fi
}

# Function to check port
check_port() {
    local name=$1
    local host=$2
    local port=$3
    
    echo -n "Checking $name port $port... "
    
    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✓ Open${NC}"
        return 0
    else
        echo -e "${RED}✗ Closed${NC}"
        return 1
    fi
}

# Check services
echo -e "${YELLOW}Checking services...${NC}"
check_service "Economy Scheduler Health" "http://localhost:8080/health"
check_service "Economy Scheduler Ready" "http://localhost:8080/ready"
check_service "Trust Engine Verifier Health" "http://localhost:8081/health"
check_service "Trust Engine Verifier Ready" "http://localhost:8081/ready"

echo ""
echo -e "${YELLOW}Checking infrastructure...${NC}"
check_port "PostgreSQL" "localhost" 5432
check_port "Redis" "localhost" 6379
check_port "MinIO" "localhost" 9000
check_port "MinIO Console" "localhost" 9001

echo ""
echo -e "${YELLOW}Checking metrics endpoints...${NC}"
check_service "Scheduler Metrics" "http://localhost:9090/metrics"
check_service "Verifier Metrics" "http://localhost:9091/metrics"

# Check GPU if available
echo ""
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        echo -e "${GREEN}✓ GPU: $line${NC}"
    done
else
    echo -e "${YELLOW}⚠ No GPU detected${NC}"
fi

# Check Docker containers
echo ""
echo -e "${YELLOW}Checking Docker containers...${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep worldshare || echo "No WorldShare containers running"

echo ""
echo -e "${GREEN}Health check complete!${NC}"