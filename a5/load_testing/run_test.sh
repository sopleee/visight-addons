#!/bin/bash

# Quick start script for running Locust load tests
# Usage: ./run_test.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Visight Load Testing - Quick Start"
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create .env file from .env.example:"
    echo "  cp .env.example .env"
    echo "Then edit .env with your credentials"
    exit 1
fi

# Load environment variables
echo -e "${GREEN}Loading environment variables...${NC}"
export $(cat .env | grep -v '^#' | grep -v '^$' | sed 's/#.*$//' | xargs)

# Check if required variables are set
if [ -z "$MODAL_ENDPOINT_URL" ]; then
    echo -e "${RED}Error: MODAL_ENDPOINT_URL not set in .env${NC}"
    exit 1
fi

# Run environment check
echo -e "\n${GREEN}Running environment check...${NC}"
python3 check_env.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Environment check failed. Please fix the issues above.${NC}"
    exit 1
fi

# Parse command line arguments
MODE=${1:-web}
USERS=${2:-5}
SPAWN_RATE=${3:-1}
RUN_TIME=${4:-10m}

echo -e "\n${GREEN}Starting Locust in $MODE mode...${NC}"
echo "Configuration:"
echo "  Host: $MODAL_ENDPOINT_URL"
echo "  Users: $USERS"
echo "  Spawn Rate: $SPAWN_RATE users/sec"
echo "  Run Time: $RUN_TIME"
echo ""

case $MODE in
    web)
        echo -e "${YELLOW}Starting Locust Web UI...${NC}"
        echo "Open http://localhost:8089 in your browser"
        locust -f locustfile.py --host=$MODAL_ENDPOINT_URL
        ;;
    headless)
        echo -e "${YELLOW}Running headless test...${NC}"
        locust -f locustfile.py \
            --host=$MODAL_ENDPOINT_URL \
            --users=$USERS \
            --spawn-rate=$SPAWN_RATE \
            --run-time=$RUN_TIME \
            --headless \
            --html=report_$(date +%Y%m%d_%H%M%S).html
        ;;
    master)
        echo -e "${YELLOW}Starting Locust Master...${NC}"
        locust -f locustfile.py --host=$MODAL_ENDPOINT_URL --master
        ;;
    worker)
        echo -e "${YELLOW}Starting Locust Worker...${NC}"
        locust -f locustfile.py --worker --master-host=localhost
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: ./run_test.sh [web|headless|master|worker] [users] [spawn_rate] [run_time]"
        echo "Examples:"
        echo "  ./run_test.sh web                    # Start web UI"
        echo "  ./run_test.sh headless 5 1 10m       # Run headless test"
        echo "  ./run_test.sh master                 # Start master node"
        echo "  ./run_test.sh worker                 # Start worker node"
        exit 1
        ;;
esac
