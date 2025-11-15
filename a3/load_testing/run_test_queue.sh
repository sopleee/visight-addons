#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Visight Load Testing - Queue-based"
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

echo ""
echo "Starting Locust in $MODE mode (QUEUE-BASED)..."
echo "Configuration:"
echo "  Host: $MODAL_ENDPOINT_URL"
echo "  Users: $USERS"
echo "  Spawn Rate: $SPAWN_RATE users/sec"
echo "  Run Time: $RUN_TIME"
echo "  Poll Interval: ${POLL_INTERVAL:-10}s"
echo "  Max Poll Time: ${MAX_POLL_TIME:-600}s"
echo ""

case $MODE in
    web)
        echo "Starting Locust Web UI..."
        echo "Open http://localhost:8089 in your browser"
        locust -f locustfile_queue.py --host=$MODAL_ENDPOINT_URL
        ;;
    headless)
        echo "Running headless load test..."
        locust -f locustfile_queue.py \
            --host=$MODAL_ENDPOINT_URL \
            --headless \
            --users=$USERS \
            --spawn-rate=$SPAWN_RATE \
            --run-time=$RUN_TIME \
            --html=load_test_report_queue.html \
            --csv=load_test_results_queue
        ;;
    master)
        echo "Starting Locust master node..."
        locust -f locustfile_queue.py --host=$MODAL_ENDPOINT_URL --master
        ;;
    worker)
        echo "Starting Locust worker node..."
        locust -f locustfile_queue.py --worker
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Usage: $0 [web|headless|master|worker] [users] [spawn-rate] [run-time]"
        echo ""
        echo "Examples:"
        echo "  $0 web                    # Start web UI"
        echo "  $0 headless 10 2 5m       # Run headless with 10 users for 5 minutes"
        echo "  $0 master                 # Start master for distributed testing"
        echo "  $0 worker                 # Start worker for distributed testing"
        exit 1
        ;;
esac
