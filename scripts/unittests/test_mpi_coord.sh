#!/bin/bash
# Test script to verify MPI coordination setup
# This simulates a multi-rank environment for testing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../"

# Source the environment to get variables
source "$SCRIPT_DIR/user_env.sh"

echo "=== Testing MPI Coordination Setup ==="
echo ""

# Test parameters
TEST_SIZE=4
COORD_PORT=15555
COORD_SCRIPT="$SCRIPT_DIR/mpi_coord_setup.py"

# Calculate expected values
MPC_SIZE=${USER_VLLM_MPC_SIZE:-2}
DP_SIZE=${USER_VLLM_DATA_PARALLEL_SIZE:-2}

echo "Configuration:"
echo "  TOTAL_SIZE: $TEST_SIZE"
echo "  MPC_SIZE: $MPC_SIZE"
echo "  DP_SIZE: $DP_SIZE"
echo ""

# Start server in background
echo "[1] Starting coordination server..."
python3 "$COORD_SCRIPT" --server --port $COORD_PORT --expected-ranks $TEST_SIZE &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Start clients in background (simulating different ranks)
echo "[2] Starting client processes..."

for RANK in $(seq 0 $((TEST_SIZE - 1))); do
    # Simulate different IPs for different ranks
    # In real scenario, each rank would use its actual VLLM_LOOPBACK_IP
    IP="172.33.0.$((11 + RANK))"

    (
        echo "  Client $RANK connecting from IP $IP..."
        export VLLM_MPI_ENV_EXPORT_FILE="/tmp/test_vllm_env_${RANK}.sh"
        python3 "$COORD_SCRIPT" --client --port $COORD_PORT --rank $RANK --ip "$IP"
    ) &
done

# Wait for all clients to complete
wait

echo ""
echo "[3] Test complete. Checking generated environment files..."
echo ""

# Display results
for RANK in $(seq 0 $((TEST_SIZE - 1))); do
    ENV_FILE="/tmp/test_vllm_env_${RANK}.sh"
    if [ -f "$ENV_FILE" ]; then
        echo "--- Rank $RANK Environment Variables ---"
        cat "$ENV_FILE"
        echo ""
    else
        echo "ERROR: Environment file for rank $RANK not found!"
    fi
done

# Cleanup
kill $SERVER_PID 2>/dev/null

echo "=== Test Complete ==="
echo "Environment files saved in /tmp/test_vllm_env_*.sh"
