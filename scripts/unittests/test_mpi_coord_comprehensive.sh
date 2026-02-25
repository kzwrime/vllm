#!/bin/bash
# Comprehensive test script for MPI coordination setup
# Tests various rank counts: 1, 4, 7, 64, 67, 128, 512

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../"

# Source the environment to get variables
source "$SCRIPT_DIR/user_env.sh" 2>/dev/null

COORD_PORT=15555
COORD_SCRIPT="$SCRIPT_DIR/mpi_coord_setup.py"
MPC_SIZE=${USER_VLLM_MPC_SIZE:-2}
DP_SIZE=${USER_VLLM_DATA_PARALLEL_SIZE:-2}

# Test cases: (rank_count, expected_mpc_groups)
TEST_CASES=(
    "1:1"
    "4:2"
    "8:2"
    "64:8"
    "68:17"
    "68:4"
    "128:8"
    "512:16"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

run_test() {
    local num_ranks=$1
    local expected_groups=$2
    local test_name="TEST_${num_ranks}_RANKS"

    echo ""
    echo "=========================================="
    echo "  Testing with $num_ranks ranks"
    echo "  Expected MPC groups: $expected_groups"
    echo "=========================================="

    # Clean up any previous test artifacts
    # rm -f /tmp/vllm_mpi_test_*.sh

    # Start server in background
    echo "[Server] Starting coordination server..."
    python3 "$COORD_SCRIPT" --server --port $COORD_PORT --expected-ranks $num_ranks \
        > /tmp/vllm_coord_server_${test_name}.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to start
    sleep 1

    # Start clients in background
    echo "[Clients] Starting $num_ranks client processes..."

    for RANK in $(seq 0 $((num_ranks - 1))); do
        # Simulate IPs: 172.33.0.11, 172.33.0.12, ...
        # For larger ranks, extend to 172.33.1.x, 172.33.2.x, etc.
        ip_base=$((11 + RANK))
        ip_third=$((ip_base / 256))
        ip_fourth=$((ip_base % 256))
        IP="172.33.${ip_third}.${ip_fourth}"

        (
            export VLLM_MPI_ENV_EXPORT_FILE="/tmp/vllm_mpi_test_rank_${RANK}.sh"
            python3 "$COORD_SCRIPT" --client --port $COORD_PORT --rank $RANK --ip "$IP" \
                > /tmp/vllm_coord_client_rank_${RANK}.log 2>&1
        ) &
    done

    # Wait for all background processes to complete
    echo "[Test] Waiting for all background processes to finish..."
    wait

    # Wait for all clients to complete (with timeout)
    echo "[Test] Checking completion status..."
    local wait_count=0
    local max_wait=10  # 10 seconds should be enough since we already waited

    while [ $wait_count -lt $max_wait ]; do
        local completed=0
        for RANK in $(seq 0 $((num_ranks - 1))); do
            if [ -f "/tmp/vllm_mpi_test_rank_${RANK}.sh" ]; then
                completed=$((completed + 1))
            fi
        done

        if [ $completed -eq $num_ranks ]; then
            echo "[Test] All $num_ranks clients completed!"
            break
        fi

        echo -n "."
        sleep 1
        wait_count=$((wait_count + 1))
    done
    echo ""

    # Check if all clients completed
    if [ $wait_count -ge $max_wait ]; then
        echo -e "${RED}[FAIL] Test timeout after ${max_wait}s${NC}"
        echo "[Server] Only $completed/$num_ranks clients completed"
        kill $SERVER_PID 2>/dev/null
        test_failed=$((test_failed + 1))
        return 1
    fi

    # Verify results
    echo "[Test] Verifying results..."
    local errors=0

    # Check 1: All env files exist
    for RANK in $(seq 0 $((num_ranks - 1))); do
        if [ ! -f "/tmp/vllm_mpi_test_rank_${RANK}.sh" ]; then
            echo -e "${RED}[FAIL] Missing env file for rank $RANK${NC}"
            errors=$((errors + 1))
        fi
    done

    # Check 2: Verify environment variables are set correctly
    local expected_dp_address="172.33.0.11"  # Rank 0's IP

    for RANK in $(seq 0 $((num_ranks - 1))); do
        local env_file="/tmp/vllm_mpi_test_rank_${RANK}.sh"
        if [ -f "$env_file" ]; then
            # Check USER_VLLM_DATA_PARALLEL_ADDRESS
            if ! grep -q "USER_VLLM_DATA_PARALLEL_ADDRESS=\"${expected_dp_address}\"" "$env_file"; then
                echo -e "${RED}[FAIL] Rank $RANK: Wrong USER_VLLM_DATA_PARALLEL_ADDRESS${NC}"
                errors=$((errors + 1))
            fi

            # Check VLLM_DP_MASTER_WORKER_IP
            if ! grep -q "VLLM_DP_MASTER_WORKER_IP=\"${expected_dp_address}\"" "$env_file"; then
                echo -e "${RED}[FAIL] Rank $RANK: Wrong VLLM_DP_MASTER_WORKER_IP${NC}"
                errors=$((errors + 1))
            fi

            # Check ExecutorIP (should be same for all ranks in same MPC group)
            local dp_rank=$((RANK / MPC_SIZE))
            local mpc_group_first_rank=$((dp_rank * MPC_SIZE))
            local ip_base=$((11 + mpc_group_first_rank))
            local ip_third=$((ip_base / 256))
            local ip_fourth=$((ip_base % 256))
            local expected_executor_ip="172.33.${ip_third}.${ip_fourth}"

            if ! grep -q "ExecutorIP=\"${expected_executor_ip}\"" "$env_file"; then
                echo -e "${RED}[FAIL] Rank $RANK: Wrong ExecutorIP (expected ${expected_executor_ip})${NC}"
                errors=$((errors + 1))
            fi
        fi
    done

    # Check 3: Verify server received all ranks
    local server_received=$(grep -c "Received rank" /tmp/vllm_coord_server_${test_name}.log 2>/dev/null || echo 0)
    if [ $server_received -ne $num_ranks ]; then
        echo -e "${RED}[FAIL] Server only received $server_received/$num_ranks ranks${NC}"
        errors=$((errors + 1))
    fi

    # Kill server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    # # Clean up test files
    # rm -f /tmp/vllm_mpi_test_*.sh
    # rm -f /tmp/vllm_coord_server_${test_name}.log
    # rm -f /tmp/vllm_coord_client_rank_*.log

    # Report results
    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}[PASS] Test with $num_ranks ranks passed!${NC}"
        test_passed=$((test_passed + 1))
        return 0
    else
        echo -e "${RED}[FAIL] Test with $num_ranks ranks failed with $errors errors${NC}"
        test_failed=$((test_failed + 1))
        return 1
    fi
}

# Run all test cases
echo "=========================================="
echo "  MPI Coordination Comprehensive Tests"
echo "=========================================="
echo "MPC_SIZE: $MPC_SIZE"
echo "Test cases: ${TEST_CASES[@]}"
echo ""

for test_case in "${TEST_CASES[@]}"; do
    IFS=':' read -r num_ranks expected_groups <<< "$test_case"
    run_test "$num_ranks" "$expected_groups"

    # Small pause between tests
    sleep 2
done

# Summary
echo ""
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}${test_passed}${NC}"
echo -e "Failed: ${RED}${test_failed}${NC}"
echo "Total:  $((test_passed + test_failed))"
echo ""

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
