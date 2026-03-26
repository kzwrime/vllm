#!/bin/bash
# VLLM 自动化测试脚本
# 用法: ./scripts/run_vllm_test.sh <preset_name> [options]
# 示例: ./scripts/run_vllm_test.sh qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager
# 示例: ./scripts/run_vllm_test.sh qwen30b_a3b_dp2_tp2_ep_statistics_only
# 示例: ./scripts/run_vllm_test.sh qwen30b_a3b_dp2_tp2_ep_statistics_only --bench
# 默认：启动服务后，调用 serve_test_template.sh 测试
# 选项:
#   --no-test    只启动服务，不运行测试
#   --bench      启动服务后，调用 serve_test_bench_template.sh 测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 解析参数
# TEST_MODE: "test" (默认) | "bench" (--bench) | "none" (--no-test)
TEST_MODE="test"
PRESET_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-test)
            TEST_MODE="none"
            shift
            ;;
        --bench)
            TEST_MODE="bench"
            shift
            ;;
        *)
            if [ -z "$PRESET_NAME" ]; then
                PRESET_NAME="$1"
            else
                log_error "未知参数: $1"
                echo "用法: $0 <preset_name> [--no-test|--bench]"
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查预设名称参数
if [ -z "$PRESET_NAME" ]; then
    log_error "缺少预设名称参数"
    echo ""
    echo "用法: $0 <preset_name> [--no-test]"
    echo ""
    echo "示例:"
    echo "  $0 qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager"
    exit 1
fi
PRESET_FILE="$SCRIPT_DIR/presets/${PRESET_NAME}.sh"

# 检查预设文件是否存在
if [ ! -f "$PRESET_FILE" ]; then
    log_error "预设文件不存在: $PRESET_FILE"
    exit 1
fi

# 创建日志目录
mkdir -p "$PROJECT_ROOT/logs"

# 测试日志文件
TEST_LOG="$PROJECT_ROOT/logs/test_$(date +%Y%m%d_%H%M%S).log"
BENCH_LOG="$PROJECT_ROOT/logs/bench_$(date +%Y%m%d_%H%M%S).log"
HEAD_LOG="$PROJECT_ROOT/logs/vllm_head_log.txt"

# 存储进程 PID
PIDS_FILE="$PROJECT_ROOT/logs/.test_pids_$$.tmp"

# MPI 清理日志文件
MPI_CLEANUP_LOG="$PROJECT_ROOT/logs/mpi_cleanup_$(date +%Y%m%d_%H%M%S).log"

# MPI Workers 日志文件
MPI_WORKERS_LOG="$PROJECT_ROOT/logs/mpi_workers_$(date +%Y%m%d_%H%M%S).log"

# 备份旧的日志文件
backup_old_logs() {
    if ls "$PROJECT_ROOT/logs"/vllm_*.txt 1> /dev/null 2>&1; then
        for log in "$PROJECT_ROOT/logs"/vllm_*.txt; do
            mv "$log" "${log}.old" 2>/dev/null || true
        done
        log_info "已备份旧的日志文件 (*.txt.old)"
    fi
}

# 清理函数
cleanup() {
    log_info "清理进程..."

    if [ -f "$PIDS_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                log_info "杀掉进程 $pid"
                # 尝试 kill 进程组（负 PID），以便同时清理子进程
                kill -- -"$pid"
                sleep 10
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi

    log_info "尝试清理其他残留进程..."

    # 额外清理可能残留的 vllm 进程，输出到 MPI 清理日志
    {
        # 首先尝试优雅地终止进程
        log_info "尝试优雅终止 vllm 相关进程..."
        pkill -TERM -f "vllm serve" 2>&1 || true
        pkill -TERM -f "VLLM" 2>&1 || true
        pkill -TERM -f "run_mp_rpc_worker.py" 2>&1 || true
        pkill -TERM -f "serve_mp_rpc_all_mpi_template.sh" 2>&1 || true

        # 等待几秒让进程清理
        sleep 8

        # 强制清理仍然存活的进程
        log_info "强制清理残留进程..."
        pkill -9 -f "vllm serve" 2>&1 || true
        pkill -9 -f "VLLM" 2>&1 || true
        pkill -9 -f "run_mp_rpc_worker.py" 2>&1 || true
        pkill -9 -f "serve_mp_rpc_all_mpi_template.sh" 2>&1 || true

        # 清理可能残留的 mpirun 进程
        pkill -9 -f "mpirun.*serve_mp_rpc" 2>&1 || true

    } >> "$MPI_CLEANUP_LOG" 2>&1

    # 等待 MPI 清理完成，将其错误输出保存到日志文件
    {
        wait
    } >> "$MPI_CLEANUP_LOG" 2>&1

    # 检查是否有 MPI 清理消息
    if [ -s "$MPI_CLEANUP_LOG" ]; then
        log_info "MPI 清理消息已保存到: $MPI_CLEANUP_LOG"
    fi

    # 最终验证：检查是否还有残留的 python3 进程与当前预设相关
    REMAINING_PYTHON=$(ps aux | grep -E "vllm serve|run_mp_rpc_worker" | grep -v grep | wc -l)
    if [ "$REMAINING_PYTHON" -gt 0 ]; then
        log_warning "检测到 $REMAINING_PYTHON 个残留的 vllm python 进程"
        log_info "残留进程信息："
        ps aux | grep -E "vllm serve|run_mp_rpc_worker" | grep -v grep >> "$MPI_CLEANUP_LOG" 2>&1 || true
    fi

    log_info "清理完成"
}

# 设置退出时清理
trap cleanup EXIT INT TERM

# ========================================
# 步骤 1: 备份旧日志
# ========================================
log_info "========================================="
log_info "  VLLM 自动化测试"
log_info "========================================="
log_info "预设: $PRESET_NAME"
log_info "测试日志: $TEST_LOG"
echo ""

backup_old_logs

# ========================================
# 步骤 2: 加载预设配置
# ========================================
# 设置预设环境变量并获取 MPI 进程数
export VLLM_CURRENT_PRESET="$PRESET_NAME"
source "$PRESET_FILE"

MPI_COUNT=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
log_info "配置: DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
log_info "MPI 进程数: $MPI_COUNT"
log_info "模型: $USER_VLLM_MODEL"
echo ""

# 清理旧的 head 日志
rm -f "$HEAD_LOG"

# ========================================
# 步骤 3: 启动 Head Server
# ========================================
log_info "[3/7] 启动 Head Server..."
cd "$PROJECT_ROOT"

setsid bash "$SCRIPT_DIR/serve_head_only_template.sh" > "$HEAD_LOG" 2>&1 &
HEAD_PID=$!
echo "$HEAD_PID" >> "$PIDS_FILE"

log_info "Head Server PID: $HEAD_PID"
log_info "日志: $HEAD_LOG"

# ========================================
# 步骤 4: 启动 MPI Workers
# ========================================
log_info "[4/7] 启动 MPI Workers ($MPI_COUNT processes)..."

sleep 2  # 等待 head server 初始化

# 使用 setsid 创建新会话，便于后续清理整个进程组
setsid mpirun -np $MPI_COUNT bash "$SCRIPT_DIR/serve_mp_rpc/serve_mp_rpc_all_mpi_template.sh" >> "$MPI_WORKERS_LOG" 2>&1 &
MPI_PID=$!
echo "$MPI_PID" >> "$PIDS_FILE"

log_info "MPI Workers PID: $MPI_PID"
log_info "启动日志: $MPI_WORKERS_LOG"

# ========================================
# 步骤 5: 等待服务启动
# ========================================
log_info "[5/7] 等待服务启动..."

# 最大等待时间（秒）；可通过 VLLM_TEST_MAX_WAIT 覆盖（大模型在 x86 上需要更长时间）
MAX_WAIT=${VLLM_TEST_MAX_WAIT:-300}
WAIT_TIME=0
CHECK_INTERVAL=5

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if [ -f "$HEAD_LOG" ]; then
        if grep -q "Application startup complete" "$HEAD_LOG"; then
            log_success "服务启动成功！"
            break
        fi

        # 检查是否有错误
        # if grep -q "Error\|Exception\|Traceback" "$HEAD_LOG"; then
        if grep -q "Error\|Traceback" "$HEAD_LOG"; then
            log_error "检测到错误，请检查日志: $HEAD_LOG"
            tail -20 "$HEAD_LOG"
            exit 1
        fi
    fi

    echo -n "."
    sleep $CHECK_INTERVAL
    WAIT_TIME=$((WAIT_TIME + CHECK_INTERVAL))
done

echo ""

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    log_error "等待超时 ($MAX_WAIT 秒)"
    log_info "Head 日志最后 30 行:"
    tail -30 "$HEAD_LOG"
    exit 1
fi

# 额外等待，确保服务完全就绪
log_info "等待服务完全就绪..."
sleep 5

# ========================================
# 步骤 6: 运行测试/Bench（可选）
# ========================================
if [ "$TEST_MODE" = "test" ]; then
    log_info "[6/7] 运行测试..."
    echo ""

    # 运行测试并记录日志
    TEST_OUTPUT=$(bash "$SCRIPT_DIR/serve_test_template.sh" 2>&1)
    TEST_EXIT_CODE=$?

    echo "$TEST_OUTPUT" | tee "$TEST_LOG"

    echo ""
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log_success "测试完成！"
        log_info "测试日志: $TEST_LOG"
    else
        log_warning "测试退出码: $TEST_EXIT_CODE"
    fi

    # ========================================
    # 步骤 6.5: 提取并显示模型回答
    # ========================================
    echo ""
    log_info "========================================"
    log_info "  模型回答 (Content)"
    log_info "========================================"

    # 尝试使用 jq 提取 content，如果失败则使用 grep+sed
    if command -v jq &> /dev/null; then
        # 使用 jq 提取 content
        CONTENT=$(echo "$TEST_OUTPUT" | jq -r '.choices[0].message.content // .choices[0].text // empty' 2>/dev/null)
        if [ -n "$CONTENT" ]; then
            echo -e "${GREEN}$CONTENT${NC}"
        else
            log_warning "无法使用 jq 提取 content，尝试备用方法..."
            CONTENT=$(echo "$TEST_OUTPUT" | grep -oP '"content":\s*"\K[^"]*' | head -1)
            [ -n "$CONTENT" ] && echo -e "${GREEN}$CONTENT${NC}" || log_warning "未能提取到 content 字段"
        fi
    else
        # 备用方法：使用 grep + sed 提取 content
        CONTENT=$(echo "$TEST_OUTPUT" | grep -oP '"content":\s*"\K[^"]*' | head -1)
        if [ -n "$CONTENT" ]; then
            echo -e "${GREEN}$CONTENT${NC}"
        else
            # 尝试匹配 multiline content (处理包含转义字符的情况)
            CONTENT=$(echo "$TEST_OUTPUT" | sed -n 's/.*"content"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
            [ -n "$CONTENT" ] && echo -e "${GREEN}$CONTENT${NC}" || log_warning "未能提取到 content 字段"
        fi
    fi

    echo ""
    log_info "完整响应请查看: $TEST_LOG"

elif [ "$TEST_MODE" = "bench" ]; then
    log_info "[6/7] 运行 Bench（serve_test_bench_template.sh）..."
    echo ""

    bash "$SCRIPT_DIR/serve_test_bench_template.sh" > "$BENCH_LOG" 2>&1
    BENCH_EXIT_CODE=$?

    echo ""
    if [ $BENCH_EXIT_CODE -eq 0 ]; then
        log_success "Bench 完成！"
    else
        log_warning "Bench 退出码: $BENCH_EXIT_CODE"
    fi
    log_info "Bench 日志: $BENCH_LOG"

else
    log_info "[6/7] 跳过测试（--no-test 模式）"
    log_info "服务已启动并就绪，您可以手动测试"
    log_info "测试命令示例:"
    log_info "  curl http://localhost:\${USER_VLLM_PORT}/v1/chat/completions \\"
    log_info "    -H \"Content-Type: application/json\" \\"
    log_info "    -d '{\"model\": \"\${USER_VLLM_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 16}'"
fi

# ========================================
# 步骤 7: 显示日志摘要
# ========================================
echo ""
log_info "========================================="
log_info "  日志文件位置"
log_info "========================================="
log_info "Head Server:      $HEAD_LOG"
if [ "$TEST_MODE" = "test" ]; then
    log_info "Test Result:      $TEST_LOG"
elif [ "$TEST_MODE" = "bench" ]; then
    log_info "Bench Result:     $BENCH_LOG"
fi
log_info "MPI Workers:      $MPI_WORKERS_LOG"
log_info "MPI Cleanup:      $MPI_CLEANUP_LOG"

# 列出所有 worker 日志（只显示本轮运行的）
if ls "$PROJECT_ROOT/logs"/vllm_worker_log_rank*.txt 1> /dev/null 2>&1; then
    log_info "Worker Logs:"
    ls -lh "$PROJECT_ROOT"/logs/vllm_worker_log_rank*.txt | awk '{print "  " $9 " (" $5 ")"}'
fi

if ls "$PROJECT_ROOT/logs"/vllm_serve_log_dp_rank*.txt 1> /dev/null 2>&1; then
    log_info "DP Server Logs:"
    ls -lh "$PROJECT_ROOT"/logs/vllm_serve_log_dp_rank*.txt | awk '{print "  " $9 " (" $5 ")"}'
fi

echo ""
log_info "========================================="
if [ "$TEST_MODE" = "none" ]; then
    log_info "  服务已启动（按 Ctrl+C 退出）"
    log_info "  注意：脚本退出时会自动清理所有进程"
else
    log_info "  测试完成"
fi
log_info "========================================="

# 如果是 --no-test 模式，保持脚本运行以便用户可以手动测试
if [ "$TEST_MODE" = "none" ]; then
    echo ""
    log_info "服务正在运行中... 按 Ctrl+C 停止"
    log_info "提示：在另一个终端中可以查看日志："
    log_info "  tail -f $HEAD_LOG"

    # 等待用户中断
    wait
fi
