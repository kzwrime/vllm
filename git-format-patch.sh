#!/bin/bash

# ==========================================
# 配置部分
# ==========================================

# 检查参数数量
if [ "$#" -lt 2 ]; then
    echo "使用方法: $0 <START_COMMIT> <END_COMMIT>"
    echo "示例: $0 b2b6dc b2b6dc"
    exit 1
fi

# 从参数获取 Commit ID
# $1: 起始 Commit ID (包含此 Commit)
# $2: 结束 Commit ID
START_COMMIT=$1
END_COMMIT=$2

# 文件名前缀 (可根据需要在此修改)
PREFIX="vllm"

# ==========================================
# 逻辑部分：自动计算日期和短哈希
# ==========================================

# 获取当前日期 (格式: YYYYMMDD)
DATE=$(date +%Y%m%d)

# 获取起始和结束 Commit 的短哈希 (前 6 位)
# 增加错误处理，确保输入的 Commit ID 有效
SHORT_START=$(git rev-parse --short=6 "$START_COMMIT" 2>/dev/null)
SHORT_END=$(git rev-parse --short=6 "$END_COMMIT" 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "错误：无效的 Commit ID。"
    exit 1
fi

# 拼接输出文件名
# 格式: 前缀_日期_开始短哈希_结束短哈希.patch
OUTPUT_FILE="${PREFIX}_${DATE}_${SHORT_START}_${SHORT_END}.patch"

echo "正在生成 Patch..."
echo "范围: $START_COMMIT^ 到 $END_COMMIT"
echo "文件名: $OUTPUT_FILE"

# 执行 git format-patch 命令
# 使用 ^.. 包含起始 commit 本身
git format-patch "${START_COMMIT}^..${END_COMMIT}" --stdout > "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "--------------------------------------"
    echo "成功！Patch 已保存至: $OUTPUT_FILE"
    echo "--------------------------------------"
else
    echo "错误：生成 Patch 失败。"
fi
