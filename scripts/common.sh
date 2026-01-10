#!/bin/bash

# 函数：检查指定环境变量是否存在，存在则打印，不存在则报错退出。
# 参数: $1 - 要检查的环境变量名称
check_and_print_env() {
    local var_name="$1"
    # 使用 ${!var_name} 来间接引用变量的值
    if [ -z "${!var_name}" ]; then
        echo "❌ 错误：必需的环境变量 ${var_name} 未设置或为空。" >&2
        echo "    请设置 ${var_name}。" >&2
        exit 1
    else
        echo "✅ ${var_name} 已设置：'${!var_name}'"
    fi
}

# 函数：加载指定的环境变量文件。如果文件不存在则报错退出。
# 参数: $1 - 要加载的文件名 (相对于 $SCRIPT_DIR)
load_env_file() {
    local file_name="$1"
    local env_file=$file_name

    if [ -f "$env_file" ]; then
        echo "✅ 正在加载环境文件: $env_file"
        # 使用 source 命令加载文件内容
        source "$env_file"
        
        # 检查 source 命令是否成功
        if [ $? -ne 0 ]; then
            echo "❌ 错误！加载文件 $env_file 失败。请检查文件内容是否有语法错误。" >&2
            exit 1
        fi
    else
        echo "❌ 错误！无法找到必需的环境文件: $env_file" >&2
        exit 1
    fi
}