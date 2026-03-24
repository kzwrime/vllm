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
    local env_file="$1"

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

# 函数：智能加载用户配置
# 优先级: user_env.sh > 预设配置 > user_env_template.sh
# 参数: 无
load_user_config() {
    local script_dir="$1"

    echo "load_user_config"

    # 优先级 1: 预设配置 (VLLM_CURRENT_PRESET)
    if [ -n "${VLLM_CURRENT_PRESET:-}" ]; then
        local preset_file="$script_dir/presets/${VLLM_CURRENT_PRESET}.sh"
        if [ -f "$preset_file" ]; then
            echo "🚀 优先加载预设配置: $preset_file"
            source "$preset_file"
        else
            echo "⚠️  警告：设置了预设但文件不存在: $preset_file"
            # 这里可以选择 exit 1 (强制报错) 或者继续尝试后续逻辑
        fi

    # 优先级 2: 用户自定义配置 (user_env.sh)
    elif [ -f "$script_dir/user_env.sh" ]; then
        echo "✅ 正在加载用户配置: $script_dir/user_env.sh"
        source "$script_dir/user_env.sh"

    # 优先级 3: 模板文件回退 (user_env_template.sh)
    elif [ -f "$script_dir/user_env_template.sh" ]; then
        echo "ℹ️  未检测到自定义配置，加载模板: $script_dir/user_env_template.sh"
        source "$script_dir/user_env_template.sh"

    # 最终兜底：报错退出
    else
        echo "❌ 错误！无法找到任何有效的配置文件。" >&2
        exit 1
    fi
}
