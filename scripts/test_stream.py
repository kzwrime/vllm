# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys

from openai import OpenAI


def source_env_with_common_logic(script_dir):
    """
    使用与 serve_head_only_template.sh 相同的逻辑加载环境变量

    加载优先级:
    1. 预设配置 (VLLM_CURRENT_PRESET)
    2. 用户自定义配置 (user_env.sh)
    3. 模板文件回退 (user_env_template.sh)
    """
    # 构建与 serve_head_only_template.sh 相同的 bash 命令
    # 需要传递当前进程的环境变量，特别是 VLLM_CURRENT_PRESET
    bash_command = f"""
export VLLM_CURRENT_PRESET="${{VLLM_CURRENT_PRESET:-}}"
source {script_dir}/common.sh
load_env_file "{script_dir}/env.sh"
load_user_config "{script_dir}"
env
"""

    result = subprocess.run(
        ["bash", "-c", bash_command],
        capture_output=True,
        text=True,
        env=os.environ,  # 传递当前进程的环境变量
    )

    if result.returncode != 0:
        print("[错误] 加载环境变量失败:")
        print(result.stderr)
        sys.exit(1)

    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    return env_vars


# 1. 使用与 serve_head_only_template.sh 相同的逻辑加载环境变量
script_dir = os.path.dirname(os.path.abspath(__file__))
env_vars = source_env_with_common_logic(script_dir)

# 2. 从环境变量中获取配置
MODEL_NAME = env_vars.get("USER_VLLM_MODEL", "你的模型名称")
PORT = env_vars.get("USER_VLLM_PORT", "8000")

# 3. 初始化客户端，指向你的 vLLM 服务地址
client = OpenAI(
    api_key="EMPTY",  # vLLM 默认不需要真实的 API Key
    base_url=f"http://localhost:{PORT}/v1",
)


def main():
    print("开始测试流式输出...\n")
    print(f"模型: {MODEL_NAME}")
    print(f"端口: {PORT}")
    print("-" * 50)

    try:
        # 3. 发起流式请求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "欧盟有多少个国家，详细展开论述欧盟现状."}
            ],
            stream=True,  # 开启流式输出
            temperature=0.5,
            max_tokens=3000,
        )

        # 4. 实时打印返回的数据块 (chunks)
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                # 使用 end="" 和 flush=True 确保文字能够逐字平滑显示
                print(content, end="", flush=True)

    except Exception as e:
        print(f"\n[错误] 请求失败: {e}")

    print("\n" + "-" * 50)
    print("生成结束！")


if __name__ == "__main__":
    main()
