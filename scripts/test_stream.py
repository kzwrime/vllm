# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys

from openai import OpenAI


def source_env_file(env_file):
    """读取 shell 环境变量文件并返回环境变量字典"""
    result = subprocess.run(
        f"bash -c 'source {env_file} && env'",
        shell=True,
        capture_output=True,
        text=True,
    )

    env_vars = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    return env_vars


# 1. 读取 user_env.sh 中的环境变量
script_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(script_dir, "user_env.sh")

if not os.path.exists(env_file):
    print(f"[错误] 找不到环境变量文件: {env_file}")
    sys.exit(1)

env_vars = source_env_file(env_file)

# 2. 从环境变量中获取配置
MODEL_NAME = env_vars.get("USER_VLLM_MODEL", "你的模型名称")
PORT = env_vars.get("USER_VLLM_PORT", "8000")

# 3. 初始化客户端，指向你的 vLLM 服务地址
client = OpenAI(
    api_key="EMPTY",  # vLLM 默认不需要真实的 API Key
    base_url=f"http://localhost:{PORT}/v1",
)


def main():
    print("开始测试流式输出...\\n")
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
