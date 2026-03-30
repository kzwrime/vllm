# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import subprocess
import sys
import time

# 移除 logging，改为直接写文件和动态终端输出
from openai import AsyncOpenAI


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

# 3. 初始化异步客户端
client = AsyncOpenAI(
    api_key="EMPTY",  # vLLM 默认不需要真实的 API Key
    base_url=f"http://localhost:{PORT}/v1",
)

# 2. 准备多组测试数据
PROMPTS = [
    "请用一段话简单介绍一下量子计算。",
    "写一首关于春天的七言绝句。",
    "欧盟有多少个国家，详细展开论述欧盟现状。",
    "解释一下相对论的核心思想。",
    "给出三个提高编程效率的建议。",
    (
        "作为一名资深的国际政治与经济评论员，"
        "请针对欧洲联盟（EU）的现状进行深度剖析。"
        "首先，"
        "请明确指出截至2026年欧盟的成员国数量，"
        "并简述近年来成员国变动（如英国脱欧后）对联盟地缘政治版图的实质性影响。"
        "接下来，"
        "请从以下三个维度详细展开论述欧盟的现状：\n\n1. **经济韧性与数字化转型**："
        "在面对全球通胀压力及能源危机后，"
        "欧盟目前的单一市场表现如何？其《数字市场法案》（DMA）和《数字服务法案》（DSA）"
        "在实施过程中对成员国经济活力产生了怎样的正面或负面效应？\n\n2. "
        "**政治一体化与内部博弈**：请探讨‘多速欧洲’概念在当前的实践情况。"
        "特别是针对匈牙利、波兰等国与欧盟总部在法治原则上的博弈，"
        "以及这种内部撕裂是否正在削弱欧盟作为一个整体在国际事务中的话语权。"
        "\n\n3. **外部安全与防务自主**：在当前的国际安全形势下，"
        "欧盟是如何平衡其对北约的依赖与对‘战略自主’（Strategic Autonomy）的追求的？"
        "欧盟防务共同体的建设目前处于什么阶段？\n\n最后，"
        "请对欧盟未来五年的发展趋势做一个简短但具前瞻性的预测，"
        "分析其是否仍能维持全球第三大经济体的核心地位。"
        "请保持语言风格专业、客观且逻辑严密。"
    ),
]
PROMPTS = PROMPTS[-1]  # 最后一条过长的测试用例暂时不执行

# 用于记录各个任务的状态，方便在屏幕上更新 tips
task_states = {}


async def fetch_stream(task_id: int, prompt: str):
    """处理单个流式请求，并将结果实时写入独立的日志文件"""
    task_states[task_id] = {"status": "请求中", "tokens": 0}
    log_file = f"vllm_task_{task_id}.log"

    try:
        # 打开独立的日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"【Prompt】: {prompt}\n")
            f.write("=" * 40 + "\n")

            # 发起异步流式请求
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,  # 开启流式输出
                temperature=0.7,
                max_tokens=2048,
            )

            # 实时接收数据块
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    f.write(content)
                    f.flush()  # 流式：立刻将内容写入磁盘
                    task_states[task_id]["tokens"] += 1
                    task_states[task_id]["status"] = "生成中"

            task_states[task_id]["status"] = "已完成"
            f.write("\n" + "=" * 40 + "\n【生成结束】")

    except Exception as e:
        task_states[task_id]["status"] = "错误"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n[请求失败]: {e}")


async def display_tips():
    """在屏幕上维持动态的提示信息 (Tips)"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    start_time = time.time()

    while True:
        active_count = 0
        status_strs = []

        for i in range(len(PROMPTS)):
            state = task_states.get(i, {"status": "等待中", "tokens": 0})
            if state["status"] not in ["已完成", "错误", "等待中"]:
                active_count += 1
            status_strs.append(f"T{i}: {state['status']}({state['tokens']}字)")

        elapsed = time.time() - start_time
        spinner = frames[idx % len(frames)]

        # 拼接单行状态栏 (\033[K 用于清除行尾残余字符，防止残留)
        line = (
            f"\r\033[K{spinner} 耗时: {elapsed:.1f}s"
            + f" | 活跃任务: {active_count}/{len(PROMPTS)} | "
            + " | ".join(status_strs)
        )

        sys.stdout.write(line)
        sys.stdout.flush()

        # 如果所有任务都已完成或出错，则退出循环
        if active_count == 0 and all(
            task_states.get(i, {}).get("status") in ["已完成", "错误"]
            for i in range(len(PROMPTS))
        ):
            sys.stdout.write(
                "\n\n所有并发任务执行完毕！请查看各自的 vllm_task_X.log 文件。\n"
            )
            break

        idx += 1
        await asyncio.sleep(0.15)


async def main():
    print("开始并发测试流式输出")
    print(f"模型: {MODEL_NAME}")
    print(f"端口: {PORT}")
    print(f"日志文件: {os.getcwd()}/vllm_task_*.log")
    print("提示：模型输出将流式写入独立的日志文件，屏幕仅显示实时状态。\n")

    # 启动状态监控任务
    monitor_task = asyncio.create_task(display_tips())

    # 并发执行所有流式请求任务
    fetch_tasks = [fetch_stream(i, prompt) for i, prompt in enumerate(PROMPTS)]
    await asyncio.gather(*fetch_tasks)

    # 等待监控任务结束
    await monitor_task


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
