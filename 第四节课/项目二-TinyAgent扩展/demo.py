# ┌─────────────────────────────────────────────────────────────────────┐
# │ 项目二扩展版：Tiny-Agent CLI 演示                                    │
# │ 配置了 8 个工具：日期 / 维基百科 / 天气 / 计算器 / 文件读取 /        │
# │ 字符串处理 / 随机数                                                   │
# └─────────────────────────────────────────────────────────────────────┘

from src.core import Agent
from src.tools import (
    get_current_datetime, search_wikipedia, get_current_temperature,
    calculator, read_file, reverse_string, word_count, random_number
)
from openai import OpenAI

if __name__ == "__main__":
    # ===== 配置：替换为你的硅基流动 API Key =====
    API_KEY = "your-siliconflow-api-key"

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.siliconflow.cn/v1",
    )

    # 创建 Agent，注册 8 个工具
    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[
            get_current_datetime,
            search_wikipedia,
            get_current_temperature,
            calculator,          # 新增：数学计算
            read_file,           # 新增：文件读取
            reverse_string,      # 新增：字符串反转
            word_count,          # 新增：字数统计
            random_number,       # 新增：随机数
        ],
        verbose=True,
    )

    print("="*50)
    print("  Tiny-Agent 扩展版 (8 个工具)")
    print("  输入问题开始对话，输入 exit 退出")
    print("="*50)

    while True:
        prompt = input("\nUser: ").strip()
        if prompt.lower() == 'exit':
            break
        if not prompt:
            continue
        response = agent.get_completion(prompt)
        print(f"Assistant: {response}")
