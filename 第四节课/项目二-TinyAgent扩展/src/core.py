#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 原始文件来源：Datawhale happy-llm 项目

from openai import OpenAI
import json
from typing import List, Dict, Any
from src.utils import function_to_json

# Agent 的身份定义：通过 System Prompt 设定角色和行为
SYSTEM_PROMPT = """
你是一个叫 TinyAgent 的智能助理。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
你可以连续调用多个工具来解决一个复杂问题。
"""


class Agent:
    """
    ReAct Agent 核心实现

    工作流程:
        1. 用户输入 → 追加到 self.messages
        2. 调用 LLM（携带工具 Schema）→ LLM 判断是否需要调用工具
        3. 如果需要工具 → 执行工具 → 结果追加到 messages → 再次调用 LLM
        4. 如果不需要 → 直接返回回答
    """

    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct",
                 tools: List = None, verbose: bool = True):
        """
        参数:
            client: OpenAI 兼容客户端实例
            model: 使用的模型名称
            tools: 工具函数列表
            verbose: 是否打印调试信息
        """
        self.client = client
        self.tools = tools or []
        self.model = model
        # 消息历史：以 System Prompt 开头，后续追加 user/assistant/tool 消息
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 工具名 → 函数对象的映射，用于 O(1) 查找
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """将所有工具函数转为 Function Calling 所需的 JSON Schema 列表"""
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        """
        处理单个工具调用请求
        
        流程:
            1. 从 tool_call 对象中提取函数名和参数 JSON 字符串
            2. 将参数 JSON 解析为 Python 字典
            3. 从 self.tool_map 中查找函数并调用
            4. 将结果包装为标准 tool message 返回
        """
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments       # 这是 JSON 字符串
        function_id = tool_call.id

        # 解析 JSON 参数（防止 LLM 返回无效 JSON）
        try:
            args_dict = json.loads(function_args)
        except json.JSONDecodeError:
            args_dict = {}

        # 从 tool_map 查找并执行函数
        func = self.tool_map.get(function_name)
        if func:
            result = func(**args_dict)                      # **解包字典为命名参数
        else:
            result = f"Error: 找不到名为 {function_name} 的工具"

        # 包装为 tool role message
        return {
            "role": "tool",
            "content": str(result),
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt: str) -> str:
        """
        Agent 主循环入口
        
        参数:
            prompt: 用户输入
        返回:
            LLM 的最终回答
        """
        # 1. 将用户消息追加到历史
        self.messages.append({"role": "user", "content": prompt})

        # 2. 调用 LLM（携带工具 Schema）
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )

        # 3. 如果 LLM 决定调用工具
        if response.choices[0].message.tool_calls:
            # 3a. 将 assistant 的工具调用消息写入历史
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            }
            self.messages.append(assistant_message)

            # 3b. 逐个执行工具调用
            tool_list = []
            for tc in response.choices[0].message.tool_calls:
                self.messages.append(self.handle_tool_call(tc))
                tool_list.append([tc.function.name, tc.function.arguments])

            # 3c. 打印调试信息
            if self.verbose:
                print(f"  [调用工具] {tool_list}")

            # 3d. 将工具结果反馈给 LLM，再次调用生成最终答案
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 4. 将最终回答追加到历史并返回
        final_answer = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": final_answer})
        return final_answer
