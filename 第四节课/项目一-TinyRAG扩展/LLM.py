#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 原始文件来源：Datawhale happy-llm 项目

import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# RAG 专用 Prompt 模板：引导 LLM 严格基于检索到的上下文作答
RAG_PROMPT_TEMPLATE = """
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答"数据库中没有这个内容，我不知道"。
有用的回答:
"""


class OpenAIChat:
    """通过 OpenAI 兼容 API 调用大语言模型"""

    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct"):
        """
        参数:
            model: 模型名称（默认 Qwen2.5-32B，通义千问最新版）
        """
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        """
        发起一次 RAG 问答请求
        
        参数:
            prompt: 用户输入的问题
            history: 对话历史（list of {"role": "...", "content": "..."}）
            content: 检索到的上下文文本（将填入 RAG_PROMPT_TEMPLATE）
        返回:
            LLM 生成的回答文本
        """
        # 初始化 OpenAI 客户端（从环境变量读取 API 配置）
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")

        # 构建消息：对话历史 + RAG Prompt
        history.append({
            'role': 'user',
            'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)
        })

        # 调用 API 生成回答
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,   # 最大生成 Token 数
            temperature=0.1,   # 低温度 → 更确定性的回答
        )
        return response.choices[0].message.content
