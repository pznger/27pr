#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 原始文件来源：Datawhale happy-llm 项目
# 说明：调用硅基流动 API 获取 BAAI/bge-m3 嵌入向量

import os
import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class OpenAIEmbedding:
    """通过 OpenAI 兼容 API 获取文本的向量表示"""

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化嵌入模型客户端
        
        参数:
            api_key: API 密钥（默认从环境变量 OPENAI_API_KEY 读取）
            base_url: API 地址（默认从环境变量 OPENAI_BASE_URL 读取）
        """
        self.client = OpenAI()
        # 如果传入参数则使用传入的，否则从环境变量读取
        self.client.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        调用 BAAI/bge-m3 模型将文本转为 1024 维向量
        
        参数:
            text: 待编码的文本
            model: 嵌入模型名称（BAAI/bge-m3 为多语言 SOTA 模型）
        返回:
            1024 维浮点数向量列表
        """
        # 去掉换行符以减少 Token
        text = text.replace("\n", " ")
        # 调用 API
        resp = self.client.embeddings.create(input=[text], model=model)
        return resp.data[0].embedding

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        公式: cos(v1, v2) = (v1·v2) / (||v1|| * ||v2||)
        范围: [-1, 1]，值越大表示越相似
        
        参数:
            v1, v2: 两个等长的浮点数向量
        返回:
            余弦相似度（0~1 之间通常表示不同程度的相似）
        """
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)

        # 检查是否有无穷大或 NaN
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 0.0

        dot = np.dot(a, b)                        # 点积: a·b
        norm_a = np.linalg.norm(a)                # ||a||
        norm_b = np.linalg.norm(b)                # ||b||

        if norm_a == 0 or norm_b == 0:            # 防止除零
            return 0.0
        return float(dot / (norm_a * norm_b))
