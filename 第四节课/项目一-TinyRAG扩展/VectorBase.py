#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 原始文件来源：Datawhale happy-llm 项目

import os
import json
import numpy as np
from typing import List
from tqdm import tqdm


class VectorStore:
    """
    简易向量数据库（内存 + JSON 持久化）
    
    存储结构:
        storage/
        ├── vectors.json   # 向量数据 [[0.1, 0.2, ...], ...]
        └── documents.json # 原始文档 ["文档块1", "文档块2", ...]
    """

    def __init__(self, documents: List[str] = None):
        """
        参数:
            documents: 文档块列表
        """
        self.document = documents or []
        self.vectors: List[List[float]] = []

    def get_vector(self, EmbeddingModel) -> List[List[float]]:
        """
        对所有文档块进行向量编码
        
        参数:
            EmbeddingModel: 嵌入模型实例（需有 get_embedding(text) 方法）
        返回:
            所有文档块的向量列表
        """
        self.vectors = []
        for doc in tqdm(self.document, desc="编码文档块"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        """
        将文档和向量保存到本地 JSON 文件
        
        参数:
            path: 存储目录路径
        """
        os.makedirs(path, exist_ok=True)
        # 保存原始文档（UTF-8，保留中文）
        with open(f"{path}/documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        # 保存向量
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        """
        从本地 JSON 文件加载向量和文档
        
        参数:
            path: 存储目录路径
        """
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/documents.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def query(self, query: str, EmbeddingModel, k: int = 1) -> List[str]:
        """
        查询与输入文本最相似的 k 个文档块
        
        流程:
            1. 将查询文本转为向量
            2. 计算查询向量与所有文档向量的余弦相似度
            3. 按相似度降序排列，取 Top-K
        
        参数:
            query: 查询文本
            EmbeddingModel: 嵌入模型实例
            k: 返回的文档块数量
        返回:
            最相似的 k 个文档块文本列表
        """
        # 第1步：编码查询
        query_vector = EmbeddingModel.get_embedding(query)

        # 第2步：批量计算余弦相似度
        scores = np.array([
            EmbeddingModel.cosine_similarity(query_vector, v)
            for v in self.vectors
        ])

        # 第3步：argsort 返回从小到大的索引，取后k个并反转 → 即 Top-K
        top_indices = scores.argsort()[-k:][::-1]
        return np.array(self.document)[top_indices].tolist()
