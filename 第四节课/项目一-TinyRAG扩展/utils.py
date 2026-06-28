#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 原始文件来源：Datawhale happy-llm 项目
# 扩展内容：添加了 BM25Index 类（关键词检索）+ semantic_chunk() 函数（语义分块）

import os
import re
import PyPDF2
import markdown
import json
import math
from typing import List, Dict, Tuple
from collections import Counter
from bs4 import BeautifulSoup
import tiktoken
from tqdm import tqdm

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """读取 data/ 目录下的 .md / .txt / .pdf 文件，并分词块"""

    def __init__(self, path: str):
        self._path = path
        self.file_list = self.get_files()

    def get_files(self) -> List[str]:
        """递归遍历目录，收集 .md / .txt / .pdf 文件"""
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.endswith(('.md', '.txt', '.pdf')):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """读取所有文件并分块"""
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunks = self.get_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunks)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """按 token 长度分块，相邻块之间有 cover_content 个 token 重叠"""
        chunk_text = []
        curr_len = 0
        curr_chunk = ''
        token_len = max_token_len - cover_content
        lines = text.splitlines()

        for line in lines:
            line = line.strip()
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                # 超长行按 token 切分
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                curr_chunk, curr_len = '', 0
                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = min(start + token_len, len(line_tokens))
                    chunk_part = enc.decode(line_tokens[start:end])
                    if i > 0 and chunk_text:
                        prev = chunk_text[-1]
                        cover = prev[-cover_content:] if len(prev) > cover_content else prev
                        chunk_part = cover + chunk_part
                    chunk_text.append(chunk_part)
            elif curr_len + line_len + 1 <= token_len:
                # 加入当前块
                if curr_chunk:
                    curr_chunk += '\n'; curr_len += 1
                curr_chunk += line; curr_len += line_len
            else:
                # 开始新块，添加重叠
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                if chunk_text:
                    prev = chunk_text[-1]
                    cover = prev[-cover_content:] if len(prev) > cover_content else prev
                    curr_chunk = cover + '\n' + line
                    curr_len = len(enc.encode(cover)) + 1 + line_len
                else:
                    curr_chunk = line; curr_len = line_len

        if curr_chunk:
            chunk_text.append(curr_chunk)
        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str) -> str:
        """根据后缀选择读取方式"""
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")

    @classmethod
    def read_pdf(cls, file_path: str) -> str:
        """读取 PDF 文件内容"""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join(page.extract_text() for page in reader.pages)
            return text

    @classmethod
    def read_markdown(cls, file_path: str) -> str:
        """读取 Markdown 并提取纯文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html = markdown.markdown(f.read())
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            return re.sub(r'http\S+', '', text)

    @classmethod
    def read_text(cls, file_path: str) -> str:
        """读取纯文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


# ========== 扩展一：语义分块 ==========

def semantic_chunk(text: str, embedding_model, threshold: float = 0.5) -> List[str]:
    """
    语义分块：在相邻句子语义相似度低于阈值的位置切分
    
    参数:
        text: 待分块的原始文本
        embedding_model: 嵌入模型实例（需有 get_embedding 方法）
        threshold: 余弦相似度阈值。低于此值表示语义断点，在此切分
    返回:
        分块后的文本列表
    """
    # 步骤1：按句号/换行/问号/感叹号将文本分割为句子序列
    sentences = re.split(r'(?<=[。！？\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # 过滤空句子

    if len(sentences) <= 1:
        return [text]

    # 步骤2：对每个句子作向量编码
    embeddings = [embedding_model.get_embedding(s) for s in sentences]

    # 步骤3：计算相邻句子的余弦相似度
    import numpy as np

    def cosine(v1, v2):
        v1, v2 = np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0

    sims = [cosine(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

    # 步骤4：在语义断点处切分
    chunks = []
    current = sentences[0]
    for i, sim in enumerate(sims):
        if sim < threshold:
            # 相似度低于阈值 → 语义断点
            chunks.append(current)
            current = sentences[i+1]
        else:
            # 相似度高 → 合并在当前块
            current += '。' + sentences[i+1]

    if current:
        chunks.append(current)
    return chunks


# ========== 扩展二：BM25 关键词检索 ==========

class BM25Index:
    """
    简易 BM25 检索引擎
    
    BM25 公式：
        score(D,Q) = Σ IDF(qi) * (f(qi,D)*(k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
    
    其中：
        IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        参数:
            documents: 文档列表
            k1: 词频饱和参数（1.2~2.0），控制词频的影响上限
            b: 文档长度归一化参数（0~1），控制长度惩罚力度
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)                                    # 文档总数

        # 对每个文档分词并统计词频
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens] # 各文档长度
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 1  # 平均文档长度

        # 计算 IDF（逆文档频率）
        self.idf = {}
        self._compute_idf()

    def _tokenize(self, text: str) -> List[str]:
        """简易分词：按非字母数字字符分割，转小写，过滤长度<2的词"""
        tokens = re.findall(r'[a-zA-Z\u4e00-\u9fff0-9]+', text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _compute_idf(self):
        """计算每个词的 IDF 值"""
        df = Counter()  # 文档频率
        for tokens in self.doc_tokens:
            df.update(set(tokens))  # 每个文档中每个词只计1次

        for term, freq in df.items():
            # IDF = log((N - n + 0.5) / (n + 0.5))
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1e-10)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        搜索与查询最相关的 top_k 个文档
        返回: [(文档索引, BM25分数), ...]
        """
        query_tokens = self._tokenize(query)
        scores = []

        for idx, tokens in enumerate(self.doc_tokens):
            score = 0.0
            tf = Counter(tokens)
            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                f = tf.get(qt, 0)  # 词频
                if f == 0:
                    continue
                # BM25 核心公式
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score += self.idf[qt] * numerator / denominator
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
