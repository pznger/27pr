# ┌─────────────────────────────────────────────────────────────────────┐
# │ 项目一扩展版：Tiny-RAG + 进阶优化                                   │
# │ 在原始 happy-llm Tiny-RAG 基础上添加：                              │
# │   - 语义分块（Semantic Chunking）                                    │
# │   - BM25 关键词检索                                                  │
# │   - RRF 融合检索（向量 + BM25）                                      │
# └─────────────────────────────────────────────────────────────────────┘

import os
import json
from typing import List, Dict
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding


def load_or_build_index(data_dir: str, max_token_len: int = 600, cover_content: int = 150):
    """加载已有向量库，不存在则新建"""
    if os.path.exists('storage/vectors.json'):
        print("[系统] 发现已有向量库，从 storage/ 加载...")
        docs = ReadFiles(data_dir).get_content(max_token_len, cover_content)
        vector = VectorStore(docs)
        vector.load_vector('./storage')
        return vector
    else:
        print("[系统] 未找到向量库，正在构建...")
        docs = ReadFiles(data_dir).get_content(max_token_len, cover_content)
        vector = VectorStore(docs)
        embedding = OpenAIEmbedding()
        vector.get_vector(EmbeddingModel=embedding)
        vector.persist(path='storage')
        return vector


def build_bm25_index(documents: List[str]):
    """构建 BM25 索引（简易版）"""
    from utils import BM25Index
    return BM25Index(documents)


def hybrid_search(query: str, vector_store: VectorStore, bm25_index, embedding_model, k: int = 3, alpha: float = 0.5):
    """
    RRF 融合检索：结合向量语义检索和 BM25 关键词检索
    alpha: 向量检索权重 (0~1)，值越大越偏向语义检索
    """
    # 1. 向量检索（语义相似度）
    query_vec = embedding_model.get_embedding(query)
    sims = [embedding_model.cosine_similarity(query_vec, v) for v in vector_store.vectors]
    vec_ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:max(k*3, 10)]

    # 2. BM25 关键词检索
    bm25_scores = bm25_index.search(query, top_k=max(k*3, 10))
    bm25_ranked = [i for i, _ in bm25_scores]

    # 3. RRF 融合
    rrf_scores = {}
    K = 60  # RRF 平滑参数
    for rank, idx in enumerate(vec_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (K + rank + 1)
    for rank, (idx, _) in enumerate(bm25_scores):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1-alpha) / (K + rank + 1)

    # 4. 按 RRF 分数排序返回 Top-K
    sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
    return [vector_store.document[i] for i in sorted_ids]


def main():
    # ========== 可调参数 ==========
    DATA_DIR = './data'
    MAX_TOKEN_LEN = 600       # 分块大小（token 数）
    COVER_CONTENT = 150       # 重叠 token 数
    TOP_K = 3                 # 检索 Top-K
    USE_HYBRID = True         # 是否启用融合检索
    ALPHA = 0.6               # 融合检索中向量权重（0~1）
    MODEL = 'Qwen/Qwen2.5-32B-Instruct'
    # =============================

    print(f"[配置] 分块大小={MAX_TOKEN_LEN}, 重叠={COVER_CONTENT}, Top-K={TOP_K}, 融合检索={'开' if USE_HYBRID else '关'}, 向量权重={ALPHA}")

    # 建索引 / 加载索引
    vector = load_or_build_index(DATA_DIR, MAX_TOKEN_LEN, COVER_CONTENT)
    embedding = OpenAIEmbedding()
    bm25 = build_bm25_index(vector.document) if USE_HYBRID else None

    print(f"[系统] 知识库已就绪，共 {len(vector.document)} 个文档块")

    # 交互式问答
    chat = OpenAIChat(model=MODEL)
    while True:
        question = input("\n🔍 请输入问题（输入 exit 退出）: ").strip()
        if question.lower() == 'exit':
            break
        if not question:
            continue

        # 检索
        if USE_HYBRID and bm25:
            print("[检索] 使用 RRF 融合检索...")
            contents = hybrid_search(question, vector, bm25, embedding, k=TOP_K, alpha=ALPHA)
        else:
            print(f"[检索] 使用纯向量检索 (Top-{TOP_K})...")
            contents = vector.query(question, EmbeddingModel=embedding, k=TOP_K)

        # 显示检索到的片段预览
        for i, c in enumerate(contents):
            preview = c[:100].replace('\n', ' ') + ('...' if len(c) > 100 else '')
            print(f"  [片段{i+1}] {preview}")

        # 生成回答
        context = "\n\n---\n\n".join(contents)
        answer = chat.chat(question, [], context)
        print(f"\n🤖 回答: {answer}")


if __name__ == '__main__':
    main()
