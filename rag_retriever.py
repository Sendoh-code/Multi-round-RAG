import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


class FaissRetriever:
    def __init__(
        self,
        index_path="faiss_index.bin",
        metadata_path="passages.jsonl",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        rerank_model_name="BAAI/bge-reranker-v2-m3",
        top_k=5,
        rerank_factor=5   # FAISS 先召回 top_k * rerank_factor
    ):
        # 1. 加载 FAISS index（粗召回）
        self.index = faiss.read_index(index_path)

        # 2. 加载文档元数据
        self.metadata = [json.loads(line) for line in open(metadata_path)]

        # 3. 加载 embedding 模型（用于召回）
        self.embed_model = SentenceTransformer(embed_model_name)

        # 4. 加载 Cross-Encoder Reranker（用于精重排）
        #    bge-reranker 是目前最强的开源 reranker
        self.reranker = FlagReranker(
            rerank_model_name,
            use_fp16=False,   # Mac MPS 不支持 FP16
            use_gpu=False      # 强制单进程 CPU 计算 → 绝对不会泄漏 semaphore
        )

        self.top_k = top_k
        self.rerank_factor = rerank_factor

    def retrieve(self, query_text):
        # =========================
        # Step 1: embed query
        # =========================
        query_emb = self.embed_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        # =========================
        # Step 2: FAISS coarse search
        # =========================
        search_k = self.top_k * self.rerank_factor
        D, I = self.index.search(query_emb, search_k)

        # 收集 top_k*r 文档
        candidate_docs = [self.metadata[idx]["text"] for idx in I[0]]

        # =========================
        # Step 3: Rerank using BGE cross-encoder
        # =========================
        # BGE reranker 输入为：[(query, doc1), (query, doc2), ...]
        pairs = [(query_text, doc) for doc in candidate_docs]

        scores = self.reranker.compute_score(pairs)

        # 根据得分排序
        ranked = sorted(
            zip(candidate_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # =========================
        # Step 4: 返回最终 top_k 文档
        # =========================
        reranked_docs = [doc for doc, score in ranked[:self.top_k]]

        return reranked_docs
