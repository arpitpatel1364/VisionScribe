"""
retrieval/hybrid_retriever.py

Combines:
  - Dense retrieval  (ChromaDB cosine similarity, BGE-M3 embeddings)
  - Sparse retrieval (BM25 over stored chunk text)
  - Reciprocal Rank Fusion (RRF) to merge results

This consistently outperforms either approach alone,
especially for technical queries with specific terminology.
"""
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")
django.setup()

from loguru import logger
from ingestion.embedder import embed_text, _get_chroma
from api.models import Chunk


CHROMA_TEXT_COL = os.getenv("CHROMA_COLLECTION_TEXT", "vision_rag_text")
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", 10))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", 10))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 5))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.6))  # 0=BM25 only, 1=dense only
RRF_K = 60  # standard RRF constant


def _reciprocal_rank_fusion(
    dense_ids: List[str], sparse_ids: List[str], alpha: float = HYBRID_ALPHA
) -> List[str]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    alpha controls weight: 1.0 = pure dense, 0.0 = pure sparse.
    """
    scores: Dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + alpha * (1.0 / (RRF_K + rank + 1))

    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1.0 / (RRF_K + rank + 1))

    return sorted(scores, key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    """Main retriever used by the Django query endpoint."""

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        include_images: bool = True,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run hybrid retrieval for a text query.

        Returns a list of chunk dicts with:
          chunk_id, text, score, page, doc_id, chunk_type
        """
        # ── Dense retrieval ──────────────────────────────
        query_emb = embed_text([query])[0]
        client = _get_chroma()
        col = client.get_or_create_collection(CHROMA_TEXT_COL)

        where = None
        if doc_filter:
            where = {"doc_id": {"$in": doc_filter}}

        dense_results = col.query(
            query_embeddings=[query_emb],
            n_results=TOP_K_DENSE,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = dense_results["ids"][0] if dense_results["ids"] else []
        dense_docs = dense_results["documents"][0] if dense_results["documents"] else []
        dense_metas = dense_results["metadatas"][0] if dense_results["metadatas"] else []

        # ── Sparse retrieval (BM25) ───────────────────────
        # Build corpus from PostgreSQL chunks (scoped to doc_filter if set)
        qs = Chunk.objects.all()
        if doc_filter:
            qs = qs.filter(document_id__in=doc_filter)
        if not include_images:
            qs = qs.filter(chunk_type="text")

        pg_chunks = list(qs.values("id", "text_content", "image_description", "chunk_type"))
        corpus_texts = [
            (c["text_content"] or c["image_description"] or "").lower().split()
            for c in pg_chunks
        ]
        corpus_ids = [str(c["id"]) for c in pg_chunks]

        sparse_ids: List[str] = []
        if corpus_texts:
            bm25 = BM25Okapi(corpus_texts)
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            top_sparse = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:TOP_K_SPARSE]
            sparse_ids = [corpus_ids[i] for i in top_sparse]

        # ── RRF merge ─────────────────────────────────────
        merged_ids = _reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]

        # ── Build result list ─────────────────────────────
        # Map chroma_id → meta for dense results
        chroma_meta = {cid: meta for cid, meta in zip(dense_ids, dense_metas)}
        chroma_docs = {cid: doc for cid, doc in zip(dense_ids, dense_docs)}

        results = []
        for cid in merged_ids:
            meta = chroma_meta.get(cid, {})
            results.append({
                "chunk_id": cid,
                "text": chroma_docs.get(cid, ""),
                "page": meta.get("page", 0),
                "doc_id": meta.get("doc_id", ""),
                "chunk_type": meta.get("type", "text"),
                "image_path": meta.get("image_path", ""),
            })

        logger.info(
            f"[Retriever] query='{query[:60]}' → "
            f"{len(dense_ids)} dense + {len(sparse_ids)} sparse → {len(results)} merged"
        )
        return results
