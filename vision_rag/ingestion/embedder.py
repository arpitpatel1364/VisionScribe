"""
ingestion/embedder.py

Embeds chunks and stores them in:
  - ChromaDB   (vector search)
  - PostgreSQL (metadata + source attribution via Django ORM)

Text  → BGE-M3 embeddings
Image → CLIP   embeddings (stored in a separate Chroma collection)
"""
import os
import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")
django.setup()

from loguru import logger
from sentence_transformers import SentenceTransformer
import open_clip
import torch
from PIL import Image

from api.models import Chunk, Document


# ── Singletons (loaded once per process) ─────────────────────
_text_model: SentenceTransformer | None = None
_clip_model = None
_clip_preprocess = None
_chroma_client: chromadb.ClientAPI | None = None

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "BAAI/bge-m3")
IMAGE_EMBED_MODEL = os.getenv("IMAGE_EMBED_MODEL", "openai/clip-vit-base-patch32")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_TEXT_COL = os.getenv("CHROMA_COLLECTION_TEXT", "vision_rag_text")
CHROMA_IMG_COL = os.getenv("CHROMA_COLLECTION_IMAGE", "vision_rag_image")


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        logger.info(f"Loading text embedding model: {TEXT_EMBED_MODEL}")
        _text_model = SentenceTransformer(TEXT_EMBED_MODEL)
    return _text_model


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        logger.info(f"Loading CLIP model: {IMAGE_EMBED_MODEL}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def embed_text(texts: List[str]) -> List[List[float]]:
    model = _get_text_model()
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True)
    return embeddings.tolist()


def embed_image(image_path: str) -> List[float]:
    model, preprocess = _get_clip()
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = model.encode_image(img)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].tolist()


def embed_and_store(chunks: List[Dict[str, Any]]) -> None:
    """
    Embeds all chunks and saves to ChromaDB + PostgreSQL.
    Separates text vs visual chunks automatically.
    """
    client = _get_chroma()
    text_col = client.get_or_create_collection(CHROMA_TEXT_COL)
    img_col = client.get_or_create_collection(CHROMA_IMG_COL)

    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    visual_chunks = [c for c in chunks if c["chunk_type"] != "text"]

    # ── Text chunks ──────────────────────────────────────────
    if text_chunks:
        texts = [c["text_content"] for c in text_chunks]
        embeddings = embed_text(texts)

        chroma_ids, docs, metas, embeds = [], [], [], []
        pg_chunks = []

        for chunk, emb in zip(text_chunks, embeddings):
            cid = str(uuid.uuid4())
            chroma_ids.append(cid)
            docs.append(chunk["text_content"])
            metas.append({
                "doc_id": chunk["doc_id"],
                "page": chunk["page_number"],
                "seq": chunk["sequence"],
                "type": "text",
            })
            embeds.append(emb)

            pg_chunks.append(Chunk(
                document_id=chunk["doc_id"],
                chunk_type="text",
                page_number=chunk["page_number"],
                sequence=chunk["sequence"],
                text_content=chunk["text_content"],
                bbox_x0=chunk["bbox"][0],
                bbox_y0=chunk["bbox"][1],
                bbox_x1=chunk["bbox"][2],
                bbox_y1=chunk["bbox"][3],
                chroma_id=cid,
            ))

        text_col.upsert(ids=chroma_ids, documents=docs, metadatas=metas, embeddings=embeds)
        Chunk.objects.bulk_create(pg_chunks, batch_size=200)
        logger.info(f"Stored {len(pg_chunks)} text chunks")

    # ── Visual chunks ─────────────────────────────────────────
    for chunk in visual_chunks:
        # Use image description as the searchable text + optionally CLIP
        desc = chunk["image_description"] or f"[{chunk['chunk_type']}]"
        text_emb = embed_text([desc])[0]

        cid = str(uuid.uuid4())
        text_col.upsert(
            ids=[cid],
            documents=[desc],
            metadatas=[{
                "doc_id": chunk["doc_id"],
                "page": chunk["page_number"],
                "seq": chunk["sequence"],
                "type": chunk["chunk_type"],
                "image_path": chunk.get("image_path", ""),
            }],
            embeddings=[text_emb],
        )

        # Also store CLIP embedding in image collection (if image exists)
        if chunk.get("image_path") and os.path.exists(chunk["image_path"]):
            try:
                img_emb = embed_image(chunk["image_path"])
                img_col.upsert(
                    ids=[cid],
                    documents=[desc],
                    metadatas=[{"doc_id": chunk["doc_id"], "page": chunk["page_number"]}],
                    embeddings=[img_emb],
                )
            except Exception as e:
                logger.warning(f"CLIP embed failed for {chunk['image_path']}: {e}")

        Chunk.objects.create(
            document_id=chunk["doc_id"],
            chunk_type=chunk["chunk_type"],
            page_number=chunk["page_number"],
            sequence=chunk["sequence"],
            text_content=chunk.get("text_content", ""),
            image_description=desc,
            image_path=chunk.get("image_path", ""),
            bbox_x0=chunk["bbox"][0],
            bbox_y0=chunk["bbox"][1],
            bbox_x1=chunk["bbox"][2],
            bbox_y1=chunk["bbox"][3],
            chroma_id=cid,
            yolo_class=chunk.get("yolo_class", ""),
            yolo_confidence=chunk.get("yolo_confidence", 0.0),
        )

    logger.info(f"Stored {len(visual_chunks)} visual chunks")
