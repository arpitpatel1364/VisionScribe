"""
ingestion/chunker.py

Splits page text blocks into overlapping chunks of ~512 tokens.
Preserves block-level bbox metadata for source attribution.
"""
import os
from typing import List, Dict, Any


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_page_content(
    text_blocks: List[Dict[str, Any]],
    page_number: int,
    doc_id: str,
) -> List[Dict[str, Any]]:
    """
    Merges text blocks from a page and splits into overlapping chunks.
    Each chunk dict is ready to be saved as a Chunk model instance.
    """
    # Concatenate blocks preserving order
    full_text = "\n".join(b["text"] for b in text_blocks)
    words = full_text.split()

    if not words:
        return []

    # Use the bounding box of the first block as representative
    bbox = text_blocks[0]["bbox"] if text_blocks else [0, 0, 1, 1]

    chunks = []
    start = 0
    seq = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "doc_id": doc_id,
            "page_number": page_number,
            "sequence": seq,
            "chunk_type": "text",
            "text_content": chunk_text,
            "image_description": "",
            "image_path": "",
            "bbox": bbox,
            "yolo_class": "",
            "yolo_confidence": 0.0,
        })
        seq += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP  # slide with overlap

    return chunks
