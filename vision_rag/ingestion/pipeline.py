"""
ingestion/pipeline.py

Orchestrates the full ingestion flow for a single document:
  1. Parse PDF pages with PyMuPDF (multiprocessing across pages)
  2. Detect visual regions with YOLO
  3. Caption visual regions with Vision-LLM (Ollama)
  4. Chunk & embed text with BGE-M3
  5. Embed images with CLIP
  6. Upsert all chunks into ChromaDB + PostgreSQL
"""
import os
import sys
import time
import django
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any

# Bootstrap Django ORM (safe to call from threads/processes)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")
django.setup()

from loguru import logger
from tqdm import tqdm

from ingestion.parser import parse_page
from ingestion.chunker import chunk_page_content
from vision.detector import YOLODetector
from vision.captioner import VLMCaptioner
from ingestion.embedder import embed_and_store
from api.models import Document, Chunk


def _process_page(args: tuple) -> List[Dict[str, Any]]:
    """
    Worker function — runs in a separate process per page.
    Parses one page, runs YOLO, captions visuals, chunks text.
    Returns a list of chunk dicts ready for embedding.
    """
    doc_path, page_num, doc_id = args
    try:
        # 1. Parse page → raw blocks
        page_data = parse_page(doc_path, page_num)

        # 2. YOLO — detect figures/tables/charts on page image
        detector = YOLODetector()
        visual_regions = detector.detect(page_data["page_image"], page_num)

        # 3. Caption each visual region
        captioner = VLMCaptioner()
        for region in visual_regions:
            region["caption"] = captioner.caption(region["crop"])

        # 4. Chunk text blocks
        text_chunks = chunk_page_content(page_data["text_blocks"], page_num, doc_id)

        # 5. Build visual chunk dicts
        visual_chunks = []
        for i, region in enumerate(visual_regions):
            visual_chunks.append({
                "doc_id": doc_id,
                "page_number": page_num,
                "sequence": len(text_chunks) + i,
                "chunk_type": region.get("class_name", "figure"),
                "text_content": "",
                "image_description": region["caption"],
                "image_path": region.get("crop_path", ""),
                "bbox": region["bbox"],
                "yolo_class": region.get("class_name", ""),
                "yolo_confidence": region.get("confidence", 0.0),
            })

        return text_chunks + visual_chunks

    except Exception as exc:
        logger.error(f"Page {page_num} failed: {exc}")
        return []


def run_ingestion_pipeline(doc_id: str, file_path: str) -> None:
    """
    Main entry point. Called from Django view in a background thread.
    Updates Document status throughout.
    """
    doc = Document.objects.get(id=doc_id)
    doc.status = "processing"
    doc.save()

    t_start = time.perf_counter()
    logger.info(f"[Ingestion] Starting: {file_path}")

    try:
        # Count pages
        import fitz  # PyMuPDF
        pdf = fitz.open(file_path)
        total_pages = len(pdf)
        pdf.close()
        doc.total_pages = total_pages
        doc.save()

        # Parallel page processing
        n_workers = min(int(os.getenv("INGESTION_WORKERS", 4)), cpu_count())
        args = [(file_path, p, doc_id) for p in range(total_pages)]

        logger.info(f"[Ingestion] Processing {total_pages} pages with {n_workers} workers")

        all_chunks: List[Dict[str, Any]] = []
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_page, args),
                total=total_pages,
                desc=f"Ingesting {Path(file_path).name}",
            ))
        for page_chunks in results:
            all_chunks.extend(page_chunks)

        logger.info(f"[Ingestion] Extracted {len(all_chunks)} chunks total")

        # Embed + store in ChromaDB and PostgreSQL
        embed_and_store(all_chunks)

        elapsed = time.perf_counter() - t_start
        doc.status = "done"
        doc.total_chunks = len(all_chunks)
        doc.meta = {
            "ingestion_time_s": round(elapsed, 2),
            "pages": total_pages,
            "chunks": len(all_chunks),
        }
        doc.save()
        logger.success(f"[Ingestion] Done in {elapsed:.1f}s — {len(all_chunks)} chunks stored")

    except Exception as exc:
        logger.exception(f"[Ingestion] Failed for {file_path}: {exc}")
        doc.status = "failed"
        doc.meta = {"error": str(exc)}
        doc.save()
