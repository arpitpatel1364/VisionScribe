"""
ingestion/parser.py

Uses PyMuPDF (fitz) to extract text blocks and a page image
from a single PDF page. Designed to be called from worker processes.
"""
import io
from pathlib import Path
from typing import Dict, List, Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


def parse_page(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """
    Extract text blocks and a rendered image from a single PDF page.

    Returns:
        {
            "page_number": int,
            "text_blocks": [{"text": str, "bbox": [x0,y0,x1,y1]}],
            "page_image": np.ndarray  (RGB, 150 DPI)
            "page_width": float,
            "page_height": float,
        }
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    page_width = page.rect.width
    page_height = page.rect.height

    # ── Text extraction ──────────────────────────────────
    blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
    text_blocks = []
    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        if block_type == 0 and text.strip():  # type 0 = text
            text_blocks.append({
                "text": text.strip(),
                "bbox": [
                    x0 / page_width,   # normalize to [0,1]
                    y0 / page_height,
                    x1 / page_width,
                    y1 / page_height,
                ],
                "block_no": block_no,
            })

    # ── Page image render (for YOLO) ──────────────────────
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    page_image = np.array(pil_img)

    doc.close()

    return {
        "page_number": page_num,
        "text_blocks": text_blocks,
        "page_image": page_image,
        "page_width": page_width,
        "page_height": page_height,
    }
