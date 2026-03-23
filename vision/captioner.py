"""
vision/captioner.py

Generates rich text descriptions for visual document regions
using a Vision-Language Model served locally via Ollama (Qwen-VL).

This is the core differentiator: every chart, figure, and table
gets a detailed description stored alongside text chunks.
"""
import os
import base64
import io
from typing import Optional

import httpx
import numpy as np
from PIL import Image
from loguru import logger


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen2.5vl")

CAPTION_PROMPT = """You are a precise technical document analyst.
Describe this image extracted from a document in detail.
Include: what type of visual it is (chart/table/diagram/photo),
key data points or labels visible, trends or patterns shown,
and any text visible in the image.
Be factual and detailed. Max 200 words."""


class VLMCaptioner:
    """
    Generates image captions using Ollama's vision-language model.
    Falls back to a placeholder if Ollama is unreachable.
    """

    def __init__(self):
        self._client = httpx.Client(base_url=OLLAMA_URL, timeout=60.0)

    def caption(self, image: np.ndarray) -> str:
        """
        Caption a single image crop.
        image: RGB numpy array
        """
        try:
            b64 = _array_to_base64(image)
            payload = {
                "model": VLM_MODEL,
                "prompt": CAPTION_PROMPT,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            }
            resp = self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

        except httpx.ConnectError:
            logger.warning("Ollama not reachable — using placeholder caption")
            return "[visual region — start Ollama for AI captions]"
        except Exception as e:
            logger.error(f"VLM captioning failed: {e}")
            return "[captioning failed]"

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass


def _array_to_base64(image: np.ndarray) -> str:
    """Convert numpy RGB array to base64 PNG string."""
    pil = Image.fromarray(image.astype("uint8"))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
