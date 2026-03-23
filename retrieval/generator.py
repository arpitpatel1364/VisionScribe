"""
retrieval/generator.py

Builds a prompt from retrieved chunks and calls the local LLM
via Ollama to generate a grounded answer with source attribution.
"""
import os
from typing import List, Dict, Any

import httpx
from loguru import logger


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen2.5vl")

SYSTEM_PROMPT = """You are a precise document QA assistant.
Answer questions using ONLY the provided context chunks.
If the answer is not in the context, say "I cannot find this in the provided documents."
Always cite which page and chunk type your answer comes from.
Be concise but complete."""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        chunk_type = chunk.get("chunk_type", "text")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        lines.append(f"[Source {i} | Page {page} | Type: {chunk_type}]\n{text}\n")
    return "\n---\n".join(lines)


def _has_visual_chunks(chunks: List[Dict[str, Any]]) -> bool:
    return any(c.get("chunk_type") not in ("text", "") for c in chunks)


class RAGGenerator:
    """Generates answers using the local Ollama LLM."""

    def __init__(self):
        self._client = httpx.Client(base_url=OLLAMA_URL, timeout=120.0)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate an answer for `query` grounded in `chunks`.
        Routes to VLM if any visual chunks are present.
        """
        if not chunks:
            return "No relevant context found in the indexed documents."

        context = _build_context(chunks)
        user_message = f"""Context:\n{context}\n\nQuestion: {query}"""

        # Use VLM if visual chunks present; plain LLM otherwise
        model = VLM_MODEL if _has_visual_chunks(chunks) else LLM_MODEL

        try:
            payload = {
                "model": model,
                "system": SYSTEM_PROMPT,
                "prompt": user_message,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                    "top_p": 0.9,
                },
            }
            resp = self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
            logger.info(f"[Generator] model={model} | {len(answer)} chars generated")
            return answer

        except httpx.ConnectError:
            return (
                "[Ollama not running] Start Ollama with: "
                f"`ollama run {model}` then retry your query."
            )
        except Exception as e:
            logger.error(f"[Generator] Error: {e}")
            return f"Generation failed: {str(e)}"

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
