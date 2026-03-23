"""
Django REST API views.

Endpoints:
  POST /api/ingest/          - Upload & ingest a PDF
  GET  /api/documents/       - List documents
  GET  /api/documents/<id>/  - Document detail + chunks
  POST /api/query/           - Run a RAG query
  GET  /api/stats/           - Pipeline stats
  GET  /api/logs/            - Query history
"""
import hashlib
import time
import os
import asyncio
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, Chunk, QueryLog
from .serializers import (
    DocumentSerializer, DocumentListSerializer,
    QuerySerializer, QueryResponseSerializer, QueryLogSerializer,
)


def _file_hash(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


class IngestView(APIView):
    """
    POST /api/ingest/
    Upload a PDF or image. Triggers background ingestion pipeline.
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"error": "No file provided."}, status=400)

        allowed_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        ext = Path(file_obj.name).suffix.lower()
        if ext not in allowed_ext:
            return Response({"error": f"Unsupported file type: {ext}"}, status=400)

        # Save file
        save_dir = Path(settings.MEDIA_ROOT)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_obj.name

        with open(save_path, "wb") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # Dedup by hash
        file_hash = _file_hash(str(save_path))
        if Document.objects.filter(file_hash=file_hash).exists():
            doc = Document.objects.get(file_hash=file_hash)
            return Response({
                "detail": "Document already ingested.",
                "document_id": str(doc.id),
                "status": doc.status,
            }, status=200)

        doc = Document.objects.create(
            filename=file_obj.name,
            file_path=str(save_path),
            file_hash=file_hash,
            status="pending",
            ingested_by=request.user,
        )

        # Kick off ingestion in a background thread (use Celery in production)
        import threading
        from ingestion.pipeline import run_ingestion_pipeline
        thread = threading.Thread(
            target=run_ingestion_pipeline,
            args=(str(doc.id), str(save_path)),
            daemon=True,
        )
        thread.start()

        return Response({
            "document_id": str(doc.id),
            "filename": doc.filename,
            "status": "pending",
            "message": "Ingestion started. Poll /api/documents/<id>/ for status.",
        }, status=202)


class DocumentListView(generics.ListAPIView):
    """GET /api/documents/ — paginated list of all documents."""
    serializer_class = DocumentListSerializer
    queryset = Document.objects.all()


class DocumentDetailView(generics.RetrieveAPIView):
    """GET /api/documents/<id>/ — full document with chunks."""
    serializer_class = DocumentSerializer
    queryset = Document.objects.prefetch_related("chunks")
    lookup_field = "id"


class QueryView(APIView):
    """
    POST /api/query/
    Body: { "query": str, "top_k": int, "include_images": bool }
    Returns: answer + source chunks with latency metrics.
    """

    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        data = serializer.validated_data
        query_text = data["query"]
        top_k = data["top_k"]
        include_images = data["include_images"]
        doc_filter = data.get("document_ids", [])

        t_start = time.perf_counter()

        # ── Retrieval ──────────────────────────────────────
        from retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        t_ret_start = time.perf_counter()
        retrieved_chunks = retriever.retrieve(
            query=query_text,
            top_k=top_k,
            include_images=include_images,
            doc_filter=[str(d) for d in doc_filter],
        )
        t_ret_end = time.perf_counter()
        retrieval_ms = (t_ret_end - t_ret_start) * 1000

        # ── Generation ─────────────────────────────────────
        from retrieval.generator import RAGGenerator
        generator = RAGGenerator()
        t_gen_start = time.perf_counter()
        answer = generator.generate(query_text, retrieved_chunks)
        t_gen_end = time.perf_counter()
        generation_ms = (t_gen_end - t_gen_start) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        # ── Log ────────────────────────────────────────────
        chunk_ids = [c["chunk_id"] for c in retrieved_chunks if "chunk_id" in c]
        log = QueryLog.objects.create(
            user=request.user,
            query_text=query_text,
            answer=answer,
            retrieved_chunk_ids=chunk_ids,
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            total_latency_ms=round(total_ms, 2),
        )

        # Fetch chunk objects for source attribution
        db_chunks = Chunk.objects.filter(id__in=chunk_ids)

        from .serializers import ChunkSerializer
        return Response({
            "query_id": str(log.id),
            "answer": answer,
            "sources": ChunkSerializer(db_chunks, many=True).data,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        })


@api_view(["GET"])
def stats_view(request):
    """GET /api/stats/ — pipeline-level metrics."""
    total_docs = Document.objects.count()
    done_docs = Document.objects.filter(status="done").count()
    total_chunks = Chunk.objects.count()
    text_chunks = Chunk.objects.filter(chunk_type="text").count()
    image_chunks = Chunk.objects.filter(chunk_type__in=["image", "figure", "table"]).count()
    total_queries = QueryLog.objects.count()

    avg_latency = QueryLog.objects.filter(
        total_latency_ms__gt=0
    ).values_list("total_latency_ms", flat=True)
    avg_ms = sum(avg_latency) / len(avg_latency) if avg_latency else 0

    return Response({
        "documents": {"total": total_docs, "done": done_docs},
        "chunks": {"total": total_chunks, "text": text_chunks, "visual": image_chunks},
        "queries": {"total": total_queries, "avg_latency_ms": round(avg_ms, 2)},
    })


class QueryLogListView(generics.ListAPIView):
    """GET /api/logs/ — query history for current user."""
    serializer_class = QueryLogSerializer

    def get_queryset(self):
        return QueryLog.objects.filter(user=self.request.user)
