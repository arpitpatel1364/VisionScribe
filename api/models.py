"""
Models: Document, Chunk, QueryLog
Tracks every ingested document, its chunks, and all queries for eval.
"""
from django.db import models
from django.contrib.auth.models import User
import uuid


class Document(models.Model):
    """Represents an ingested PDF or image file."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=512)
    file_path = models.CharField(max_length=1024)
    file_hash = models.CharField(max_length=64, unique=True)  # SHA-256
    total_pages = models.IntegerField(default=0)
    total_chunks = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    ingested_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    meta = models.JSONField(default=dict)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.filename} ({self.status})"


class Chunk(models.Model):
    """A single chunk of content extracted from a document."""

    CHUNK_TYPES = [
        ("text", "Text"),
        ("image", "Image"),
        ("table", "Table"),
        ("figure", "Figure"),
        ("mixed", "Mixed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_type = models.CharField(max_length=20, choices=CHUNK_TYPES, default="text")
    page_number = models.IntegerField()
    sequence = models.IntegerField()  # order within page

    # Raw content
    text_content = models.TextField(blank=True)
    image_description = models.TextField(blank=True)  # VLM-generated caption
    image_path = models.CharField(max_length=1024, blank=True)

    # Bounding box on original page (normalized 0-1)
    bbox_x0 = models.FloatField(default=0)
    bbox_y0 = models.FloatField(default=0)
    bbox_x1 = models.FloatField(default=0)
    bbox_y1 = models.FloatField(default=0)

    # ChromaDB reference
    chroma_id = models.CharField(max_length=128, blank=True)

    # YOLO detection class for visual chunks
    yolo_class = models.CharField(max_length=64, blank=True)
    yolo_confidence = models.FloatField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    meta = models.JSONField(default=dict)

    class Meta:
        ordering = ["document", "page_number", "sequence"]
        indexes = [
            models.Index(fields=["document", "page_number"]),
            models.Index(fields=["chunk_type"]),
        ]

    def __str__(self):
        return f"Chunk {self.sequence} / page {self.page_number} ({self.chunk_type})"

    @property
    def combined_content(self) -> str:
        """Returns the best text representation of this chunk."""
        if self.chunk_type == "text":
            return self.text_content
        if self.image_description:
            return f"[{self.chunk_type.upper()}] {self.image_description}"
        return self.text_content


class QueryLog(models.Model):
    """Logs every query for analytics and RAGAS evaluation."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    query_text = models.TextField()
    query_image_path = models.CharField(max_length=1024, blank=True)
    answer = models.TextField(blank=True)
    retrieved_chunk_ids = models.JSONField(default=list)
    retrieval_latency_ms = models.FloatField(default=0)
    generation_latency_ms = models.FloatField(default=0)
    total_latency_ms = models.FloatField(default=0)

    # RAGAS scores (populated async after eval)
    faithfulness_score = models.FloatField(null=True)
    context_recall_score = models.FloatField(null=True)
    answer_relevancy_score = models.FloatField(null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
