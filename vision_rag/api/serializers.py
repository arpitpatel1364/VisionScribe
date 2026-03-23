from rest_framework import serializers
from .models import Document, Chunk, QueryLog


class ChunkSerializer(serializers.ModelSerializer):
    combined_content = serializers.ReadOnlyField()

    class Meta:
        model = Chunk
        fields = [
            "id", "chunk_type", "page_number", "sequence",
            "text_content", "image_description", "image_path",
            "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
            "yolo_class", "yolo_confidence", "combined_content",
        ]


class DocumentSerializer(serializers.ModelSerializer):
    chunks = ChunkSerializer(many=True, read_only=True)

    class Meta:
        model = Document
        fields = [
            "id", "filename", "file_hash", "total_pages",
            "total_chunks", "status", "created_at", "meta", "chunks",
        ]
        read_only_fields = ["id", "file_hash", "total_pages", "total_chunks",
                            "status", "created_at"]


class DocumentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer (no chunks) for list views."""
    class Meta:
        model = Document
        fields = ["id", "filename", "total_pages", "total_chunks", "status", "created_at"]


class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(max_length=2048)
    top_k = serializers.IntegerField(default=5, min_value=1, max_value=20)
    include_images = serializers.BooleanField(default=True)
    document_ids = serializers.ListField(
        child=serializers.UUIDField(), required=False, allow_empty=True
    )


class QueryResponseSerializer(serializers.ModelSerializer):
    sources = ChunkSerializer(many=True)

    class Meta:
        model = QueryLog
        fields = [
            "id", "query_text", "answer", "sources",
            "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
            "created_at",
        ]


class QueryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryLog
        fields = "__all__"
