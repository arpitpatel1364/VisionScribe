<div align="center">

<br/>

```
 ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗███████╗ ██████╗██████╗ ██╗██████╗ ███████╗
 ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗██║██╔══██╗██╔════╝
 ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║███████╗██║     ██████╔╝██║██████╔╝█████╗
 ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║╚════██║██║     ██╔══██╗██║██╔══██╗██╔══╝
  ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║███████║╚██████╗██║  ██║██║██████╔╝███████╗
   ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═════╝╚══════╝
```

**Ask questions over documents that contain charts, tables, diagrams, and text.**
*Most RAG systems only read. VisionScribe sees.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2-092E20?style=flat-square&logo=django&logoColor=white)](https://djangoproject.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ollama](https://img.shields.io/badge/Ollama-local_LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6B35?style=flat-square)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

</div>

---

## The problem with standard RAG

Every RAG tutorial indexes text. But real-world documents — research papers, financial reports, technical manuals — hide critical information inside **charts, tables, diagrams, and figures**. Standard pipelines go blind the moment a relevant answer lives inside a chart instead of a paragraph.

VisionScribe solves this by treating every page as both a text document and an image — detecting visual regions with YOLO, captioning them with a Vision-Language Model, and embedding everything into a unified hybrid search index.

<br/>

## How it works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE  (multiprocessing)                    │
│                                                                             │
│  PDF / Image                                                                │
│      │                                                                      │
│      ├──► PyMuPDF ──► Text blocks ──► BGE-M3 embedder ──────────────────┐  │
│      │    parser                                                         │  │
│      │                                                                   ▼  │
│      └──► YOLO ──────► Crops ────────► Qwen-VL ──► Caption ──► ChromaDB    │
│           detector      (figures,       captioner               +           │
│                          tables,                               PostgreSQL   │
│                          charts)                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE  (Django REST API)                      │
│                                                                             │
│  User query                                                                 │
│      │                                                                      │
│      ├──► Django API ──► BGE-M3 embed ──► ChromaDB dense retrieval ──┐     │
│      │    (auth + routing)                                            ▼     │
│      └──────────────────────────────► BM25 sparse retrieval ──► RRF merge  │
│                                                                       │     │
│                                                        top-k chunks ◄┘     │
│                                                               │             │
│                                                               ▼             │
│                                           Ollama LLM ──► Answer + sources  │
└─────────────────────────────────────────────────────────────────────────────┘
```

<br/>

## What sets it apart

| | Standard RAG | VisionScribe |
|---|:---:|:---:|
| Reads document text | ✅ | ✅ |
| Detects visual regions (YOLO) | ❌ | ✅ |
| Captions charts & figures (VLM) | ❌ | ✅ |
| Hybrid dense + sparse retrieval | ❌ | ✅ |
| Source image attribution | ❌ | ✅ |
| RAGAS evaluation metrics | ❌ | ✅ |
| Fully local — zero API cost | ❌ | ✅ |
| Parallel ingestion (multiprocessing) | ❌ | ✅ |

<br/>

## Stack

| Layer | Tool | Role |
|---|---|---|
| Document parsing | PyMuPDF | Fast, layout-aware page extraction |
| Visual detection | YOLO (Ultralytics) | Detects figures, tables, charts, formulas |
| Visual captioning | Qwen-VL via Ollama | Generates rich text descriptions of visual regions |
| Text embeddings | BGE-M3 (Hugging Face) | State-of-the-art multilingual dense embeddings |
| Image embeddings | CLIP (open_clip) | Shared text/image vector space |
| Vector store | ChromaDB | Persistent local vector search |
| Sparse retrieval | BM25 (rank-bm25) | Keyword-aware retrieval over stored chunks |
| Hybrid merge | Reciprocal Rank Fusion | Combines dense + sparse rankings |
| Local LLM | Llama 3 via Ollama | Private, no API cost, runs on your machine |
| Backend API | Django REST Framework | Token auth, routing, query logging |
| Concurrency | `multiprocessing.Pool` | Parallel page ingestion across CPU cores |
| Metadata store | PostgreSQL | Chunk metadata, query logs, RAGAS scores |
| Frontend | Streamlit | Query UI, source viewer, eval dashboard |
| Evaluation | RAGAS | Faithfulness, context recall, answer relevancy |

<br/>

## Quickstart

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- [Ollama](https://ollama.com) installed

### 1 — Clone & bootstrap

```bash
git clone https://github.com/arpitbhojani/visionscribe.git
cd visionscribe
bash scripts/setup.sh
```

`setup.sh` handles everything: virtual environment, pip install, Django migrations, and Ollama model pulls.

### 2 — Configure environment

```bash
cp .env.example .env
# Minimum required: SECRET_KEY and DB_PASSWORD
```

Key `.env` options:

```env
OLLAMA_LLM_MODEL=llama3        # main LLM for text queries
OLLAMA_VLM_MODEL=qwen2.5vl     # vision-language model for captioning
INGESTION_WORKERS=4            # set to your CPU core count
HYBRID_ALPHA=0.6               # 0.0 = pure BM25  →  1.0 = pure dense
CHUNK_SIZE=512                 # tokens per text chunk
```

### 3 — Get your API token

```bash
source venv/bin/activate
python manage.py createsuperuser
python manage.py drf_create_token <your-username>
# → Token 9a4f2c...  ← paste this into the Streamlit sidebar
```

### 4 — Start everything

```bash
# Terminal 1 — local LLM
ollama serve

# Terminal 2 — Django API
source venv/bin/activate
python manage.py runserver

# Terminal 3 — Streamlit UI
streamlit run streamlit_app/app.py
```

Open **[http://localhost:8501](http://localhost:8501)** → paste token → upload a PDF → ask anything.

<br/>

## Docker

```bash
ollama pull llama3 && ollama pull qwen2.5vl
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| Django API | http://localhost:8000 |
| PostgreSQL | localhost:5432 |

> **Linux note:** replace `host.docker.internal` in `docker-compose.yml` with your Docker bridge IP (`ip route | grep docker`).

<br/>

## API reference

All endpoints require `Authorization: Token <your-token>`.

**Ingest a document**
```bash
curl -X POST http://localhost:8000/api/ingest/ \
  -H "Authorization: Token <token>" \
  -F "file=@report.pdf"
```

**Query**
```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Authorization: Token <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the Q3 revenue chart show?",
    "top_k": 5,
    "include_images": true
  }'
```

**Response shape**
```json
{
  "query_id": "3f8a1c...",
  "answer": "The Q3 revenue chart shows a 23% increase vs Q2...",
  "sources": [
    {
      "chunk_type": "figure",
      "page_number": 7,
      "image_description": "Bar chart showing Q3 revenue of $4.2M, up from $3.4M in Q2",
      "image_path": "/data/processed/crops/page7_a3f2.png"
    }
  ],
  "latency": {
    "retrieval_ms": 42,
    "generation_ms": 1823,
    "total_ms": 1865
  }
}
```

**All endpoints**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest/` | Upload and ingest a PDF or image |
| `POST` | `/api/query/` | Run a hybrid RAG query |
| `GET` | `/api/documents/` | List all indexed documents |
| `GET` | `/api/documents/<id>/` | Document detail with all chunks |
| `GET` | `/api/stats/` | Pipeline metrics (docs, chunks, queries) |
| `GET` | `/api/logs/` | Query history with latency and RAGAS scores |

<br/>

## Benchmarks

```bash
python scripts/benchmark_ingestion.py --docs ./data/raw --workers 1 2 4 8
```

50-page technical PDF, M2 MacBook Pro:

| Workers | Time (s) | Pages/s | Speedup |
|:---:|:---:|:---:|:---:|
| 1 | 12.4 | 4.0 | 1× |
| 2 | 7.1 | 7.0 | 1.8× |
| 4 | 4.2 | 11.9 | 3.0× |
| 8 | 3.1 | 16.1 | **4.0×** |

<br/>

## Evaluation

```bash
python scripts/run_eval.py
```

Computes RAGAS scores on stored queries and saves them back to PostgreSQL. All scores surface in the Streamlit eval tab.

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer stay grounded in retrieved context? |
| **Context recall** | Did retrieval surface the chunks needed to answer? |
| **Answer relevancy** | Is the answer on-topic for the question asked? |

<br/>

## Project structure

```
visionscribe/
│
├── api/                        ← Django app
│   ├── models.py               · Document, Chunk, QueryLog
│   ├── views.py                · Ingest, Query, Stats, Logs endpoints
│   ├── serializers.py          · DRF serializers
│   ├── urls.py                 · URL routing
│   └── settings.py             · Django configuration
│
├── ingestion/                  ← Offline ingestion pipeline
│   ├── pipeline.py             · Multiprocessing orchestrator
│   ├── parser.py               · PyMuPDF page extraction
│   ├── chunker.py              · Overlapping 512-token text chunker
│   └── embedder.py             · BGE-M3 + CLIP → ChromaDB + PostgreSQL
│
├── vision/                     ← Visual intelligence layer
│   ├── detector.py             · YOLO region detection + crop saving
│   └── captioner.py            · Qwen-VL captions via Ollama
│
├── retrieval/                  ← Live query pipeline
│   ├── hybrid_retriever.py     · Dense + BM25 + RRF merge
│   └── generator.py            · Ollama LLM grounded answer generation
│
├── streamlit_app/
│   └── app.py                  ← 4-tab UI: query, ingest, docs, eval
│
├── scripts/
│   ├── setup.sh                · One-command project bootstrap
│   ├── benchmark_ingestion.py  · Worker count benchmarks
│   └── run_eval.py             · RAGAS scoring
│
├── data/                       · raw/, processed/, chroma_db/  [gitignored]
├── models/                     · YOLO weights  [gitignored]
├── .env.example
├── SETUP_TIPS.md               ← Common issues + fixes
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

<br/>

## Roadmap

- [ ] Async ingestion queue with Celery + Redis
- [ ] Multi-query decomposition for complex questions
- [ ] Fine-tuned YOLO on DocLayNet for document-specific layout detection
- [ ] Cross-encoder re-ranker after hybrid retrieval
- [ ] LangGraph multi-step reasoning agent
- [ ] Streaming response endpoint (`text/event-stream`)
- [ ] REST API usage analytics dashboard

<br/>

---

<div align="center">

Built by [Arpit Bhojani](https://linkedin.com/in/arpit-bhojani) &nbsp;·&nbsp; Python · Django · YOLO · RAG

*Questions? Open an issue. Stars are appreciated.*

</div>