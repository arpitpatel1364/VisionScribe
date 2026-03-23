# VisionScribe — Setup Tips & Troubleshooting

This file covers every common issue you'll hit setting this up for the first time.

---

## Step 0 — System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB |
| GPU | Not required | CUDA GPU speeds YOLO + embeddings |
| PostgreSQL | 14+ | 16 |
| Ollama | Latest | Latest |

---

## Step 1 — Install Ollama (required for LLM + VLM)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com
```

Then pull the models (do this before running setup.sh):

```bash
ollama pull llama3          # ~4 GB — main LLM for text queries
ollama pull qwen2.5vl       # ~6 GB — vision-language model for image captions
```

Start Ollama (it must run in the background the whole time):

```bash
ollama serve
# Leave this terminal open
```

**Tip:** If you're on a low-RAM machine, use `ollama pull llama3.2` (smaller) instead of `llama3`.
Update `OLLAMA_LLM_MODEL=llama3.2` in your `.env`.

---

## Step 2 — PostgreSQL setup

### macOS
```bash
brew install postgresql@16
brew services start postgresql@16
createdb vision_rag
```

### Ubuntu/Debian
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb vision_rag
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'yourpassword';"
```

Then set in `.env`:
```
DB_NAME=vision_rag
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
```

---

## Step 3 — Python environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Common install errors:**

| Error | Fix |
|---|---|
| `error: legacy-install-failure` on `psycopg2-binary` | `brew install libpq` (Mac) or `sudo apt install libpq-dev` (Linux) |
| `fitz` import error | Uninstall `fitz`, keep `PyMuPDF`: `pip uninstall fitz && pip install PyMuPDF` |
| `torch` OOM during install | Install CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `open_clip` fails | `pip install open-clip-torch --no-deps && pip install timm` |

---

## Step 4 — Django setup

```bash
# Copy and fill in .env
cp .env.example .env

# Set a real SECRET_KEY in .env:
python -c "import secrets; print(secrets.token_urlsafe(50))"

# Run migrations
python manage.py makemigrations api
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Generate API token for your user
python manage.py drf_create_token <your-username>
# Output: Token abc123... — copy this, paste into Streamlit sidebar
```

---

## Step 5 — YOLO model

The project defaults to `yolov8n.pt` (auto-downloads on first run, ~6 MB).

For **better document layout detection**, download a DocLayNet-trained model:

```bash
# Option A: Use the default YOLOv8n (works, lower accuracy on documents)
# Nothing to do — auto-downloads on first ingestion

# Option B: Use a document-layout fine-tuned model
# Download from: https://huggingface.co/nickmuchi/yolos-small-finetuned-tables
# Place it at: ./models/yolo_doc_layout.pt
# Set in .env: YOLO_MODEL_PATH=./models/yolo_doc_layout.pt
```

---

## Step 6 — Start everything

Open **3 terminals**, all with `source venv/bin/activate`:

```bash
# Terminal 1
ollama serve

# Terminal 2
python manage.py runserver
# → Django running at http://localhost:8000

# Terminal 3
streamlit run streamlit_app/app.py
# → Streamlit at http://localhost:8501
```

---

## Step 7 — First ingestion test

1. Open http://localhost:8501
2. Paste your API token in the sidebar
3. Go to "Ingest" tab → upload a small PDF (start with 5-10 pages)
4. Watch the "Documents" tab — status goes: `pending → processing → done`
5. Go to "Query" tab → ask something from your PDF

**Tip for first test:** Use a PDF with charts or tables to see the vision module in action.

---

## Performance tuning

### Speed up ingestion

In `.env`:
```
INGESTION_WORKERS=4     # Set to your CPU core count
CHUNK_SIZE=512          # Increase to 1024 for longer context
```

### Speed up embeddings (if you have a GPU)

The BGE-M3 and CLIP models automatically use CUDA if available. Verify:
```python
import torch; print(torch.cuda.is_available())
```

### Reduce RAM usage

If you're hitting memory limits:
- Use `llama3.2` instead of `llama3` (2 GB smaller)
- Set `INGESTION_WORKERS=2`
- Reduce `TOP_K_DENSE=5`

---

## Common runtime errors

### "Ollama not reachable"
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
# If not: ollama serve
```

### "relation api_document does not exist"
```bash
python manage.py migrate
```

### "No module named 'ingestion'"
Make sure you're running from the project root:
```bash
cd vision_rag
python manage.py runserver   # NOT: cd api && python manage.py...
```

### ChromaDB "collection does not exist"
ChromaDB creates collections on first ingestion. Run at least one document through the pipeline first.

### Streamlit "Connection refused" to API
Make sure Django is running on port 8000 before starting Streamlit.

---

## Docker troubleshooting

### Ollama not accessible from container
`docker-compose.yml` uses `host.docker.internal:11434` which works on Mac/Windows Docker Desktop.

On Linux:
```yaml
# In docker-compose.yml, change:
OLLAMA_BASE_URL: http://172.17.0.1:11434
# (use your Docker bridge gateway IP)
```

Find it with: `ip route | grep docker`

---

## Making it GitHub-ready

1. Record a demo GIF (use QuickTime or OBS) showing a chart query being answered
2. Paste the GIF at the top of README.md: `![VisionScribe Demo](demo.gif)`
3. Run `python scripts/benchmark_ingestion.py` and paste the table in README
4. Add your actual RAGAS scores once you've run `python scripts/run_eval.py`
5. Tag your first release: `git tag v0.1.0 && git push --tags`
