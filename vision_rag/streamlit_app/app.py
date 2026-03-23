"""
streamlit_app/app.py

Full-featured Streamlit dashboard for Vision-RAG.
  - Upload & ingest PDFs
  - Run queries with source attribution
  - View retrieved chunks with page images
  - Live pipeline metrics + eval scores
"""
import os
import time
import requests
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(
    page_title="VisionScribe",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = ""
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def api_headers():
    return {"Authorization": f"Token {st.session_state.token}"}


def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", headers=api_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, json=None, files=None, data=None):
    try:
        r = requests.post(
            f"{API_BASE}{path}", headers=api_headers(),
            json=json, files=files, data=data, timeout=120
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar: Auth ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 VisionScribe")
    st.caption("Multimodal document intelligence")
    st.divider()

    st.subheader("Authentication")
    token_input = st.text_input("API Token", type="password", value=st.session_state.token)
    if token_input:
        st.session_state.token = token_input
    if st.session_state.token:
        st.success("Token set ✓")
    else:
        st.warning("Enter your Django API token")

    st.divider()

    # Stats
    if st.session_state.token:
        stats = api_get("/api/stats/")
        if stats:
            st.metric("Documents", stats["documents"]["done"])
            st.metric("Chunks", stats["chunks"]["total"])
            c1, c2 = st.columns(2)
            c1.metric("Text", stats["chunks"]["text"])
            c2.metric("Visual", stats["chunks"]["visual"])
            st.metric("Queries", stats["queries"]["total"])
            st.metric("Avg latency", f"{stats['queries']['avg_latency_ms']:.0f} ms")


# ── Main tabs ─────────────────────────────────────────────────
tab_query, tab_ingest, tab_docs, tab_eval = st.tabs([
    "💬 Query", "📄 Ingest", "📚 Documents", "📊 Evaluation"
])

# ── TAB 1: Query ──────────────────────────────────────────────
with tab_query:
    st.header("Ask your documents")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_text = st.text_area(
            "Your question",
            placeholder="What does the chart on page 5 show about revenue trends?",
            height=100,
        )
    with col2:
        top_k = st.slider("Sources to retrieve", 1, 15, 5)
        include_images = st.checkbox("Include visual chunks", value=True)

    if st.button("🔍 Search", type="primary", disabled=not st.session_state.token):
        if not query_text.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                t0 = time.time()
                result = api_post("/api/query/", json={
                    "query": query_text,
                    "top_k": top_k,
                    "include_images": include_images,
                })

            if result:
                st.session_state.query_history.append({
                    "query": query_text,
                    "answer": result["answer"],
                    "latency": result["latency"],
                })

                # Answer
                st.subheader("Answer")
                st.markdown(result["answer"])

                # Latency metrics
                lat = result["latency"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval", f"{lat['retrieval_ms']:.0f} ms")
                c2.metric("Generation", f"{lat['generation_ms']:.0f} ms")
                c3.metric("Total", f"{lat['total_ms']:.0f} ms")

                # Sources
                st.subheader(f"Sources ({len(result['sources'])})")
                for i, src in enumerate(result["sources"], 1):
                    chunk_type = src.get("chunk_type", "text")
                    page = src.get("page_number", "?")
                    icon = {"text": "📝", "figure": "📊", "table": "📋", "formula": "∑"}.get(chunk_type, "📄")

                    with st.expander(f"{icon} Source {i} — Page {page} ({chunk_type})"):
                        content = src.get("image_description") or src.get("text_content", "")
                        st.write(content)

                        if src.get("image_path") and Path(src["image_path"]).exists():
                            st.image(src["image_path"], caption=f"Page {page} visual region")

    # History
    if st.session_state.query_history:
        with st.expander("Query history"):
            for h in reversed(st.session_state.query_history[-10:]):
                st.markdown(f"**Q:** {h['query']}")
                st.markdown(f"**A:** {h['answer'][:200]}...")
                st.caption(f"Total: {h['latency']['total_ms']:.0f} ms")
                st.divider()


# ── TAB 2: Ingest ─────────────────────────────────────────────
with tab_ingest:
    st.header("Ingest documents")
    st.info("Upload PDFs or images. They'll be parsed, region-detected with YOLO, "
            "captioned with Qwen-VL, and embedded into ChromaDB.")

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("📥 Start Ingestion", type="primary"):
        for f in uploaded:
            with st.spinner(f"Uploading {f.name}..."):
                result = api_post(
                    "/api/ingest/",
                    files={"file": (f.name, f.getvalue(), f.type)},
                )
            if result:
                if result.get("detail") == "Document already ingested.":
                    st.warning(f"⚠️ {f.name} — already in index")
                else:
                    st.success(f"✅ {f.name} — ingestion started (ID: {result['document_id'][:8]}...)")
                    st.caption("Ingestion runs in the background. Check Documents tab for status.")


# ── TAB 3: Documents ──────────────────────────────────────────
with tab_docs:
    st.header("Indexed documents")

    if st.button("🔄 Refresh"):
        st.rerun()

    docs = api_get("/api/documents/")
    if docs and docs.get("results"):
        status_emoji = {"done": "✅", "processing": "⏳", "pending": "🕐", "failed": "❌"}
        for doc in docs["results"]:
            emoji = status_emoji.get(doc["status"], "❓")
            with st.expander(f"{emoji} {doc['filename']} — {doc['total_chunks']} chunks"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Pages", doc["total_pages"])
                c2.metric("Chunks", doc["total_chunks"])
                c3.metric("Status", doc["status"])
                st.caption(f"ID: {doc['id']} | Ingested: {doc['created_at'][:19]}")
    elif docs:
        st.info("No documents ingested yet. Go to the Ingest tab to add some.")


# ── TAB 4: Evaluation ─────────────────────────────────────────
with tab_eval:
    st.header("Pipeline evaluation (RAGAS)")
    st.info("RAGAS scores are computed asynchronously after each query. "
            "Scores appear here once available.")

    logs = api_get("/api/logs/")
    if logs and logs.get("results"):
        import pandas as pd

        rows = []
        for log in logs["results"][:50]:
            rows.append({
                "query": log["query_text"][:60] + "...",
                "faithfulness": log.get("faithfulness_score"),
                "context_recall": log.get("context_recall_score"),
                "answer_relevancy": log.get("answer_relevancy_score"),
                "total_ms": log["total_latency_ms"],
                "timestamp": log["created_at"][:19],
            })
        df = pd.DataFrame(rows)

        # Summary metrics
        scored = df.dropna(subset=["faithfulness"])
        if not scored.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Faithfulness", f"{scored['faithfulness'].mean():.2f}")
            c2.metric("Avg Context Recall", f"{scored['context_recall'].mean():.2f}")
            c3.metric("Avg Answer Relevancy", f"{scored['answer_relevancy'].mean():.2f}")

        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run some queries to populate evaluation metrics.")
