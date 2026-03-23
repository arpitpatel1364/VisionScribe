"""
scripts/benchmark_ingestion.py

Benchmarks ingestion speed across different worker counts.
Run: python scripts/benchmark_ingestion.py --docs ./data/raw --workers 1 2 4 8
"""
import argparse
import time
import os
import sys
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")

import django
django.setup()

from loguru import logger
from ingestion.parser import parse_page
import fitz


def _count_pages(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


def _parse_worker(args):
    pdf_path, page_num = args
    try:
        parse_page(pdf_path, page_num)
        return True
    except Exception:
        return False


def benchmark(pdf_paths: list, worker_counts: list):
    results = {}

    for n_workers in worker_counts:
        all_args = []
        for path in pdf_paths:
            pages = _count_pages(path)
            all_args.extend([(path, p) for p in range(pages)])

        logger.info(f"Benchmarking {len(all_args)} pages with {n_workers} workers...")
        t0 = time.perf_counter()

        with Pool(n_workers) as pool:
            outcomes = pool.map(_parse_worker, all_args)

        elapsed = time.perf_counter() - t0
        pages_per_sec = len(all_args) / elapsed
        results[n_workers] = {
            "pages": len(all_args),
            "time_s": round(elapsed, 2),
            "pages_per_sec": round(pages_per_sec, 1),
            "success": sum(outcomes),
        }
        logger.success(
            f"Workers={n_workers} | {len(all_args)} pages in {elapsed:.2f}s "
            f"({pages_per_sec:.1f} pages/s)"
        )

    print("\n── Benchmark Results ──────────────────────────")
    print(f"{'Workers':>8} {'Pages':>8} {'Time (s)':>10} {'Pages/s':>10}")
    print("-" * 45)
    for w, r in sorted(results.items()):
        print(f"{w:>8} {r['pages']:>8} {r['time_s']:>10} {r['pages_per_sec']:>10}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ingestion pipeline")
    parser.add_argument("--docs", default="./data/raw", help="Directory of PDFs")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4], help="Worker counts to test")
    args = parser.parse_args()

    pdfs = list(Path(args.docs).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.docs}")
        sys.exit(1)

    benchmark([str(p) for p in pdfs], args.workers)
