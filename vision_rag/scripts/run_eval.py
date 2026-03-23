"""
scripts/run_eval.py

Runs RAGAS evaluation on stored QueryLogs that don't yet have scores.
Computes: faithfulness, context_recall, answer_relevancy.

Run: python scripts/run_eval.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")

import django
django.setup()

from loguru import logger
from api.models import QueryLog, Chunk

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Run: pip install ragas")


def run_eval(batch_size: int = 10):
    if not RAGAS_AVAILABLE:
        print("Install ragas: pip install ragas datasets")
        return

    # Fetch unscored logs
    unscored = QueryLog.objects.filter(
        faithfulness_score__isnull=True,
        answer__gt="",
    )[:batch_size]

    if not unscored:
        logger.info("No unscored queries found.")
        return

    logger.info(f"Evaluating {len(unscored)} queries...")

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for log in unscored:
        chunk_ids = log.retrieved_chunk_ids or []
        chunks = Chunk.objects.filter(id__in=chunk_ids)
        contexts = [
            (c.image_description or c.text_content or "")[:500]
            for c in chunks
        ]
        if not contexts:
            continue

        rows["question"].append(log.query_text)
        rows["answer"].append(log.answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(log.query_text)  # placeholder — swap with real GT

    if not rows["question"]:
        logger.info("No evaluatable queries (missing chunks).")
        return

    ds = Dataset.from_dict(rows)

    try:
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        df = result.to_pandas()
        logger.success(f"Eval complete:\n{df[['faithfulness','answer_relevancy','context_recall']].describe()}")

        # Save scores back to DB
        for i, log in enumerate(list(unscored)[:len(df)]):
            row = df.iloc[i]
            log.faithfulness_score = float(row.get("faithfulness", 0) or 0)
            log.answer_relevancy_score = float(row.get("answer_relevancy", 0) or 0)
            log.context_recall_score = float(row.get("context_recall", 0) or 0)
            log.save()

        logger.success(f"Saved scores for {len(df)} queries.")

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")


if __name__ == "__main__":
    run_eval()
