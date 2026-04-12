#!/usr/bin/env python3
"""Flat vector-store baseline for LongMemEval.

Replaces Memento's knowledge graph with a naive vector store approach:
1. Chunk each conversation turn as a separate document
2. Embed with the same model Memento uses (all-MiniLM-L6-v2)
3. For each question, embed the query and retrieve top-k by cosine similarity
4. Stuff results into the same ANSWER_PROMPT and generate with the same LLM

This is the standard RAG baseline: no entity resolution, no relationships,
no temporal tracking — just similarity search over text chunks.

Usage:
    python run_vector_baseline.py --variant oracle --output results_vector.jsonl
    python run_vector_baseline.py --variant oracle --output results_vector_sample.jsonl --sample 50

Then evaluate with the shared judge:
    python ../run_benchmark.py evaluate --results results_vector.jsonl --variant oracle
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "longmemeval"))

from memento.config import LLMConfig  # noqa: E402
from memento.embedder import SentenceTransformerEmbedder  # noqa: E402
from memento.llm import create_llm_client, get_default_model  # noqa: E402

# Reuse the same prompt and dataset helpers
from run_benchmark import (  # noqa: E402
    ANSWER_PROMPT,
    ANSWER_SYSTEM,
    _stratified_sample,
    load_dataset,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Vector store
# ═══════════════════════════════════════════════════════════════════════════


class FlatVectorStore:
    """Minimal in-memory vector store with cosine similarity search."""

    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self.embedder = embedder
        self.texts: list[str] = []
        self.embeddings: list[np.ndarray] = []

    def add(self, text: str) -> None:
        self.texts.append(text)
        self.embeddings.append(self.embedder.embed(text))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.embeddings:
            return []
        q = self.embedder.embed(query)
        matrix = np.vstack(self.embeddings)
        scores = matrix @ q  # already normalized, so this is cosine similarity
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.texts[i], float(scores[i])) for i in top_idx]


def ingest_sessions(store: FlatVectorStore, sessions: list[list[dict]], dates: list[str]) -> None:
    """Chunk each turn as a separate document, prefixed with the session date."""
    for session, date in zip(sessions, dates):
        prefix = f"[Conversation date: {date}] " if date else ""
        for turn in session:
            role = turn["role"].capitalize()
            store.add(f"{prefix}{role}: {turn['content']}")


def build_context(results: list[tuple[str, float]], token_budget: int = 4000) -> str:
    """Concatenate retrieved chunks within a token budget (~4 chars per token)."""
    char_budget = token_budget * 4
    lines = ["## Related Conversation Turns"]
    total_chars = len(lines[0])
    for text, _score in results:
        cost = len(text) + 2
        if total_chars + cost > char_budget:
            break
        lines.append(text)
        total_chars += cost
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════


def run(
    variant: str,
    output_path: str,
    *,
    sample: int | None = None,
    limit: int | None = None,
    resume: bool = True,
    token_budget: int = 4000,
    top_k: int = 30,
    answer_model: str | None = None,
) -> None:
    dataset = load_dataset(variant)
    if sample and sample < len(dataset):
        dataset = _stratified_sample(dataset, sample)
    elif limit:
        dataset = dataset[:limit]

    out = Path(output_path)
    done_ids: set[str] = set()
    if resume and out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["question_id"])
        print(f"  Resuming — {len(done_ids)} questions already completed")

    remaining = [e for e in dataset if e["question_id"] not in done_ids]
    if not remaining:
        print("  All questions already answered.")
        return

    # LLM client for answer generation (uses env vars like ANTHROPIC_API_KEY)
    llm_config = LLMConfig()
    llm_client = create_llm_client(llm_config)
    provider = llm_config.provider or "anthropic"
    model = answer_model or get_default_model(provider, "chat")

    print()
    print("=" * 60)
    print(f"  VECTOR BASELINE — variant={variant}")
    print(f"  ANSWER MODEL: {model}  ({provider})")
    print(f"  QUESTIONS:    {len(remaining)} (of {len(dataset)})")
    print(f"  TOP-K:        {top_k}")
    print(f"  TOKEN BUDGET: {token_budget}")
    print("=" * 60)

    # Load the embedder once — reused across all questions
    print("\n  Loading embedding model...")
    embedder = SentenceTransformerEmbedder()

    for i, entry in enumerate(remaining):
        qid = entry["question_id"]
        sessions = entry["haystack_sessions"]
        dates = entry["haystack_dates"]

        print(f"\n  [{i+1}/{len(remaining)}] {qid} ({len(sessions)} sessions)", flush=True)

        t0 = time.time()
        store = FlatVectorStore(embedder)
        try:
            ingest_sessions(store, sessions, dates)

            question = entry["question"]
            current_date = entry.get("question_date", "")
            results = store.search(question, top_k=top_k)
            memory_context = build_context(results, token_budget=token_budget)

            user_msg = ANSWER_PROMPT.format(
                memory_context=memory_context,
                current_date=current_date,
                question=question,
            )
            answer = llm_client.complete(
                messages=[{"role": "user", "content": user_msg}],
                model=model,
                system=ANSWER_SYSTEM,
                temperature=0.0,
                max_tokens=1024,
            )
            _append_result(out, qid, answer)
            print(f"    {time.time()-t0:.1f}s  {answer[:100]}...", flush=True)
        except Exception as e:
            logger.error("Error on %s: %s", qid, e)
            traceback.print_exc()
            _append_result(out, qid, f"Error: {e}")

    print(f"\nDone. Results written to {out}")


def _append_result(path: Path, question_id: str, hypothesis: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": question_id, "hypothesis": hypothesis}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector store baseline for LongMemEval")
    parser.add_argument("--variant", required=True, choices=["oracle", "s", "m"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--answer-model", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run(
        variant=args.variant,
        output_path=args.output,
        sample=args.sample,
        limit=args.limit,
        resume=not args.no_resume,
        token_budget=args.token_budget,
        top_k=args.top_k,
        answer_model=args.answer_model,
    )


if __name__ == "__main__":
    main()
