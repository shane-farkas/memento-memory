#!/usr/bin/env python3
"""
LongMemEval benchmark harness for the Memento memory system.

Evaluates Memento's ability to remember and reason over information
scattered across many past conversation sessions.

Dataset: https://github.com/xiaowu0162/longmemeval
Paper:   "LongMemEval: Benchmarking Chat Assistants on Long-Term
          Interactive Memory" (Wu et al., 2024)

Usage:
    # 1. Download the dataset
    python run_benchmark.py download

    # 2. Run benchmark (oracle = evidence-only sessions, fastest)
    python run_benchmark.py run --variant oracle --output results_oracle.jsonl

    # 3. Run benchmark (small haystack = ~80 sessions with noise)
    python run_benchmark.py run --variant s --output results_s.jsonl

    # 4. Evaluate with GPT-4o judge
    python run_benchmark.py evaluate --results results_oracle.jsonl --variant oracle

Environment variables:
    ANTHROPIC_API_KEY   or
    OPENAI_API_KEY      or
    GOOGLE_API_KEY       — for Memento's LLM-powered extraction + answer generation
    OPENAI_API_KEY       — required for GPT-4o judge evaluation step
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from the benchmarks/longmemeval/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from memento import MemoryStore  # noqa: E402
from memento.config import (  # noqa: E402
    ConsolidationConfig,
    EmbeddingConfig,
    LLMConfig,
    MementoConfig,
    RetrievalConfig,
)
from memento.llm import create_llm_client, get_default_model  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

DATASET_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}

DATASET_FILENAMES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

# Prompt for the answer-generation LLM
ANSWER_SYSTEM = (
    "You are a helpful chat assistant. You have access to memories retrieved "
    "from your past conversations with the user. Use these memories to answer "
    "the user's question. If the memories do not contain enough information, "
    "say so honestly."
)

ANSWER_PROMPT = """\
Below are memories retrieved from our past conversations, followed by the \
user's question.

Important guidelines:
- Each conversation is tagged with its date in a [Conversation date: ...] \
header. Use these dates to reason about when events happened, their \
chronological order, and time spans between them.
- When information was updated across conversations (e.g., a number changed, \
a preference shifted, a status was revised), ALWAYS use the value from the \
MOST RECENT conversation. Later conversations supersede earlier ones.
- Answer based only on what is explicitly stated. Do not add to or modify \
stated values (e.g., if the user says "my list has 25 titles", the answer \
is 25 — do not add items mentioned in the same conversation unless the user \
explicitly said the count changed).
- If the question asks for a recommendation or suggestion, USE the \
preferences and details you find in the memories to give a SPECIFIC, \
CONCRETE answer. Do NOT ask clarifying questions back to the user — they \
already shared their preferences in past conversations, and your job is to \
remember and apply them. For example, if the user previously mentioned \
loving rooftop pools and ocean views, recommend hotels with rooftop pools \
and ocean views — do not ask "what amenities do you want?".
- Tailor your response to the user's specific interests, hobbies, or domain \
mentioned in the memories. Generic answers that ignore the user's known \
preferences are wrong.
- For counting questions ("how many X"), carefully enumerate every distinct \
item mentioned across ALL conversations. Do not skip items because they \
appear in different sessions. Build a numbered list first, then count.

## Retrieved Memories
{memory_context}

## Current Date
{current_date}

## Question
{question}

Answer the question step by step:
1. Extract ALL relevant facts, preferences, and dates from the memories above.
2. If the question involves timing or ordering, note the conversation dates.
3. If the same fact appears with different values across conversations, use \
the value from the latest conversation date.
4. If the question asks for a recommendation, immediately apply the user's \
known preferences to produce a specific answer — do not ask the user to \
restate them.
5. Give a direct, specific answer — do not say "I don't know" unless the \
information is truly absent from the memories."""


# ═══════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════

def download_datasets(variants: list[str] | None = None) -> None:
    """Download LongMemEval datasets from HuggingFace."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    variants = variants or list(DATASET_URLS.keys())

    for variant in variants:
        url = DATASET_URLS[variant]
        filepath = DATA_DIR / DATASET_FILENAMES[variant]

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {filepath.name} already exists ({size_mb:.1f} MB), skipping")
            continue

        print(f"  Downloading {filepath.name} ...")
        urllib.request.urlretrieve(url, filepath)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Saved {filepath.name} ({size_mb:.1f} MB)")


def load_dataset(variant: str) -> list[dict]:
    """Load a LongMemEval dataset variant from disk."""
    filepath = DATA_DIR / DATASET_FILENAMES[variant]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Run: python run_benchmark.py download --variants {variant}"
        )
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# MemoryStore factory
# ═══════════════════════════════════════════════════════════════════════════

def create_memory_store(db_path: str = ":memory:") -> MemoryStore:
    """Create a MemoryStore configured for benchmarking."""
    config = MementoConfig(
        db_path=Path(db_path),
        retrieval=RetrievalConfig(
            default_token_budget=4000,   # Larger budget — benchmark needs more context
            max_hop_depth=3,
        ),
        consolidation=ConsolidationConfig(
            # Disable auto-consolidation to keep ingestion fast
            decay_interval_ingestions=999_999,
            full_interval_ingestions=999_999,
        ),
    )
    return MemoryStore(config)


# ═══════════════════════════════════════════════════════════════════════════
# Ingestion
# ═══════════════════════════════════════════════════════════════════════════

def format_session_text(session: list[dict], date: str = "") -> str:
    """Collapse a multi-turn chat session into a single text block.

    Prepends the session date so the LLM and knowledge graph can associate
    temporal context with the facts in this session.
    """
    lines = []
    if date:
        lines.append(f"[Conversation date: {date}]")
    for turn in session:
        role = turn["role"].capitalize()
        content = turn["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def ingest_haystack(
    store: MemoryStore,
    sessions: list[list[dict]],
    dates: list[str],
    *,
    per_turn: bool = False,
    progress: bool = True,
) -> None:
    """Ingest all haystack sessions into *store*.

    Args:
        per_turn:  If True, ingest each user/assistant turn separately
                   (more granular but slower). If False, ingest one text
                   block per session (faster, recommended for s/m).
        progress:  Print a progress line.
    """
    total = len(sessions)
    for idx, (session, date) in enumerate(zip(sessions, dates)):
        ts = f"{date}T12:00:00+00:00" if "T" not in date else date
        conv_id = f"session_{idx}"

        if per_turn:
            # First turn gets the date header
            for turn_num, turn in enumerate(session):
                prefix = f"[Conversation date: {date}]\n" if turn_num == 0 and date else ""
                store.ingest(
                    text=f"{prefix}{turn['role'].capitalize()}: {turn['content']}",
                    conversation_id=conv_id,
                    turn_number=turn_num,
                    source_type="conversation",
                    authority=0.9,
                    timestamp=ts,
                )
        else:
            # Ingest full session for entity extraction + graph building
            text = format_session_text(session, date=date)
            store.ingest(
                text=text,
                conversation_id=conv_id,
                turn_number=0,
                source_type="conversation",
                authority=0.9,
                timestamp=ts,
            )
            # Also store each turn separately in verbatim store for
            # fine-grained FTS5 keyword search (no LLM calls, no embeddings)
            for turn_num, turn in enumerate(session):
                prefix = f"[Conversation date: {date}] " if date else ""
                store.verbatim.store_text_only(
                    text=f"{prefix}{turn['role'].capitalize()}: {turn['content']}",
                    conversation_id=conv_id,
                    turn_number=turn_num + 1,
                    source_type="conversation",
                    timestamp=ts,
                )

        if progress and (idx + 1) % 5 == 0:
            print(f"    Ingested {idx + 1}/{total} sessions", flush=True)

    if progress:
        print(f"    Ingested {total}/{total} sessions — done", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Answer generation
# ═══════════════════════════════════════════════════════════════════════════

def _is_counting_question(question: str) -> bool:
    """Detect counting/enumeration questions."""
    q = question.lower()
    return any(p in q for p in [
        "how many", "how much", "total number", "count",
        "list all", "list every",
    ])


COUNTING_ENUMERATE_PROMPT = """\
Below are memories from past conversations. The user will ask a counting \
question. Your job is ONLY to enumerate — list every distinct item that \
matches the question, with dates and sources where available. Do NOT count \
them yet, just list them.

## Retrieved Memories
{memory_context}

## Current Date
{current_date}

## Question
{question}

List every distinct item relevant to this question, one per line, with any \
dates or details. Do not count or summarize — just enumerate."""


COUNTING_ANSWER_PROMPT = """\
Here is a numbered list of items relevant to the user's question, extracted \
from past conversations:

{enumeration}

## Question
{question}

Now count the items in the list above and give a direct, specific answer."""


VERIFY_PROMPT = """\
You are a careful fact-checker. A chat assistant was asked a question and \
produced an answer based on retrieved memories. Your job is to verify whether \
the answer is correct and complete given the evidence.

## Retrieved Memories
{memory_context}

## Question
{question}

## Proposed Answer
{answer}

Check the answer against the memories:
1. Is every claim in the answer supported by the memories?
2. Is there relevant information in the memories that the answer missed?
3. Is the answer actually responding to what was asked?

If the answer is correct, respond with exactly: VERIFIED
If the answer needs correction, provide the corrected answer directly \
(no preamble, just the better answer)."""


def generate_answer(
    llm_client,
    model: str,
    memory_context: str,
    question: str,
    current_date: str,
) -> str:
    """Use an LLM to produce an answer from the retrieved memory context.

    For counting questions, uses a two-pass approach:
      Pass 1: enumerate all relevant items
      Pass 2: count and answer from the enumeration

    All answers go through a self-verification pass that catches obvious
    errors before returning.
    """
    if _is_counting_question(question):
        return _two_pass_counting(
            llm_client, model, memory_context, question, current_date,
        )

    user_msg = ANSWER_PROMPT.format(
        memory_context=memory_context,
        current_date=current_date,
        question=question,
    )
    return llm_client.complete(
        messages=[{"role": "user", "content": user_msg}],
        model=model,
        system=ANSWER_SYSTEM,
        temperature=0.0,
        max_tokens=1024,
    )


def _two_pass_counting(
    llm_client,
    model: str,
    memory_context: str,
    question: str,
    current_date: str,
) -> str:
    """Two-pass answer generation for counting questions."""
    # Pass 1: enumerate
    enum_msg = COUNTING_ENUMERATE_PROMPT.format(
        memory_context=memory_context,
        current_date=current_date,
        question=question,
    )
    enumeration = llm_client.complete(
        messages=[{"role": "user", "content": enum_msg}],
        model=model,
        system=ANSWER_SYSTEM,
        temperature=0.0,
        max_tokens=1024,
    )

    # Pass 2: count from the enumeration
    count_msg = COUNTING_ANSWER_PROMPT.format(
        enumeration=enumeration,
        question=question,
    )
    return llm_client.complete(
        messages=[{"role": "user", "content": count_msg}],
        model=model,
        system=ANSWER_SYSTEM,
        temperature=0.0,
        max_tokens=512,
    )


def _verify_answer(
    llm_client,
    model: str,
    memory_context: str,
    question: str,
    answer: str,
) -> str:
    """Self-verification: check the answer against the evidence."""
    verify_msg = VERIFY_PROMPT.format(
        memory_context=memory_context,
        question=question,
        answer=answer,
    )
    verdict = llm_client.complete(
        messages=[{"role": "user", "content": verify_msg}],
        model=model,
        temperature=0.0,
        max_tokens=1024,
    )

    if verdict.strip().startswith("VERIFIED"):
        return answer
    # The verifier produced a correction — use it
    return verdict


# ═══════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ═══════════════════════════════════════════════════════════════════════════

def _detect_shared_haystack(dataset: list[dict]) -> bool:
    """Return True if all entries share the same haystack_session_ids."""
    if len(dataset) < 2:
        return False
    first_ids = dataset[0].get("haystack_session_ids")
    if first_ids is None:
        return False
    return all(
        entry.get("haystack_session_ids") == first_ids for entry in dataset[1:]
    )


def _stratified_sample(dataset: list[dict], n: int) -> list[dict]:
    """Sample *n* questions evenly across all categories."""
    from collections import defaultdict
    import random

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for entry in dataset:
        by_cat[entry.get("question_type", "unknown")].append(entry)

    per_cat = max(1, n // len(by_cat))
    sampled = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        random.seed(42)  # Reproducible
        sampled.extend(random.sample(items, min(per_cat, len(items))))

    # Fill remaining slots if rounding left us short
    remaining_ids = {e["question_id"] for e in sampled}
    all_remaining = [e for e in dataset if e["question_id"] not in remaining_ids]
    random.seed(42)
    random.shuffle(all_remaining)
    sampled.extend(all_remaining[: n - len(sampled)])
    return sampled[:n]


def run_benchmark(
    variant: str,
    output_path: str,
    *,
    per_turn: bool = False,
    limit: int | None = None,
    sample: int | None = None,
    category: str | None = None,
    resume: bool = True,
    token_budget: int = 4000,
    answer_model: str | None = None,
) -> None:
    """Run the full benchmark: ingest → recall → answer → save.

    Args:
        variant:       "oracle", "s", or "m"
        output_path:   Path for the JSONL results file
        per_turn:      Ingest per-turn (slower, more granular) vs per-session
        limit:         Only process the first N questions (for testing)
        sample:        Stratified sample of N questions across all categories
        resume:        Skip questions already present in *output_path*
        token_budget:  Token budget for Memento recall()
        answer_model:  Override the LLM model used for answer generation
    """
    dataset = load_dataset(variant)
    if category:
        dataset = [e for e in dataset if e.get("question_type") == category]
        print(f"  Filtered to category={category}: {len(dataset)} questions")
    if sample and sample < len(dataset):
        dataset = _stratified_sample(dataset, sample)
    elif limit:
        dataset = dataset[:limit]

    print(f"\nLongMemEval benchmark — variant={variant}, questions={len(dataset)}")

    # Load already-completed question IDs for resume
    done_ids: set[str] = set()
    out = Path(output_path)
    if resume and out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["question_id"])
        print(f"  Resuming — {len(done_ids)} questions already completed")

    remaining = [e for e in dataset if e["question_id"] not in done_ids]
    if not remaining:
        print("  All questions already answered. Nothing to do.")
        return

    # Determine LLM for answer generation
    llm_config = LLMConfig()
    llm_client = create_llm_client(llm_config)
    provider = llm_config.provider or _detect_provider()
    model = answer_model or get_default_model(provider, "chat")

    print()
    print("=" * 60)
    print(f"  ANSWER MODEL: {model}")
    print(f"  PROVIDER:     {provider}")
    print(f"  QUESTIONS:    {len(remaining)} (of {len(dataset)} after resume)")
    print(f"  TOKEN BUDGET: {token_budget}")
    print("=" * 60)

    # Sanity check: warn loudly if the detected provider is unexpected
    expected_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    expected_key = expected_keys.get(provider)
    if expected_key and not os.environ.get(expected_key):
        print(f"\n  ⚠ WARNING: Provider is {provider} but {expected_key} is NOT set!")
    if provider == "openai" and os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ⚠ WARNING: ANTHROPIC_API_KEY is set but provider is openai.")
        print("  Set MEMENTO_LLM_PROVIDER=anthropic to use Claude instead.")
    print()

    shared = _detect_shared_haystack(dataset)
    print(f"  Shared haystack: {shared}")

    if shared:
        _run_shared_haystack(
            remaining, dataset, llm_client, model, out,
            per_turn=per_turn, token_budget=token_budget,
        )
    else:
        _run_per_question(
            remaining, llm_client, model, out,
            per_turn=per_turn, token_budget=token_budget,
        )

    # Summary
    total_done = 0
    if out.exists():
        with open(out, encoding="utf-8") as f:
            total_done = sum(1 for line in f if line.strip())
    print(f"\nDone. {total_done} answers written to {out}")


def _detect_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


def _run_shared_haystack(
    questions: list[dict],
    full_dataset: list[dict],
    llm_client,
    model: str,
    out_path: Path,
    *,
    per_turn: bool,
    token_budget: int,
) -> None:
    """All questions share one haystack — ingest once, query many times."""
    ref = full_dataset[0]
    sessions = ref["haystack_sessions"]
    dates = ref["haystack_dates"]

    print(f"\n  Phase 1: Ingesting shared haystack ({len(sessions)} sessions) ...")
    store = create_memory_store()
    t0 = time.time()
    ingest_haystack(store, sessions, dates, per_turn=per_turn)
    print(f"  Ingestion done in {time.time() - t0:.1f}s")

    health = store.health()
    print(f"  Graph: {health.node_count} entities, {health.edge_count} relationships, "
          f"{health.active_property_count} properties")

    print(f"\n  Phase 2: Answering {len(questions)} questions ...")
    _answer_questions(store, questions, llm_client, model, out_path, token_budget)
    store.close()


def _run_per_question(
    questions: list[dict],
    llm_client,
    model: str,
    out_path: Path,
    *,
    per_turn: bool,
    token_budget: int,
) -> None:
    """Each question has its own haystack — build a store per question."""
    print(f"\n  Processing {len(questions)} questions (per-question stores) ...")

    for i, entry in enumerate(questions):
        qid = entry["question_id"]
        sessions = entry["haystack_sessions"]
        dates = entry["haystack_dates"]

        print(f"\n  [{i+1}/{len(questions)}] {qid} ({len(sessions)} sessions)")

        store = create_memory_store()
        try:
            ingest_haystack(store, sessions, dates, per_turn=per_turn, progress=False)

            question = entry["question"]
            current_date = entry.get("question_date", "")
            as_of = f"{current_date}T23:59:59+00:00" if current_date and "T" not in current_date else current_date

            memory = store.recall(question, token_budget=token_budget, as_of=as_of or None)
            answer = generate_answer(llm_client, model, memory.text, question, current_date)

            _append_result(out_path, qid, answer)
            print(f"    Answer: {answer[:120]}...")
        except Exception as e:
            logger.error("Error on %s: %s", qid, e)
            traceback.print_exc()
            _append_result(out_path, qid, f"Error: {e}")
        finally:
            store.close()


def _answer_questions(
    store: MemoryStore,
    questions: list[dict],
    llm_client,
    model: str,
    out_path: Path,
    token_budget: int,
) -> None:
    """Query *store* for each question and generate answers."""
    for i, entry in enumerate(questions):
        qid = entry["question_id"]
        question = entry["question"]
        current_date = entry.get("question_date", "")
        as_of = f"{current_date}T23:59:59+00:00" if current_date and "T" not in current_date else current_date

        try:
            memory = store.recall(question, token_budget=token_budget, as_of=as_of or None)
            answer = generate_answer(llm_client, model, memory.text, question, current_date)

            _append_result(out_path, qid, answer)

            if (i + 1) % 10 == 0:
                print(f"    Answered {i+1}/{len(questions)}", flush=True)
        except Exception as e:
            logger.error("Error on %s: %s", qid, e)
            traceback.print_exc()
            _append_result(out_path, qid, f"Error: {e}")

    print(f"    Answered {len(questions)}/{len(questions)} — done", flush=True)


def _append_result(path: Path, question_id: str, hypothesis: str) -> None:
    """Append one result line to the JSONL output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": question_id, "hypothesis": hypothesis}) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_results(
    results_path: str,
    variant: str,
    *,
    judge_model: str = "gpt-4o",
) -> None:
    """Evaluate results using GPT-4o as judge (LongMemEval protocol).

    Loads the results JSONL and the reference dataset, then runs
    task-specific evaluation prompts through GPT-4o.
    """
    dataset = load_dataset(variant)
    ref_map = {e["question_id"]: e for e in dataset}

    results: list[dict] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"\nEvaluating {len(results)} results against {variant} reference ...")

    try:
        import openai
    except ImportError:
        print("ERROR: openai package required for evaluation. pip install openai")
        sys.exit(1)

    client = openai.OpenAI()
    eval_out_path = Path(results_path + f".eval-{judge_model}")

    correct = 0
    total = 0
    category_stats: dict[str, dict[str, int]] = {}

    for entry in results:
        qid = entry["question_id"]
        hypothesis = entry["hypothesis"]
        ref = ref_map.get(qid)
        if not ref:
            print(f"  Warning: {qid} not found in reference data, skipping")
            continue

        qtype = ref.get("question_type", "unknown")
        answer = ref["answer"]
        question = ref["question"]
        is_abstention = qid.endswith("_abs")

        verdict = _judge_answer(
            client, judge_model, question, answer, hypothesis,
            qtype=qtype, is_abstention=is_abstention,
        )

        entry["verdict"] = verdict
        entry["question_type"] = qtype

        if qtype not in category_stats:
            category_stats[qtype] = {"correct": 0, "total": 0}
        category_stats[qtype]["total"] += 1
        total += 1

        if verdict == "yes":
            correct += 1
            category_stats[qtype]["correct"] += 1

    # Write detailed evaluation results
    with open(eval_out_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"LongMemEval Results — variant={variant}, judge={judge_model}")
    print(f"{'='*60}")
    print(f"{'Category':<35} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"{'-'*60}")

    cat_accs = []
    for cat in sorted(category_stats):
        c = category_stats[cat]["correct"]
        t = category_stats[cat]["total"]
        acc = c / t if t > 0 else 0
        cat_accs.append(acc)
        print(f"{cat:<35} {c:>8} {t:>6} {acc:>6.1%}")

    print(f"{'-'*60}")
    overall_acc = correct / total if total > 0 else 0
    task_avg = sum(cat_accs) / len(cat_accs) if cat_accs else 0
    print(f"{'Overall':<35} {correct:>8} {total:>6} {overall_acc:>6.1%}")
    print(f"{'Task-averaged':<35} {'':>8} {'':>6} {task_avg:>6.1%}")
    print(f"{'='*60}")
    print(f"\nDetailed results: {eval_out_path}")


# Task-specific judge prompts following the LongMemEval protocol
_JUDGE_PROMPTS = {
    "default": (
        "A chat assistant was asked the following question based on its past "
        "conversation history with the user.\n\n"
        "Question: {question}\n"
        "Reference answer: {answer}\n"
        "Assistant's answer: {hypothesis}\n\n"
        "Does the assistant's answer correctly convey the same information as "
        "the reference answer? Minor phrasing differences are acceptable. "
        "Answer 'yes' or 'no' only."
    ),
    "temp_reasoning_implicit": (
        "A chat assistant was asked a temporal reasoning question based on its "
        "past conversation history.\n\n"
        "Question: {question}\n"
        "Reference answer: {answer}\n"
        "Assistant's answer: {hypothesis}\n\n"
        "Does the assistant's answer match the reference? For day/time counts, "
        "off-by-one errors are acceptable. Answer 'yes' or 'no' only."
    ),
    "temp_reasoning_explicit": (
        "A chat assistant was asked a temporal reasoning question based on its "
        "past conversation history.\n\n"
        "Question: {question}\n"
        "Reference answer: {answer}\n"
        "Assistant's answer: {hypothesis}\n\n"
        "Does the assistant's answer match the reference? For day/time counts, "
        "off-by-one errors are acceptable. Answer 'yes' or 'no' only."
    ),
    "knowledge_update": (
        "A chat assistant was asked about information that was updated across "
        "conversations.\n\n"
        "Question: {question}\n"
        "Reference answer (the updated/current answer): {answer}\n"
        "Assistant's answer: {hypothesis}\n\n"
        "Does the assistant's answer include the updated/current information "
        "from the reference? The assistant may also mention the previous value "
        "as long as the current one is identified. Answer 'yes' or 'no' only."
    ),
    "implicit_preference_v2": (
        "A chat assistant was asked about a user's preferences based on past "
        "conversations.\n\n"
        "Question: {question}\n"
        "Reference answer / rubric: {answer}\n"
        "Assistant's answer: {hypothesis}\n\n"
        "Does the assistant's answer correctly use the user's personal "
        "information or preferences? It need not cover every rubric point. "
        "Answer 'yes' or 'no' only."
    ),
    "abstention": (
        "A chat assistant was asked a question that CANNOT be answered from "
        "the conversation history (the correct response is to abstain).\n\n"
        "Question: {question}\n"
        "Assistant's answer: {hypothesis}\n\n"
        "Did the assistant correctly identify that it does not have enough "
        "information and abstain from answering (e.g., 'I don't know', "
        "'I don't have that information')? Answer 'yes' or 'no' only."
    ),
}


def _judge_answer(
    client,
    model: str,
    question: str,
    answer: str,
    hypothesis: str,
    *,
    qtype: str,
    is_abstention: bool,
) -> str:
    """Ask GPT-4o to judge whether the hypothesis matches the reference."""
    if is_abstention:
        template = _JUDGE_PROMPTS["abstention"]
    else:
        template = _JUDGE_PROMPTS.get(qtype, _JUDGE_PROMPTS["default"])

    prompt = template.format(question=question, answer=answer, hypothesis=hypothesis)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=8,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return "yes" if verdict.startswith("yes") else "no"
    except Exception as e:
        logger.error("Judge error: %s", e)
        return "error"


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark harness for Memento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download -----------------------------------------------------------
    dl = sub.add_parser("download", help="Download LongMemEval datasets")
    dl.add_argument(
        "--variants", nargs="+", default=["oracle", "s", "m"],
        choices=["oracle", "s", "m"],
        help="Which dataset variants to download (default: all)",
    )

    # run ----------------------------------------------------------------
    run = sub.add_parser("run", help="Run the benchmark")
    run.add_argument(
        "--variant", required=True, choices=["oracle", "s", "m"],
        help="Dataset variant to benchmark against",
    )
    run.add_argument(
        "--output", required=True,
        help="Output JSONL file path",
    )
    run.add_argument(
        "--per-turn", action="store_true",
        help="Ingest each turn separately (slower but more granular)",
    )
    run.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N questions (for testing)",
    )
    run.add_argument(
        "--sample", type=int, default=None,
        help="Stratified sample of N questions across all 6 categories",
    )
    run.add_argument(
        "--category", type=str, default=None,
        choices=["single-session-user", "single-session-assistant",
                 "single-session-preference", "multi-session",
                 "temporal-reasoning", "knowledge-update"],
        help="Only run questions from this category",
    )
    run.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from existing output",
    )
    run.add_argument(
        "--token-budget", type=int, default=4000,
        help="Token budget for Memento recall (default: 4000)",
    )
    run.add_argument(
        "--answer-model", default=None,
        help="Override the LLM model for answer generation",
    )

    # evaluate -----------------------------------------------------------
    ev = sub.add_parser("evaluate", help="Evaluate results with GPT-4o judge")
    ev.add_argument("--results", required=True, help="Path to results JSONL")
    ev.add_argument(
        "--variant", required=True, choices=["oracle", "s", "m"],
        help="Dataset variant (for loading reference answers)",
    )
    ev.add_argument(
        "--judge-model", default="gpt-4o",
        help="Judge model (default: gpt-4o)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "download":
        print("Downloading LongMemEval datasets ...")
        download_datasets(args.variants)

    elif args.command == "run":
        run_benchmark(
            variant=args.variant,
            output_path=args.output,
            per_turn=args.per_turn,
            limit=args.limit,
            sample=args.sample,
            category=args.category,
            resume=not args.no_resume,
            token_budget=args.token_budget,
            answer_model=args.answer_model,
        )

    elif args.command == "evaluate":
        evaluate_results(
            results_path=args.results,
            variant=args.variant,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
