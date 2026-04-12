#!/usr/bin/env python3
"""Markdown file memory baseline for LongMemEval.

Simulates the CLAUDE.md / USER.md / mem0 "append facts to a markdown file"
memory pattern:
1. For each session, an LLM extracts key facts and appends them to a markdown file
2. For each question, the entire markdown file is passed as context
3. Same ANSWER_PROMPT and answer LLM as Memento — only the memory layer differs

This is how most AI coding agents handle "persistent memory" today: edit a
file. It works fine for small amounts of info but breaks down as the file
grows beyond the context window.

Usage:
    python run_markdown_baseline.py --variant oracle --output results_markdown.jsonl
    python run_markdown_baseline.py --variant oracle --output results_markdown_sample.jsonl --sample 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "longmemeval"))

from memento.config import LLMConfig  # noqa: E402
from memento.llm import create_llm_client, get_default_model  # noqa: E402

from run_benchmark import (  # noqa: E402
    ANSWER_PROMPT,
    ANSWER_SYSTEM,
    _stratified_sample,
    load_dataset,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Fact extraction prompt (simulates what mem0 / Claude Desktop do internally)
# ═══════════════════════════════════════════════════════════════════════════

FACT_EXTRACTION_SYSTEM = (
    "You are a memory extraction tool. From a conversation, extract any facts "
    "worth remembering long-term: preferences, decisions, events, relationships, "
    "things that happened, and specific values (numbers, dates, names). "
    "Output a bulleted markdown list. Be concise and specific. Do NOT include "
    "general chit-chat or speculation. If nothing is worth remembering, respond "
    "with just: (nothing notable)"
)

FACT_EXTRACTION_PROMPT = """\
Extract facts worth remembering from this conversation. Output as a markdown \
bulleted list. Include the date in each bullet so the reader knows when the \
fact was recorded.

Date: {date}

Conversation:
{conversation}

Facts (markdown bullets):"""


def format_session(session: list[dict]) -> str:
    lines = []
    for turn in session:
        lines.append(f"{turn['role'].capitalize()}: {turn['content']}")
    return "\n".join(lines)


def extract_facts(
    llm_client,
    model: str,
    session: list[dict],
    date: str,
) -> str:
    """Ask the LLM to extract markdown-bulleted facts from a session."""
    conversation = format_session(session)
    prompt = FACT_EXTRACTION_PROMPT.format(date=date or "unknown", conversation=conversation)
    try:
        return llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system=FACT_EXTRACTION_SYSTEM,
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception as e:
        logger.error("Fact extraction failed: %s", e)
        return "(extraction failed)"


def build_memory_file(
    llm_client,
    model: str,
    sessions: list[list[dict]],
    dates: list[str],
) -> str:
    """Build a single markdown memory file from all sessions."""
    sections = ["# User Memory\n"]
    for session, date in zip(sessions, dates):
        facts = extract_facts(llm_client, model, session, date)
        if facts.strip() and "nothing notable" not in facts.lower():
            sections.append(f"## Session {date or '?'}\n{facts.strip()}\n")
    return "\n".join(sections)


def truncate_to_budget(text: str, token_budget: int = 4000) -> str:
    """Crude truncation at a character budget (~4 chars per token)."""
    char_budget = token_budget * 4
    if len(text) <= char_budget:
        return text
    return text[:char_budget] + "\n...(truncated)"


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

    llm_config = LLMConfig()
    llm_client = create_llm_client(llm_config)
    provider = llm_config.provider or "anthropic"
    model = answer_model or get_default_model(provider, "chat")

    print()
    print("=" * 60)
    print(f"  MARKDOWN BASELINE — variant={variant}")
    print(f"  ANSWER MODEL: {model}  ({provider})")
    print(f"  QUESTIONS:    {len(remaining)} (of {len(dataset)})")
    print(f"  TOKEN BUDGET: {token_budget}")
    print("=" * 60)

    for i, entry in enumerate(remaining):
        qid = entry["question_id"]
        sessions = entry["haystack_sessions"]
        dates = entry["haystack_dates"]

        print(f"\n  [{i+1}/{len(remaining)}] {qid} ({len(sessions)} sessions)", flush=True)

        t0 = time.time()
        try:
            memory_file = build_memory_file(llm_client, model, sessions, dates)
            memory_context = truncate_to_budget(memory_file, token_budget)

            question = entry["question"]
            current_date = entry.get("question_date", "")

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
    parser = argparse.ArgumentParser(description="Markdown file memory baseline for LongMemEval")
    parser.add_argument("--variant", required=True, choices=["oracle", "s", "m"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--token-budget", type=int, default=4000)
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
        answer_model=args.answer_model,
    )


if __name__ == "__main__":
    main()
