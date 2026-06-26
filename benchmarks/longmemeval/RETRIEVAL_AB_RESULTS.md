# Retrieval-improvements A/B — results & findings

A/B comparison of five gbrain-inspired changes against the pre-change baseline
(`e640aeb`). Baseline run from a detached git worktree so its code is
unmodified; both arms use identical samples (stratified, seed 42) and identical
models per run.

## The five changes (branch `longmemeval-improvements`)

1. **Recency relative to `as_of`** (`retrieval.py`) — recency decay was measured
   against wall-clock `now()`, scoring every benchmark fact ~0 (conversations are
   dated years in the past). Now decays relative to the query's `as_of` date.
   *Correctness fix regardless of score.*
2. **Two-pass counting + self-verification** (`run_benchmark.py`) — **reverted to
   off by default.** See findings below.
3. **FTS keyword tokenization** (`verbatim_store.py`) — whole-question phrase
   match (which almost never hit) → stopword-filtered OR-ed terms. *Correctness fix.*
4. **Cross-encoder reranker** (`reranker.py`) — optional, process-cached, graceful
   fallback; reranks verbatim chunks + adds a query-relevance signal to graph facts.
5. **Semantic entity recall + cross-channel fusion** (`retrieval.py`) — seed graph
   expansion from entities appearing in semantically-retrieved chunks, so the graph
   is reachable for queries that share no surface tokens with any entity name.

## Experiment 1 — oracle, 100-question sample (Anthropic answers, GPT-4o judge)

| Config | Overall | vs baseline |
|---|---|---|
| Baseline | 93.0% | — |
| All changes, as first shipped | 81.0% | **−12.0** |
| After fix (retrieval-only) | 90.0% | −3.0 (≈noise) |

**Finding: the self-verification pass (item 2) caused a −12 regression.**
`_verify_answer` returned the verifier's raw text whenever it wasn't exactly
`VERIFIED`, but the verifier emits a *critique* ("The proposed answer is
reasonable, but…") rather than a clean replacement — which leaked into **23/100**
final answers (11 judged wrong), collapsing single-session-preference 100%→50%.
The original authors had deliberately left this infrastructure unwired.

**Fix (commit `30451ba`):** verify + two-pass default OFF, flipped to opt-in
`--verify` / `--two-pass`, plus a guard that keeps the original answer if the
verifier replies with a critique.

**Why oracle can't validate retrieval:** oracle haystacks are evidence-only (no
distractor sessions), so better retrieval has nothing to filter and can only
break even or hurt slightly. The −3 retrieval-only result is expected and
inconclusive — the wrong test for these changes.

## Experiment 2 — `s` variant (noisy ~48-session haystacks), the real test

`s` adds distractor sessions, which is what the retrieval changes are built to
cut through. Run entirely on **open models via Together** (extraction + answers
on the same model in both arms; embeddings + reranker local; judge GPT-4o).

### 2a. 30-question stratified sample (Qwen3-235B)

Baseline 73.3% → New 80.0%, **+6.7pp, +2 questions, 0 regressions.** First
positive signal, but n=30 is within noise and gains were confined to one
category — not conclusive.

### 2b. Full-500 run, stopped at n=264 common questions (Llama-3.3-70B-Instruct-Turbo)

Run halted partway to save Together credits once the result was clear. The 264
completed questions happen to be the *hard* categories (multi-session,
temporal-reasoning, preference, user) — the run processes the dataset in order
and the easy categories (knowledge-update, single-session-assistant) come later
— so absolute scores are lower than the balanced 30-sample, but the **delta is a
fair paired comparison on identical questions**.

| Category | n | Baseline | New | Δ |
|---|---|---|---|---|
| multi-session | 132 | 52.3% | 59.1% | +6.8 |
| temporal-reasoning | 34 | 35.3% | 44.1% | +8.8 |
| single-session-preference | 30 | 50.0% | 60.0% | +10.0 |
| single-session-user | 68 | 77.9% | 89.7% | +11.8 |
| **Overall** | **264** | **56.4%** | **65.2%** | **+8.8** |

Paired flips: **NEW gained 49, lost 26 (net +23).** McNemar χ² ≈ 6.4,
**p ≈ 0.01** — statistically significant. Gains are broad-based, including the
theoretically-targeted categories (multi-session +9 net, temporal +3 net).

## Verdict

- **Item 2 (verify/two-pass): a real regression — reverted.**
- **Items 1/3/4/5 (recency, FTS, reranker, semantic recall): neutral on oracle
  (no noise to filter), and a significant +8.8pp on noisy `s` haystacks.** The
  changes help where they're designed to: cross-session synthesis and temporal
  reasoning amid distractors.
- Recency (1) and FTS (3) are correctness fixes independent of the score.

## Notes for future runs

- **Harness bug:** `--session-workers > 1` corrupts ingestion — multiple threads
  write to one question's SQLite connection (`sqlite3.InterfaceError: bad
  parameter`). Use cross-question `--workers` instead (each question gets its own
  store/connection). Worth fixing the thread-safety if fast parallel session
  ingestion is wanted.
- Whole pipeline runs on open models (Together) for a fraction of frontier-model
  cost; only the GPT-4o judge needs OpenAI. API keys via `memento.secret_store`.
- To get a full-500 headline number, finish the `s` run (the remaining ~236 are
  the easy categories where both arms are near-ceiling, so expect the delta to
  shrink toward the mean but stay positive).
