# Benchmarks

## LongMemEval

I have been evaluating Memento on [LongMemEval](https://github.com/xiaowu0162/longmemeval), a benchmark designed to test chat assistants on long-term interactive memory. The benchmark presents 500 questions that require recalling and reasoning over information scattered across multiple past conversation sessions.

**Paper:** "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" - [arXiv](https://arxiv.org/abs/2410.10813)

### What "accuracy" means here

Every number in this document is **end-to-end accuracy**, not retrieval-only. There's an important distinction:

- **Retrieval metrics** (`recall@k`, `R@5`, MRR, nDCG) measure whether the correct evidence was somewhere in the top-k retrieved items. They're typically 90–98% on well-tuned systems and are relatively easy to win — any competent vector store gets high scores on them.
- **End-to-end accuracy** measures whether the full pipeline produces a correct answer. A question is only counted as correct if:
  1. The retrieval step surfaced enough evidence to answer, *and*
  2. The LLM composed a correct answer from the retrieved context, *and*
  3. An independent judge model agreed the answer matches the reference.

All three stages have to succeed. Retrieval failures, reasoning failures, and formatting mismatches all count as wrong. The full 500-question run is uniform: no per-question tuning, no hand-curated prompts, no oracle routing, one harness file ([`benchmarks/longmemeval/run_benchmark.py`](benchmarks/longmemeval/run_benchmark.py)), one answer model, one judge.

This matters because benchmark numbers on LongMemEval from different projects are often not directly comparable - some report retrieval-only metrics, some report on 50-question samples, some hand-tune per category. Everything below is the same thing across every row: ingest → recall → generate → judge, across all 500 questions, and is reproducible.

### Results

**Oracle variant, 500 questions, GPT-4o judge:**

| Category | Correct | Total | Accuracy |
|---|--:|--:|--:|
| single-session-assistant | 55 | 56 | 98.2% |
| single-session-user | 68 | 70 | 97.1% |
| single-session-preference | 28 | 30 | 93.3% |
| temporal-reasoning | 119 | 133 | 89.5% |
| knowledge-update | 69 | 78 | 88.5% |
| multi-session | 115 | 133 | 86.5% |
| **Overall** | **454** | **500** | **90.8%** |
| **Task-averaged** | | | **92.2%** |

### Comparison vs Baselines

To isolate what Memento's knowledge graph actually contributes, we ran the same 500 oracle questions through two simpler memory strategies. Only the recall layer differs, so every baseline uses the same dataset, the same haystack ingestion loop, the same answer model (Claude Sonnet 4.6), the same `ANSWER_PROMPT`, the same 4,000-token context budget, and the same GPT-4o judge with task-specific prompts.

**Vector store baseline** ([`baselines/run_vector_baseline.py`](benchmarks/longmemeval/baselines/run_vector_baseline.py)) - a minimal in-memory RAG system. Each haystack conversation turn is embedded individually with `sentence-transformers/all-MiniLM-L6-v2` (the same model Memento uses) and stored as a `(text, embedding)` pair in a numpy array. At query time, the question is embedded, cosine similarity is computed against all turns via a single matrix multiplication, and the top-30 results are concatenated (each turn prefixed by its `[Conversation date: ...]` header) into the context block. No chunking, no reranker, no graph, no temporal awareness — pure similarity search.

**Markdown baseline** ([`baselines/run_markdown_baseline.py`](benchmarks/longmemeval/baselines/run_markdown_baseline.py)) - a per-session LLM extraction pipeline. For each conversation session, an LLM is prompted to extract bulleted facts worth remembering (with each bullet tagged by session date) and those bullets are appended to a single markdown file. At query time, the entire accumulated markdown file is truncated to a 4,000-token budget and passed into the answer prompt. This simulates the CLAUDE.md / USER.md / mem0 "append facts to a file" pattern that most AI coding agents use for persistent memory today.

The only thing that changes between the three runs is the recall step of how the system stores and retrieves information before handing it to the answer LLM. That isolates the contribution of structured memory from everything else in the pipeline.

| Category | Markdown | Vector Store | Memento |
|---|--:|--:|--:|
| single-session-assistant | **41.1%** | 100.0% | 98.2% |
| single-session-preference | 100.0% | 100.0% | 93.3% |
| single-session-user | 94.3% | 94.3% | 97.1% |
| knowledge-update | 88.5% | 87.2% | 88.5% |
| multi-session | 80.5% | 67.7% | **86.5%** |
| temporal-reasoning | 82.0% | 66.9% | **89.5%** |
| **Overall** | **80.8%** | **79.8%** | **90.8%** |
| **Task-averaged** | **81.0%** | **86.0%** | **92.2%** |

**What the gaps tell us:**

- **Single-session questions are mostly easy for all three approaches** - except the markdown baseline, which catastrophically drops to 41.1% on assistant-side questions. The LLM fact extractor is biased toward capturing what the *user* said and loses assistant statements. This is a structural weakness of LLM-distilled summaries: extraction is lossy, and information that doesn't match the "memorable fact" pattern gets dropped.
- **Vector retrieval is surprisingly strong on single-session questions** (94–100%). When the needle is in one haystack, cosine similarity just works.
- **Vector falls apart on multi-session and temporal reasoning** (67.7% and 66.9%). These questions require composing information across conversations - flat similarity search has no way to connect chunks or reason about time.
- **Markdown holds up better than vector on multi-session and temporal** (80.5% and 82.0%) because the LLM extraction captures some structure across sessions. But it pays the price in the extraction step, catastrophically missing certain categories.
- **Memento is the only approach without a catastrophic category failure.** Its worst category is 86.5% vs 41.1% (markdown) and 66.9% (vector). The knowledge graph retains all facts with their provenance and temporal anchoring, so retrieval doesn't depend on an upfront "what's memorable" decision or a pure similarity signal.

The overall 10-point gap over both baselines isolates the value of structured, bitemporal memory. Vector and markdown approaches each win on specific categories, but neither covers the full space. Memento's worst category is still 86.5% - every approach to memory eventually fails somewhere, but structure keeps the floor high.

### Question Categories

The 500 questions span six categories of increasing difficulty:

- **single-session-user** (70): Recall facts stated by the user in a single conversation
- **single-session-assistant** (56): Recall facts stated by the assistant in a single conversation
- **single-session-preference** (30): Apply user preferences revealed in a single conversation
- **multi-session** (133): Synthesize information scattered across multiple conversations
- **knowledge-update** (78): Return the most recent value when a fact changes over time
- **temporal-reasoning** (133): Reason about when events happened, their order, or time spans between them

Each question also has an abstention variant (suffixed `_abs`) where the correct answer is "I don't know" which tests that the system doesn't hallucinate.

### Dataset Variants

| Variant | Description | Haystack |
|---|---|---|
| `oracle` | Evidence-only sessions (no noise) | 1-6 sessions per question, only the sessions containing the answer |
| `s` | Small haystack with distractor sessions | ~80 sessions per question |
| `m` | Medium haystack with more distractors | ~170 sessions per question |

The `oracle` variant isolates retrieval quality from needle-in-haystack search. The `s` and `m` variants additionally test whether Memento can find the right information among irrelevant conversations.

### Methodology

#### Pipeline

Each question is processed through this pipeline:

1. **Ingest** - All haystack sessions for a question are ingested into a fresh in-memory MemoryStore. Each session runs through Memento's full pipeline: entity extraction, entity resolution, relationship extraction, temporal tagging, and verbatim storage. Session dates are preserved as `[Conversation date: ...]` headers.

2. **Recall** - `store.recall(question, token_budget=4000, as_of=question_date)` retrieves relevant context. This uses Memento's compositional retrieval: keyword search (FTS5), semantic search (embeddings), and knowledge graph traversal up to 3 hops. The `as_of` parameter ensures temporal correctness so the system only sees information available at the question's date.

3. **Answer** - An LLM generates an answer from the retrieved context. The prompt instructs the model to use conversation dates for temporal reasoning, prefer the most recent value for updated facts, apply known preferences concretely, and enumerate before counting. Temperature is set to 0.0 for reproducibility.

4. **Judge** - A GPT-4o judge compares the generated answer against the reference using task-specific prompts (e.g., temporal questions allow off-by-one tolerance, knowledge-update questions accept mentioning old values if the current one is identified). The judge outputs "yes" or "no".

#### MemoryStore Configuration

The benchmark uses an in-memory SQLite database with these settings:

- **Token budget:** 4,000 tokens for recall context
- **Max hop depth:** 3 (knowledge graph traversal)
- **Auto-consolidation:** Disabled (ingestion speed over maintenance)
- **Ingestion mode:** Per-session (full session as one text block for entity extraction, plus individual turns stored in verbatim for fine-grained FTS5 search)

#### Answer Generation

- **Model:** Configurable via `--answer-model` (defaults to provider's default chat model)
- **Temperature:** 0.0
- **Two-pass counting:** Questions detected as counting/enumeration ("how many X") use a two-pass approach — first enumerate all items, then count from the enumeration
- **Self-verification:** Not currently active in the main path since it didn't seem to improve the overall accuracy (available but not invoked by default)

### Reproduction

#### Prerequisites

```bash
# Install Memento with your preferred LLM provider
pip install memento-memory[anthropic]   # or [openai] or [gemini]

# Set your provider's API key (pick one)
export ANTHROPIC_API_KEY=your-key       # For Anthropic
export OPENAI_API_KEY=your-key          # For OpenAI
export GOOGLE_API_KEY=your-key          # For Gemini

# Also needed for the evaluation step (GPT-4o judge)
export OPENAI_API_KEY=your-key
```

#### Step 1: Download the Dataset

```bash
cd memento/benchmarks/longmemeval
python run_benchmark.py download
```

Downloads datasets from HuggingFace to `benchmarks/longmemeval/data/`. The oracle variant is ~15 MB, the small variant ~265 MB.

#### Step 2: Run the Benchmark

Full run (all 500 questions):

```bash
python run_benchmark.py run --variant oracle --output results.jsonl
```

Quick test (30 questions, stratified across all categories):

```bash
python run_benchmark.py run --variant oracle --output results_sample.jsonl --sample 30
```

Single category:

```bash
python run_benchmark.py run --variant oracle --output results_temporal.jsonl --category temporal-reasoning
```

Parallel execution (big speedup on the `s` and `m` variants, where ingestion dominates):

```bash
python run_benchmark.py run --variant s --output results_s.jsonl --workers 10
```

`--workers N` runs N questions in parallel via a thread pool. Each worker creates its own in-memory SQLite store, so there's no shared graph state — only the JSONL write is serialized. On the oracle variant this gives roughly linear speedup up to 5-10 workers. On the `s` and `m` variants (where ingestion is 15-20x more LLM work per question), 10 workers can turn a multi-day sequential run into an overnight one. Start conservative — Anthropic and your answer-model provider both have rate limits. If you see 429 errors, drop the worker count.

The run supports resuming — if interrupted, re-running the same command skips already-completed questions. Use `--no-resume` to start fresh.

#### Step 3: Evaluate

```bash
python run_benchmark.py evaluate --results results.jsonl --variant oracle
```

This calls GPT-4o to judge each answer. Results are written to `results.jsonl.eval-gpt-4o` and a per-category accuracy table is printed to stdout.

To use a different judge model:

```bash
python run_benchmark.py evaluate --results results.jsonl --variant oracle --judge-model gpt-4o-mini
```

#### Check Progress

Count completed questions during a run:

```bash
wc -l results.jsonl
```

#### Full CLI Reference

```
run_benchmark.py run
  --variant {oracle,s,m}        Dataset variant (required)
  --output PATH                 Output JSONL path (required)
  --per-turn                    Ingest each turn separately (slower, more granular)
  --limit N                     Only process first N questions
  --sample N                    Stratified sample of N questions across all 6 categories
  --category CAT                Only run one category
  --no-resume                   Start fresh, ignore existing output
  --token-budget N              Token budget for recall (default: 4000)
  --answer-model MODEL          Override LLM model for answer generation

run_benchmark.py evaluate
  --results PATH                Path to results JSONL (required)
  --variant {oracle,s,m}        Dataset variant for reference answers (required)
  --judge-model MODEL           Judge model (default: gpt-4o)
```

### Cost and Runtime

The oracle variant processes each question independently (separate MemoryStore per question, 1-6 sessions each). Expect:

- **Ingestion:** ~1-3 LLM calls per session (entity extraction)
- **Recall:** Embedding search + graph traversal (no LLM calls)
- **Answer generation:** 1 LLM call per question (2 for counting questions)
- **Evaluation:** 1 GPT-4o call per question (500 total, very cheap)

Total wall time depends on the LLM provider and rate limits. A full 500-question oracle run typically takes 2-4 hours with Anthropic or OpenAI APIs.
