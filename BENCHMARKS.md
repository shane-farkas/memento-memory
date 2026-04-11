# Benchmarks

## LongMemEval

We evaluate Memento on [LongMemEval](https://github.com/xiaowu0162/longmemeval) (Wu et al., 2024), a benchmark designed to test chat assistants on long-term interactive memory. The benchmark presents 500 questions that require recalling and reasoning over information scattered across multiple past conversation sessions.

**Paper:** "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" — [arXiv](https://arxiv.org/abs/2410.10813)

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

### Question Categories

The 500 questions span six categories of increasing difficulty:

- **single-session-user** (70) — Recall facts stated by the user in a single conversation
- **single-session-assistant** (56) — Recall facts stated by the assistant in a single conversation
- **single-session-preference** (30) — Apply user preferences revealed in a single conversation
- **multi-session** (133) — Synthesize information scattered across multiple conversations
- **knowledge-update** (78) — Return the most recent value when a fact changes over time
- **temporal-reasoning** (133) — Reason about when events happened, their order, or time spans between them

Each question also has an abstention variant (suffixed `_abs`) where the correct answer is "I don't know" — testing that the system doesn't hallucinate.

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

1. **Ingest** — All haystack sessions for a question are ingested into a fresh in-memory MemoryStore. Each session runs through Memento's full pipeline: entity extraction, entity resolution, relationship extraction, temporal tagging, and verbatim storage. Session dates are preserved as `[Conversation date: ...]` headers.

2. **Recall** — `store.recall(question, token_budget=4000, as_of=question_date)` retrieves relevant context. This uses Memento's compositional retrieval: keyword search (FTS5), semantic search (embeddings), and knowledge graph traversal up to 3 hops. The `as_of` parameter ensures temporal correctness — the system only sees information available at the question's date.

3. **Answer** — An LLM generates an answer from the retrieved context. The prompt instructs the model to use conversation dates for temporal reasoning, prefer the most recent value for updated facts, apply known preferences concretely, and enumerate before counting. Temperature is set to 0.0 for reproducibility.

4. **Judge** — A GPT-4o judge compares the generated answer against the reference using task-specific prompts (e.g., temporal questions allow off-by-one tolerance, knowledge-update questions accept mentioning old values if the current one is identified). The judge outputs "yes" or "no".

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
- **Self-verification:** Not currently active in the main path (available but not invoked by default)

### Reproduction

#### Prerequisites

```bash
# Install Memento with your preferred LLM provider
pip install memento-memory[anthropic]   # or [openai] or [gemini]

# Set API keys
export ANTHROPIC_API_KEY=your-key       # For Memento extraction + answer generation
export OPENAI_API_KEY=your-key          # Required for GPT-4o judge in evaluation step
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
