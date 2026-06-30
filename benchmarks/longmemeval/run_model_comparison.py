"""Run LongMemEval across multiple models via Together AI for comparison.

API keys are resolved from the encrypted secret store (see
memento.secret_store), falling back to environment variables. Store them once:

    python -m memento.secret_store set together     # answer generation
    python -m memento.secret_store set anthropic     # entity extraction (ingestion)
    python -m memento.secret_store set openai        # GPT-4o judge (evaluate step)

Then, from this directory:
    python run_model_comparison.py

Environment variables (TOGETHER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY)
still override the stored values when set.
"""

import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
_HERE = Path(__file__).resolve().parent
BENCHMARK = str(_HERE / "run_benchmark.py")

# Make the memento package importable so we can read the encrypted secret store.
sys.path.insert(0, str(_HERE.parents[1] / "src"))
from memento.secret_store import get_secret  # noqa: E402

# Models to test via Together AI
MODELS = [
    ("MiniMaxAI/MiniMax-M3", "minimax_m3"),
    ("zai-org/GLM-5.2", "glm52"),
]

# Use a sample for quick comparison (set to None for full 500)
SAMPLE = None


def main():
    # Resolve from the encrypted store (env var wins if set).
    together_key = get_secret("together")
    anthropic_key = get_secret("anthropic")

    if not together_key:
        print("ERROR: No Together AI key. Run: python -m memento.secret_store set together")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: No Anthropic key (needed for entity extraction). "
              "Run: python -m memento.secret_store set anthropic")
        sys.exit(1)

    # Child benchmark processes read ANTHROPIC_API_KEY from the environment for
    # MemoryStore's extraction LLM, so export it for the subprocesses we spawn.
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    for model_id, short_name in MODELS:
        output = f"results_compare_{short_name}.jsonl"

        if os.path.exists(output):
            with open(output) as f:
                done = sum(1 for line in f if line.strip())
            target = SAMPLE or 500
            if done >= target:
                print(f"\n{'='*60}")
                print(f"  SKIP {short_name} -- already have {done} results")
                print(f"{'='*60}")
                continue

        print(f"\n{'='*60}")
        print(f"  RUNNING: {model_id}")
        print(f"  OUTPUT:  {output}")
        print(f"{'='*60}\n")

        cmd = [
            PYTHON, BENCHMARK, "run",
            "--variant", "oracle",
            "--output", output,
            "--answer-model", model_id,
            "--answer-provider", "openai-compatible",
            "--answer-base-url", "https://api.together.xyz/v1",
            "--answer-api-key", together_key,
            "--workers", "8",
        ]
        if SAMPLE:
            cmd.extend(["--sample", str(SAMPLE)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: {short_name} exited with code {result.returncode}")

    # Summary
    print(f"\n{'='*60}")
    print("  ALL RUNS COMPLETE")
    print(f"{'='*60}\n")

    for _, short_name in MODELS:
        output = f"results_compare_{short_name}.jsonl"
        if os.path.exists(output):
            with open(output) as f:
                count = sum(1 for line in f if line.strip())
            print(f"  {short_name}: {count} results")

    print(f"\nTo evaluate (needs OPENAI_API_KEY for GPT-4o judge):")
    for _, short_name in MODELS:
        output = f"results_compare_{short_name}.jsonl"
        if os.path.exists(output):
            print(f"  python {BENCHMARK} evaluate --results {output} --variant oracle")


if __name__ == "__main__":
    main()
