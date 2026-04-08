"""Stage 15 tests: Benchmark suite runs end-to-end."""

from __future__ import annotations

from pathlib import Path

from tests.benchmarks.generate_dataset import (
    generate_benchmark_questions,
    generate_characters,
    generate_conversations,
    save_dataset,
)
from tests.benchmarks.run_benchmarks import run_benchmarks


def test_dataset_generation(tmp_path):
    """Dataset generator produces expected output."""
    result = save_dataset(tmp_path, seed=42)
    assert result["characters"] >= 10
    assert result["conversations"] == 50
    assert result["questions"] >= 5

    # Files should exist
    assert (tmp_path / "characters.json").exists()
    assert (tmp_path / "conversations.json").exists()
    assert (tmp_path / "questions.json").exists()


def test_dataset_is_reproducible(tmp_path):
    """Same seed produces same dataset."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    save_dataset(dir1, seed=42)
    save_dataset(dir2, seed=42)

    import json
    with open(dir1 / "conversations.json") as f1, open(dir2 / "conversations.json") as f2:
        assert json.load(f1) == json.load(f2)


def test_benchmark_suite_runs():
    """Benchmark suite runs end-to-end and produces scores."""
    suite = run_benchmarks()
    assert len(suite.results) >= 5
    assert 0.0 <= suite.overall_score <= 1.0

    # Category breakdown
    cats = suite.by_category
    assert "factual" in cats
    assert "compositional" in cats

    # Summary renders
    summary = suite.summary()
    assert "Overall recall" in summary


def test_benchmark_factual_recall():
    """Factual questions should have decent recall on pre-populated graph."""
    suite = run_benchmarks()
    factual = [r for r in suite.results if r.category == "factual"]
    assert len(factual) >= 2

    # At least some factual questions should pass
    passing = sum(1 for r in factual if r.passed)
    assert passing >= 1, f"Expected at least 1 factual question to pass, got {passing}"
