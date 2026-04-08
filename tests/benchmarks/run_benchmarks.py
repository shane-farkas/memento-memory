"""Benchmark runner: evaluate retrieval quality against synthetic dataset.

Runs without LLM calls by using the graph store directly
(entities pre-populated, testing retrieval only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType
from memento.retrieval import RetrievalEngine
from memento.verbatim_store import VerbatimStore


class BenchmarkEmbedder:
    """Simple word-hash embedder for benchmarks (no model downloads)."""

    @property
    def dimension(self) -> int:
        return 16

    def embed(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        vec = np.zeros(16, dtype=np.float32)
        for word in words:
            h = hash(word) % 16
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@dataclass
class BenchmarkResult:
    """Result for a single benchmark question."""

    question: str
    category: str
    entity_recall: float  # fraction of expected entities found
    fact_recall: float    # fraction of expected facts found
    passed: bool


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.fact_recall for r in self.results) / len(self.results)

    @property
    def by_category(self) -> dict[str, float]:
        cats: dict[str, list[float]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.fact_recall)
        return {k: sum(v) / len(v) for k, v in cats.items()}

    def summary(self) -> str:
        lines = [
            f"Overall recall: {self.overall_score:.1%}",
            f"Questions: {len(self.results)}",
            "",
            "By category:",
        ]
        for cat, score in sorted(self.by_category.items()):
            count = sum(1 for r in self.results if r.category == cat)
            lines.append(f"  {cat}: {score:.1%} ({count} questions)")
        return "\n".join(lines)


def populate_graph_from_characters(
    graph: GraphStore, characters: list[dict], month: int = 6
) -> dict[str, str]:
    """Populate the graph with character data up to a given month.

    Sets properties at each month they change (creating supersession chains)
    so temporal queries can retrieve historical states.
    Returns a name→entity_id mapping.
    """
    entity_map = {}

    for char in characters:
        etype = EntityType(char["type"]) if char["type"] in [e.value for e in EntityType] else EntityType.CONCEPT
        entity = graph.create_entity(
            char["name"], etype, aliases=char.get("aliases", [])
        )
        entity_map[char["name"]] = entity.id

        for key, values in char.get("properties", {}).items():
            # Set each value at its introduction month (creates supersession chain)
            for val, m in sorted(values, key=lambda x: x[1]):
                if m <= month:
                    ts = f"2025-{m:02d}-15T00:00:00+00:00"
                    prop = graph.set_property(entity.id, key, val, as_of=ts)
                    # Backdate recorded_at so point-in-time queries work
                    graph.db.execute(
                        "UPDATE properties SET recorded_at = ? WHERE id = ?",
                        (ts, prop.id),
                    )
                    graph.db.conn.commit()

    # Create relationships based on "company" properties
    for char in characters:
        if char["type"] == "person":
            for key, values in char.get("properties", {}).items():
                if key == "company":
                    for val, m in values:
                        if m <= month and val in entity_map:
                            graph.create_relationship(
                                entity_map[char["name"]],
                                entity_map[val],
                                "works_at",
                            )

    return entity_map


def run_benchmarks(data_dir: Path | None = None) -> BenchmarkSuite:
    """Run the benchmark suite against pre-populated graph."""
    # Generate data if not provided
    if data_dir and (data_dir / "characters.json").exists():
        with open(data_dir / "characters.json") as f:
            characters = json.load(f)
        with open(data_dir / "questions.json") as f:
            questions = json.load(f)
        with open(data_dir / "conversations.json") as f:
            conversations = json.load(f)
    else:
        from tests.benchmarks.generate_dataset import (
            generate_benchmark_questions,
            generate_characters,
            generate_conversations,
        )
        chars = generate_characters()
        characters = [
            {"name": c.name, "type": c.type, "aliases": c.aliases,
             "properties": {k: [(v, m) for v, m in vals] for k, vals in c.properties.items()}}
            for c in chars
        ]
        questions = []
        for q in generate_benchmark_questions(chars):
            entry = {"question": q.question, "expected_entities": q.expected_entities,
                     "expected_facts": q.expected_facts, "category": q.category}
            if q.as_of:
                entry["as_of"] = q.as_of
            questions.append(entry)
        conversations = [
            {"id": c.id, "month": c.month, "turns": c.turns}
            for c in generate_conversations(chars, 50)
        ]

    # Set up the system
    db = Database(":memory:")
    graph = GraphStore(db)
    embedder = BenchmarkEmbedder()
    verbatim = VerbatimStore(db, embedder)
    engine = RetrievalEngine(graph, verbatim=verbatim)

    # Populate graph with character data at month 6
    entity_map = populate_graph_from_characters(graph, characters, month=6)

    # Store conversations as verbatim
    for conv in conversations:
        for i, turn in enumerate(conv["turns"]):
            verbatim.store(turn, conversation_id=conv["id"], turn_number=i + 1)

    # Run queries
    suite = BenchmarkSuite()
    for q in questions:
        as_of = q.get("as_of")
        result = engine.recall(q["question"], as_of=as_of)
        text = result.text.lower()

        # Evaluate entity recall
        entities_found = sum(
            1 for e in q["expected_entities"]
            if e.lower() in text
        )
        entity_recall = entities_found / len(q["expected_entities"]) if q["expected_entities"] else 1.0

        # Evaluate fact recall
        facts_found = sum(
            1 for f in q["expected_facts"]
            if f.lower() in text
        )
        fact_recall = facts_found / len(q["expected_facts"]) if q["expected_facts"] else 1.0

        suite.results.append(BenchmarkResult(
            question=q["question"],
            category=q["category"],
            entity_recall=entity_recall,
            fact_recall=fact_recall,
            passed=fact_recall >= 0.5,
        ))

    db.close()
    return suite


if __name__ == "__main__":
    suite = run_benchmarks()
    print(suite.summary())
    print()
    for r in suite.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.question} (entity: {r.entity_recall:.0%}, fact: {r.fact_recall:.0%})")
