"""Generate a synthetic conversation dataset for benchmarking.

Creates a cast of characters with evolving situations, intentional
contradictions, and cross-references across conversations.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Character:
    name: str
    type: str  # person, organization
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    # properties maps key → [(value, month_introduced)]


@dataclass
class Conversation:
    id: str
    month: int  # simulated month (1-6)
    turns: list[str] = field(default_factory=list)
    entities_mentioned: list[str] = field(default_factory=list)


def generate_characters() -> list[Character]:
    """Generate a cast of ~30 recurring characters."""
    characters = [
        Character("John Smith", "person", ["John", "JS"], {
            "title": [("Director of Sales", 1), ("VP of Sales", 4)],
            "location": [("Chicago", 1), ("Austin", 3)],
            "company": [("Acme Corp", 1)],
        }),
        Character("Jane Doe", "person", ["Jane"], {
            "title": [("CTO", 1)],
            "company": [("Beta Inc", 1), ("Acme Corp", 5)],
        }),
        Character("Bob Park", "person", ["Bob", "Robert"], {
            "title": [("Software Engineer", 1), ("Senior Engineer", 3)],
            "company": [("Gamma Labs", 1)],
        }),
        Character("Alice Chen", "person", ["Alice"], {
            "title": [("Product Manager", 1)],
            "company": [("Acme Corp", 1)],
        }),
        Character("David Kim", "person", ["David", "Dave"], {
            "title": [("Founder", 1)],
            "company": [("StartupX", 1)],
        }),
        Character("Sarah Johnson", "person", ["Sarah"], {
            "title": [("Marketing Director", 1)],
            "company": [("Beta Inc", 1)],
        }),
        Character("Michael Brown", "person", ["Mike", "Michael"], {
            "title": [("Architect", 1)],
            "location": [("Seattle", 1), ("Portland", 4)],
        }),
        Character("Acme Corp", "organization", ["Acme"], {
            "industry": [("Technology", 1)],
            "status": [("acquiring Beta Inc", 3)],
        }),
        Character("Beta Inc", "organization", ["Beta"], {
            "industry": [("SaaS", 1)],
            "status": [("being acquired by Acme", 3)],
        }),
        Character("Gamma Labs", "organization", ["Gamma"], {
            "industry": [("AI Research", 1)],
        }),
        Character("StartupX", "organization", [], {
            "industry": [("Fintech", 1)],
            "funding": [("Series A", 1), ("Series B", 4)],
        }),
        Character("Project Falcon", "project", ["Falcon"], {
            "status": [("planning", 1), ("in progress", 3), ("launched", 5)],
            "lead": [("John Smith", 1)],
        }),
        Character("Project Mercury", "project", ["Mercury"], {
            "status": [("active", 1), ("paused", 4)],
            "lead": [("Alice Chen", 1)],
        }),
    ]
    return characters


TEMPLATES = [
    "I had a meeting with {name} today. {pronoun} mentioned that {fact}.",
    "{name} told me {pronoun_lower} role is {title} at {company}.",
    "Spoke with {name} about {project}. {pronoun} said it's {status}.",
    "{name} is moving to {location} next month.",
    "Heard from {name} that {company} is {company_status}.",
    "Quick call with {name}. {pronoun} seems excited about {project}.",
    "Ran into {alias} at the conference. We discussed {topic}.",
    "{name} mentioned that {other_name} joined {company}.",
]


def generate_conversations(
    characters: list[Character],
    num_conversations: int = 50,
    seed: int = 42,
) -> list[Conversation]:
    """Generate synthetic conversations."""
    rng = random.Random(seed)
    people = [c for c in characters if c.type == "person"]
    orgs = [c for c in characters if c.type == "organization"]
    projects = [c for c in characters if c.type == "project"]
    topics = ["AI strategy", "Q2 planning", "the partnership deal",
              "hiring plans", "the product roadmap", "market trends"]

    conversations = []
    for i in range(num_conversations):
        month = (i % 6) + 1
        conv_id = f"conv-{i+1:03d}"
        turns = []
        mentioned = set()

        num_turns = rng.randint(3, 8)
        for t in range(num_turns):
            person = rng.choice(people)
            mentioned.add(person.name)
            pronoun = "He" if rng.random() > 0.5 else "She"
            pronoun_lower = pronoun.lower()

            # Pick a template
            template = rng.choice(TEMPLATES)

            # Get current properties for this month
            title = "working on things"
            for val, m in person.properties.get("title", []):
                if m <= month:
                    title = val

            company = "their company"
            for val, m in person.properties.get("company", []):
                if m <= month:
                    company = val

            location = "their city"
            for val, m in person.properties.get("location", []):
                if m <= month:
                    location = val

            project = rng.choice(projects) if projects else None
            project_name = project.name if project else "the project"
            project_status = "in progress"
            if project:
                for val, m in project.properties.get("status", []):
                    if m <= month:
                        project_status = val
                mentioned.add(project_name)

            org = rng.choice(orgs) if orgs else None
            company_status = "growing"
            if org:
                for val, m in org.properties.get("status", []):
                    if m <= month:
                        company_status = val

            other = rng.choice([p for p in people if p != person])
            alias = rng.choice(person.aliases) if person.aliases else person.name

            try:
                turn = template.format(
                    name=person.name,
                    alias=alias,
                    pronoun=pronoun,
                    pronoun_lower=pronoun_lower,
                    fact=f"{pronoun_lower} title is now {title}",
                    title=title,
                    company=company,
                    location=location,
                    project=project_name,
                    status=project_status,
                    company_status=company_status,
                    topic=rng.choice(topics),
                    other_name=other.name,
                )
                turns.append(turn)
            except (KeyError, IndexError):
                turns.append(f"Had a conversation with {person.name} today.")

        conversations.append(Conversation(
            id=conv_id,
            month=month,
            turns=turns,
            entities_mentioned=list(mentioned),
        ))

    return conversations


@dataclass
class BenchmarkQuestion:
    """A question with a known correct answer for evaluation."""

    question: str
    expected_entities: list[str]  # Entity names that should appear in the answer
    expected_facts: list[str]  # Facts that should appear
    category: str  # factual, compositional, temporal, contradiction
    as_of: str | None = None  # ISO timestamp for temporal queries


def generate_benchmark_questions(characters: list[Character]) -> list[BenchmarkQuestion]:
    """Generate questions with known correct answers."""
    return [
        BenchmarkQuestion(
            question="What is John Smith's current title?",
            expected_entities=["John Smith"],
            expected_facts=["VP of Sales"],
            category="factual",
        ),
        BenchmarkQuestion(
            question="Where does John Smith live?",
            expected_entities=["John Smith"],
            expected_facts=["Austin"],
            category="factual",
        ),
        BenchmarkQuestion(
            question="What should I know before meeting John Smith?",
            expected_entities=["John Smith", "Acme Corp"],
            expected_facts=["VP of Sales", "Acme"],
            category="compositional",
        ),
        BenchmarkQuestion(
            question="What was John Smith's title at the beginning?",
            expected_entities=["John Smith"],
            expected_facts=["Director of Sales"],
            category="temporal",
            as_of="2025-02-01T00:00:00+00:00",
        ),
        BenchmarkQuestion(
            question="Who works at Acme Corp?",
            expected_entities=["John Smith", "Alice Chen"],
            expected_facts=["Acme Corp"],
            category="compositional",
        ),
        BenchmarkQuestion(
            question="What is the status of Project Falcon?",
            expected_entities=["Project Falcon"],
            expected_facts=["launched"],
            category="factual",
        ),
        BenchmarkQuestion(
            question="Where did John Smith used to live before Austin?",
            expected_entities=["John Smith"],
            expected_facts=["Chicago"],
            category="temporal",
            as_of="2025-02-01T00:00:00+00:00",
        ),
        BenchmarkQuestion(
            question="What company is Jane Doe at now?",
            expected_entities=["Jane Doe"],
            expected_facts=["Acme Corp"],
            category="factual",
        ),
    ]


def save_dataset(output_dir: Path, seed: int = 42) -> dict:
    """Generate and save the complete benchmark dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    characters = generate_characters()
    conversations = generate_conversations(characters, num_conversations=50, seed=seed)
    questions = generate_benchmark_questions(characters)

    # Save characters
    chars_data = []
    for c in characters:
        chars_data.append({
            "name": c.name,
            "type": c.type,
            "aliases": c.aliases,
            "properties": {k: [(v, m) for v, m in vals] for k, vals in c.properties.items()},
        })
    with open(output_dir / "characters.json", "w") as f:
        json.dump(chars_data, f, indent=2)

    # Save conversations
    convs_data = []
    for c in conversations:
        convs_data.append({
            "id": c.id,
            "month": c.month,
            "turns": c.turns,
            "entities_mentioned": c.entities_mentioned,
        })
    with open(output_dir / "conversations.json", "w") as f:
        json.dump(convs_data, f, indent=2)

    # Save questions
    qs_data = []
    for q in questions:
        entry = {
            "question": q.question,
            "expected_entities": q.expected_entities,
            "expected_facts": q.expected_facts,
            "category": q.category,
        }
        if q.as_of:
            entry["as_of"] = q.as_of
        qs_data.append(entry)
    with open(output_dir / "questions.json", "w") as f:
        json.dump(qs_data, f, indent=2)

    return {
        "characters": len(characters),
        "conversations": len(conversations),
        "questions": len(questions),
    }


if __name__ == "__main__":
    result = save_dataset(Path("benchmark_data"))
    print(f"Generated: {result}")
