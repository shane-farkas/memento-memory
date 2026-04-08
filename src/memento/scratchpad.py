"""Session scratchpad: intra-conversation entity tracking without LLM calls.

Uses regex + heuristics for lightweight entity spotting and coreference
tracking within a single conversation session. Flushes to the ingestion
pipeline on session end.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from memento.graph_store import GraphStore
from memento.models import Entity, EntityType, _new_id

logger = logging.getLogger(__name__)


@dataclass
class SessionMention:
    """A mention of an entity within a session."""

    tentative_id: str
    text: str
    turn_number: int
    surrounding_context: str = ""
    resolved_to: str | None = None  # Entity.id if matched to graph
    type_hint: EntityType | None = None


@dataclass
class SessionScratchpad:
    """Lightweight intra-conversation entity cache.

    No LLM calls. Uses regex + heuristics for entity spotting
    and simple pronoun tracking for coreference.
    """

    session_id: str = field(default_factory=_new_id)
    mentions: list[SessionMention] = field(default_factory=list)
    coref_chains: dict[str, list[SessionMention]] = field(default_factory=dict)
    _last_person_id: str | None = field(default=None, repr=False)
    _last_entity_id: str | None = field(default=None, repr=False)
    _graph: GraphStore | None = field(default=None, repr=False)

    def __init__(self, graph: GraphStore | None = None) -> None:
        self.session_id = _new_id()
        self.mentions = []
        self.coref_chains = {}
        self._last_person_id = None
        self._last_entity_id = None
        self._graph = graph

    def on_turn(self, text: str, turn_number: int) -> list[SessionMention]:
        """Process a conversation turn. Returns new mentions found."""
        new_mentions = []

        # 1. Extract named entities (capitalized multi-word sequences)
        names = self._extract_names(text)
        for name, type_hint in names:
            mention = self._create_mention(name, turn_number, text, type_hint)
            new_mentions.append(mention)

            # Update last referenced entity for pronoun tracking
            if type_hint == EntityType.PERSON:
                self._last_person_id = mention.tentative_id
            self._last_entity_id = mention.tentative_id

        # 2. Track pronouns (link to most recent named entity)
        pronouns = self._extract_pronouns(text)
        for pronoun, ptype in pronouns:
            ref_id = None
            if ptype == "person" and self._last_person_id:
                ref_id = self._last_person_id
            elif self._last_entity_id:
                ref_id = self._last_entity_id

            if ref_id:
                mention = SessionMention(
                    tentative_id=ref_id,  # Same ID as the referent
                    text=pronoun,
                    turn_number=turn_number,
                    surrounding_context=text[:200],
                    type_hint=EntityType.PERSON if ptype == "person" else None,
                )
                self.mentions.append(mention)
                if ref_id in self.coref_chains:
                    self.coref_chains[ref_id].append(mention)
                new_mentions.append(mention)

        return new_mentions

    def resolve_mention(self, text: str) -> Entity | None:
        """Try to resolve a mention against the graph via exact/alias lookup."""
        if self._graph is None:
            return None

        results = self._graph.search_entities(name=text, fuzzy=False)
        if results:
            return results[0]
        return None

    def get_coreference_chains(self) -> dict[str, list[SessionMention]]:
        """Get all coreference chains built during this session."""
        return dict(self.coref_chains)

    def get_unique_entities(self) -> list[tuple[str, str, EntityType | None]]:
        """Get deduplicated list of (tentative_id, best_name, type_hint)."""
        seen = {}
        for mention in self.mentions:
            tid = mention.tentative_id
            if tid not in seen:
                seen[tid] = (tid, mention.text, mention.type_hint)
            else:
                # Prefer longer names (more complete)
                _, existing_name, existing_type = seen[tid]
                if len(mention.text) > len(existing_name):
                    seen[tid] = (tid, mention.text, mention.type_hint or existing_type)
        return list(seen.values())

    def _create_mention(
        self,
        name: str,
        turn_number: int,
        context: str,
        type_hint: EntityType | None,
    ) -> SessionMention:
        """Create a new mention, potentially linking to an existing entity."""
        # Try to resolve against graph
        resolved_entity = self.resolve_mention(name) if self._graph else None

        # Check if we've seen this name before in this session
        existing_id = None
        for m in self.mentions:
            if m.text.lower() == name.lower():
                existing_id = m.tentative_id
                break

        tentative_id = (
            resolved_entity.id if resolved_entity
            else existing_id or _new_id()
        )

        mention = SessionMention(
            tentative_id=tentative_id,
            text=name,
            turn_number=turn_number,
            surrounding_context=context[:200],
            resolved_to=resolved_entity.id if resolved_entity else None,
            type_hint=type_hint,
        )

        self.mentions.append(mention)

        # Add to coreference chain
        if tentative_id not in self.coref_chains:
            self.coref_chains[tentative_id] = []
        self.coref_chains[tentative_id].append(mention)

        return mention

    def _extract_names(self, text: str) -> list[tuple[str, EntityType | None]]:
        """Extract potential entity names using regex heuristics."""
        results = []

        # Multi-word capitalized sequences (potential names)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(pattern, text):
            name = match.group(1)
            words = name.split()
            if len(words) <= 4:
                org_suffixes = {"Corp", "Inc", "LLC", "Ltd", "Company", "Group"}
                if any(w in org_suffixes for w in words):
                    results.append((name, EntityType.ORGANIZATION))
                else:
                    results.append((name, EntityType.PERSON))

        # Single capitalized words that look like proper nouns
        # (not at sentence start, not common words)
        common_words = {
            "The", "This", "That", "These", "Those", "What", "When", "Where",
            "Which", "Who", "How", "Why", "Yes", "No", "Not", "But", "And",
            "Also", "Just", "Only", "Very", "Much", "Many", "Some", "Any",
            "All", "Each", "Every", "Both", "Few", "More", "Most", "Other",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December",
        }
        # Already captured in multi-word matches, skip duplicates
        already_captured = {name for name, _ in results}

        return results

    def _extract_pronouns(self, text: str) -> list[tuple[str, str]]:
        """Extract pronouns and their likely referent type."""
        results = []
        person_pronouns = re.findall(
            r'\b(he|him|his|she|her|they|them|their)\b', text, re.IGNORECASE
        )
        for p in person_pronouns:
            results.append((p, "person"))

        # "it" is too ambiguous — skip
        return results
