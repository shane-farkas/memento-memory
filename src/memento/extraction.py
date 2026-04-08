"""LLM-powered entity and relation extraction from raw text."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from memento.models import EntityType

logger = logging.getLogger(__name__)

ENTITY_TYPES = [t.value for t in EntityType]

ENTITY_EXTRACTION_PROMPT = """Extract all entities mentioned in the following text. For each entity, provide:
- name: The most complete name mentioned
- type: One of {entity_types}
- properties: Key facts mentioned about this entity (as key-value pairs)

Return a JSON array. If no entities are found, return an empty array [].

Examples:
Text: "John Smith works at Acme Corp as Director of Sales. He started in March 2026."
Output: [
  {{"name": "John Smith", "type": "person", "properties": {{"title": "Director of Sales", "start_date": "March 2026"}}}},
  {{"name": "Acme Corp", "type": "organization", "properties": {{}}}}
]

Text: "The Project Falcon launch is scheduled for April in Austin."
Output: [
  {{"name": "Project Falcon", "type": "project", "properties": {{"launch_date": "April", "launch_location": "Austin"}}}},
  {{"name": "Austin", "type": "location", "properties": {{}}}}
]

Now extract entities from this text:
{text}"""

RELATION_EXTRACTION_PROMPT = """Given the following text and extracted entities, identify relationships between them.

For each relationship, provide:
- source: Name of the source entity
- target: Name of the target entity
- type: A short relationship label (e.g., "works_at", "knows", "located_in", "manages", "invested_in", "part_of", "attended")
- properties: Any additional details about the relationship (as key-value pairs)

Return a JSON array. If no relationships are found, return an empty array [].

Entities found:
{entities}

Text:
{text}"""


@dataclass
class ExtractedEntity:
    """An entity extracted from text by the LLM."""

    name: str
    type: EntityType
    properties: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.8  # Default extraction confidence


@dataclass
class ExtractedRelation:
    """A relationship extracted from text by the LLM."""

    source: str
    target: str
    type: str
    properties: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.7  # Relation extraction is less reliable


def _parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse LLM response as JSON: %s", text[:200])
        return []


class EntityExtractor:
    """Extracts entities from text using an LLM."""

    def __init__(self, model: str = "", llm_client=None) -> None:
        self.model = model
        self._llm = llm_client

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text using a single LLM call."""
        if not text.strip():
            return []

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            text=text,
        )

        try:
            content = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error("Entity extraction LLM call failed: %s", e)
            return []

        raw_entities = _parse_json_response(content)
        return self._parse_entities(raw_entities)

    def _parse_entities(self, raw: list[dict]) -> list[ExtractedEntity]:
        entities = []
        for item in raw:
            name = item.get("name", "").strip()
            type_str = item.get("type", "").strip().lower()
            if not name:
                continue

            try:
                etype = EntityType(type_str)
            except ValueError:
                etype = EntityType.CONCEPT  # Fallback for unknown types

            properties = item.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=etype,
                    properties={k: str(v) for k, v in properties.items() if v},
                )
            )
        return entities


class RelationExtractor:
    """Extracts relationships between entities from text using an LLM."""

    def __init__(self, model: str = "", llm_client=None) -> None:
        self.model = model
        self._llm = llm_client

    def extract(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedRelation]:
        """Extract relationships from text given known entities."""
        if not text.strip() or not entities:
            return []

        entity_list = "\n".join(
            f"- {e.name} ({e.type.value})" for e in entities
        )

        prompt = RELATION_EXTRACTION_PROMPT.format(
            entities=entity_list,
            text=text,
        )

        try:
            content = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error("Relation extraction LLM call failed: %s", e)
            return []

        raw_relations = _parse_json_response(content)
        return self._parse_relations(raw_relations, entities)

    def _parse_relations(
        self, raw: list[dict], entities: list[ExtractedEntity]
    ) -> list[ExtractedRelation]:
        entity_names = {e.name.lower() for e in entities}
        relations = []

        for item in raw:
            source = item.get("source", "").strip()
            target = item.get("target", "").strip()
            rel_type = item.get("type", "").strip().lower().replace(" ", "_")

            if not source or not target or not rel_type:
                continue

            # Validate that source and target match known entities
            if (
                source.lower() not in entity_names
                or target.lower() not in entity_names
            ):
                # Try partial matching
                source_match = self._fuzzy_match_entity(source, entities)
                target_match = self._fuzzy_match_entity(target, entities)
                if source_match:
                    source = source_match
                if target_match:
                    target = target_match

            properties = item.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    type=rel_type,
                    properties={k: str(v) for k, v in properties.items() if v},
                )
            )
        return relations

    def _fuzzy_match_entity(
        self, name: str, entities: list[ExtractedEntity]
    ) -> str | None:
        """Try to match a name to a known entity (case-insensitive, partial)."""
        name_lower = name.lower()
        for e in entities:
            if name_lower in e.name.lower() or e.name.lower() in name_lower:
                return e.name
        return None
