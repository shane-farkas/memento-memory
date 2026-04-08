"""Tests for LLM-powered entity and relation extraction."""

from __future__ import annotations

import pytest

from memento.extraction import (
    EntityExtractor,
    ExtractedEntity,
    RelationExtractor,
    _parse_json_response,
)
from memento.models import EntityType
from tests.conftest import FakeLLMClient


# ── JSON Parsing ─────────────────────────────────────────────────


def test_parse_json_plain_array():
    text = '[{"name": "John", "type": "person", "properties": {}}]'
    result = _parse_json_response(text)
    assert len(result) == 1
    assert result[0]["name"] == "John"


def test_parse_json_with_code_fence():
    text = '```json\n[{"name": "John", "type": "person", "properties": {}}]\n```'
    result = _parse_json_response(text)
    assert len(result) == 1
    assert result[0]["name"] == "John"


def test_parse_json_with_surrounding_text():
    text = 'Here are the entities:\n[{"name": "John", "type": "person", "properties": {}}]\nDone!'
    result = _parse_json_response(text)
    assert len(result) == 1


def test_parse_json_empty_array():
    assert _parse_json_response("[]") == []


def test_parse_json_invalid():
    assert _parse_json_response("this is not json at all") == []


def test_parse_json_empty_string():
    assert _parse_json_response("") == []


# ── Entity Extraction ────────────────────────────────────────────


def test_extract_entities_basic():
    llm = FakeLLMClient(responses=[
        '[{"name": "John Smith", "type": "person", "properties": {"title": "Director of Sales"}},'
        '{"name": "Acme Corp", "type": "organization", "properties": {}}]'
    ])
    extractor = EntityExtractor(model="test", llm_client=llm)
    entities = extractor.extract("John Smith works at Acme Corp as Director of Sales.")

    assert len(entities) == 2
    john = next(e for e in entities if e.name == "John Smith")
    assert john.type == EntityType.PERSON
    assert john.properties["title"] == "Director of Sales"


def test_extract_entities_with_code_fence():
    llm = FakeLLMClient(responses=[
        '```json\n[{"name": "Project Falcon", "type": "project", "properties": {"launch_date": "April"}}]\n```'
    ])
    extractor = EntityExtractor(model="test", llm_client=llm)
    entities = extractor.extract("Project Falcon launches in April.")

    assert len(entities) == 1
    assert entities[0].name == "Project Falcon"
    assert entities[0].type == EntityType.PROJECT


def test_extract_entities_empty_text():
    llm = FakeLLMClient()
    extractor = EntityExtractor(model="test", llm_client=llm)
    assert extractor.extract("") == []


def test_extract_entities_no_entities_found():
    llm = FakeLLMClient(responses=["[]"])
    extractor = EntityExtractor(model="test", llm_client=llm)
    assert extractor.extract("The weather is nice today.") == []


def test_extract_entities_llm_error():
    class ErrorLLM:
        def complete(self, **kwargs):
            raise Exception("API error")
    extractor = EntityExtractor(model="test", llm_client=ErrorLLM())
    assert extractor.extract("Some text") == []


def test_extract_entities_unknown_type_fallback():
    llm = FakeLLMClient(responses=[
        '[{"name": "Quantum Computing", "type": "topic", "properties": {}}]'
    ])
    extractor = EntityExtractor(model="test", llm_client=llm)
    entities = extractor.extract("We discussed quantum computing.")
    assert entities[0].type == EntityType.CONCEPT


def test_extract_entities_multiple():
    llm = FakeLLMClient(responses=[
        '['
        '{"name": "John Smith", "type": "person", "properties": {}},'
        '{"name": "Jane Doe", "type": "person", "properties": {}},'
        '{"name": "Acme Corp", "type": "organization", "properties": {}},'
        '{"name": "Austin", "type": "location", "properties": {}},'
        '{"name": "Project Alpha", "type": "project", "properties": {}}'
        ']'
    ])
    extractor = EntityExtractor(model="test", llm_client=llm)
    entities = extractor.extract("John and Jane work at Acme in Austin on Alpha.")
    assert len(entities) == 5


# ── Relation Extraction ──────────────────────────────────────────


def test_extract_relations_basic():
    llm = FakeLLMClient(responses=[
        '[{"source": "John Smith", "target": "Acme Corp", "type": "works_at", "properties": {}}]'
    ])
    entities = [
        ExtractedEntity("John Smith", EntityType.PERSON),
        ExtractedEntity("Acme Corp", EntityType.ORGANIZATION),
    ]
    extractor = RelationExtractor(model="test", llm_client=llm)
    relations = extractor.extract("John Smith works at Acme Corp.", entities)
    assert len(relations) == 1
    assert relations[0].type == "works_at"


def test_extract_relations_empty():
    llm = FakeLLMClient()
    extractor = RelationExtractor(model="test", llm_client=llm)
    assert extractor.extract("Some text", []) == []
    assert extractor.extract("", [ExtractedEntity("John", EntityType.PERSON)]) == []


def test_extract_relations_normalizes_type():
    llm = FakeLLMClient(responses=[
        '[{"source": "John", "target": "Acme", "type": "Works At", "properties": {}}]'
    ])
    entities = [
        ExtractedEntity("John", EntityType.PERSON),
        ExtractedEntity("Acme", EntityType.ORGANIZATION),
    ]
    extractor = RelationExtractor(model="test", llm_client=llm)
    relations = extractor.extract("John works at Acme.", entities)
    assert relations[0].type == "works_at"


# ── Integration: Extraction → Graph ──────────────────────────────


def test_extraction_to_graph_integration():
    from memento.db import Database
    from memento.graph_store import GraphStore
    from memento.models import SourceRef

    llm = FakeLLMClient(responses=[
        '[{"name": "John Smith", "type": "person", "properties": {"title": "Director"}},'
        '{"name": "Acme Corp", "type": "organization", "properties": {}}]',
        '[{"source": "John Smith", "target": "Acme Corp", "type": "works_at", "properties": {}}]',
    ])

    db = Database(":memory:")
    graph = GraphStore(db)
    entity_extractor = EntityExtractor(model="test", llm_client=llm)
    relation_extractor = RelationExtractor(model="test", llm_client=llm)

    text = "John Smith works at Acme Corp as Director."
    extracted_entities = entity_extractor.extract(text)
    assert len(extracted_entities) == 2

    source_ref = SourceRef(conversation_id="conv-1", turn_number=1, verbatim=text)
    entity_map = {}
    for ee in extracted_entities:
        entity = graph.create_entity(ee.name, ee.type, source_ref=source_ref)
        entity_map[ee.name] = entity
        for key, value in ee.properties.items():
            graph.set_property(entity.id, key, value, source_ref=source_ref)

    extracted_relations = relation_extractor.extract(text, extracted_entities)
    assert len(extracted_relations) == 1

    for er in extracted_relations:
        source_entity = entity_map.get(er.source)
        target_entity = entity_map.get(er.target)
        if source_entity and target_entity:
            graph.create_relationship(source_entity.id, target_entity.id, er.type, source_ref=source_ref)

    john = graph.search_entities(name="John Smith")[0]
    full_john = graph.get_entity(john.id)
    assert full_john.properties["title"].value == "Director"
    rels = graph.get_relationships(john.id, type="works_at")
    assert len(rels) == 1

    db.close()
