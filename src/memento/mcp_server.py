"""MCP server exposing Memento memory tools via the full MemoryStore API."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from memento.models import EntityType

# Lazy-initialized store
_store = None

mcp = FastMCP(
    "memento",
    instructions=(
        "Long-term memory system for AI agents. "
        "Use memory_ingest to store facts from conversations. "
        "Use memory_recall to retrieve relevant context before responding. "
        "The system extracts entities, resolves them against a knowledge graph, "
        "detects contradictions, and assembles composed briefings."
    ),
)


def _get_store():
    """Lazily initialize the MemoryStore on first use."""
    global _store
    if _store is not None:
        return _store

    from memento.memory_store import MemoryStore

    _store = MemoryStore()
    return _store


@mcp.tool()
def memory_ingest(text: str, source_type: str = "conversation") -> str:
    """Store text in memory. Extracts entities, resolves them against the knowledge graph, detects contradictions, and stores the raw text for later recall.

    Args:
        text: The text to remember (conversation turn, note, fact, etc.)
        source_type: Type of source ("conversation", "note", "document")

    Returns:
        Summary of what was ingested.
    """
    store = _get_store()
    result = store.ingest(text, source_type=source_type)

    parts = [f"Stored text ({len(text)} chars)"]
    if result.entities_created:
        parts.append(f"New entities: {', '.join(result.entities_created)}")
    if result.entities_resolved:
        parts.append(f"Matched to existing: {', '.join(result.entities_resolved)}")
    if result.relationships_created:
        parts.append(f"Relationships: {result.relationships_created}")
    if result.conflicts_detected:
        parts.append(f"Conflicts detected: {result.conflicts_detected}")

    return ". ".join(parts)


@mcp.tool()
def memory_recall(query: str, token_budget: int = 2000) -> str:
    """Recall relevant memories for a query. Returns a composed briefing from the knowledge graph and raw conversation text.

    Args:
        query: What you want to know or the context for retrieval.
        token_budget: Maximum tokens for the returned context (default 2000).

    Returns:
        Relevant memories assembled from the knowledge graph and raw text.
    """
    store = _get_store()
    memory = store.recall(query, token_budget=token_budget)
    return memory.text


@mcp.tool()
def memory_recall_as_of(query: str, as_of: str, token_budget: int = 2000) -> str:
    """Recall what was known at a specific point in time.

    Args:
        query: What you want to know.
        as_of: ISO 8601 timestamp for point-in-time retrieval (e.g., "2025-06-01T00:00:00Z").
        token_budget: Maximum tokens for the returned context.

    Returns:
        Memories as they were known at the specified time.
    """
    store = _get_store()
    memory = store.recall(query, token_budget=token_budget, as_of=as_of)
    return memory.text


@mcp.tool()
def memory_correct(entity_id: str, property_key: str, new_value: str, reason: str = "") -> str:
    """Correct a fact about an entity. Use when the system has wrong information.

    Args:
        entity_id: The ID of the entity to correct.
        property_key: The property to correct (e.g., "title", "location").
        new_value: The correct value.
        reason: Why this correction is being made.

    Returns:
        Confirmation of the correction.
    """
    store = _get_store()
    store.correct(entity_id, property_key, new_value, reason)
    return f"Corrected {property_key} = {new_value} for entity {entity_id}"


@mcp.tool()
def memory_forget(entity_id: str) -> str:
    """Forget an entity (soft delete — archives it from retrieval).

    Args:
        entity_id: The ID of the entity to forget.

    Returns:
        Confirmation.
    """
    store = _get_store()
    store.forget(entity_id=entity_id)
    return f"Entity {entity_id} archived"


@mcp.tool()
def memory_merge(entity_a_id: str, entity_b_id: str) -> str:
    """Merge two entities that are the same real-world thing.

    Args:
        entity_a_id: First entity ID (will be the survivor).
        entity_b_id: Second entity ID (will be absorbed).

    Returns:
        Summary of the merge operation.
    """
    store = _get_store()
    result = store.merge(entity_a_id, entity_b_id)
    return (
        f"Merged into {result['survivor_id']}. "
        f"Edges moved: {result['edges_re_parented']}, "
        f"Properties moved: {result['properties_moved']}"
    )


@mcp.tool()
def memory_conflicts() -> str:
    """List all unresolved contradictions in the knowledge graph.

    Returns:
        List of conflicts with details.
    """
    store = _get_store()
    conflicts = store.conflicts()
    if not conflicts:
        return "No unresolved conflicts."

    lines = []
    for c in conflicts:
        lines.append(f"- **{c.property_key}** on entity {c.entity_id[:12]}... (since {c.created_at[:10]})")
    return "\n".join(lines)


@mcp.tool()
def memory_entities(type_filter: str = "") -> str:
    """List all known entities in the knowledge graph.

    Args:
        type_filter: Optional filter (person, organization, project, location, concept, event).

    Returns:
        List of known entities with their types and IDs.
    """
    store = _get_store()
    etype = EntityType(type_filter) if type_filter else None
    entities = store.entity_list(type=etype)

    if not entities:
        return "No entities found."

    lines = []
    for e in entities:
        aliases = f" (aka {', '.join(e.aliases)})" if e.aliases else ""
        lines.append(f"- **{e.name}** [{e.type.value}] `{e.id}`{aliases}")
    return "\n".join(lines)


@mcp.tool()
def memory_entity(entity_id: str) -> str:
    """Get detailed information about a specific entity.

    Args:
        entity_id: The ID of the entity to look up.

    Returns:
        Detailed entity profile including properties and relationships.
    """
    store = _get_store()
    entity = store.recall_entity(entity_id)

    if entity is None:
        return f"Entity not found: {entity_id}"

    lines = [
        f"# {entity.name}",
        f"Type: {entity.type.value}",
        f"ID: {entity.id}",
        f"Confidence: {entity.confidence:.0%}",
        f"Created: {entity.created_at[:10]}",
        f"Last seen: {entity.last_seen[:10]}",
    ]

    if entity.aliases:
        lines.append(f"Aliases: {', '.join(entity.aliases)}")

    if entity.properties:
        lines.append("\n## Properties")
        for key, prop in entity.properties.items():
            conf = f" *(confidence: {prop.confidence:.0%})*" if prop.confidence < 0.9 else ""
            lines.append(f"- {key}: {prop.value}{conf}")

    rels = store.graph.get_relationships(entity.id)
    if rels:
        lines.append("\n## Relationships")
        for rel in rels:
            other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            other = store.graph.get_entity(other_id)
            other_name = other.name if other else other_id[:12]
            direction = "→" if rel.source_id == entity.id else "←"
            active = "" if rel.valid_to is None else f" *(ended {rel.valid_to[:10]})*"
            lines.append(f"- {direction} [{rel.type}] {other_name}{active}")

    return "\n".join(lines)


@mcp.tool()
def memory_health() -> str:
    """Get health statistics for the knowledge graph.

    Returns:
        Node count, edge count, confidence stats, and conflict count.
    """
    store = _get_store()
    h = store.health()
    return (
        f"Entities: {h.node_count}\n"
        f"Relationships: {h.edge_count}\n"
        f"Active properties: {h.active_property_count}\n"
        f"Avg confidence: {h.avg_confidence:.0%}\n"
        f"Unresolved conflicts: {h.unresolved_conflicts}"
    )


def main():
    """Run the MCP server via stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
