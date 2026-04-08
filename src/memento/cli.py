"""Command-line interface for Memento."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memento.config import MementoConfig
from memento.db import Database
from memento.graph_store import GraphStore


def get_store(args) -> GraphStore:
    """Create a GraphStore from CLI args."""
    db_path = getattr(args, "db", None) or MementoConfig().db_path
    db = Database(db_path)
    return GraphStore(db)


def cmd_entities(store, args):
    from memento.models import EntityType

    type_filter = EntityType(args.type) if args.type else None
    entities = store.search_entities(type=type_filter)

    if not entities:
        print("No entities found.")
        return

    print(f"{'ID':<40} {'Type':<15} {'Name':<30} {'Confidence'}")
    print("-" * 95)
    for e in entities:
        print(f"{e.id:<40} {e.type.value:<15} {e.name:<30} {e.confidence:.2f}")


def cmd_entity(store, args):
    entity = store.get_entity(args.id)

    if entity is None:
        print(f"Entity not found: {args.id}")
        sys.exit(1)

    print(f"Entity: {entity.name}")
    print(f"  ID:         {entity.id}")
    print(f"  Type:       {entity.type.value}")
    print(f"  Confidence: {entity.confidence:.2f}")
    print(f"  Created:    {entity.created_at}")
    print(f"  Last seen:  {entity.last_seen}")
    print(f"  Access ct:  {entity.access_count}")

    if entity.aliases:
        print(f"  Aliases:    {', '.join(entity.aliases)}")

    if entity.properties:
        print("\n  Properties:")
        for key, prop in entity.properties.items():
            conf = f" (confidence: {prop.confidence:.2f})" if prop.confidence < 1.0 else ""
            print(f"    {key}: {prop.value}{conf}")

    rels = store.get_relationships(entity.id)
    if rels:
        print("\n  Relationships:")
        for rel in rels:
            direction = "→" if rel.source_id == entity.id else "←"
            other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            other = store.get_entity(other_id)
            other_name = other.name if other else other_id[:8]
            active = "" if rel.valid_to is None else f" (ended: {rel.valid_to})"
            print(f"    {direction} [{rel.type}] {other_name}{active}")


def cmd_history(store, args):
    history = store.get_property_history(args.id, args.key)

    if not history:
        print(f"No history found for {args.id}:{args.key}")
        return

    print(f"History for property '{args.key}':")
    print(f"{'Value':<30} {'As Of':<28} {'Recorded At':<28} {'Conf':<6} {'Status'}")
    print("-" * 120)
    for prop in history:
        status = "current" if prop.superseded_by_id is None else "superseded"
        print(
            f"{str(prop.value):<30} {prop.as_of:<28} {prop.recorded_at:<28} {prop.confidence:<6.2f} {status}"
        )


def cmd_snapshot(store, args):
    entity = store.point_in_time_snapshot(args.id, args.as_of)

    if entity is None:
        print(f"Entity not found: {args.id}")
        sys.exit(1)

    print(f"Snapshot of {entity.name} as of {args.as_of}:")
    if entity.properties:
        for key, prop in entity.properties.items():
            print(f"  {key}: {prop.value}")
    else:
        print("  (no properties known at this time)")


def cmd_merge(store, args):
    result = store.merge_entities(args.id_a, args.id_b, reason="cli_merge")
    print(f"Merged entities.")
    print(f"  Survivor:     {result['survivor_id']}")
    print(f"  Absorbed:     {result['absorbed_id']}")
    print(f"  Edges moved:  {result['edges_re_parented']}")
    print(f"  Props moved:  {result['properties_moved']}")
    print(f"  Merge log ID: {result['merge_log_id']}")


def cmd_undo_merge(store, args):
    result = store.split_entity(args.merge_log_id)
    print(f"Undid merge.")
    print(f"  Survivor:  {result['survivor_id']}")
    print(f"  Restored:  {result['restored_id']}")


def cmd_stats(store, args):
    s = store.stats()
    print("Graph Statistics:")
    print(f"  Entities:          {s['node_count']}")
    print(f"  Relationships:     {s['edge_count']}")
    print(f"  Active properties: {s['active_property_count']}")
    print(f"  Avg confidence:    {s['avg_confidence']:.2f}")


def cmd_consolidate(store, args):
    from memento.consolidation import ConsolidationEngine

    engine = ConsolidationEngine(store)
    result = engine.run_full()
    print("Consolidation complete:")
    print(f"  Facts decayed:           {result.facts_decayed}")
    print(f"  Redundancies merged:     {result.redundancies_merged}")
    print(f"  Orphans archived:        {result.orphans_archived}")
    print(f"  Contradictions resolved: {result.contradictions_resolved}")
    print(f"  Centrality entries:      {result.centrality_entries}")


def cmd_export(store, args):
    from memento.privacy import PrivacyLayer

    privacy = PrivacyLayer(store)
    export = privacy.export_entity_data(args.id)
    if export is None:
        print(f"Entity not found: {args.id}")
        sys.exit(1)
    print(export.to_json())


def cmd_audit(store, args):
    from memento.privacy import PrivacyLayer
    import json

    privacy = PrivacyLayer(store)
    chain = privacy.audit_belief(args.id, args.key)
    if not chain.chain:
        print(f"No history found for {args.id}:{args.key}")
        return

    print(f"Belief chain for '{args.key}' on entity {args.id[:12]}...")
    for entry in chain.chain:
        status = "current" if entry["superseded_by"] is None else "superseded"
        print(f"\n  [{status}] {entry['value']}")
        print(f"    Recorded: {entry['recorded_at'][:19]}")
        print(f"    Confidence: {entry['confidence']:.2f}")
        if entry.get("source"):
            src = entry["source"]
            print(f"    Source: conv={src['conversation_id'][:12]}..., turn={src['turn_number']}")
            if src["verbatim"]:
                print(f"    Verbatim: \"{src['verbatim'][:100]}\"")


def cmd_delete(store, args):
    from memento.privacy import PrivacyLayer

    privacy = PrivacyLayer(store)
    if args.hard:
        receipt = privacy.delete_entity_cascade(args.id)
        if receipt is None:
            print(f"Entity not found: {args.id}")
            sys.exit(1)
        print(f"Hard deleted: {receipt.entity_name}")
        print(f"  Items deleted: {receipt.items_deleted}")
        print(f"  Content hash: {receipt.content_hash}")
        print(f"  Deleted at:   {receipt.deleted_at}")
    else:
        store.db.execute(
            "UPDATE entities SET archived = 1 WHERE id = ?", (args.id,)
        )
        store.db.conn.commit()
        print(f"Archived entity: {args.id}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="memento",
        description="Memento — Temporal knowledge graph memory system",
    )
    parser.add_argument(
        "--db", type=str, default=None, help="Path to SQLite database"
    )

    sub = parser.add_subparsers(dest="command")

    # entities
    p = sub.add_parser("entities", help="List entities")
    p.add_argument("--type", type=str, default=None, help="Filter by entity type")

    # entity
    p = sub.add_parser("entity", help="Show entity details")
    p.add_argument("id", help="Entity ID")

    # history
    p = sub.add_parser("history", help="Show property history")
    p.add_argument("id", help="Entity ID")
    p.add_argument("key", help="Property key")

    # snapshot
    p = sub.add_parser("snapshot", help="Point-in-time entity snapshot")
    p.add_argument("id", help="Entity ID")
    p.add_argument("--as-of", required=True, help="ISO 8601 timestamp")

    # merge
    p = sub.add_parser("merge", help="Merge two entities")
    p.add_argument("id_a", help="First entity ID")
    p.add_argument("id_b", help="Second entity ID")

    # undo-merge
    p = sub.add_parser("undo-merge", help="Undo an entity merge")
    p.add_argument("merge_log_id", help="Merge log entry ID")

    # stats
    sub.add_parser("stats", help="Show graph statistics")

    # consolidate
    sub.add_parser("consolidate", help="Run consolidation pass")

    # export
    p = sub.add_parser("export", help="Export all entity data (JSON)")
    p.add_argument("id", help="Entity ID")

    # audit
    p = sub.add_parser("audit", help="Trace belief provenance chain")
    p.add_argument("id", help="Entity ID")
    p.add_argument("key", help="Property key")

    # delete
    p = sub.add_parser("delete", help="Delete an entity")
    p.add_argument("id", help="Entity ID")
    p.add_argument("--hard", action="store_true", help="Hard delete (irreversible, with compliance receipt)")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "entities": cmd_entities,
        "entity": cmd_entity,
        "history": cmd_history,
        "snapshot": cmd_snapshot,
        "merge": cmd_merge,
        "undo-merge": cmd_undo_merge,
        "stats": cmd_stats,
        "consolidate": cmd_consolidate,
        "export": cmd_export,
        "audit": cmd_audit,
        "delete": cmd_delete,
    }

    store = get_store(args)
    try:
        # Pass store to command so we control the lifecycle
        commands[args.command](store, args)
    finally:
        store.db.close()


if __name__ == "__main__":
    main()
