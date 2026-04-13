"""Web viewer for the Memento knowledge graph.

Serves a single-page app with:
- Entity list with search/filter
- Entity detail view (properties, relationships, history)
- Force-directed graph visualization (d3.js)
- Timeline view (when facts were learned vs when they were true)

Usage:
    memento-web                  # http://localhost:8765
    memento-web --port 9000      # custom port
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from memento.config import MementoConfig
from memento.memory_store import MemoryStore
from memento.models import EntityType

# Content Security Policy — forbids inline scripts, permits d3 from CDN.
# Once d3 is bundled locally we can drop the d3js.org allowance.
_CSP = (
    "default-src 'none'; "
    "script-src 'self' https://d3js.org; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response


app = FastAPI(title="Memento Knowledge Graph Viewer")
app.add_middleware(_SecurityHeadersMiddleware)
# Defeats DNS-rebinding: only accepts loopback Host headers by default.
# Overridden at startup in main() when --unsafe-network is explicitly passed.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "::1"],
)

_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


# ── API Endpoints ───────────────────────────────────────────


@app.get("/api/health")
def api_health():
    store = _get_store()
    h = store.health()
    return {
        "node_count": h.node_count,
        "edge_count": h.edge_count,
        "active_property_count": h.active_property_count,
        "avg_confidence": round(h.avg_confidence, 3),
        "unresolved_conflicts": h.unresolved_conflicts,
    }


@app.get("/api/entities")
def api_entities(
    type: str | None = None,
    q: str | None = None,
):
    store = _get_store()
    etype = EntityType(type) if type else None
    entities = store.entity_list(type=etype)

    if q:
        q_lower = q.lower()
        entities = [
            e for e in entities
            if q_lower in e.name.lower()
            or any(q_lower in a.lower() for a in e.aliases)
        ]

    return [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type.value,
            "aliases": e.aliases,
            "confidence": round(e.confidence, 3),
            "created_at": e.created_at,
            "last_seen": e.last_seen,
            "property_count": len(e.properties),
        }
        for e in entities
    ]


@app.get("/api/entities/{entity_id}")
def api_entity_detail(entity_id: str):
    store = _get_store()
    entity = store.recall_entity(entity_id, depth=1)
    if not entity:
        return {"error": "Entity not found"}

    properties = {}
    for key, prop in entity.properties.items():
        properties[key] = {
            "value": prop.value,
            "confidence": round(prop.confidence, 3),
            "as_of": prop.as_of,
            "recorded_at": prop.recorded_at,
        }

    rels = store.graph.get_relationships(entity_id)
    relationships = []
    for rel in rels:
        other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
        other = store.graph.get_entity(other_id)
        relationships.append({
            "id": rel.id,
            "type": rel.type,
            "direction": "outgoing" if rel.source_id == entity_id else "incoming",
            "other_id": other_id,
            "other_name": other.name if other else other_id[:12],
            "other_type": other.type.value if other else "unknown",
            "valid_from": rel.valid_from,
            "valid_to": rel.valid_to,
            "confidence": round(rel.confidence, 3),
        })

    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases,
        "confidence": round(entity.confidence, 3),
        "created_at": entity.created_at,
        "last_seen": entity.last_seen,
        "access_count": entity.access_count,
        "properties": properties,
        "relationships": relationships,
    }


@app.get("/api/entities/{entity_id}/history/{property_key}")
def api_property_history(entity_id: str, property_key: str):
    store = _get_store()
    history = store.graph.get_property_history(entity_id, property_key)
    return [
        {
            "id": pv.id,
            "value": pv.value,
            "as_of": pv.as_of,
            "recorded_at": pv.recorded_at,
            "confidence": round(pv.confidence, 3),
            "superseded_by_id": pv.superseded_by_id,
        }
        for pv in history
    ]


@app.get("/api/graph")
def api_graph(
    center: str | None = None,
    hops: int = 2,
):
    """Return nodes and links for d3 force graph."""
    store = _get_store()
    hops = max(1, min(hops, 4))  # Clamp to prevent recursive CTE explosion

    if center:
        # Subgraph around a specific entity
        center_entity = store.graph.get_entity(center)
        if not center_entity:
            return {"nodes": [], "links": []}

        neighbors = store.graph.get_neighbors(center, max_hops=hops)
        entity_list = [center_entity] + neighbors
        entity_ids = {e.id for e in entity_list}
    else:
        # Full graph (capped for performance)
        entity_list = store.entity_list()[:200]
        entity_ids = {e.id for e in entity_list}

    nodes = [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type.value,
            "confidence": round(e.confidence, 3),
        }
        for e in entity_list
    ]

    links = []
    seen_rels = set()
    for e in entity_list:
        rels = store.graph.get_relationships(e.id)
        for rel in rels:
            if rel.id in seen_rels:
                continue
            if rel.source_id in entity_ids and rel.target_id in entity_ids:
                seen_rels.add(rel.id)
                links.append({
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.type,
                    "confidence": round(rel.confidence, 3),
                    "active": rel.valid_to is None,
                })

    return {"nodes": nodes, "links": links}


@app.get("/api/conflicts")
def api_conflicts():
    store = _get_store()
    conflicts = store.conflicts()
    results = []
    for c in conflicts:
        entity = store.graph.get_entity(c.entity_id)
        val_a = store.graph.get_property(c.entity_id, c.property_key)
        results.append({
            "id": c.id,
            "entity_id": c.entity_id,
            "entity_name": entity.name if entity else c.entity_id[:12],
            "property_key": c.property_key,
            "status": c.status.value if hasattr(c.status, 'value') else c.status,
            "created_at": c.created_at,
        })
    return results


@app.get("/api/timeline")
def api_timeline(entity_id: str | None = None):
    """Return timeline events: facts learned and when they were true."""
    store = _get_store()

    if entity_id:
        entities = [store.graph.get_entity(entity_id)]
        entities = [e for e in entities if e]
    else:
        entities = store.entity_list()[:100]

    events = []
    for entity in entities:
        for key, prop in entity.properties.items():
            events.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.type.value,
                "property_key": key,
                "value": prop.value,
                "as_of": prop.as_of,
                "recorded_at": prop.recorded_at,
                "confidence": round(prop.confidence, 3),
                "type": "property",
            })

    rels = []
    seen = set()
    for entity in entities:
        for rel in store.graph.get_relationships(entity.id):
            if rel.id not in seen:
                seen.add(rel.id)
                source = store.graph.get_entity(rel.source_id)
                target = store.graph.get_entity(rel.target_id)
                events.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "entity_type": entity.type.value,
                    "property_key": rel.type,
                    "value": f"{source.name if source else '?'} -> {target.name if target else '?'}",
                    "as_of": rel.valid_from,
                    "recorded_at": rel.valid_from,
                    "confidence": round(rel.confidence, 3),
                    "type": "relationship",
                })

    events.sort(key=lambda e: e["recorded_at"], reverse=True)
    return events[:500]


# ── Frontend ────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index():
    return _FRONTEND_HTML


# ── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Memento Knowledge Graph Viewer")
    parser.add_argument("--port", type=int, default=8766, help="Port (default: 8766)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument(
        "--unsafe-network",
        action="store_true",
        help="Explicitly allow binding to a non-loopback address. The web "
             "viewer has no authentication; only use this on a trusted "
             "network (e.g. Tailscale, LAN behind firewall).",
    )
    args = parser.parse_args()

    import sys
    loopback = args.host in ("127.0.0.1", "localhost", "::1", "[::1]")
    if not loopback and not args.unsafe_network:
        print(
            "\n  ERROR: --host is a non-loopback address "
            f"({args.host}) but --unsafe-network was not passed.\n"
            "\n  The web viewer has no authentication. Exposing it on a "
            "network interface\n  means anyone who can reach that interface "
            "can read your entire knowledge\n  graph. If you understand the "
            "risk (e.g. Tailscale-only network), re-run with:\n\n"
            f"    memento-web --host {args.host} --unsafe-network\n"
        )
        sys.exit(2)

    # If user opted in to non-loopback, widen the TrustedHost allowlist to
    # include the bound address (DNS-rebinding protection relaxed by consent).
    if args.unsafe_network and not loopback:
        for mw in app.user_middleware:
            if mw.cls is TrustedHostMiddleware:
                mw.kwargs["allowed_hosts"] = ["*"]
                break
        print(
            "\n  WARNING: Binding to non-loopback address without authentication.\n"
            "  Anyone who can reach this host:port can read your knowledge graph.\n"
        )

    import uvicorn
    print(f"\n  Memento Knowledge Graph Viewer")
    print(f"  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ── Embedded Frontend HTML ──────────────────────────────────

_FRONTEND_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memento — Knowledge Graph Viewer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #f0883e; --purple: #d2a8ff;
}
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }
a { color: var(--accent); text-decoration: none; cursor: pointer; }
a:hover { text-decoration: underline; }

.app { display: grid; grid-template-columns: 320px 1fr; grid-template-rows: 56px 1fr; height: 100vh; }

/* Header */
.header { grid-column: 1 / -1; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 20px; gap: 20px; }
.header h1 { font-size: 16px; font-weight: 600; }
.header .stats { display: flex; gap: 16px; font-size: 13px; color: var(--text-muted); }
.header .stats span { color: var(--accent); font-weight: 600; }
.header nav { margin-left: auto; display: flex; gap: 4px; }
.header nav button { background: none; border: 1px solid var(--border); color: var(--text-muted); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.header nav button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.header nav button:hover:not(.active) { border-color: var(--text-muted); color: var(--text); }

/* Sidebar */
.sidebar { background: var(--surface); border-right: 1px solid var(--border); overflow-y: auto; }
.sidebar .search { padding: 12px; position: sticky; top: 0; background: var(--surface); z-index: 1; }
.sidebar input { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 13px; outline: none; }
.sidebar input:focus { border-color: var(--accent); }
.sidebar .filters { padding: 0 12px 8px; display: flex; gap: 4px; flex-wrap: wrap; }
.sidebar .filters button { background: none; border: 1px solid var(--border); color: var(--text-muted); padding: 2px 8px; border-radius: 12px; cursor: pointer; font-size: 11px; }
.sidebar .filters button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.entity-list { list-style: none; }
.entity-item { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid var(--border); }
.entity-item:hover { background: var(--bg); }
.entity-item.selected { background: var(--bg); border-left: 3px solid var(--accent); }
.entity-item .name { font-size: 14px; font-weight: 500; }
.entity-item .meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.type-badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
.type-person { background: #1f3d5c; color: #58a6ff; }
.type-organization { background: #2a1f3d; color: #d2a8ff; }
.type-project { background: #1f3d2a; color: #3fb950; }
.type-location { background: #3d2a1f; color: #f0883e; }
.type-concept { background: #3d3d1f; color: #e3b341; }
.type-event { background: #3d1f1f; color: #f85149; }

/* Main content */
.main { overflow-y: auto; padding: 24px; }

/* Detail view */
.detail h2 { font-size: 22px; margin-bottom: 4px; }
.detail .entity-type { color: var(--text-muted); font-size: 14px; margin-bottom: 16px; }
.detail .aliases { color: var(--text-muted); font-size: 13px; margin-bottom: 16px; }
.detail section { margin-bottom: 24px; }
.detail section h3 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.detail table { width: 100%; border-collapse: collapse; }
.detail td, .detail th { text-align: left; padding: 6px 12px; font-size: 13px; border-bottom: 1px solid var(--border); }
.detail th { color: var(--text-muted); font-weight: 500; width: 160px; }
.confidence { font-size: 11px; padding: 1px 6px; border-radius: 8px; }
.confidence-high { background: #1f3d2a; color: var(--green); }
.confidence-mid { background: #3d3d1f; color: var(--orange); }
.confidence-low { background: #3d1f1f; color: var(--red); }
.rel-row { cursor: pointer; }
.rel-row:hover { background: var(--bg); }
.history-btn { background: none; border: 1px solid var(--border); color: var(--text-muted); padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; }
.history-btn:hover { border-color: var(--accent); color: var(--accent); }

/* Graph view */
.graph-container { width: 100%; height: calc(100vh - 56px); }
.graph-container svg { width: 100%; height: 100%; }
.node-label { font-size: 11px; fill: var(--text); pointer-events: none; paint-order: stroke; stroke: #0d1117; stroke-width: 3px; stroke-linejoin: round; }
.link-label { font-size: 9px; fill: var(--text-muted); pointer-events: none; paint-order: stroke; stroke: #0d1117; stroke-width: 2px; stroke-linejoin: round; }

/* Timeline view */
.timeline-event { display: grid; grid-template-columns: 140px 1fr; gap: 16px; padding: 12px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.timeline-date { color: var(--text-muted); font-size: 12px; text-align: right; }
.timeline-content .entity { color: var(--accent); font-weight: 500; }
.timeline-content .prop-key { color: var(--purple); }
.timeline-content .prop-val { color: var(--text); }
.timeline-badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; margin-left: 6px; }
.timeline-badge.property { background: #1f3d5c; color: #58a6ff; }
.timeline-badge.relationship { background: #2a1f3d; color: #d2a8ff; }

/* History modal */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; }
.modal h3 { margin-bottom: 16px; }
.modal .close { float: right; background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 18px; }

.empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
.empty-state h2 { font-size: 18px; margin-bottom: 8px; color: var(--text); }
</style>
</head>
<body>
<div class="app" id="app">
  <header class="header">
    <h1>Memento</h1>
    <div class="stats" id="stats"></div>
    <nav id="view-nav">
      <button class="active" data-view="detail">Detail</button>
      <button data-view="graph">Graph</button>
      <button data-view="timeline">Timeline</button>
    </nav>
  </header>
  <aside class="sidebar">
    <div class="search"><input type="text" id="search" placeholder="Search entities..."></div>
    <div class="filters" id="filters"></div>
    <ul class="entity-list" id="entity-list"></ul>
  </aside>
  <main class="main" id="main">
    <div class="empty-state"><h2>Select an entity</h2><p>Choose an entity from the sidebar to view details, or switch to Graph or Timeline view.</p></div>
  </main>
</div>
<div class="modal-overlay" id="modal" style="display:none">
  <div class="modal">
    <button class="close" id="modal-close">&times;</button>
    <div id="modal-content"></div>
  </div>
</div>

<script>
const API = '';
let allEntities = [];
let currentFilter = null;
let currentView = 'detail';
let selectedId = null;

// ── Init ──
async function init() {
  const [health, entities] = await Promise.all([
    fetch(`${API}/api/health`).then(r => r.json()),
    fetch(`${API}/api/entities`).then(r => r.json()),
  ]);
  allEntities = entities;
  document.getElementById('stats').innerHTML =
    `<div>Entities: <span>${health.node_count}</span></div>
     <div>Relationships: <span>${health.edge_count}</span></div>
     <div>Properties: <span>${health.active_property_count}</span></div>
     <div>Conflicts: <span>${health.unresolved_conflicts}</span></div>`;

  const types = [...new Set(entities.map(e => e.type))].sort();
  const filtersEl = document.getElementById('filters');
  filtersEl.innerHTML =
    `<button class="active" data-filter-type="">All</button>` +
    types.map(t => `<button data-filter-type="${esc(t)}">${esc(t)}</button>`).join('');

  renderEntityList(entities);
  document.getElementById('search').addEventListener('input', onSearch);

  // ── Event delegation (no inline handlers; CSP forbids them) ──
  document.getElementById('view-nav').addEventListener('click', e => {
    const btn = e.target.closest('button[data-view]');
    if (btn) switchView(btn.dataset.view, btn);
  });
  filtersEl.addEventListener('click', e => {
    const btn = e.target.closest('button[data-filter-type]');
    if (btn) setFilter(btn.dataset.filterType || null, btn);
  });
  document.getElementById('entity-list').addEventListener('click', e => {
    const li = e.target.closest('li[data-entity-id]');
    if (li) selectEntity(li.dataset.entityId);
  });
  document.getElementById('main').addEventListener('click', e => {
    const relRow = e.target.closest('[data-entity-id]');
    const histBtn = e.target.closest('button[data-history-key]');
    const timelineLink = e.target.closest('a[data-timeline-entity]');
    if (histBtn) {
      showHistory(histBtn.dataset.historyEntity, histBtn.dataset.historyKey);
    } else if (timelineLink) {
      selectEntity(timelineLink.dataset.timelineEntity);
    } else if (relRow && relRow.dataset.entityId) {
      selectEntity(relRow.dataset.entityId);
    }
  });
  const modal = document.getElementById('modal');
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.getElementById('modal-close').addEventListener('click', closeModal);
}

// ── Sidebar ──
function renderEntityList(entities) {
  const list = document.getElementById('entity-list');
  list.innerHTML = entities.map(e => `
    <li class="entity-item ${e.id === selectedId ? 'selected' : ''}" data-entity-id="${esc(e.id)}">
      <div class="name"><span class="type-badge type-${esc(e.type)}">${esc(e.type)}</span> ${esc(e.name)}</div>
      <div class="meta">${e.property_count} properties &middot; conf: ${(e.confidence * 100).toFixed(0)}%</div>
    </li>
  `).join('');
}

function onSearch(e) {
  const q = e.target.value.toLowerCase();
  let filtered = allEntities;
  if (currentFilter) filtered = filtered.filter(e => e.type === currentFilter);
  if (q) filtered = filtered.filter(e =>
    e.name.toLowerCase().includes(q) ||
    e.aliases.some(a => a.toLowerCase().includes(q))
  );
  renderEntityList(filtered);
}

function setFilter(type, btn) {
  currentFilter = type;
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  onSearch({ target: document.getElementById('search') });
}

// ── Views ──
function switchView(view, btn) {
  currentView = view;
  document.querySelectorAll('.header nav button').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  if (view === 'graph') renderGraph();
  else if (view === 'timeline') renderTimeline();
  else if (selectedId) selectEntity(selectedId);
  else document.getElementById('main').innerHTML = '<div class="empty-state"><h2>Select an entity</h2><p>Choose an entity from the sidebar.</p></div>';
}

// ── Detail View ──
async function selectEntity(id) {
  selectedId = id;
  renderEntityList(getFilteredEntities());
  if (currentView === 'graph') { renderGraph(id); return; }
  if (currentView === 'timeline') { renderTimeline(id); return; }

  const data = await fetch(`${API}/api/entities/${encodeURIComponent(id)}`).then(r => r.json());
  if (data.error) return;

  const props = Object.entries(data.properties);
  const rels = data.relationships;

  document.getElementById('main').innerHTML = `
    <div class="detail">
      <h2>${esc(data.name)}</h2>
      <div class="entity-type"><span class="type-badge type-${esc(data.type)}">${esc(data.type)}</span>
        &nbsp; ${confBadge(data.confidence)} &nbsp;
        <span style="color:var(--text-muted);font-size:12px">ID: ${esc(data.id.slice(0,12))}...</span>
      </div>
      ${data.aliases.length ? `<div class="aliases">Also known as: ${data.aliases.map(esc).join(', ')}</div>` : ''}
      <section>
        <h3>Properties (${props.length})</h3>
        ${props.length ? `<table>${props.map(([k, v]) => `
          <tr>
            <th>${esc(k)} <button class="history-btn" data-history-entity="${esc(id)}" data-history-key="${esc(k)}">history</button></th>
            <td>${esc(String(v.value))} ${confBadge(v.confidence)}</td>
          </tr>
        `).join('')}</table>` : '<p style="color:var(--text-muted);font-size:13px">No properties</p>'}
      </section>
      <section>
        <h3>Relationships (${rels.length})</h3>
        ${rels.length ? `<table>
          <tr><th>Type</th><th>Entity</th><th>Status</th><th>Confidence</th></tr>
          ${rels.map(r => `
            <tr class="rel-row" data-entity-id="${esc(r.other_id)}">
              <td>${r.direction === 'outgoing' ? '&rarr;' : '&larr;'} ${esc(r.type)}</td>
              <td><span class="type-badge type-${esc(r.other_type)}">${esc(r.other_type)}</span> ${esc(r.other_name)}</td>
              <td>${r.valid_to ? '<span style="color:var(--red)">ended</span>' : '<span style="color:var(--green)">active</span>'}</td>
              <td>${confBadge(r.confidence)}</td>
            </tr>
          `).join('')}
        </table>` : '<p style="color:var(--text-muted);font-size:13px">No relationships</p>'}
      </section>
      <section>
        <h3>Metadata</h3>
        <table>
          <tr><th>Created</th><td>${esc(data.created_at.slice(0,19))}</td></tr>
          <tr><th>Last seen</th><td>${esc(data.last_seen.slice(0,19))}</td></tr>
          <tr><th>Access count</th><td>${data.access_count}</td></tr>
        </table>
      </section>
    </div>
  `;
}

// ── History Modal ──
async function showHistory(entityId, key) {
  const url = `${API}/api/entities/${encodeURIComponent(entityId)}/history/${encodeURIComponent(key)}`;
  const history = await fetch(url).then(r => r.json());
  const mc = document.getElementById('modal-content');
  mc.innerHTML = `
    <h3>History: ${esc(key)}</h3>
    <table style="width:100%">
      <tr><th>Value</th><th>Valid from</th><th>Recorded at</th><th>Confidence</th><th>Status</th></tr>
      ${history.map(h => `
        <tr>
          <td>${esc(String(h.value))}</td>
          <td style="font-size:12px">${esc(h.as_of.slice(0,19))}</td>
          <td style="font-size:12px">${esc(h.recorded_at.slice(0,19))}</td>
          <td>${confBadge(h.confidence)}</td>
          <td>${h.superseded_by_id ? '<span style="color:var(--red)">superseded</span>' : '<span style="color:var(--green)">current</span>'}</td>
        </tr>
      `).join('')}
    </table>
  `;
  document.getElementById('modal').style.display = 'flex';
}

function closeModal() { document.getElementById('modal').style.display = 'none'; }

// ── Graph View ──
async function renderGraph(centerId) {
  const url = centerId ? `${API}/api/graph?center=${centerId}&hops=2` : `${API}/api/graph`;
  const data = await fetch(url).then(r => r.json());

  if (!data.nodes.length) {
    document.getElementById('main').innerHTML = '<div class="empty-state"><h2>No entities</h2><p>Ingest some conversations first.</p></div>';
    return;
  }

  const main = document.getElementById('main');
  main.innerHTML = '<div class="graph-container"><svg></svg></div>';
  const svg = d3.select('.graph-container svg');
  const width = main.clientWidth;
  const height = main.clientHeight;

  const typeColor = {
    person: '#58a6ff', organization: '#d2a8ff', project: '#3fb950',
    location: '#f0883e', concept: '#e3b341', event: '#f85149',
  };

  // Scale forces with node count so dense graphs get more breathing room
  const nodeCount = data.nodes.length;
  const linkDistance = Math.max(140, 80 + Math.sqrt(nodeCount) * 8);
  const chargeStrength = -Math.max(600, 200 + nodeCount * 4);
  const collisionRadius = 40;

  const simulation = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.links).id(d => d.id).distance(linkDistance))
    .force('charge', d3.forceManyBody().strength(chargeStrength))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(collisionRadius))
    .force('x', d3.forceX(width / 2).strength(0.05))
    .force('y', d3.forceY(height / 2).strength(0.05));

  const g = svg.append('g');

  // Zoom — tracks current scale so we can show/hide labels based on zoom
  let currentZoom = 1;
  svg.call(d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => {
    g.attr('transform', e.transform);
    currentZoom = e.transform.k;
    g.selectAll('.node-label').style('display', currentZoom < 0.6 ? 'none' : null);
    g.selectAll('.link-label').style('display', currentZoom < 1.0 ? 'none' : null);
  }));

  const link = g.append('g').selectAll('line')
    .data(data.links).join('line')
    .attr('stroke', d => d.active ? '#30363d' : '#21262d')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', d => d.active ? null : '4,4');

  const linkLabel = g.append('g').selectAll('text')
    .data(data.links).join('text')
    .attr('class', 'link-label')
    .attr('text-anchor', 'middle')
    .text(d => d.type);

  const node = g.append('g').selectAll('g')
    .data(data.nodes).join('g')
    .attr('cursor', 'pointer')
    .on('click', (e, d) => selectEntity(d.id))
    .on('mouseover', function(e, d) {
      d3.select(this).select('circle').attr('stroke', '#fff').attr('stroke-width', 2);
      d3.select(this).select('.node-label').style('display', null).style('font-weight', 'bold');
    })
    .on('mouseout', function(e, d) {
      if (d.id !== centerId) {
        d3.select(this).select('circle').attr('stroke', 'none');
      }
      d3.select(this).select('.node-label').style('font-weight', null)
        .style('display', currentZoom < 0.6 ? 'none' : null);
    })
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  node.append('circle')
    .attr('r', d => d.id === centerId ? 14 : 9)
    .attr('fill', d => typeColor[d.type] || '#8b949e')
    .attr('stroke', d => d.id === centerId ? '#fff' : 'none')
    .attr('stroke-width', 2);

  node.append('text')
    .attr('class', 'node-label')
    .attr('dx', 14).attr('dy', 4)
    .text(d => d.name.length > 24 ? d.name.slice(0, 22) + '…' : d.name)
    .append('title').text(d => d.name);

  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    linkLabel.attr('x', d => (d.source.x + d.target.x) / 2)
             .attr('y', d => (d.source.y + d.target.y) / 2);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

// ── Timeline View ──
async function renderTimeline(entityId) {
  const url = entityId ? `${API}/api/timeline?entity_id=${entityId}` : `${API}/api/timeline`;
  const events = await fetch(url).then(r => r.json());

  if (!events.length) {
    document.getElementById('main').innerHTML = '<div class="empty-state"><h2>No timeline events</h2></div>';
    return;
  }

  document.getElementById('main').innerHTML = `
    <div>
      <h2 style="margin-bottom:16px">Timeline${entityId ? '' : ' (all entities)'}</h2>
      ${events.map(e => `
        <div class="timeline-event">
          <div class="timeline-date">
            <div>${e.recorded_at.slice(0,10)}</div>
            <div style="font-size:10px;margin-top:2px">${e.recorded_at.slice(11,19)}</div>
          </div>
          <div class="timeline-content">
            <a class="entity" data-timeline-entity="${esc(e.entity_id)}">${esc(e.entity_name)}</a>
            <span class="timeline-badge ${e.type}">${e.type}</span><br>
            <span class="prop-key">${esc(e.property_key)}</span>:
            <span class="prop-val">${esc(String(e.value))}</span>
            ${e.as_of !== e.recorded_at ? `<br><span style="font-size:11px;color:var(--text-muted)">valid from: ${e.as_of.slice(0,19)}</span>` : ''}
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// ── Helpers ──
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function confBadge(c) {
  const cls = c >= 0.8 ? 'high' : c >= 0.5 ? 'mid' : 'low';
  return `<span class="confidence confidence-${cls}">${(c * 100).toFixed(0)}%</span>`;
}
function getFilteredEntities() {
  let filtered = allEntities;
  if (currentFilter) filtered = filtered.filter(e => e.type === currentFilter);
  const q = document.getElementById('search').value.toLowerCase();
  if (q) filtered = filtered.filter(e => e.name.toLowerCase().includes(q) || e.aliases.some(a => a.toLowerCase().includes(q)));
  return filtered;
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
