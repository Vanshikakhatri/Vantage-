"""
ui/graph_panel.py — Knowledge Graph workspace.

Reads the same pickled networkx graph GraphBuilder/GraphRetriever write to
(data/graph/knowledge_graph.gpickle) and renders it as a self-contained,
dependency-free interactive SVG panel (no CDN, no extra pip package — works
fully offline like the rest of the app):

  - entities + relation edges laid out with networkx's spring_layout
  - click a node -> side panel lists every chunk (source + page) it's
    mentioned in, and its direct relation edges
  - matched entities for the current chat query are highlighted (amber ring)
  - pan (drag) + zoom (wheel / buttons), light/dark aware via CSS variables
    already defined in ui/assets/style.css
"""

from __future__ import annotations

import html
import json
import pickle
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

_CANVAS_W, _CANVAS_H = 900, 560


def _load_graph(graph_path: str) -> nx.MultiDiGraph:
    p = Path(graph_path)
    if not p.exists():
        return nx.MultiDiGraph()
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return nx.MultiDiGraph()


def _entity_view(graph: nx.MultiDiGraph) -> tuple[list[str], list[tuple[str, str, str]], dict]:
    """Reduce the raw storage graph (entities + chunk-id nodes + mentioned_in
    edges) down to: entity list, entity-entity relation triples, and an
    entity -> [{source, page, chunk_id}] lookup for the side panel."""
    entities = [n for n, d in graph.nodes(data=True) if d.get("type") == "entity"]
    entity_set = set(entities)

    triples: list[tuple[str, str, str]] = []
    mentions: dict[str, list[dict]] = {e: [] for e in entities}

    for u, v, data in graph.edges(data=True):
        relation = data.get("relation", "related_to")
        if u in entity_set and v in entity_set:
            triples.append((u, v, relation))
        elif u in entity_set and relation == "mentioned_in":
            mentions[u].append({
                "chunk_id": v, "source": data.get("source", "?"), "page": data.get("page", 0),
            })

    return entities, triples, mentions


def _layout(entities: list[str], triples: list[tuple[str, str, str]]) -> dict[str, tuple[float, float]]:
    g = nx.Graph()
    g.add_nodes_from(entities)
    g.add_edges_from([(a, b) for a, b, _ in triples])
    if not entities:
        return {}
    pos = nx.spring_layout(g, seed=42, k=1.4 / max(len(entities) ** 0.5, 1), iterations=100)
    margin = 60
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs) or 1
    y_min, y_max = min(ys), max(ys) or 1
    x_span = (x_max - x_min) or 1
    y_span = (y_max - y_min) or 1
    scaled = {}
    for node, (x, y) in pos.items():
        sx = margin + (x - x_min) / x_span * (_CANVAS_W - 2 * margin)
        sy = margin + (y - y_min) / y_span * (_CANVAS_H - 2 * margin)
        scaled[node] = (sx, sy)
    return scaled


def _build_svg(entities, triples, positions, highlighted: set[str]) -> str:
    edge_svg = []
    for a, b, relation in triples:
        if a not in positions or b not in positions:
            continue
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        edge_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="vt-graph-edge" />'
            f'<text x="{mx:.1f}" y="{my:.1f}" class="vt-graph-edge-label">'
            f'{html.escape(relation)}</text>'
        )

    node_svg = []
    for e in entities:
        if e not in positions:
            continue
        x, y = positions[e]
        cls = "vt-graph-node highlighted" if e in highlighted else "vt-graph-node"
        label = html.escape(e if len(e) <= 18 else e[:16] + "…")
        safe_id = html.escape(e).replace('"', "&quot;")
        node_svg.append(
            f'<g class="{cls}" data-entity="{safe_id}" onclick="vtSelectEntity(this)" '
            f'transform="translate({x:.1f},{y:.1f})">'
            f'<circle r="16"></circle>'
            f'<text y="28" text-anchor="middle" class="vt-graph-node-label">{label}</text>'
            f'</g>'
        )

    return (
        f'<svg id="vtGraphSvg" viewBox="0 0 {_CANVAS_W} {_CANVAS_H}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<g id="vtViewport">{"".join(edge_svg)}{"".join(node_svg)}</g>'
        f'</svg>'
    )


_CSS = """
:root {
  --vg-bg: var(--vt-card, #FFFDF8);
  --vg-border: var(--vt-border, #D9CEC2);
  --vg-text: var(--vt-text, #2F372F);
  --vg-text-dim: var(--vt-text-dim, #706861);
  --vg-accent: var(--vt-forest, #235B46);
  --vg-warn: var(--vt-warn, #C18A45);
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "Inter","Segoe UI",sans-serif; }
.vt-graph-wrap { display:flex; gap:14px; background:var(--vg-bg); border:1px solid var(--vg-border);
  border-radius:22px; padding:14px; height:600px; box-shadow:0 10px 30px -12px rgba(47,55,47,0.16); }
.vt-graph-canvas { flex: 1 1 auto; position:relative; border-radius:16px; overflow:hidden;
  background: radial-gradient(circle at 30% 10%, rgba(35,91,70,0.06), transparent 60%); cursor:grab; }
.vt-graph-canvas.dragging { cursor:grabbing; }
.vt-graph-canvas svg { width:100%; height:100%; display:block; }
.vt-graph-edge { stroke: var(--vg-border); stroke-width: 1.4; }
.vt-graph-edge-label { fill: var(--vg-text-dim); font-size: 8px; text-anchor: middle; }
.vt-graph-node circle { fill: rgba(35,91,70,0.12); stroke: var(--vg-accent); stroke-width: 1.6; }
.vt-graph-node:hover circle { fill: rgba(35,91,70,0.24); }
.vt-graph-node.highlighted circle { stroke: var(--vg-warn); stroke-width: 2.6;
  filter: drop-shadow(0 0 4px rgba(193,138,69,0.5)); }
.vt-graph-node { cursor: pointer; }
.vt-graph-node-label { fill: var(--vg-text); font-size: 10px; text-anchor: middle; font-family:"Inter",sans-serif; }
.vt-graph-toolbar { position:absolute; top:10px; right:10px; display:flex; gap:6px; }
.vt-graph-toolbar button { background: var(--vg-bg); border:1px solid var(--vg-border);
  color: var(--vg-text); border-radius:999px; width:28px; height:28px; cursor:pointer; font-size:14px; }
.vt-graph-toolbar button:hover { border-color: var(--vg-accent); color: var(--vg-accent); }
.vt-graph-side { flex: 0 0 260px; border-left:1px solid var(--vg-border); padding-left:14px;
  overflow-y:auto; color: var(--vg-text); font-size:13px; font-family:"Inter",sans-serif; }
.vt-graph-side h4 { margin:0 0 8px 0; font-size:14px; color: var(--vg-accent); font-family:"Georgia",serif; }
.vt-graph-empty { color: var(--vg-text-dim); font-size:12px; }
.vt-graph-chip { display:inline-block; background:rgba(35,91,70,0.08); border:1px solid var(--vg-border);
  border-radius:999px; padding:3px 9px; margin:3px 4px 3px 0; font-size:11px; color: var(--vg-text-dim); }
"""

_JS = """
<script>
const graphData = __GRAPH_DATA__;
let scale = 1, panX = 0, panY = 0, isDragging = false, lastX = 0, lastY = 0;
const canvas = document.getElementById('vtCanvas');
const viewport = document.getElementById('vtViewport');

function applyTransform() {
  viewport.setAttribute('transform', `translate(${panX},${panY}) scale(${scale})`);
}
function zoom(delta) {
  scale = Math.min(3, Math.max(0.3, scale + delta));
  applyTransform();
}
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoom(e.deltaY < 0 ? 0.1 : -0.1);
}, { passive: false });
canvas.addEventListener('mousedown', (e) => {
  isDragging = true; lastX = e.clientX; lastY = e.clientY; canvas.classList.add('dragging');
});
window.addEventListener('mouseup', () => { isDragging = false; canvas.classList.remove('dragging'); });
window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  panX += (e.clientX - lastX); panY += (e.clientY - lastY);
  lastX = e.clientX; lastY = e.clientY;
  applyTransform();
});

function vtSelectEntity(el) {
  const name = el.getAttribute('data-entity');
  const info = graphData.mentions[name] || [];
  const rels = graphData.relations[name] || [];
  const side = document.getElementById('vtSide');
  let html = `<h4>${name}</h4>`;
  if (rels.length) {
    html += '<div style="margin-bottom:10px;">' +
      rels.map(r => `<span class="vt-graph-chip">${r}</span>`).join('') + '</div>';
  }
  if (info.length) {
    html += `<div class="vt-graph-empty" style="margin-bottom:6px;">${info.length} chunk(s):</div>`;
    html += info.map(m => `<div class="vt-graph-chip">📄 ${m.source} · p.${m.page}</div>`).join('');
  } else {
    html += '<div class="vt-graph-empty">No indexed chunks for this entity.</div>';
  }
  side.innerHTML = html;
}
</script>
"""


def render_graph_panel(graph_path: str, highlight_query: str | None = None) -> None:
    """Renders the Knowledge Graph tab. Call from app.py inside its own tab/container."""
    graph = _load_graph(graph_path)
    entities, triples, mentions = _entity_view(graph)

    if not entities:
        st.markdown(
            '<div class="vt-empty-state">🕸️ <b>No knowledge graph yet.</b><br>'
            'Ingest a document — entities and relationships extracted during '
            'ingestion will appear here automatically.</div>',
            unsafe_allow_html=True,
        )
        return

    highlighted: set[str] = set()
    if highlight_query:
        lowered = highlight_query.lower()
        highlighted = {e for e in entities if e.lower() in lowered}

    positions = _layout(entities, triples)
    svg = _build_svg(entities, triples, positions, highlighted)

    relations_by_entity: dict[str, list[str]] = {e: [] for e in entities}
    for a, b, relation in triples:
        relations_by_entity.setdefault(a, []).append(f"{relation} → {b}")
        relations_by_entity.setdefault(b, []).append(f"{a} → {relation}")

    graph_data = {"mentions": mentions, "relations": relations_by_entity}

    st.caption(f"{len(entities)} entities · {len(triples)} relations extracted from your documents")

    html_doc = f"""
    <style>{_CSS}</style>
    <div class="vt-graph-wrap">
      <div class="vt-graph-canvas" id="vtCanvas">
        {svg}
        <div class="vt-graph-toolbar">
          <button onclick="zoom(0.15)">+</button>
          <button onclick="zoom(-0.15)">−</button>
        </div>
      </div>
      <div class="vt-graph-side" id="vtSide">
        <div class="vt-graph-empty">Click a node to see its relations and source chunks.</div>
      </div>
    </div>
    {_JS.replace("__GRAPH_DATA__", json.dumps(graph_data))}
    """
    components.html(html_doc, height=620, scrolling=False)
