"""Graph export helpers for DOT and simple standalone HTML outputs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List

from .storage import GraphStore


def export_dot(store: GraphStore, output_file: Path, focus: str = "") -> None:
    nodes = {row["node_id"]: row for row in store.get_nodes()}
    edges = [dict(e) for e in store.get_edges()]

    selected = _focused_subgraph(nodes, edges, focus)

    lines = ["digraph CodeGraph {"]
    lines.append("  rankdir=LR;")

    for node_id in selected["nodes"]:
        if node_id not in nodes:
            continue
        node = nodes[node_id]
        label = f"{node['node_type']}\\n{node['qualname']}"
        lines.append(f'  "{node_id}" [label="{_esc(label)}"];')

    for edge in selected["edges"]:
        if edge["src"] not in nodes or edge["dst"] not in nodes:
            continue
        lines.append(
            f'  "{edge["src"]}" -> "{edge["dst"]}" [label="{_esc(edge["edge_type"])}"];'
        )

    lines.append("}")
    output_file.write_text("\n".join(lines), encoding="utf-8")


def export_html(store: GraphStore, output_file: Path, focus: str = "") -> None:
    """Export graph to interactive HTML visualization using vis.js."""
    nodes = {row["node_id"]: row for row in store.get_nodes()}
    edges = [dict(e) for e in store.get_edges()]

    selected = _focused_subgraph(nodes, edges, focus)
    graph_payload = {
        "nodes": [
            {
                "id": node_id,
                "label": f"{nodes[node_id]['node_type']}: {nodes[node_id]['qualname']}",
                "title": nodes[node_id]["file_path"],
            }
            for node_id in selected["nodes"]
            if node_id in nodes
        ],
        "edges": [e for e in selected["edges"] if e["src"] in nodes and e["dst"] in nodes],
    }

    # Load interactive template
    template_path = Path(__file__).parent / "templates" / "graph_interactive.html"
    
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        # Inject graph data
        doc = template.replace("{{ GRAPH_DATA }}", json.dumps(graph_payload, indent=2))
    else:
        # Fallback to basic HTML if template not found
        doc = _basic_html_export(graph_payload)
    
    output_file.write_text(doc, encoding="utf-8")


def _basic_html_export(graph_payload: dict) -> str:
    """Fallback basic HTML export."""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CodeGraph Export</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 20px; }}
    #container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; }}
    ul {{ list-style: none; padding: 0; margin: 0; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>CodeGraph Export</h1>
  <div id="container">
    <div class="panel">
      <h2>Nodes</h2>
      <ul id="nodes"></ul>
    </div>
    <div class="panel">
      <h2>Edges</h2>
      <ul id="edges"></ul>
    </div>
  </div>
  <script>
    const graph = {json.dumps(graph_payload)};
    const nodesEl = document.getElementById('nodes');
    const edgesEl = document.getElementById('edges');
    graph.nodes.forEach(n => {{
      const li = document.createElement('li');
      li.textContent = `${{n.id}} -> ${{n.label}} (${{n.title}})`;
      nodesEl.appendChild(li);
    }});
    graph.edges.forEach(e => {{
      const li = document.createElement('li');
      li.textContent = `${{e.src}} --${{e.edge_type}}--> ${{e.dst}}`;
      edgesEl.appendChild(li);
    }});
  </script>
</body>
</html>
"""


def _focused_subgraph(nodes: Dict[str, dict], edges: List[dict], focus: str) -> Dict[str, List]:
    if not focus:
        return {"nodes": list(nodes.keys()), "edges": edges}

    focus_ids = {
        node_id
        for node_id, node in nodes.items()
        if focus in node_id or focus in node["name"] or focus in node["qualname"]
    }

    if not focus_ids:
        return {"nodes": list(nodes.keys()), "edges": edges}

    edge_subset = [e for e in edges if e["src"] in focus_ids or e["dst"] in focus_ids]
    node_subset = set(focus_ids)
    for e in edge_subset:
        if e["src"] in nodes:
            node_subset.add(e["src"])
        if e["dst"] in nodes:
            node_subset.add(e["dst"])
    return {"nodes": sorted(node_subset), "edges": edge_subset}


def _esc(text: str) -> str:
    return text.replace('"', '\\"')
