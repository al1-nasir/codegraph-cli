"""Visual Code Explorer — browser-based UI for navigating indexed projects.

Launches a local web server serving a self-contained HTML page with:
- Directory tree sidebar with expandable folders
- File analysis: AI explanations, dependencies, syntax-highlighted code
- Mermaid diagrams (file-level and system-level)
- Export features (Mermaid, Excalidraw link, HTML)

Uses Starlette + Uvicorn (already installed) — zero extra dependencies.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

logger = logging.getLogger(__name__)

explore_app = typer.Typer(
    help="🌐 Visual code explorer in your browser.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)


# ===================================================================
# Backend helpers — build data from GraphStore
# ===================================================================


def _build_api_tree(store: Any) -> Dict:
    """Build JSON directory tree from indexed nodes."""
    nodes_by_file = store.all_by_file()
    root: Dict[str, Any] = {"name": "root", "type": "dir", "path": "", "children": []}
    dir_cache: Dict[str, Dict] = {"": root}

    all_paths: set[str] = set()
    for fp in nodes_by_file:
        parts = Path(fp).parts
        for i in range(len(parts) - 1):
            all_paths.add(str(Path(*parts[: i + 1])))
        all_paths.add(fp)

    # Create directory entries
    for p in sorted(all_paths):
        pp = Path(p)
        parent_key = str(pp.parent) if str(pp.parent) != "." else ""
        if p in nodes_by_file:
            # It's a file
            entry = {"name": pp.name, "type": "file", "path": p, "children": []}
        else:
            entry = {"name": pp.name, "type": "dir", "path": p, "children": []}
            dir_cache[p] = entry

        parent = dir_cache.get(parent_key, root)
        parent["children"].append(entry)

    return root


def _analyze_file(store: Any, file_path: str) -> Dict:
    """Deep analysis of a single file: code, deps, dependents, symbols."""
    nodes_by_file = store.all_by_file()
    file_nodes = nodes_by_file.get(file_path, [])

    # Read source code from the first node or reconstruct
    content_lines: list[str] = []
    functions: list[dict] = []
    classes: list[dict] = []

    for node in sorted(file_nodes, key=lambda n: n.get("start_line", 0)):
        ntype = node.get("node_type", "")
        name = node.get("name", "")
        qualname = node.get("qualname", name)
        start = node.get("start_line", 0)
        end = node.get("end_line", 0)
        code = node.get("code", "")

        if ntype == "function":
            functions.append({"name": name, "qualname": qualname, "line": start, "end_line": end})
        elif ntype == "class":
            classes.append({"name": name, "qualname": qualname, "line": start, "end_line": end})

        if code:
            content_lines.append(code)

    # Build full content — try reading the actual file first
    content = ""
    metadata = store.get_metadata()
    source_root = metadata.get("source_path", "")
    if source_root:
        actual_file = Path(source_root) / file_path
        if actual_file.exists():
            try:
                content = actual_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = "\n\n".join(content_lines)
    if not content:
        content = "\n\n".join(content_lines)

    # Dependencies — symbols this file calls (outgoing edges)
    deps: list[str] = []
    dependents: list[str] = []
    node_ids = {n.get("node_id") for n in file_nodes}
    for nid in node_ids:
        for edge in store.neighbors(nid):
            dst = edge["dst"] if isinstance(edge, dict) else edge[1]
            dst_node = store.get_node(dst)
            if dst_node:
                deps.append(dst_node["qualname"])
        for edge in store.reverse_neighbors(nid):
            src = edge["src"] if isinstance(edge, dict) else edge[0]
            src_node = store.get_node(src)
            if src_node:
                dependents.append(src_node["qualname"])

    deps = sorted(set(deps))
    dependents = sorted(set(dependents))

    # Generate Mermaid diagram for this file
    mermaid = _generate_file_mermaid(store, file_path, file_nodes)

    return {
        "name": Path(file_path).name,
        "path": file_path,
        "content": content,
        "explanation": None,  # filled by /api/explain
        "deps": deps,
        "dependents": dependents,
        "mermaid": mermaid,
        "functions": functions,
        "classes": classes,
    }


def _generate_file_mermaid(store: Any, file_path: str, file_nodes: list) -> Optional[str]:
    """Generate a Mermaid diagram for a single file's internal structure."""
    if not file_nodes:
        return None

    lines = ["graph TD"]
    node_ids_in_file = set()
    id_to_label: dict[str, str] = {}

    for node in file_nodes:
        nid = node.get("node_id", "")
        name = node.get("name", "unknown")
        ntype = node.get("node_type", "")
        safe_id = nid.replace(".", "_").replace("/", "_").replace("-", "_").replace(":", "_")
        node_ids_in_file.add(nid)
        id_to_label[nid] = safe_id

        if ntype == "class":
            lines.append(f'    {safe_id}["{name} (class)"]')
        elif ntype == "function":
            lines.append(f'    {safe_id}("{name}()")')
        else:
            lines.append(f'    {safe_id}["{name}"]')

    # Add edges between nodes in this file
    edge_count = 0
    for nid in node_ids_in_file:
        for edge in store.neighbors(nid):
            dst = edge["dst"] if isinstance(edge, dict) else edge[1]
            if dst in node_ids_in_file and edge_count < 50:
                src_safe = id_to_label.get(nid, "")
                dst_safe = id_to_label.get(dst, "")
                if src_safe and dst_safe:
                    lines.append(f"    {src_safe} --> {dst_safe}")
                    edge_count += 1

    if len(lines) <= 1:
        return None
    return "\n".join(lines)


def _generate_system_mermaid(store: Any) -> str:
    """Generate a system-level architecture Mermaid diagram (file-to-file deps)."""
    nodes_by_file = store.all_by_file()
    edges = store.get_edges()

    # Map node_id -> file_path
    nid_to_file: dict[str, str] = {}
    for fp, nodes in nodes_by_file.items():
        for n in nodes:
            nid_to_file[n.get("node_id", "")] = fp

    # Build file-level edges
    file_edges: set[tuple[str, str]] = set()
    for edge in edges:
        src = edge["src"] if isinstance(edge, dict) else edge[0]
        dst = edge["dst"] if isinstance(edge, dict) else edge[1]
        src_file = nid_to_file.get(src, "")
        dst_file = nid_to_file.get(dst, "")
        if src_file and dst_file and src_file != dst_file:
            file_edges.add((src_file, dst_file))

    lines = ["graph LR"]
    file_ids: dict[str, str] = {}
    for fp in sorted(nodes_by_file.keys()):
        safe = fp.replace("/", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
        file_ids[fp] = safe
        short = Path(fp).name
        lines.append(f'    {safe}["{short}"]')

    for src_f, dst_f in sorted(file_edges):
        src_id = file_ids.get(src_f)
        dst_id = file_ids.get(dst_f)
        if src_id and dst_id:
            lines.append(f"    {src_id} --> {dst_id}")

    return "\n".join(lines)


def _explain_file(store: Any, file_path: str, llm_provider: str, llm_model: str, llm_api_key: str) -> Optional[str]:
    """Use LLM to generate a plain-English explanation of a file."""
    try:
        from .llm import LocalLLM
        llm = LocalLLM(model=llm_model, provider=llm_provider, api_key=llm_api_key)

        # Get file content
        metadata = store.get_metadata()
        source_root = metadata.get("source_path", "")
        content = ""
        if source_root:
            actual = Path(source_root) / file_path
            if actual.exists():
                content = actual.read_text(encoding="utf-8", errors="replace")[:4000]

        if not content:
            nodes = store.all_by_file().get(file_path, [])
            content = "\n".join(n.get("code", "")[:500] for n in nodes[:5])

        if not content:
            return None

        prompt = (
            f"Explain what this file does in 2-3 concise sentences. "
            f"Focus on its purpose and key functionality.\n\n"
            f"File: {file_path}\n```\n{content[:3000]}\n```"
        )
        return llm.explain(prompt)
    except Exception as e:
        logger.warning("LLM explanation failed: %s", e)
        return None


# ===================================================================
# HTML Template — complete self-contained SPA
# ===================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodeGraph Explorer</title>
<script src="https://cdn.tailwindcss.com"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/typescript.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: { extend: { colors: { 'cg-bg': '#0d1117', 'cg-sidebar': '#161b22', 'cg-card': '#1c2128', 'cg-border': '#30363d', 'cg-accent': '#58a6ff', 'cg-green': '#3fb950', 'cg-purple': '#bc8cff', 'cg-orange': '#d29922' } } }
}
</script>
<style>
  html, body { margin:0; padding:0; height:100%; overflow:hidden; background:#0d1117; color:#c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #161b22; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #484f58; }
  .tree-item { transition: background 0.1s; }
  .tree-item:hover { background: #1c2128; }
  .tree-item.active { background: #1c2128; border-right: 2px solid #58a6ff; }
  .fade-in { animation: fadeIn 0.2s ease-in; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  .slide-down { overflow: hidden; transition: max-height 0.25s ease-out; }
  .toast { animation: toastIn 0.3s ease-out, toastOut 0.3s ease-in 1.7s; }
  @keyframes toastIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  @keyframes toastOut { from { opacity:1; } to { opacity:0; } }
  pre code.hljs { background: transparent !important; padding: 0 !important; }
  .mermaid svg { max-width: 100%; }
</style>
</head>
<body x-data="explorer()" x-init="init()" @keydown.escape.window="selectedFile = null">

<!-- Toast -->
<div x-show="toast" x-transition x-text="toast" class="fixed bottom-6 right-6 z-50 bg-cg-green text-black px-4 py-2 rounded-lg font-medium shadow-lg toast"></div>

<!-- Export Modal -->
<div x-show="showExportModal" x-transition class="fixed inset-0 z-40 flex items-center justify-center bg-black/60" @click.self="showExportModal = false" @keydown.escape.window="showExportModal = false">
  <div class="bg-cg-sidebar border border-cg-border rounded-xl shadow-2xl w-[480px] max-w-[95vw] p-6 fade-in">
    <div class="flex items-center justify-between mb-5">
      <h2 class="text-lg font-bold text-white">📄 Export to DOCX</h2>
      <button @click="showExportModal = false" class="text-gray-500 hover:text-white text-xl">&times;</button>
    </div>

    <div class="space-y-4">
      <!-- Include Code -->
      <label class="flex items-center gap-3 cursor-pointer">
        <input type="checkbox" x-model="exportOpts.includeCode" class="rounded border-cg-border bg-cg-bg text-cg-accent focus:ring-cg-accent">
        <div>
          <div class="text-sm text-gray-200">Include source code</div>
          <div class="text-xs text-gray-500">Adds full file content (larger file)</div>
        </div>
      </label>

      <!-- Include Diagram -->
      <label class="flex items-center gap-3 cursor-pointer">
        <input type="checkbox" x-model="exportOpts.includeDiagram" class="rounded border-cg-border bg-cg-bg text-cg-accent focus:ring-cg-accent">
        <div>
          <div class="text-sm text-gray-200">Include architecture diagram</div>
          <div class="text-xs text-gray-500">Embeds system diagram as image</div>
        </div>
      </label>

      <!-- Enhanced (LLM) -->
      <label class="flex items-center gap-3 cursor-pointer" :class="!exportStatus.llm_available && 'opacity-50'">
        <input type="checkbox" x-model="exportOpts.enhanced" :disabled="!exportStatus.llm_available" class="rounded border-cg-border bg-cg-bg text-cg-purple focus:ring-cg-purple">
        <div>
          <div class="text-sm text-gray-200">AI-enhanced explanations</div>
          <div class="text-xs text-gray-500" x-text="exportStatus.llm_available ? 'Using ' + (exportStatus.provider || '') + '/' + (exportStatus.model || '') : 'No LLM configured — run cg config setup'"></div>
        </div>
      </label>

      <!-- Depth (only if enhanced) -->
      <div x-show="exportOpts.enhanced && exportStatus.llm_available" x-transition class="pl-8">
        <div class="text-sm text-gray-300 mb-2">Explanation depth:</div>
        <div class="flex gap-2">
          <template x-for="d in ['overview', 'modules', 'files']" :key="d">
            <button @click="exportOpts.depth = d"
                    :class="exportOpts.depth === d ? 'bg-cg-accent text-black' : 'bg-cg-bg text-gray-400 border border-cg-border'"
                    class="px-3 py-1.5 text-xs rounded transition capitalize" x-text="d"></button>
          </template>
        </div>
        <div class="text-xs text-gray-600 mt-1" x-text="exportOpts.depth === 'overview' ? 'Project-level summary only' : exportOpts.depth === 'modules' ? 'Per-module summaries' : 'Per-file explanations (slowest)'"></div>
      </div>
    </div>

    <!-- Actions -->
    <div class="mt-6 flex items-center justify-between">
      <button @click="showExportModal = false" class="px-4 py-2 text-sm text-gray-400 hover:text-white transition">Cancel</button>
      <button @click="startExport()" :disabled="exporting" class="px-5 py-2 text-sm bg-cg-accent text-black font-medium rounded-lg hover:bg-blue-400 transition disabled:opacity-50 flex items-center gap-2">
        <span x-show="exporting" class="animate-spin">⏳</span>
        <span x-text="exporting ? 'Exporting...' : '📄 Export DOCX'"></span>
      </button>
    </div>
  </div>
</div>

<div class="flex h-screen">

  <!-- Sidebar -->
  <div class="w-[300px] min-w-[300px] bg-cg-sidebar border-r border-cg-border flex flex-col">
    <!-- Header -->
    <div class="p-4 border-b border-cg-border">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-xl">🧠</span>
        <h1 class="text-sm font-bold text-white">CodeGraph Explorer</h1>
      </div>
      <div class="text-xs text-gray-500" x-text="projectName"></div>
      <div class="text-xs text-gray-600" x-text="projectStats"></div>
    </div>

    <!-- Search -->
    <div class="p-3 border-b border-cg-border">
      <input type="text" x-model="treeFilter" placeholder="Filter files..."
        class="w-full bg-cg-bg border border-cg-border rounded px-3 py-1.5 text-sm text-gray-300 placeholder-gray-600 focus:border-cg-accent focus:outline-none">
    </div>

    <!-- File Tree -->
    <div class="flex-1 overflow-y-auto py-2">
      <template x-if="tree && tree.children">
        <div>
          <template x-for="child in filteredTree(tree.children)" :key="child.path">
            <div x-data="{ open: false }">
              <div @click="child.type === 'dir' ? open = !open : loadFile(child.path)"
                   :class="{'active': selectedFile === child.path}"
                   class="tree-item flex items-center gap-1.5 px-3 py-1 cursor-pointer text-sm select-none">
                <span class="w-4 text-center text-xs text-gray-500" x-show="child.type === 'dir'" x-text="open ? '▾' : '▸'"></span>
                <span class="w-4 text-center" x-show="child.type !== 'dir'"></span>
                <span x-text="child.type === 'dir' ? '📁' : fileIcon(child.name)" class="text-sm"></span>
                <span class="truncate" :class="child.type === 'dir' ? 'text-gray-300 font-medium' : 'text-gray-400'" x-text="child.name"></span>
              </div>
              <!-- Children -->
              <div x-show="open && child.type === 'dir'" x-transition class="pl-4">
                <template x-for="sub in filteredTree(child.children || [])" :key="sub.path">
                  <div x-data="{ subOpen: false }">
                    <div @click="sub.type === 'dir' ? subOpen = !subOpen : loadFile(sub.path)"
                         :class="{'active': selectedFile === sub.path}"
                         class="tree-item flex items-center gap-1.5 px-3 py-1 cursor-pointer text-sm select-none">
                      <span class="w-4 text-center text-xs text-gray-500" x-show="sub.type === 'dir'" x-text="subOpen ? '▾' : '▸'"></span>
                      <span class="w-4 text-center" x-show="sub.type !== 'dir'"></span>
                      <span x-text="sub.type === 'dir' ? '📁' : fileIcon(sub.name)" class="text-sm"></span>
                      <span class="truncate" :class="sub.type === 'dir' ? 'text-gray-300 font-medium' : 'text-gray-400'" x-text="sub.name"></span>
                    </div>
                    <div x-show="subOpen && sub.type === 'dir'" x-transition class="pl-4">
                      <template x-for="deep in filteredTree(sub.children || [])" :key="deep.path">
                        <div @click="deep.type === 'dir' ? null : loadFile(deep.path)"
                             :class="{'active': selectedFile === deep.path}"
                             class="tree-item flex items-center gap-1.5 px-3 py-1 cursor-pointer text-sm select-none">
                          <span class="w-4 text-center" x-show="deep.type === 'dir'">📁</span>
                          <span class="w-4 text-center" x-show="deep.type !== 'dir'"></span>
                          <span x-text="deep.type === 'dir' ? '📁' : fileIcon(deep.name)" class="text-sm"></span>
                          <span class="truncate text-gray-400" x-text="deep.name"></span>
                        </div>
                      </template>
                    </div>
                  </div>
                </template>
              </div>
            </div>
          </template>
        </div>
      </template>
      <div x-show="loading" class="p-4 text-center text-gray-500 text-sm">Loading tree...</div>
    </div>

    <!-- System Diagram Button -->
    <div class="p-3 border-t border-cg-border space-y-2">
      <button @click="loadSystemDiagram()" class="w-full bg-cg-card hover:bg-gray-700 border border-cg-border text-sm text-gray-300 rounded py-2 transition">
        🗺️ System Architecture
      </button>
      <button @click="showExportModal = true; loadExportStatus()" class="w-full bg-blue-900/30 hover:bg-blue-900/50 border border-blue-800/40 text-sm text-cg-accent rounded py-2 transition">
        📄 Export to DOCX
      </button>
    </div>
  </div>

  <!-- Main Panel -->
  <div class="flex-1 flex flex-col overflow-hidden">

    <!-- Top Bar -->
    <div class="flex items-center justify-between px-5 py-3 border-b border-cg-border bg-cg-sidebar">
      <div class="flex items-center gap-3">
        <span class="text-sm text-gray-400" x-text="selectedFile || 'Select a file to explore'"></span>
        <span x-show="fileLoading" class="text-xs text-gray-500">⏳ Loading...</span>
      </div>
      <div class="flex items-center gap-2" x-show="fileData">
        <button @click="triggerExplain()" class="px-3 py-1 text-xs bg-purple-900/40 text-cg-purple border border-purple-800/50 rounded hover:bg-purple-900/60 transition" title="AI Explain">
          🤖 Explain
        </button>
        <button @click="exportMermaid()" x-show="fileData && fileData.mermaid" class="px-3 py-1 text-xs bg-cg-card text-gray-300 border border-cg-border rounded hover:bg-gray-700 transition">
          📋 Copy Mermaid
        </button>
        <button @click="openMermaidLive(fileData?.mermaid)" x-show="fileData && fileData.mermaid" class="px-3 py-1 text-xs bg-cg-card text-gray-300 border border-cg-border rounded hover:bg-gray-700 transition" title="Open in Mermaid Live Editor">
          🔗 Mermaid Live
        </button>
        <button @click="exportDiagramSVG('fileMermaid', fileData?.name || 'diagram')" x-show="fileData && fileData.mermaid" class="px-3 py-1 text-xs bg-cg-card text-gray-300 border border-cg-border rounded hover:bg-gray-700 transition" title="Download as SVG">
          🖼️ SVG
        </button>
      </div>
    </div>

    <!-- Content -->
    <div class="flex-1 overflow-y-auto p-6 space-y-6">

      <!-- Welcome -->
      <div x-show="!selectedFile && !systemDiagram" class="flex flex-col items-center justify-center h-full text-center fade-in">
        <div class="text-6xl mb-4">🧠</div>
        <h2 class="text-2xl font-bold text-white mb-2">CodeGraph Explorer</h2>
        <p class="text-gray-500 max-w-md">Select a file from the sidebar to explore its structure, dependencies, and AI-powered explanations.</p>
        <div class="mt-6 flex gap-3">
          <div class="px-4 py-2 bg-cg-card rounded border border-cg-border text-sm text-gray-400">📁 Browse files in sidebar</div>
          <div class="px-4 py-2 bg-cg-card rounded border border-cg-border text-sm text-gray-400">🗺️ View system architecture</div>
        </div>
        <div class="mt-8 text-xs text-gray-600">
          <span class="text-gray-500">Shortcuts:</span> Esc to deselect
        </div>
      </div>

      <!-- System Diagram View -->
      <div x-show="systemDiagram && !selectedFile" class="fade-in">
        <div class="bg-cg-card rounded-lg border border-cg-border p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-bold text-white">🗺️ System Architecture</h3>
            <div class="flex items-center gap-2">
              <button @click="copyToClipboard(systemDiagram); showToast('Mermaid copied!')" class="px-3 py-1 text-xs bg-cg-bg text-gray-300 border border-cg-border rounded hover:bg-gray-700 transition" title="Copy Mermaid code">📋 Mermaid</button>
              <button @click="exportDiagramSVG('systemMermaid', 'system-architecture')" class="px-3 py-1 text-xs bg-cg-bg text-gray-300 border border-cg-border rounded hover:bg-gray-700 transition" title="Download as SVG">🖼️ SVG</button>
              <button @click="openMermaidLive(systemDiagram)" class="px-3 py-1 text-xs bg-cg-bg text-gray-300 border border-cg-border rounded hover:bg-gray-700 transition" title="Open in Mermaid Live Editor">🔗 Mermaid Live</button>
            </div>
          </div>
          <div class="bg-cg-bg rounded p-4 overflow-x-auto">
            <div class="mermaid" x-ref="systemMermaid"></div>
          </div>
        </div>
      </div>

      <!-- File View -->
      <div x-show="selectedFile && fileData" class="space-y-6 fade-in">

        <!-- File Header -->
        <div class="flex items-center gap-3">
          <span class="text-2xl" x-text="fileIcon(fileData?.name || '')"></span>
          <div>
            <h2 class="text-xl font-bold text-white" x-text="fileData?.name"></h2>
            <span class="text-xs text-gray-500" x-text="fileData?.path"></span>
          </div>
        </div>

        <!-- AI Explanation -->
        <div x-show="fileData?.explanation" class="bg-purple-900/20 border border-purple-800/40 rounded-lg p-4 fade-in">
          <div class="flex items-center gap-2 mb-2">
            <span class="text-sm">🤖</span>
            <span class="text-sm font-medium text-cg-purple">AI Explanation</span>
          </div>
          <p class="text-sm text-gray-300 leading-relaxed" x-text="fileData?.explanation"></p>
        </div>

        <!-- Symbols -->
        <div x-show="(fileData?.functions?.length || 0) + (fileData?.classes?.length || 0) > 0" class="bg-cg-card rounded-lg border border-cg-border p-4">
          <h3 class="text-sm font-semibold text-white mb-3">📌 Symbols</h3>
          <div class="flex flex-wrap gap-2">
            <template x-for="fn in (fileData?.functions || [])" :key="fn.name">
              <span class="inline-flex items-center gap-1 px-2 py-1 bg-blue-900/30 text-blue-400 text-xs rounded border border-blue-800/40">
                <span>ƒ</span> <span x-text="fn.name"></span>
                <span class="text-blue-600" x-text="'L' + fn.line"></span>
              </span>
            </template>
            <template x-for="cls in (fileData?.classes || [])" :key="cls.name">
              <span class="inline-flex items-center gap-1 px-2 py-1 bg-orange-900/30 text-cg-orange text-xs rounded border border-orange-800/40">
                <span>◆</span> <span x-text="cls.name"></span>
                <span class="text-orange-600" x-text="'L' + cls.line"></span>
              </span>
            </template>
          </div>
        </div>

        <!-- Dependencies Grid -->
        <div class="grid grid-cols-2 gap-4" x-show="(fileData?.deps?.length || 0) + (fileData?.dependents?.length || 0) > 0">
          <div class="bg-cg-card rounded-lg border border-cg-border p-4">
            <h3 class="text-sm font-semibold text-cg-green mb-3">⬆️ Dependencies <span class="text-gray-600 font-normal" x-text="'(' + (fileData?.deps?.length || 0) + ')'"></span></h3>
            <div class="space-y-1 max-h-40 overflow-y-auto">
              <template x-for="dep in (fileData?.deps || [])" :key="dep">
                <div class="text-xs text-gray-400 font-mono truncate" x-text="dep"></div>
              </template>
              <div x-show="!(fileData?.deps?.length)" class="text-xs text-gray-600 italic">None</div>
            </div>
          </div>
          <div class="bg-cg-card rounded-lg border border-cg-border p-4">
            <h3 class="text-sm font-semibold text-cg-accent mb-3">⬇️ Dependents <span class="text-gray-600 font-normal" x-text="'(' + (fileData?.dependents?.length || 0) + ')'"></span></h3>
            <div class="space-y-1 max-h-40 overflow-y-auto">
              <template x-for="d in (fileData?.dependents || [])" :key="d">
                <div class="text-xs text-gray-400 font-mono truncate" x-text="d"></div>
              </template>
              <div x-show="!(fileData?.dependents?.length)" class="text-xs text-gray-600 italic">None</div>
            </div>
          </div>
        </div>

        <!-- Source Code (collapsible) -->
        <div class="bg-cg-card rounded-lg border border-cg-border overflow-hidden">
          <div class="flex items-center justify-between px-4 py-2 border-b border-cg-border cursor-pointer select-none" @click="codeOpen = !codeOpen">
            <div class="flex items-center gap-2">
              <span class="text-xs text-gray-500 transition-transform duration-200" :class="codeOpen && 'rotate-90'" style="display:inline-block">▶</span>
              <h3 class="text-sm font-semibold text-white">📝 Source Code</h3>
            </div>
            <div class="flex items-center gap-1" @click.stop>
              <button @click="copyToClipboard(fileData?.content || ''); showToast('Code copied!')" class="px-2 py-1 text-xs text-gray-400 hover:text-white transition" title="Copy source code">📋 Copy</button>
            </div>
          </div>
          <div x-show="codeOpen" x-transition:enter="transition ease-out duration-200" x-transition:leave="transition ease-in duration-150" class="overflow-x-auto">
            <pre class="p-4 text-sm leading-relaxed" style="max-height:600px; overflow-y:auto"><code x-ref="codeBlock" class="hljs"></code></pre>
          </div>
        </div>

        <!-- Mermaid Diagram -->
        <div x-show="fileData?.mermaid" class="bg-cg-card rounded-lg border border-cg-border p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-semibold text-white">📊 File Diagram</h3>
            <div class="flex items-center gap-2">
              <button @click="exportMermaid()" class="px-2 py-1 text-xs bg-cg-bg text-gray-400 border border-cg-border rounded hover:text-white hover:bg-gray-700 transition" title="Copy Mermaid code">📋 Mermaid</button>
              <button @click="exportDiagramSVG('fileMermaid', fileData?.name || 'diagram')" class="px-2 py-1 text-xs bg-cg-bg text-gray-400 border border-cg-border rounded hover:text-white hover:bg-gray-700 transition" title="Download as SVG">🖼️ SVG</button>
              <button @click="openMermaidLive(fileData?.mermaid)" class="px-2 py-1 text-xs bg-cg-bg text-gray-400 border border-cg-border rounded hover:text-white hover:bg-gray-700 transition" title="Open in Mermaid Live Editor">🔗 Mermaid Live</button>
            </div>
          </div>
          <div class="bg-cg-bg rounded p-4 overflow-x-auto">
            <div class="mermaid" x-ref="fileMermaid"></div>
          </div>
        </div>

      </div>
    </div>
  </div>
</div>

<script>
mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });

function explorer() {
  return {
    tree: null,
    selectedFile: null,
    fileData: null,
    fileLoading: false,
    loading: true,
    toast: null,
    treeFilter: '',
    systemDiagram: null,
    projectName: '',
    projectStats: '',
    codeOpen: true,
    showExportModal: false,
    exporting: false,
    exportStatus: { llm_available: false, provider: null, model: null },
    exportOpts: { includeCode: false, includeDiagram: true, enhanced: false, depth: 'modules' },

    async init() {
      try {
        const res = await fetch('/api/tree');
        const data = await res.json();
        this.tree = data.tree;
        this.projectName = data.project || '';
        const stats = data.stats || {};
        this.projectStats = `${stats.files || 0} files · ${stats.functions || 0} functions · ${stats.classes || 0} classes`;
      } catch(e) {
        console.error('Failed to load tree:', e);
      }
      this.loading = false;
    },

    filteredTree(children) {
      if (!this.treeFilter) return children || [];
      const q = this.treeFilter.toLowerCase();
      return (children || []).filter(c => {
        if (c.name.toLowerCase().includes(q)) return true;
        if (c.type === 'dir' && c.children) return this.filteredTree(c.children).length > 0;
        return false;
      });
    },

    async loadFile(path) {
      this.selectedFile = path;
      this.systemDiagram = null;
      this.fileLoading = true;
      this.fileData = null;
      this.codeOpen = true;
      try {
        const res = await fetch('/api/file/' + encodeURIComponent(path));
        this.fileData = await res.json();
        this.$nextTick(() => {
          this.highlightCode();
          this.renderFileMermaid();
        });
      } catch(e) {
        console.error('Failed to load file:', e);
        this.fileData = { name: path.split('/').pop(), path: path, content: 'Error loading file', deps: [], dependents: [], functions: [], classes: [] };
      }
      this.fileLoading = false;
    },

    async loadSystemDiagram() {
      this.selectedFile = null;
      this.fileData = null;
      try {
        const res = await fetch('/api/diagram');
        const data = await res.json();
        this.systemDiagram = data.mermaid;
        this.$nextTick(() => this.renderSystemMermaid());
      } catch(e) {
        console.error('Failed to load diagram:', e);
      }
    },

    async triggerExplain() {
      if (!this.selectedFile) return;
      try {
        const res = await fetch('/api/explain/' + encodeURIComponent(this.selectedFile));
        const data = await res.json();
        if (data.explanation && this.fileData) {
          this.fileData.explanation = data.explanation;
        } else if (data.error) {
          this.showToast('LLM not available');
        }
      } catch(e) { this.showToast('Explain failed'); }
    },

    highlightCode() {
      if (!this.fileData?.content || !this.$refs.codeBlock) return;
      const ext = (this.fileData.name || '').split('.').pop();
      const langMap = { py: 'python', js: 'javascript', ts: 'typescript', jsx: 'javascript', tsx: 'typescript' };
      const lang = langMap[ext] || ext || 'plaintext';
      this.$refs.codeBlock.className = 'hljs language-' + lang;
      this.$refs.codeBlock.textContent = this.fileData.content;
      hljs.highlightElement(this.$refs.codeBlock);
    },

    async renderFileMermaid() {
      if (!this.fileData?.mermaid || !this.$refs.fileMermaid) return;
      try {
        const id = 'fm-' + Date.now();
        const { svg } = await mermaid.render(id, this.fileData.mermaid);
        this.$refs.fileMermaid.innerHTML = svg;
      } catch(e) { this.$refs.fileMermaid.innerHTML = '<span class="text-red-400 text-xs">Diagram render error</span>'; }
    },

    async renderSystemMermaid() {
      if (!this.systemDiagram || !this.$refs.systemMermaid) return;
      try {
        const id = 'sm-' + Date.now();
        const { svg } = await mermaid.render(id, this.systemDiagram);
        this.$refs.systemMermaid.innerHTML = svg;
      } catch(e) { this.$refs.systemMermaid.innerHTML = '<span class="text-red-400 text-xs">Diagram render error</span>'; }
    },

    fileIcon(name) {
      const ext = (name || '').split('.').pop();
      const icons = { py: '🐍', js: '📜', ts: '🔷', jsx: '⚛️', tsx: '⚛️', json: '📋', md: '📝', yml: '⚙️', yaml: '⚙️', toml: '⚙️', html: '🌐', css: '🎨', go: '🔹', rs: '🦀', java: '☕', rb: '💎' };
      return icons[ext] || '📄';
    },

    exportMermaid() {
      if (this.fileData?.mermaid) {
        this.copyToClipboard(this.fileData.mermaid);
        this.showToast('Mermaid copied!');
      }
    },

    exportDiagramSVG(refName, filename) {
      const el = this.$refs[refName];
      if (!el) return;
      const svg = el.querySelector('svg');
      if (!svg) { this.showToast('No diagram rendered'); return; }
      const svgData = new XMLSerializer().serializeToString(svg);
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (filename || 'diagram') + '.svg';
      a.click();
      URL.revokeObjectURL(a.href);
      this.showToast('SVG downloaded!');
    },

    openMermaidLive(mermaidCode) {
      if (!mermaidCode) return;
      try {
        const state = JSON.stringify({
          code: mermaidCode,
          mermaid: JSON.stringify({ theme: 'dark' }),
          autoSync: true,
          updateDiagram: true
        });
        const data = new TextEncoder().encode(state);
        const compressed = pako.deflate(data, { level: 9 });
        let b64 = '';
        for (let i = 0; i < compressed.length; i++) {
          b64 += String.fromCharCode(compressed[i]);
        }
        const encoded = btoa(b64).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
        window.open('https://mermaid.live/edit#pako:' + encoded, '_blank');
      } catch(e) {
        console.error('Mermaid Live encoding failed:', e);
        this.copyToClipboard(mermaidCode);
        this.showToast('Copied — paste at mermaid.live/edit');
      }
    },

    async loadExportStatus() {
      try {
        const res = await fetch('/api/export/status');
        this.exportStatus = await res.json();
      } catch(e) { console.error('Export status failed:', e); }
    },

    async startExport() {
      this.exporting = true;
      try {
        const params = new URLSearchParams({
          code: this.exportOpts.includeCode,
          enhanced: this.exportOpts.enhanced,
          depth: this.exportOpts.depth,
          no_diagram: !this.exportOpts.includeDiagram,
        });
        const res = await fetch('/api/export?' + params.toString());
        if (!res.ok) {
          const err = await res.json();
          this.showToast('Export failed: ' + (err.error || 'unknown'));
          this.exporting = false;
          return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const cd = res.headers.get('Content-Disposition') || '';
        const filenameMatch = cd.match(/filename="(.+?)"/);
        a.download = filenameMatch ? filenameMatch[1] : 'project-docs.docx';
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('DOCX downloaded!');
        this.showExportModal = false;
      } catch(e) {
        console.error('Export failed:', e);
        this.showToast('Export failed');
      }
      this.exporting = false;
    },

    copyToClipboard(text) { navigator.clipboard.writeText(text).catch(() => {}); },

    showToast(msg) {
      this.toast = msg;
      setTimeout(() => this.toast = null, 2000);
    }
  };
}
</script>
</body>
</html>"""


# ===================================================================
# Starlette app + Uvicorn server
# ===================================================================


def _create_server(store: Any, llm_provider: str, llm_model: str, llm_api_key: str):
    """Create the Starlette ASGI application."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    async def homepage(request):
        return HTMLResponse(HTML_TEMPLATE)

    async def api_tree(request):
        try:
            tree = _build_api_tree(store)
            nodes = store.get_nodes()
            files = set(n["file_path"] for n in nodes)
            functions = sum(1 for n in nodes if n["node_type"] == "function")
            classes = sum(1 for n in nodes if n["node_type"] == "class")
            meta = store.get_metadata()
            return JSONResponse({
                "tree": tree,
                "project": meta.get("project_name", ""),
                "stats": {"files": len(files), "functions": functions, "classes": classes, "nodes": len(nodes)},
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_file(request):
        file_path = urllib.parse.unquote(request.path_params["path"])
        try:
            data = _analyze_file(store, file_path)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_diagram(request):
        try:
            mermaid = _generate_system_mermaid(store)
            return JSONResponse({"mermaid": mermaid})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_explain(request):
        file_path = urllib.parse.unquote(request.path_params["path"])
        try:
            explanation = _explain_file(store, file_path, llm_provider, llm_model, llm_api_key)
            if explanation:
                return JSONResponse({"explanation": explanation})
            return JSONResponse({"error": "LLM not configured or unavailable"}, status_code=200)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_export(request):
        """Generate and return DOCX as a file download."""
        try:
            import tempfile
            from .cli_export import generate_basic_docx, generate_enhanced_docx
            from starlette.responses import Response

            params = request.query_params
            include_code = params.get("code", "false").lower() == "true"
            enhanced = params.get("enhanced", "false").lower() == "true"
            depth = params.get("depth", "modules")
            no_diagram = params.get("no_diagram", "false").lower() == "true"

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            if enhanced and llm_provider:
                generate_enhanced_docx(
                    store=store,
                    output_path=tmp_path,
                    include_code=include_code,
                    include_diagram=not no_diagram,
                    explanation_depth=depth,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                )
            else:
                generate_basic_docx(
                    store=store,
                    output_path=tmp_path,
                    include_code=include_code,
                    include_diagram=not no_diagram,
                )

            data = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)

            meta = store.get_metadata()
            filename = f"{meta.get('project_name', 'project')}-docs.docx"

            return Response(
                content=data,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as e:
            logger.exception("Export failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def api_export_status(request):
        """Return export capabilities (whether LLM is available etc)."""
        return JSONResponse({
            "llm_available": bool(llm_provider and llm_model),
            "provider": llm_provider or None,
            "model": llm_model or None,
        })

    app = Starlette(routes=[
        Route("/", homepage),
        Route("/api/tree", api_tree),
        Route("/api/file/{path:path}", api_file),
        Route("/api/diagram", api_diagram),
        Route("/api/explain/{path:path}", api_explain),
        Route("/api/export", api_export),
        Route("/api/export/status", api_export_status),
    ])
    return app


# ===================================================================
# Typer command
# ===================================================================


@explore_app.command("open")
def explore_open(
    port: int = typer.Option(8421, "--port", "-p", help="Port for the local web server."),
):
    """🌐 Open the visual code explorer in your browser.

    Launches a local web server and opens a modern UI for navigating
    your indexed codebase with syntax highlighting, dependency graphs,
    AI explanations, and Mermaid diagrams.

    Example:
      cg explore open
      cg explore open --port 9000
    """
    import socket
    import threading
    import time
    import webbrowser

    from rich.console import Console

    from .storage import GraphStore, ProjectManager
    from . import config as cfg

    console = Console()

    pm = ProjectManager()
    project = pm.get_current_project()
    if not project:
        console.print("[red]No project loaded.[/red] Run [cyan]cg project index <path>[/cyan] first.")
        raise typer.Exit(code=1)

    project_dir = pm.project_dir(project)
    if not project_dir.exists():
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(code=1)

    store = GraphStore(project_dir)

    # Verify the project has nodes
    nodes = store.get_nodes()
    if not nodes:
        console.print("[red]Project is empty — no indexed nodes found.[/red] Re-run [cyan]cg project index[/cyan].")
        store.close()
        raise typer.Exit(code=1)

    # Check if port is available
    actual_port = port
    for attempt in range(5):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", actual_port)) != 0:
                break
            actual_port += 1
    else:
        console.print(f"[red]Ports {port}-{port+4} are all in use.[/red]")
        store.close()
        raise typer.Exit(code=1)

    server_app = _create_server(
        store,
        llm_provider=cfg.LLM_PROVIDER,
        llm_model=cfg.LLM_MODEL,
        llm_api_key=cfg.LLM_API_KEY,
    )

    url = f"http://127.0.0.1:{actual_port}"
    console.print(f"\n[bold green]🌐 CodeGraph Explorer[/bold green]")
    console.print(f"   Project: [cyan]{project}[/cyan]")
    console.print(f"   URL:     [link={url}]{url}[/link]")
    console.print(f"   Nodes:   {len(nodes)}")
    console.print(f"\n   [dim]Press Ctrl+C to stop the server[/dim]\n")

    # Open browser after a short delay
    def _open_browser():
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        import uvicorn
        uvicorn.run(server_app, host="127.0.0.1", port=actual_port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        store.close()
        console.print("\n[dim]Server stopped.[/dim]")
