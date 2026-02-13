"""Core data models used by indexing, retrieval, and orchestration layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Node:
    node_id: str
    node_type: str
    name: str
    qualname: str
    file_path: str
    start_line: int
    end_line: int
    code: str
    docstring: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    edge_type: str


@dataclass
class SearchResult:
    node_id: str
    score: float
    node_type: str
    qualname: str
    file_path: str
    start_line: int
    end_line: int
    snippet: str


@dataclass
class ImpactReport:
    root: str
    impacted: List[str]
    explanation: str
    ascii_graph: str
