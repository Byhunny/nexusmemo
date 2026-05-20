"""In-memory knowledge graph backed by NetworkX."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx

from nexusmemo.database import Database, NodeRow, EdgeRow

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    name: str
    type: str
    importance: float = 0.5


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""

    source: str
    target: str
    relation: str
    weight: float = 1.0


class GraphManager:
    """Manage an in-memory knowledge graph with NetworkX."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def load_from_db(self, db: Database) -> None:
        """Load the full graph from database into memory."""
        self.graph.clear()

        with db.session() as session:
            nodes = session.query(NodeRow).all()
            for node in nodes:
                self.graph.add_node(
                    node.id,
                    name=node.name,
                    type=node.type,
                    importance=node.importance_score,
                    description=node.description or "",
                )

            edges = session.query(EdgeRow).all()
            for edge in edges:
                if self.graph.has_node(edge.source_id) and self.graph.has_node(
                    edge.target_id
                ):
                    self.graph.add_edge(
                        edge.source_id,
                        edge.target_id,
                        relation=edge.relation,
                        weight=edge.weight,
                        context=edge.context or "",
                    )

        logger.info(
            "Loaded graph: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def add_node(self, node_id: str, name: str, node_type: str, **attrs) -> None:
        """Add or update a node in the graph."""
        self.graph.add_node(node_id, name=name, type=node_type, **attrs)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        context: str = "",
    ) -> None:
        """Add an edge between two nodes."""
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(
                source_id,
                target_id,
                relation=relation,
                weight=weight,
                context=context,
            )

    def get_neighbors(
        self, node_id: str, depth: int = 1
    ) -> list[dict]:
        """Get neighbors of a node up to a given depth."""
        if not self.graph.has_node(node_id):
            return []

        visited = set()
        neighbors = []
        queue = [(node_id, 0)]

        while queue:
            current, current_depth = queue.pop(0)
            if current in visited or current_depth > depth:
                continue
            visited.add(current)

            if current != node_id:
                node_data = self.graph.nodes[current]
                # find the edge that led here
                edge_info = []
                for pred in self.graph.predecessors(current):
                    if pred in visited or pred == node_id:
                        edge_data = self.graph.edges[pred, current]
                        edge_info.append(
                            {
                                "from": self.graph.nodes[pred].get("name", pred),
                                "relation": edge_data.get("relation", "RELATED_TO"),
                            }
                        )
                for succ in self.graph.successors(current):
                    if succ in visited:
                        edge_data = self.graph.edges[current, succ]
                        edge_info.append(
                            {
                                "to": self.graph.nodes[succ].get("name", succ),
                                "relation": edge_data.get("relation", "RELATED_TO"),
                            }
                        )

                neighbors.append(
                    {
                        "id": current,
                        "name": node_data.get("name", current),
                        "type": node_data.get("type", "concept"),
                        "importance": node_data.get("importance", 0.5),
                        "depth": current_depth,
                        "edges": edge_info,
                    }
                )

            if current_depth < depth:
                for successor in self.graph.successors(current):
                    queue.append((successor, current_depth + 1))
                for predecessor in self.graph.predecessors(current):
                    queue.append((predecessor, current_depth + 1))

        return neighbors

    def find_node_by_name(self, name: str) -> str | None:
        """Find a node ID by its name (case-insensitive)."""
        name_lower = name.lower()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("name", "").lower() == name_lower:
                return node_id
        return None

    def get_all_relations_for_node(self, node_id: str) -> list[dict]:
        """Get all incoming and outgoing relations for a node."""
        if not self.graph.has_node(node_id):
            return []

        relations = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            relations.append(
                {
                    "entity": self.graph.nodes[target].get("name", target),
                    "relation": data.get("relation", "RELATED_TO"),
                    "direction": "outgoing",
                }
            )
        for source, _, data in self.graph.in_edges(node_id, data=True):
            relations.append(
                {
                    "entity": self.graph.nodes[source].get("name", source),
                    "relation": data.get("relation", "RELATED_TO"),
                    "direction": "incoming",
                }
            )
        return relations

    def get_subgraph_text(self, node_ids: list[str]) -> str:
        """Generate a text representation of a subgraph for LLM context."""
        lines = []
        seen_edges = set()

        for node_id in node_ids:
            if not self.graph.has_node(node_id):
                continue
            data = self.graph.nodes[node_id]
            lines.append(f"- {data.get('name', node_id)} ({data.get('type', 'entity')})")

            for _, target, edata in self.graph.out_edges(node_id, data=True):
                edge_key = (node_id, target)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    source_name = data.get("name", node_id)
                    target_name = self.graph.nodes[target].get("name", target)
                    relation = edata.get("relation", "RELATED_TO")
                    lines.append(f"  → {source_name} --{relation}--> {target_name}")

        return "\n".join(lines)
