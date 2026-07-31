from __future__ import annotations

from typing import Any

from app.models import GraphConfig, GraphEdge, TransitionCondition


def graph_is_acyclic(graph: GraphConfig) -> bool:
    nodes = {node.key for node in graph.nodes}
    if not nodes or graph.entry_node_key not in nodes:
        return False
    adjacency: dict[str, list[str]] = {key: [] for key in nodes}
    for edge in graph.edges:
        if edge.from_node not in nodes or edge.to_node not in nodes:
            return False
        adjacency[edge.from_node].append(edge.to_node)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        if not all(visit(next_node) for next_node in adjacency[node]):
            return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in nodes)


def choose_transition(
    graph: GraphConfig, current_node: str, captured: dict[str, Any]
) -> GraphEdge | None:
    candidates = sorted(
        (edge for edge in graph.edges if edge.from_node == current_node),
        key=lambda edge: edge.priority,
        reverse=True,
    )
    return next(
        (edge for edge in candidates if condition_matches(edge.condition, captured)),
        None,
    )


def condition_matches(condition: TransitionCondition, captured: dict[str, Any]) -> bool:
    actual = _get_path(captured, condition.field)
    match condition.operator:
        case "eq":
            return _normalized(actual) == _normalized(condition.value)
        case "neq":
            return _normalized(actual) != _normalized(condition.value)
        case "in":
            if isinstance(actual, list):
                return _normalized(condition.value) in {
                    _normalized(item) for item in actual
                }
            if isinstance(condition.value, list):
                return _normalized(actual) in {
                    _normalized(item) for item in condition.value
                }
            return False
        case "exists":
            return actual not in (None, "", [], {})
        case "truthy":
            return bool(actual)
        case _:
            return False


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _normalized(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value
