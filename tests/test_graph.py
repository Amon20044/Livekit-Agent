from app.graph import choose_transition, graph_is_acyclic
from app.models import GraphConfig, GraphEdge, GraphNode, TransitionCondition


def test_graph_is_acyclic_for_single_and_multi_agent_flows() -> None:
    assert graph_is_acyclic(
        GraphConfig(
            entry_node_key="a", nodes=[GraphNode(key="a", agent_id="agent-a")], edges=[]
        )
    )
    assert graph_is_acyclic(
        GraphConfig(
            entry_node_key="a",
            nodes=[
                GraphNode(key="a", agent_id="agent-a"),
                GraphNode(key="b", agent_id="agent-b"),
            ],
            edges=[
                GraphEdge(
                    from_node="a",
                    to_node="b",
                    priority=10,
                    condition=TransitionCondition(
                        field="intent", operator="eq", value="orders"
                    ),
                )
            ],
        )
    )


def test_graph_rejects_cycle() -> None:
    graph = GraphConfig(
        entry_node_key="a",
        nodes=[
            GraphNode(key="a", agent_id="agent-a"),
            GraphNode(key="b", agent_id="agent-b"),
        ],
        edges=[
            GraphEdge(
                from_node="a",
                to_node="b",
                priority=10,
                condition=TransitionCondition(
                    field="intent", operator="eq", value="orders"
                ),
            ),
            GraphEdge(
                from_node="b",
                to_node="a",
                priority=10,
                condition=TransitionCondition(
                    field="intent", operator="eq", value="shopping"
                ),
            ),
        ],
    )
    assert not graph_is_acyclic(graph)


def test_choose_transition_is_deterministic_and_priority_ordered() -> None:
    graph = GraphConfig(
        entry_node_key="triage",
        nodes=[
            GraphNode(key="triage", agent_id="a"),
            GraphNode(key="general", agent_id="b"),
            GraphNode(key="orders", agent_id="c"),
        ],
        edges=[
            GraphEdge(
                from_node="triage",
                to_node="general",
                priority=1,
                condition=TransitionCondition(field="intent", operator="exists"),
            ),
            GraphEdge(
                from_node="triage",
                to_node="orders",
                priority=20,
                condition=TransitionCondition(
                    field="intent", operator="eq", value="order_help"
                ),
            ),
        ],
    )
    edge = choose_transition(graph, "triage", {"intent": "order_help"})
    assert edge is not None
    assert edge.to_node == "orders"


def test_choose_transition_supports_in_and_truthy() -> None:
    graph = GraphConfig(
        entry_node_key="a",
        nodes=[GraphNode(key="a", agent_id="a"), GraphNode(key="b", agent_id="b")],
        edges=[
            GraphEdge(
                from_node="a",
                to_node="b",
                priority=1,
                condition=TransitionCondition(
                    field="tags", operator="in", value="urgent"
                ),
            )
        ],
    )
    assert choose_transition(graph, "a", {"tags": ["order", "urgent"]}) is not None
    assert choose_transition(graph, "a", {"tags": ["order"]}) is None
