"""LangGraph v13 workflow builder."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.graph import legal_graph_v5 as v5
from backend.graph import legal_graph_v6 as v6
from backend.graph import legal_graph_v10 as v10
from backend.graph import legal_graph_v11 as v11


def build_legal_graph(AgentState, reranker_node):
    workflow = StateGraph(AgentState)
    for name, node in (
        ("prepare_turn", v11.prepare_turn_node),
        ("reference_guard", v11.reference_guard_node),
        ("missing_response", v6.missing_response_node),
        ("classifier", v5.classifier_node),
        ("query_rewriter", v5.query_rewriter_node),
        ("researcher", v10.researcher_node),
        ("reranker", reranker_node),
        ("generator", v11.generator_node),
        ("finish_turn", v11.finish_turn_node),
    ):
        workflow.add_node(name, node)
    workflow.set_entry_point("prepare_turn")
    workflow.add_edge("prepare_turn", "reference_guard")
    workflow.add_conditional_edges(
        "reference_guard", v6.route_after_guard,
        {"missing": "missing_response", "continue": "classifier"},
    )
    workflow.add_edge("missing_response", "finish_turn")
    for start, end in (
        ("classifier", "query_rewriter"),
        ("query_rewriter", "researcher"),
        ("researcher", "reranker"),
        ("reranker", "generator"),
        ("generator", "finish_turn"),
        ("finish_turn", END),
    ):
        workflow.add_edge(start, end)
    legal_graph = workflow.compile(checkpointer=MemorySaver())
    return workflow, legal_graph
