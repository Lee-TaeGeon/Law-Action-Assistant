import pytest

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.graph import legal_graph_v11 as v11


@pytest.fixture
def memory_graph():
    """외부 DB와 LLM 없이 실제 대화 기억 노드만 검증."""
    workflow = StateGraph(v11.AgentState)

    workflow.add_node(
        "prepare_turn", v11.prepare_turn_node
    )
    workflow.add_node(
        "finish_turn", v11.finish_turn_node
    )

    workflow.set_entry_point("prepare_turn")
    workflow.add_edge("prepare_turn", "finish_turn")
    workflow.add_edge("finish_turn", END)

    return workflow.compile(checkpointer=MemorySaver())


def ask(graph, session_id, question):
    return graph.invoke(
        {"incoming_question": question},
        config={
            "configurable": {"thread_id": session_id}
        },
    )


def test_same_session_memory(memory_graph):
    first = ask(
        memory_graph, "session-a",
        "중고거래 판매자에게 50만 원을 송금했습니다."
    )

    assert not first["followup_context_used"]

    second = ask(
        memory_graph, "session-a",
        "아직 물건을 받지 못했습니다."
    )

    assert second["followup_context_used"]
    assert "50만 원" in second["question"]
    assert "아직 물건" in second["question"]


def test_different_session_isolation(memory_graph):
    ask(
        memory_graph, "session-a",
        "중고거래 사기 피해를 입었습니다."
    )

    result = ask(
        memory_graph, "session-b",
        "퇴직했는데 임금을 받지 못했습니다."
    )

    assert not result["followup_context_used"]
    assert "중고거래" not in result["question"]
    assert "퇴직" in result["question"]


def test_new_case_resets_context(memory_graph):
    ask(
        memory_graph, "session-a",
        "중고거래 사기 피해를 입었습니다."
    )

    result = ask(
        memory_graph, "session-a",
        "새 사건: 회사에서 임금을 받지 못했습니다."
    )

    assert not result["followup_context_used"]
    assert "중고거래" not in result["question"]
    assert len(result["user_turn_history"]) == 1


def test_max_three_previous_turns(memory_graph):
    for i in range(1, 6):
        result = ask(
            memory_graph, "session-a",
            f"추가 진술 {i}"
        )

    assert result["user_turn_history"] == [
        "추가 진술 3",
        "추가 진술 4",
        "추가 진술 5",
    ]


def test_reference_guard_uses_current_question(monkeypatch):
    received = []

    def fake_guard(state):
        received.append(state.get("question"))
        return {"missing_explicit_references": []}

    monkeypatch.setattr(
        v11.v9,
        "reference_guard_node",
        fake_guard,
    )

    state = {
        "incoming_question": "민법 제390조를 설명해 주세요.",
        "question": (
            "이전 질문: 형법 제347조\n"
            "현재 질문: 민법 제390조"
        ),
    }

    v11.reference_guard_node(state)

    assert received == [
        "민법 제390조를 설명해 주세요."
    ]


def test_invalid_question_rejected():
    with pytest.raises(ValueError):
        v11.prepare_turn_node({
            "incoming_question": ""
        })

    with pytest.raises(ValueError):
        v11.prepare_turn_node({
            "incoming_question": "A" * 2001
        })