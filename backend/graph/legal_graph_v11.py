"""LangGraph v11 (격리 테스트): RAG v6 + Generator v9 + 대화 문맥.

설치: backend/graph/legal_graph_v11.py
기존 main.py/legal_graph.py/SQLite/ChromaDB 및 v10 파일은 변경하지 않는다.

- 동일한 session_id에서는 *사용자 진술* 최근 최대 4개만 문맥으로 유지한다.
- 다른 session_id는 MemorySaver의 다른 thread_id로 격리한다.
- 명시 법령/조문 미수록 검사는 반드시 '이번 사용자 질문'만 대상으로 한다.
- v10의 RAG v6 검색, v9의 인용 검증/안전한 원문 대체 경로를 그대로 사용한다.
- 이 기억은 서버 프로세스의 메모리 안에서만 유지되고 재시작 시 사라진다.
- 사용자가 '새 사건' 등으로 시작하면 해당 thread의 이전 사용자 진술을 제외한다.
- 사용자 진술은 검증된 사실이 아니며 실제 사건의 법적 판단·현행성은 별도 확인해야 한다.

주의: 실험용. 인증 없는 session_id는 실제 서비스의 개인정보 격리 수단이 아니다.
필수 DB 환경변수는 이 파일 최초 import 전에 지정해야 한다.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.graph import legal_graph_v5 as v5
from backend.graph import legal_graph_v6 as v6
from backend.graph import legal_graph_v9 as v9
from backend.graph import legal_graph_v10 as v10


MAX_PREVIOUS_TURNS = 3
MAX_TURN_CHARS = 600
MAX_CURRENT_CHARS = 2000
RESET_PATTERN = re.compile(
    r"^\s*(?:새\s*(?:사건|상담|질문)|다른\s*사건|별도\s*사건|"
    r"이전\s*사건과\s*(?:무관|별개))\b"
)


class AgentState(v9.AgentState, total=False):
    incoming_question: str            # 이 API 요청에서 사용자가 실제 입력한 문장
    user_turn_history: list[str]      # 서버에 남는 사용자 진술만 (AI 답변은 저장하지 않음)
    followup_context_used: bool       # 이번 요청에서 이전 진술을 참조했는가


def _clean_turn(value: Any, max_chars: int = MAX_TURN_CHARS) -> str:
    """과거 진술에 과도한 길이/개행이 들어오지 않게 제한한다."""
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_chars]


def _history(state: dict) -> list[str]:
    raw = state.get("user_turn_history")
    if not isinstance(raw, list):
        return []
    result = [_clean_turn(t) for t in raw[-MAX_PREVIOUS_TURNS:]]
    return [t for t in result if t]


def _is_new_case(text: str) -> bool:
    return bool(RESET_PATTERN.match(text))


def contextual_question(previous: list[str], current: str) -> str:
    """이전 진술을 사용자 진술로만 표시한다. AI 답변을 사건 사실로 합치지 않는다."""
    if not previous:
        return current
    parts = [
        "[이전 대화의 사용자 진술 - 사실 확인 전이며 이번 질문의 문맥 참고용]",
    ]
    parts.extend(f"{idx}. {item}" for idx, item in enumerate(previous, 1))
    parts.extend(["", "[이번 사용자 질문 또는 추가 진술]", current])
    return "\n".join(parts)


def prepare_turn_node(state: AgentState) -> dict[str, Any]:
    current = str(state.get("incoming_question") or "").strip()
    if not current:
        raise ValueError("이번 사용자 질문이 없습니다. run_legal_chat()을 사용해 주세요.")
    if len(current) > MAX_CURRENT_CHARS:
        raise ValueError(f"사용자 질문은 최대 {MAX_CURRENT_CHARS}자까지 허용합니다.")
    previous = [] if _is_new_case(current) else _history(state)
    return {
        "question": contextual_question(previous, current),
        "user_turn_history": previous,
        "followup_context_used": bool(previous),
    }


def reference_guard_node(state: AgentState) -> dict[str, Any]:
    """구형 가드가 이전 메시지의 법령명을 '이번 명시 질의'로 오인하지 않게 한다."""
    current = str(state.get("incoming_question") or "").strip()
    return v9.reference_guard_node({**state, "question": current})


def _improve_followup_questions(answer: str, current: str, previous: list[str]) -> str:
    """배송 약속을 이미 밝힌 경우, 반복 질문을 더 구체적 사실 확인으로 바꾼다.

    어떤 법령도 추가하지 않고, 사용자 설명에서 확인된 표현만 사용한다.
    """
    if not previous or "## 사건 확인에 필요한 사실" not in answer:
        return answer
    c = _clean_turn(current)
    # 좁은 규칙: '판매자가 물건을 보내겠다고 여러 번 약속했다'는 추가 진술에 한정.
    if (
        "판매자" in c
        and "약속" in c
        and any(term in c for term in ("보내", "배송", "발송"))
    ):
        answer = answer.replace(
            "1. 거래 당시 판매자가 어떤 설명과 약속을 했나요?",
            "1. 판매자가 배송을 약속한 구체적인 날짜와 각 연락 내용은 무엇인가요?",
        )
        answer = answer.replace(
            "2. 대금 지급일과 물품의 실제 배송·인도 상황은 어떠한가요?",
            "2. 대금을 지급한 날짜와 마지막 배송 약속일은 언제인가요?",
        )
    return answer


def generator_node(state: AgentState) -> dict[str, Any]:
    """v9 인용·원문 복구는 그대로 사용하고 후속 질문의 표시만 조정한다."""
    result = v9.generator_node(state)
    if result.get("answer_validation_status") != "ok":
        return result
    previous = _history(state)
    current = str(state.get("incoming_question") or "").strip()
    if not previous:
        return result
    answer = _improve_followup_questions(str(result.get("answer") or ""), current, previous)
    # 실제 입력한 사용자 표현만 보여주며, 사실 인정이나 법적 평가를 추가하지 않는다.
    raw_statement = _clean_turn(current, max_chars=350)
    if raw_statement:
        answer = (
            "## 이번에 추가로 알려준 내용 (사용자 진술, 사실관계 미검증)\n"
            f"{raw_statement}\n\n" + answer
        )
    return {**result, "answer": answer}


def finish_turn_node(state: AgentState) -> dict[str, Any]:
    """답변 생성이 끝난 시점에만 이번 입력을 다음 턴용 문맥에 기록한다."""
    current = _clean_turn(state.get("incoming_question"))
    previous = _history(state)
    return {"user_turn_history": (previous + [current])[-MAX_PREVIOUS_TURNS:]}


workflow = StateGraph(AgentState)
for name, node in (
    ("prepare_turn", prepare_turn_node),
    ("reference_guard", reference_guard_node),
    ("missing_response", v6.missing_response_node),
    ("classifier", v5.classifier_node),
    ("query_rewriter", v5.query_rewriter_node),
    ("researcher", v10.researcher_node),
    ("reranker", v5.reranker_node),
    ("generator", generator_node),
    ("finish_turn", finish_turn_node),
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


def run_legal_chat(question: str, session_id: str) -> dict:
    if not str(question or "").strip() or not str(session_id or "").strip():
        raise ValueError("질문과 session_id가 필요합니다.")
    if len(question) > MAX_CURRENT_CHARS:
        raise ValueError(f"질문은 {MAX_CURRENT_CHARS}자 이하여야 합니다.")
    return legal_graph.invoke(
        {"incoming_question": question.strip()},
        config={"configurable": {"thread_id": session_id}},
    )


def self_test() -> None:
    """외부 DB/Ollama 없이 문맥·명시법령 검사·질문 보정 및 노드 연결 검사."""
    first = "중고거래 판매자가 돈을 받고 물건을 보내지 않았어요. 형사상 사기와 민사상 손해배상을 알고 싶어요."
    followup = "판매자가 물건을 보내겠다고 여러 번 약속했는데 아직 보내지 않았어요. 어떤 사실을 확인해야 하나요?"
    state1 = prepare_turn_node({"incoming_question": first, "user_turn_history": []})
    assert state1["question"] == first and not state1["followup_context_used"]
    saved1 = finish_turn_node({**state1, "incoming_question": first})
    state2 = prepare_turn_node({**saved1, "incoming_question": followup})
    assert state2["followup_context_used"]
    assert first in state2["question"] and followup in state2["question"]
    assert "[이번 사용자 질문" in state2["question"]
    assert _is_new_case("새 사건: 임금체불 관련 질문")
    assert not prepare_turn_node({**saved1, "incoming_question": "새 사건: 임금체불 관련 질문"})["followup_context_used"]
    assert len(finish_turn_node({"incoming_question": "추가", "user_turn_history": ["a", "b", "c"]})["user_turn_history"]) == 3
    print("[통과] 같은 세션의 사용자 진술 누적 / 신규 사건 문맥 초기화 / 최대 3개 과거 발화 유지")

    orig_guard = v9.reference_guard_node
    received = []
    try:
        def fake_guard(s):
            received.append(s.get("question"))
            return {"sources": [], "answer": "", "answer_validation_status": "pending"}
        v9.reference_guard_node = fake_guard
        reference_guard_node({**state2, "incoming_question": "의료법 제27조를 알려줘"})
        assert received == ["의료법 제27조를 알려줘"], received
    finally:
        v9.reference_guard_node = orig_guard
    print("[통과] 명시 법령·조문 검사는 이번 사용자 질문만 대상으로 수행")

    old = """## 확인된 법령 원문\n- **형법 제347조 (사기)**\n  - 법문 원문: “사람을 기망하여 재물을 교부받은 경우”\n\n## 사건 확인에 필요한 사실\n1. 거래 당시 판매자가 어떤 설명과 약속을 했나요?\n2. 대금 지급일과 물품의 실제 배송·인도 상황은 어떠한가요?\n\n## 검증 범위와 주의사항\n인용문 검증"""
    refined = _improve_followup_questions(old, followup, [first])
    assert "판매자가 배송을 약속한 구체적인 날짜" in refined
    assert "거래 당시 판매자가 어떤 설명과 약속" not in refined
    assert "**형법 제347조 (사기)**" in refined and "인용문 검증" in refined
    assert _improve_followup_questions(old, followup, []) == old
    print("[통과] 이미 알려준 배송 약속을 반복 질문하지 않고 후속 사실확인 질문으로 조정")

    expected = {
        "researcher": v10.researcher_node,
        "reranker": v5.reranker_node,
        "generator": generator_node,
        "reference_guard": reference_guard_node,
        "finish_turn": finish_turn_node,
    }
    for name, fn in expected.items():
        spec = workflow.nodes.get(name)
        assert spec is not None, name
        actual = getattr(getattr(spec, "runnable", None), "func", None)
        assert actual is fn, f"노드 연결 오류: {name}"
    print("[통과] RAG v6 / 기존 Reranker / v9 검증형 Generator / 문맥 노드 연결")
    print("단위 테스트 완료. Ollama, 실제 SQLite·ChromaDB는 호출하지 않았습니다.")


if __name__ == "__main__":
    self_test()
