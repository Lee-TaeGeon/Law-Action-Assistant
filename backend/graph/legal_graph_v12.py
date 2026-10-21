"""격리 시험용 LangGraph v12: v11 대화 문맥 + 제한적 적용범위 Reranker 가드.

저장 위치: backend/graph/legal_graph_v12.py

- v11의 사용자 진술 문맥·명시 조문 검사·RAG v6·v9 인용 검증/복구 유지.
- v4 Reranker가 선택하기 전/후에 '명백히 다른 특수 법률쟁점' 일부를 검토.
- 법령을 명시하여 질문한 경우 무조건 배제하지 않음.
- 이 가드는 모든 법률의 적용범위를 판정하거나 최신성을 보증하지 않음.
- 운영 main.py, legal_graph.py, SQLite, ChromaDB 및 이전 테스트 버전 변경 없음.
- DB 환경변수는 최초 import 전에 설정.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.graph import legal_graph_v3 as v3
from backend.graph import legal_graph_v4 as v4
from backend.graph import legal_graph_v5 as v5
from backend.graph import legal_graph_v6 as v6
from backend.graph import legal_graph_v10 as v10
from backend.graph import legal_graph_v11 as v11

logger = logging.getLogger(__name__)
AgentState = v11.AgentState


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(_compact(word) in text for word in words)


def _out_of_scope_reason(state: AgentState, source: dict) -> str | None:
    """명시적 적용대상이 불일치하는 일부 후보만 제한적으로 거른다.

    법령 해석을 자동으로 검증하는 함수가 아니다. 특히 명시 법령 질문은
    배제하지 않고 후속 단계가 적용 여부를 검토하도록 유지한다.
    """
    current = str(state.get("incoming_question") or state.get("question") or "")
    context = str(state.get("question") or current)
    # 사용자가 특정 법령과 조문을 직접 질문했다면 그 조문을 확인할 기회를 남긴다.
    if v3._explicit_ref(current, source):
        return None

    eligible, reason = v3.scope_eligibility(context, source)
    if not eligible:
        return reason or "특수 적용범위 확인되지 않음"

    law = _compact(source.get("law_name"))
    article = _compact(source.get("article"))
    title = _compact(source.get("article_title"))
    q = _compact(context)

    # 추상적인 '사기' 또는 '승인'이라는 단어가 법률상 서로 다른 분야의
    # 조문을 동일 사건의 직접 근거로 끌어올리는 문제를 제한적으로 방지.
    if law == "민법":
        if "혼인" in title and not _has_any(q, ("혼인", "결혼", "배우자", "부부")):
            return "혼인 관계의 사실관계가 확인되지 않음"
        if "이혼" in title and not _has_any(q, ("이혼", "혼인", "배우자", "부부")):
            return "이혼 관련 사실관계가 확인되지 않음"
        if "약혼" in title and not _has_any(q, ("약혼", "혼약")):
            return "약혼 관련 사실관계가 확인되지 않음"
        if "상속" in title and not _has_any(q, ("상속", "유산", "피상속", "상속인")):
            return "상속 관련 사실관계가 확인되지 않음"
        if "상계" in title and not _has_any(q, ("상계", "채권채무를서로", "서로공제")):
            return "상계 쟁점이 확인되지 않음"
        # 이번에 재현된 '민법 제854조' 혼입: 본문의 '승인'은 일반적인
        # 중고거래 배송 약속과 같은 의미가 아니다. 특정 신분관계나
        # 이 조문 자체에 대한 질문인 경우에는 배제하지 않는다.
        if article == "제854조" and "승인의취소" in title and not _has_any(
            q, ("승인", "인지", "친생", "양육", "친권", "가족관계", "제854조")
        ):
            return "승인 취소에 관한 별도 법률관계가 질문에 확인되지 않음"

    if law == "형법" and title.startswith("전시") and not _has_any(
        q, ("전시", "전쟁", "전시군수", "전시공수")
    ):
        return "전시 상황이 확인되지 않음"

    return None


def reranker_node(state: AgentState) -> dict[str, Any]:
    """v4 Reranker를 그대로 사용하되 명백한 다른 쟁점만 전후로 검사."""
    original_candidates = list(state.get("candidate_sources") or [])
    candidates: list[dict] = []
    for candidate in original_candidates:
        if not isinstance(candidate, dict):
            continue
        reason = _out_of_scope_reason(state, candidate)
        if reason:
            logger.info("Reranker v12 특수 적용범위 후보 보류: %s %s (%s)",
                        candidate.get("law_name"), candidate.get("article"), reason)
        else:
            candidates.append(candidate)

    if not candidates:
        return {"sources": [], "reranker_status": "none"}

    # 파싱·재시도와 LLM 조문 선택의 안전장치는 기존 v4 로직 재사용.
    selected = v4.reranker_node({**state, "candidate_sources": candidates})
    if not isinstance(selected, dict):
        return {"sources": [], "reranker_status": "format_error"}
    if selected.get("reranker_status") != "ok":
        return selected

    verified: list[dict] = []
    for source in selected.get("sources") or []:
        if not isinstance(source, dict):
            continue
        reason = _out_of_scope_reason(state, source)
        if reason:
            logger.warning("Reranker v12 선정 후 보류: %s %s (%s)",
                           source.get("law_name"), source.get("article"), reason)
        else:
            verified.append(source)
    return {"sources": verified,
            "reranker_status": "ok" if verified else "none"}


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


def run_legal_chat(question: str, session_id: str) -> dict:
    if not str(question or "").strip() or not str(session_id or "").strip():
        raise ValueError("질문과 session_id가 필요합니다.")
    if len(question) > v11.MAX_CURRENT_CHARS:
        raise ValueError(f"질문은 {v11.MAX_CURRENT_CHARS}자 이하여야 합니다.")
    return legal_graph.invoke(
        {"incoming_question": question.strip()},
        config={"configurable": {"thread_id": session_id}},
    )


def self_test() -> None:
    """Ollama·SQLite·ChromaDB 호출 없이 필터/선정/노드 연결 확인."""
    from types import SimpleNamespace

    case = "중고거래 판매자가 돈을 받고 물건을 보내지 않았어요. 형사상 사기와 민사상 손해배상을 알고 싶어요."
    followup = "판매자가 물건을 보내겠다고 여러 번 약속했는데 아직 보내지 않았어요. 어떤 사실을 추가로 확인하나요?"
    state = {"incoming_question": followup, "question": v11.contextual_question([case], followup), "category": "형사"}

    def source(law, article, title):
        return {"law_name": law, "article": article, "article_title": title,
                "section_scope": "", "content": "실제 근거 판단이 필요한 조문 본문"}

    fraud = source("형법", "제347조", "사기")
    tort = source("민법", "제750조", "불법행위의 내용")
    family = source("민법", "제854조", "사기, 강박으로 인한 승인의 취소")
    marriage = source("민법", "제823조", "사기, 강박으로 인한 혼인취소청구권의 소멸")
    offset = source("민법", "제496조", "불법행위채권을 수동채권으로 하는 상계의 금지")
    wartime = source("형법", "제103조", "전시군수계약불이행")
    kept = [s for s in (fraud, tort, family, marriage, offset, wartime) if not _out_of_scope_reason(state, s)]
    assert kept == [fraud, tort], kept
    print("[통과] 중고거래 후속 질문에서 일반 사기·불법행위 후보 유지, 무관한 특수조문 보류")

    assert _out_of_scope_reason({"incoming_question": "민법 제854조를 설명해 줘"}, family) is None
    assert _out_of_scope_reason({"incoming_question": "혼인 취소는 언제 가능한가요?"}, marriage) is None
    assert _out_of_scope_reason({"incoming_question": "전시군수계약불이행은 어떤 죄인가요?"}, wartime) is None
    assert _out_of_scope_reason({"incoming_question": "상계가 가능한가요?"}, offset) is None
    print("[통과] 명시적 조문 질문·실제 특수 사실관계에서는 관련 후보 보존")

    orig = v4.original.get_fast_llm
    try:
        v4.original.get_fast_llm = lambda: SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="C00,C01")
        )
        result = reranker_node({**state, "candidate_sources": [fraud, family, tort, offset]})
        assert result["reranker_status"] == "ok" and result["sources"] == [fraud, tort], result
        none = reranker_node({**state, "candidate_sources": [family, marriage]})
        assert none == {"sources": [], "reranker_status": "none"}, none
    finally:
        v4.original.get_fast_llm = orig
    print("[통과] 기존 v4 Reranker 연결·조문 선정/무근거 응답 처리")

    expected = {
        "prepare_turn": v11.prepare_turn_node,
        "researcher": v10.researcher_node,
        "reranker": reranker_node,
        "generator": v11.generator_node,
        "finish_turn": v11.finish_turn_node,
    }
    for name, func in expected.items():
        spec = workflow.nodes.get(name)
        assert spec is not None, name
        actual = getattr(getattr(spec, "runnable", None), "func", None)
        assert actual is func, f"노드 연결 불일치: {name}"
    print("[통과] v11 대화 문맥·RAG v6·v9 인용 검증 복구 노드 유지")
    print("단위 테스트 완료. 실제 DB·Ollama는 호출하지 않았습니다.")


if __name__ == "__main__":
    self_test()
