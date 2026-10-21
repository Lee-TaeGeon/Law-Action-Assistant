"""LangGraph v9 테스트 버전: RAG v5 + 인용 검증 및 검증된 원문 자동 복구.

설치 위치: backend/graph/legal_graph_v9.py

기존 legal_graph_v8.py, 운영 main.py, 원본 SQLite/ChromaDB는 변경하지 않는다.
- v8의 질문 분류, RAG v5 검색, 누락 법령 감지, Reranker를 그대로 사용한다.
- LLM이 선택한 원문 인용을 v8 방식으로 엄격하게 검증한다.
- 인용 선택이 두 번 실패하면, 서버가 이미 선정된 출처의 저장된 본문에서
  짧은 *원문 일부*를 그대로 뽑아 동일한 검증 함수로 재검사한다.
- 복구가 불가능하면 기존처럼 안전하게 차단한다.
- 복구 결과는 '적용 법령의 확인'이나 법률적 의미 검증이 아니다.

이 모듈은 실험용이며 기존 운영 API에서 자동으로 사용되지 않는다.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.graph import legal_graph as original
from backend.graph import legal_graph_v5 as v5
from backend.graph import legal_graph_v6 as v6
from backend.graph import legal_graph_v7 as v7
from backend.graph import legal_graph_v8 as v8

logger = logging.getLogger(__name__)
class AgentState(v8.AgentState, total=False):
    evidence_selection_mode: str  # llm_verified / deterministic_fallback / none


def _candidate_quotes(source: dict) -> list[str]:
    """저장된 본문에서만 연속 원문 부분 문자열을 반환한다.

    요약/번역/법률 판단은 하지 않는다. 인용 후보에 머리말이 섞이지 않도록
    SQLite content의 메타데이터 헤더는 v8의 기존 파서로 제외한다.
    """
    body = v8._law_body(source).strip()
    if not body:
        return []

    candidates: list[str] = []
    # 본문 줄과 첫 문장부터 시도해 '짧은 원문 발췌'가 되도록 한다.
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # 조문 전체가 아주 길더라도 문장 경계가 있는 부분까지 우선 채택.
        first_sentence = re.split(r"(?<=[.!?。])\s+", line, maxsplit=1)[0].strip()
        for text in (first_sentence, line):
            if text and len(v8._compact(text)) >= 12:
                candidates.append(text[:350].strip())
        if candidates:
            break

    if not candidates and len(v8._compact(body)) >= 12:
        candidates.append(body[:350].strip())
    # 길이나 연속 인용 여부는 최종적으로 v8 validator에서 검사한다.
    return list(dict.fromkeys(candidates))


def deterministic_verified_evidence(sources: list[dict], max_quotes: int = 2) -> list[dict]:
    """LLM 실패 시 검증된 발췌문을 생성한다. 법령 적용을 판단하지 않는다."""
    result: list[dict] = []
    if not isinstance(sources, list) or not sources:
        return result
    for index, src in enumerate(sources[:4], start=1):
        for quote in _candidate_quotes(src):
            payload = {"findings": [{"source_id": f"S{index}", "evidence": quote}]}
            try:
                checked = v8.validate_evidence_draft(payload, sources)
            except v8.EvidenceError:
                continue
            result.extend(checked)
            break
        if len(result) >= max_quotes:
            break
    return result


def answer_from_sources(question: str, sources: list[dict], llm: Any) -> dict[str, str]:
    """출처가 있으면 LLM 인용을 먼저 시도하고, 검증 실패 시 원문으로 복구."""
    if not sources:
        return {"answer": "검색·검증된 법령 근거가 없어 답변하지 않았습니다.",
                "answer_validation_status": "no_sources", "evidence_selection_mode": "none"}

    verified, status = v8.select_verified_evidence(question, sources, llm)
    if status == "ok" and verified:
        return {"answer": v8.render_evidence_answer(verified, question),
                "answer_validation_status": "ok", "evidence_selection_mode": "llm_verified"}

    fallback = deterministic_verified_evidence(sources)
    if fallback:
        answer = v8.render_evidence_answer(fallback, question)
        answer = (
            "※ AI의 인용문 생성이 검증에 실패하여 검색된 법령 본문에서 "
            "직접 추출한 원문 일부를 표시합니다. 법적 적용 판단은 아닙니다.\n\n" + answer
        )
        return {"answer": answer, "answer_validation_status": "ok",
                "evidence_selection_mode": "deterministic_fallback"}

    return {
        "answer": "관련 법령을 검색했지만 검증 가능한 원문 구절을 확보하지 못해 답변하지 않았습니다.",
        "answer_validation_status": "blocked", "evidence_selection_mode": "none",
    }


def generator_node(state: AgentState) -> dict[str, str]:
    sources = list(state.get("sources") or [])
    if not sources:
        # 기존 v8의 no_sources / reranker_error 처리 보존.
        return v8.generator_node(state)
    question = str(state.get("question") or "").strip()
    return answer_from_sources(question, sources, original.get_answer_llm())


def reference_guard_node(state: AgentState) -> dict:
    """새 질문마다 이전 검증 경로 표시를 초기화한다."""
    return {**v6.reference_guard_node(state), "evidence_selection_mode": "none"}


# v8의 참조 검증·RAG v5·Reranker 구조를 그대로 유지한다.
workflow = StateGraph(AgentState)
for name, node in (
    ("reference_guard", reference_guard_node),
    ("missing_response", v6.missing_response_node),
    ("classifier", v5.classifier_node),
    ("query_rewriter", v5.query_rewriter_node),
    ("researcher", v7.researcher_node),
    ("reranker", v5.reranker_node),
    ("generator", generator_node),
):
    workflow.add_node(name, node)
workflow.set_entry_point("reference_guard")
workflow.add_conditional_edges(
    "reference_guard", v6.route_after_guard,
    {"missing": "missing_response", "continue": "classifier"},
)
workflow.add_edge("missing_response", END)
for start, end in (
    ("classifier", "query_rewriter"),
    ("query_rewriter", "researcher"),
    ("researcher", "reranker"),
    ("reranker", "generator"),
    ("generator", END),
):
    workflow.add_edge(start, end)
legal_graph = workflow.compile(checkpointer=MemorySaver())


def run_legal_chat(question: str, session_id: str) -> dict:
    if not str(question or "").strip() or not str(session_id or "").strip():
        raise ValueError("질문과 session_id가 필요합니다.")
    return legal_graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": session_id}},
    )


def self_test() -> None:
    """Ollama·SQLite·ChromaDB 호출 없이 복구 동작을 모의 검증."""
    from types import SimpleNamespace
    import json

    source = {"law_name": "민법", "article": "제394조", "article_title": "손해배상의 방법",
              "content": "<민법>\n제394조 (손해배상의 방법)\n다른 의사표시가 없으면 손해는 금전으로 배상한다."}
    source2 = {"law_name": "형법", "article": "제347조", "article_title": "사기",
               "content": "<형법>\n제347조 (사기)\n사람을 기망하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는 처벌받는다."}

    class FakeLLM:
        def __init__(self, outputs):
            self.outputs = iter(outputs)
            self.calls = 0
        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content=next(self.outputs))

    bad = '{"findings":[{"source_id":"S1","evidence":"금전"}]}'
    model = FakeLLM([bad, bad])
    recovered = answer_from_sources("계약에서 손해배상은?", [source, source2], model)
    assert recovered["answer_validation_status"] == "ok"
    assert recovered["evidence_selection_mode"] == "deterministic_fallback"
    assert model.calls == 2
    assert "다른 의사표시가 없으면 손해는 금전으로 배상한다." in recovered["answer"]
    assert "민법 제394조" in recovered["answer"]
    assert "금전배상만 인정" not in recovered["answer"]
    print("[통과] 짧은 인용 2회 실패 후 저장 원문으로 복구")

    good = {"findings": [{"source_id": "S1", "evidence": "다른 의사표시가 없으면 손해는 금전으로 배상한다."}]}
    model = FakeLLM([json.dumps(good, ensure_ascii=False)])
    result = answer_from_sources("손해배상", [source], model)
    assert result["evidence_selection_mode"] == "llm_verified" and model.calls == 1
    print("[통과] 정상적인 LLM 인용은 기존 검증 경로 유지")

    broken = [{"law_name": "민법", "article": "제0조", "content": "짧음"}]
    model = FakeLLM([bad, bad])
    result = answer_from_sources("질문", broken, model)
    assert result["answer_validation_status"] == "blocked"
    assert result["evidence_selection_mode"] == "none"
    assert deterministic_verified_evidence([]) == []
    print("[통과] 검증할 본문이 없으면 창작하지 않고 차단")
