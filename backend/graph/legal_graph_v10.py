"""검증용 LangGraph v10: Hybrid RAG v6 + v9의 검증형 Generator.

설치: backend/graph/legal_graph_v10.py

- v9의 명시 법령·조문 미수록 감지, 분류, 재작성, 재정렬,
  원문 인용 검증 및 검증된 법문 복구를 유지한다.
- researcher 노드에서만 RAG v5 대신 RAG v6의 hybrid_search_laws를 사용한다.
- 검색된 조문의 SQLite 내부 문서 ID를 기준으로 중복을 제거한다.
- 법령이 실제 사건에 적용되는지/시행 시점이 맞는지는 판정하지 않는다.
- 기존 운영 backend/main.py, legal_graph.py 및 법령 DB를 변경하지 않는다.

필수: LAW_SEARCH_DB_PATH, LAW_VECTOR_DB_PATH는 최초 모듈 import 전에 설정.
"""
from __future__ import annotations

import logging
import re
from contextlib import closing
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.graph import legal_graph_v3 as v3
from backend.graph import legal_graph_v5 as v5
from backend.graph import legal_graph_v6 as v6
from backend.graph import legal_graph_v7 as v7
from backend.graph import legal_graph_v9 as v9
from backend.services import rag_service_v6 as rag_v6

logger = logging.getLogger(__name__)
AgentState = v9.AgentState


def _search_candidates(question: str, search_query: str, category: str) -> list[dict]:
    """RAG v6 단일 검색 진입점. 테스트에서 이 함수만 대체할 수 있다."""
    return rag_v6.hybrid_search_laws(
        question=question, search_query=search_query, category=category, k=16
    )


def researcher_node(state: AgentState) -> dict[str, Any]:
    """RAG v6 검색 결과에서 중복/빈 본문/삭제된 조문을 제외하고 절 범위를 보강."""
    question = str(state.get("question") or "").strip()
    if not question:
        return {"candidate_sources": []}

    search_query = str(state.get("search_query") or "").strip()
    category = str(state.get("category") or "기타")
    raw_candidates = _search_candidates(question, search_query, category)
    usable: list[dict] = []
    seen: set[tuple] = set()

    with closing(rag_v6.get_search_connection()) as db:
        for source in raw_candidates:
            if not isinstance(source, dict):
                continue
            content = str(source.get("content") or "")
            if not content.strip():
                continue
            identity = v7._identity(source)
            if identity in seen:
                continue
            # 원문에서 '제N조 삭제'로 확인되는 삭제 조문은 근거 후보로 사용하지 않는다.
            # 모든 개정·삭제 이력과 사건 시점까지 포괄하는 검사는 별도 필요하다.
            if re.search(r"제\s*\d+\s*조(?:의\s*\d+)?\s*삭제\b", content):
                continue
            candidate = dict(source)
            candidate["section_scope"] = v3._section_scope(db, candidate)
            usable.append(candidate)
            seen.add(identity)

    logger.info("RAG v6 후보 %d건 중 유효 후보 %d건", len(raw_candidates), len(usable))
    return {"candidate_sources": usable}


workflow = StateGraph(AgentState)
for node_name, node in (
    ("reference_guard", v9.reference_guard_node),
    ("missing_response", v6.missing_response_node),
    ("classifier", v5.classifier_node),
    ("query_rewriter", v5.query_rewriter_node),
    ("researcher", researcher_node),
    ("reranker", v5.reranker_node),
    ("generator", v9.generator_node),
):
    workflow.add_node(node_name, node)
workflow.set_entry_point("reference_guard")
workflow.add_conditional_edges(
    "reference_guard", v6.route_after_guard,
    {"missing": "missing_response", "continue": "classifier"},
)
workflow.add_edge("missing_response", END)
for source_node, dest_node in (
    ("classifier", "query_rewriter"),
    ("query_rewriter", "researcher"),
    ("researcher", "reranker"),
    ("reranker", "generator"),
    ("generator", END),
):
    workflow.add_edge(source_node, dest_node)

legal_graph = workflow.compile(checkpointer=MemorySaver())


def run_legal_chat(question: str, session_id: str) -> dict:
    if not str(question or "").strip() or not str(session_id or "").strip():
        raise ValueError("질문과 session_id가 필요합니다.")
    return legal_graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": session_id}},
    )


def self_test() -> None:
    """SQLite, ChromaDB, Ollama를 호출하지 않고 v6 연결/검색 후보 전달만 검사."""
    from types import SimpleNamespace

    fraud = {"article_id": 1, "law_name": "형법", "article": "제347조",
             "article_title": "사기", "content": "사람을 기망하여 재물을 교부받은 경우"}
    tort = {"article_id": 2, "law_name": "민법", "article": "제750조",
            "article_title": "불법행위의 내용", "content": "고의 또는 과실로 인한 위법행위"}
    unnumbered = {"article_id": 3, "law_name": "예시 무조문 법령", "article": "",
                  "content": "번호 없이 작성된 공식 법령 본문"}
    deleted = {"article_id": 4, "law_name": "민법", "article": "제999조",
               "content": "제999조 삭제"}

    class DummyDb:
        def close(self):
            pass

    original_search = globals()["_search_candidates"]
    original_connection = rag_v6.get_search_connection
    original_scope = v3._section_scope
    try:
        seen_args: list[tuple[str, str, str]] = []

        def fake_search(q: str, query: str, category: str):
            seen_args.append((q, query, category))
            return [fraud, tort, dict(tort), deleted, unnumbered, {"article_id": 6, "content": "  "}]

        globals()["_search_candidates"] = fake_search
        rag_v6.get_search_connection = lambda: DummyDb()
        v3._section_scope = lambda _db, _src: "모의 장·절"
        question = "중고거래 사기와 민사상 손해배상"
        out = researcher_node({"question": question, "search_query": "사기, 손해배상",
                               "category": "형사"})
        items = out["candidate_sources"]
        assert seen_args == [(question, "사기, 손해배상", "형사")], seen_args
        assert [s["article_id"] for s in items] == [1, 2, 3], items
        assert all(s["section_scope"] == "모의 장·절" for s in items)
        assert researcher_node({"question": ""}) == {"candidate_sources": []}
        print("[통과] RAG v6 연결 / 민형사 후보 유지 / 내부 ID 중복 제거 / 번호 없는 본문 보존")

        # LangGraph 1.x는 add_node()에 전달된 함수를 StateNodeSpec과
        # RunnableCallable로 감싸 저장한다. wrapper와 원래 함수를 바로
        # `is`로 비교하면 항상 실패하므로 실제 등록된 runnable.func를 검사.
        expected_nodes = {
            "generator": v9.generator_node,
            "reference_guard": v9.reference_guard_node,
            "reranker": v5.reranker_node,
            "missing_response": v6.missing_response_node,
        }
        for node_name, expected_func in expected_nodes.items():
            node_spec = workflow.nodes.get(node_name)
            assert node_spec is not None, f"노드가 등록되지 않았습니다: {node_name}"
            actual_func = getattr(getattr(node_spec, "runnable", None), "func", None)
            assert actual_func is expected_func, f"잘못 연결된 노드: {node_name}"
        print("[통과] v9 원문 인용 복구 / 법령 누락 가드 / 기존 재정렬 노드 유지")
    finally:
        globals()["_search_candidates"] = original_search
        rag_v6.get_search_connection = original_connection
        v3._section_scope = original_scope


if __name__ == "__main__":
    self_test()
