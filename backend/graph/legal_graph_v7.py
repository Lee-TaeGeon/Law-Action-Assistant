"""검증용 LangGraph v7: 전체 법령용 Hybrid RAG v5를 v6 안전장치에 연결한다.

설치: backend/graph/legal_graph_v7.py
- 운영 legal_graph.py, main.py 및 법령 SQLite/ChromaDB는 수정하지 않는다.
- LangGraph v6의 명시적 조문 누락 가드, v4의 재정렬, v5의 검증형 Generator 유지.
- 새 검색 단계에서만 rag_service_v5.hybrid_search_laws 사용.
- 환경변수 LAW_SEARCH_DB_PATH / LAW_VECTOR_DB_PATH는 최초 import 전에 지정.
- 아직 법률 해석의 정확성, 과거시점의 적용, 법령 최신성을 보증하지 않는다.
"""

from __future__ import annotations

import logging
import re
from contextlib import closing
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from backend.graph import legal_graph_v3 as v3
from backend.graph import legal_graph_v5 as v5
from backend.graph import legal_graph_v6 as v6
from backend.services import rag_service_v5 as rag

logger = logging.getLogger(__name__)
AgentState = v6.AgentState


def _identity(source: dict) -> tuple:
    """공식 조문번호가 없는 본문도 SQLite 내부 문서 ID로 구별한다."""
    article_id = source.get("article_id")
    if article_id is not None:
        return ("sqlite", int(article_id))
    return ("legacy", str(source.get("law_name") or ""), str(source.get("article") or ""))


def _search_candidates(question: str, search_query: str, category: str) -> list[dict]:
    """검색 모듈의 단일 진입점. unittest에서 이 함수만 교체해도 됨."""
    return rag.hybrid_search_laws(
        question=question, search_query=search_query, category=category, k=16,
    )


def researcher_node(state: AgentState) -> dict[str, Any]:
    """v5 Hybrid 검색 후보를 읽기 전용 SQLite로 장·절 문맥까지 보강한다."""
    question = str(state.get("question") or "").strip()
    if not question:
        return {"candidate_sources": []}

    search_query = str(state.get("search_query") or "").strip()
    category = str(state.get("category") or "기타")
    raw_candidates = _search_candidates(question, search_query, category)
    usable: list[dict] = []
    seen: set[tuple] = set()

    with closing(rag.get_search_connection()) as db:
        for source in raw_candidates:
            if not isinstance(source, dict):
                continue
            content = str(source.get("content") or "")
            if not content.strip():
                continue
            identity = _identity(source)
            if identity in seen:
                continue
            # 삭제된 조문은 유효한 실체적 근거로 사용하지 않는다.
            # 장차 공식 개정 상태 메타데이터로 검증 범위를 확대해야 한다.
            if re.search(r"제\s*\d+\s*조(?:의\s*\d+)?\s*삭제\b", content):
                continue
            candidate = dict(source)
            candidate["section_scope"] = v3._section_scope(db, candidate)
            usable.append(candidate)
            seen.add(identity)

    logger.info("RAG v5 후보 %d건 중 유효 후보 %d건", len(raw_candidates), len(usable))
    return {"candidate_sources": usable}


workflow = StateGraph(AgentState)
for name, node in (
    ("reference_guard", v6.reference_guard_node),
    ("missing_response", v6.missing_response_node),
    ("classifier", v5.classifier_node),
    ("query_rewriter", v5.query_rewriter_node),
    ("researcher", researcher_node),
    ("reranker", v5.reranker_node),
    ("generator", v5.generator_node),
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
        raise ValueError("question과 session_id가 필요합니다.")
    return legal_graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": session_id}},
    )
