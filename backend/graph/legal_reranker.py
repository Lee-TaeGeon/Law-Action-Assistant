"""Legal source reranking extracted from LangGraph v13."""
from __future__ import annotations

import logging
from typing import Any, TypeAlias

from backend.graph import legal_graph_v4 as v4
from backend.graph import legal_graph_v11 as v11
from backend.graph.legal_scope_guard import _out_of_scope_reason

logger = logging.getLogger(__name__)
AgentState: TypeAlias = v11.AgentState



def reranker_node(state: AgentState) -> dict[str, Any]:
    """v4 Reranker를 그대로 사용하되 명백한 다른 쟁점만 전후로 검사."""
    original_candidates = list(state.get("candidate_sources") or [])
    candidates: list[dict] = []
    for candidate in original_candidates:
        if not isinstance(candidate, dict):
            continue
        reason = _out_of_scope_reason(state, candidate)
        if reason:
            logger.info("Reranker v13 특수 적용범위 후보 보류: %s %s (%s)",
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
            logger.warning("Reranker v13 선정 후 보류: %s %s (%s)",
                           source.get("law_name"), source.get("article"), reason)
        else:
            verified.append(source)
    return {"sources": verified,
            "reranker_status": "ok" if verified else "none"}
