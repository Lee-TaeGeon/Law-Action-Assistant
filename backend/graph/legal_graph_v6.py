"""실험용 LangGraph v6: 명시한 법령·조문의 DB 수록 여부를 검색 전에 확인한다.

배치: backend/graph/legal_graph_v6.py

- v5 검색·적용범위·재정렬·Generator를 그대로 재사용한다.
- 질문에 명시된 특정 법령/조문이 새 SQLite에 없는 경우, 검색 전에 분기하여
  관련 없는 법령을 대신 인용하지 않는다.
- 검색 데이터의 '미수록'을 실제 법령의 부존재와 혼동하지 않는다.
- 운영 main.py, 법령 DB 및 v5 파일은 변경하지 않는다.
- LAW_SEARCH_DB_PATH / LAW_VECTOR_DB_PATH는 이 모듈 import 전에 설정한다.
- 이 검사는 한국 법령명을 인식하는 초기 규칙이며 모든 표기·별칭을 지원하지 않는다.
"""

from __future__ import annotations

import re
from contextlib import closing
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.graph import legal_graph_v5 as v5
from backend.services import rag_service_v2 as rag


class AgentState(v5.AgentState, total=False):
    missing_explicit_references: list[dict[str, str]]


# 따옴표로 감싼 법령명은 공백을 포함할 수 있다.
QUOTED_REFERENCE = re.compile(
    r"[「『](?P<law>[^」』\r\n]{2,150})[」』]\s*"
    r"제\s*(?P<no>\d{1,5})\s*조(?:\s*의\s*(?P<branch>\d{1,4}))?"
)
# 따옴표 없는 단순 법령명: '의료법 제27조', '민법 제390조' 등.
# 띄어쓰기가 들어간 긴 법령명과 별칭은 우선 인식 범위 밖이다.
BARE_REFERENCE = re.compile(
    r"(?<![가-힣A-Za-z0-9])"
    r"(?P<law>[가-힣A-Za-z0-9·ㆍ]{1,80}(?:법률|법|령|규칙|조례))\s*"
    r"제\s*(?P<no>\d{1,5})\s*조(?:\s*의\s*(?P<branch>\d{1,4}))?"
)


def extract_cited_laws(question: str) -> list[dict[str, str]]:
    """질문에서 직접 명시한 법령명 + 조문번호만 보수적으로 추출한다."""
    text = str(question or "")
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for pattern in (QUOTED_REFERENCE, BARE_REFERENCE):
        for match in pattern.finditer(text):
            name = re.sub(r"\s+", " ", match.group("law")).strip()
            number = str(int(match.group("no")))
            if match.group("branch"):
                number += "의" + str(int(match.group("branch")))
            # 짧은 '이 법 제N조'처럼 실제 법령명이 아닌 표현은 검사 대상에서 제외한다.
            if name in {"이법", "그법", "동법", "본법", "법"}:
                continue
            key = (re.sub(r"\s+", "", name), number)
            if key not in seen:
                results.append({"law_name": name, "article_no": number})
                seen.add(key)
    return results


def check_explicit_references(question: str) -> list[dict[str, str]]:
    """현재 SQLite에 *정확히 그 법령명/조문*이 들어 있는지 검사한다.

    missing status: law_not_indexed / article_not_indexed
    잘못된 철자, 약칭이나 별칭도 미수록으로 보일 수 있으므로 법령 부존재로 말하지 않는다.
    """
    refs = extract_cited_laws(question)
    if not refs:
        return []

    missing: list[dict[str, str]] = []
    with closing(rag.get_search_connection()) as db:
        for ref in refs:
            # 공백 차이는 허용하지만 임의의 법령 이름 유사성 매칭은 하지 않는다.
            law = db.execute(
                "SELECT law_name FROM laws WHERE REPLACE(law_name, ' ', '') = ? LIMIT 1",
                (re.sub(r"\s+", "", ref["law_name"]),),
            ).fetchone()
            if law is None:
                missing.append({**ref, "status": "law_not_indexed"})
                continue
            # 실제 DB에 저장된 정식 법령명을 사용한다.
            official_name = str(law["law_name"])
            article = db.execute(
                "SELECT 1 FROM articles WHERE law_name = ? AND article_no = ? LIMIT 1",
                (official_name, ref["article_no"]),
            ).fetchone()
            if article is None:
                missing.append({**ref, "law_name": official_name, "status": "article_not_indexed"})
    return missing


def _format_article(number: str) -> str:
    return "제" + number.replace("의", "조의", 1) + ("" if "의" in number else "조")


def render_missing_message(missing: list[dict[str, str]]) -> str:
    """완전히 새로운 사실을 생성하지 않고 DB의 수록 상태만 설명한다."""
    lines = [
        "## 현재 법령 데이터에서 확인할 수 없는 내용",
        "질문에서 명시한 다음 법령·조문을 현재 연결된 검색 DB에서 확인하지 못했습니다.",
        "",
    ]
    for item in missing:
        law_name = item["law_name"]
        display = _format_article(item["article_no"])
        if item["status"] == "law_not_indexed":
            lines.append(f"- **{law_name} {display}**: '{law_name}' 법령 자체가 현재 검색 DB에 수록되어 있지 않거나, 법령명 표기가 일치하지 않습니다.")
        else:
            lines.append(f"- **{law_name} {display}**: 해당 법령은 DB에 있지만 이 조문은 현재 데이터에서 확인되지 않습니다.")
    lines.extend([
        "",
        "이는 대한민국에 해당 법령·조문이 존재하지 않는다는 의미가 아닙니다.",
        "다른 법령의 유사한 조문으로 대신 답변하지 않았습니다.",
        "공식 국가법령정보센터에서 명칭·조문·시행일을 확인하거나, 검증된 법령 데이터를 추가한 뒤 다시 검색해야 합니다.",
    ])
    return "\n".join(lines)


def reference_guard_node(state: AgentState) -> dict[str, Any]:
    """기존 체크포인트에 남은 이전 질문의 출처도 초기화한다."""
    missing = check_explicit_references(str(state.get("question") or ""))
    return {
        "missing_explicit_references": missing,
        "sources": [],
        "candidate_sources": [],
        "answer": "",
        "answer_validation_status": "pending",
        "reranker_status": "pending",
        "search_query": "",
    }


def route_after_guard(state: AgentState) -> str:
    return "missing" if state.get("missing_explicit_references") else "continue"


def missing_response_node(state: AgentState) -> dict[str, Any]:
    return {
        "answer": render_missing_message(list(state.get("missing_explicit_references") or [])),
        "answer_validation_status": "missing_reference",
        "reranker_status": "skipped",
        "sources": [],
    }


workflow = StateGraph(AgentState)
for name, node in (
    ("reference_guard", reference_guard_node),
    ("missing_response", missing_response_node),
    ("classifier", v5.classifier_node),
    ("query_rewriter", v5.query_rewriter_node),
    ("researcher", v5.researcher_node),
    ("reranker", v5.reranker_node),
    ("generator", v5.generator_node),
):
    workflow.add_node(name, node)

workflow.set_entry_point("reference_guard")
workflow.add_conditional_edges(
    "reference_guard", route_after_guard,
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
    """읽기 전용 DB로 누락/존재/혼합 질문을 검사한다. Ollama는 호출하지 않는다."""
    assert extract_cited_laws("의료법 제27조의 규정을 설명해 주세요") == [
        {"law_name": "의료법", "article_no": "27"}
    ]
    assert extract_cited_laws("「근로기준법」 제36조, 민법 제390조의2") == [
        {"law_name": "근로기준법", "article_no": "36"},
        {"law_name": "민법", "article_no": "390의2"},
    ]
    medical = check_explicit_references("의료법 제27조를 알려 주세요")
    if medical:
        assert medical[0]["status"] == "law_not_indexed"
        state = reference_guard_node({"question": "의료법 제27조를 알려 주세요"})
        assert route_after_guard(state) == "missing"
        answer = missing_response_node(state)
        assert answer["answer_validation_status"] == "missing_reference"
        assert "의료법 제27조" in answer["answer"]
        print("[통과] DB 미수록 법령: 답변 차단 및 수록 범위 안내")
    else:
        print("[참고] 의료법은 현재 DB에 있음: '미수록' 테스트 조건이 아님")

    known = check_explicit_references("민법 제390조와 형법 제347조")
    assert not known, f"기존 조문을 누락으로 잘못 판정함: {known}"
    assert route_after_guard(reference_guard_node({"question": "민법 제390조"})) == "continue"
    print("[통과] 수록된 민법 제390조 및 형법 제347조는 정상 검색으로 전달")

    unknown_article = check_explicit_references("민법 제99999조를 알려 주세요")
    assert unknown_article and unknown_article[0]["status"] == "article_not_indexed"
    print("[통과] 법령은 있지만 조문이 미수록인 경우 구분")
    print("테스트 완료: Ollama 호출 없음, 기존 운영 코드/DB 변경 없음")
