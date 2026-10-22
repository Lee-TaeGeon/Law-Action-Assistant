
import json
import os
from pathlib import Path

from langchain_core.tools import tool

from backend.graph.legal_graph_v10 import researcher_node
from backend.graph.legal_graph_v5 import reranker_node

def check_law_data_paths() -> None:
    """지정된 법령 DB와 테스트용 벡터 DB 경로 확인."""

    search_path = os.getenv("LAW_SEARCH_DB_PATH")
    vector_path = os.getenv("LAW_VECTOR_DB_PATH")

    if not search_path or not vector_path:
        raise RuntimeError(
            "LAW_SEARCH_DB_PATH 및 LAW_VECTOR_DB_PATH 설정이 필요합니다."
        )

    if not Path(search_path).is_file():
        raise RuntimeError("SQLite 법령 DB 경로가 올바르지 않습니다.")

    if not Path(vector_path).is_dir():
        raise RuntimeError("ChromaDB 경로가 올바르지 않습니다.")


@tool
def search_laws(query: str) -> str:
    """한국 법령과 조문을 기존 Hybrid RAG로 검색합니다.

    법적 쟁점에 관한 법령 근거가 필요할 때 사용합니다.
    검색 결과는 법적 적용 여부를 확정하지 않습니다.
    """

    query = query.strip()

    if not query:
        return json.dumps(
            {
                "status": "invalid_input",
                "sources": [],
            },
            ensure_ascii=False,
        )

    check_law_data_paths()

    result = researcher_node(
        {
            "question": query,
            "search_query": query,
            "category": "기타",
        }
    )

    
    reranked = reranker_node(
        {
            "question": query,
            "search_query": query,
            "category": "기타",
            **result,
        }
    )

    reranker_status = reranked.get(
        "reranker_status",
        "format_error",
    )

    candidates = (
        reranked.get("sources", [])
        if reranker_status == "ok"
        else []
    )


    sources = []

    for index, source in enumerate(candidates[:5], start=1):
        sources.append(
            {
                "source_id": f"S{index}",
                "article_id": source.get("article_id"),
                "law_name": source.get("law_name"),
                "article": source.get("article"),
                "article_title": source.get("article_title"),
                "content": str(
                    source.get("content") or ""
                )[:800],
                "section_scope": source.get("section_scope"),
                "source_type": source.get("source_type"),
            }
        )

   
    status = (
        "reranker_error"
        if reranker_status not in ("ok", "none")
        else "ok" if sources else "no_results"
    )

    return json.dumps(
        {
            "status": status,
            "reranker_status": reranker_status,
            "query": query,
            "sources": sources,
        },
        ensure_ascii=False,
    )

