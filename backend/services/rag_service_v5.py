"""법령 Hybrid RAG v5 (검증용): 복합 불법행위 검색의 과도한 제목 가산점 보정.

v4의 일반 계약/사기/노동/선원 검색은 그대로 재사용한다.
'민사상 불법행위'를 명시한 복합 질문에서는 일반 불법행위 책임 근거와
별도 쟁점(상계, 가족관계 등)을 구분해 검색 결과 *순위*를 조정한다.
법률 적용 여부/시행일/법률적 정확성을 판정하지 않는다.

운영 rag_service.py, SQLite, ChromaDB는 읽거나 수정하지 않는다.
환경변수 LAW_SEARCH_DB_PATH, LAW_VECTOR_DB_PATH는 최초 import 전에 설정.
"""
from __future__ import annotations

from . import rag_service_v4 as v4

SEARCH_DB_PATH = v4.SEARCH_DB_PATH
VECTOR_DB_PATH = v4.VECTOR_DB_PATH
EMBEDDING_MODEL_NAME = v4.EMBEDDING_MODEL_NAME
get_search_connection = v4.get_search_connection
get_vector_db = v4.get_vector_db
get_embedding_model = v4.get_embedding_model
get_law_names = v4.get_law_names
extract_explicit_law_refs = v4.extract_explicit_law_refs
exact_search_laws = v4.exact_search_laws
keyword_search_laws = v4.keyword_search_laws
search_laws = v4.search_laws


def _civil_tort_adjust(question: str, item: dict) -> float:
    """불법행위의 일반 규정과 별도 쟁점의 검색 우선도를 구별하는 제한적 힌트.

    특정 법률 조문번호를 임의로 만들거나 법률 적용 요건을 판단하지 않는다.
    질문에 명시된 '상계'와 가족관계는 불일치로 취급하지 않는다.
    """
    if item.get("law_name") != "민법":
        return 0.0
    norm = v4.previous.base.compact_text
    title = norm(item.get("article_title", ""))
    q = norm(question)
    if not title or "불법행위" not in title:
        return 0.0
    if "상계" in title and "상계" not in q:
        return -0.70
    if title == "불법행위의내용":
        return 1.08
    if "손해배상" in title or "배상책임" in title:
        return 0.70
    # '불법행위'라는 단어만 공유하는 조문에 큰 보너스를 주지 않는다.
    return 0.02


def _offtopic_family_penalty(question: str, item: dict) -> float:
    """명시되지 않은 가족관계 특수 쟁점은 복합 중고거래 질의에서만 약하게 감점."""
    if item.get("law_name") != "민법":
        return 0.0
    norm = v4.previous.base.compact_text
    title, q = norm(item.get("article_title", "")), norm(question)
    for word in ("혼인", "이혼", "약혼", "입양", "상속"):
        if word in title and word not in q:
            return -0.35
    return 0.0


def hybrid_search_laws(question: str, search_query: str | None = None,
                       category: str = "", k: int = 8) -> list[dict]:
    """v4의 복합 쟁점 검색을 재사용하되 일반 불법행위의 무관한 상계 조문 과우대를 보정."""
    if k <= 0 or not (question or "").strip():
        return []
    facets = v4._facets(question)
    if not any(f.reason == "civil_tort" for f in facets):
        return v4.hybrid_search_laws(question, search_query, category, k)

    baseline = v4.previous.hybrid_search_laws(question, search_query, category,
                                               k=max(40, k * 5))
    additional = v4._facet_candidates(facets)
    candidates: dict[tuple, dict] = {}
    for rank, item in enumerate(baseline):
        key = v4.previous._identity(item)
        candidates[key] = {
            "item": dict(item),
            "score": 1.0 / (1.0 + 0.075 * rank),
            "methods": set(item.get("retrieval_methods", [item.get("source_type", "unknown")])),
        }
    for facet in facets:
        for item in additional.get(facet.reason, []):
            key = v4.previous._identity(item)
            if key not in candidates:
                candidates[key] = {"item": dict(item), "score": 0.16, "methods": set()}
            candidates[key]["methods"].add("facet:" + facet.reason)

    for entry in candidates.values():
        item = entry["item"]
        for facet in facets:
            if facet.reason == "civil_tort":
                entry["score"] += _civil_tort_adjust(question, item)
            else:
                entry["score"] += v4._boost(question, item, facet)
        entry["score"] += _offtopic_family_penalty(question, item)

    ordered = sorted(candidates.values(), key=lambda entry: -entry["score"])
    results = []
    for entry in ordered[:k]:
        item = dict(entry["item"])
        item["retrieval_methods"] = sorted(entry["methods"])
        results.append(item)
    return results
