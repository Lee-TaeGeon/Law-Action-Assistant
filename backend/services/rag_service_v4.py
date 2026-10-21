"""법령 Hybrid RAG v4 (실험용): 복합 쟁점 및 명시된 특수 업종 후보 보강.

- 기존 rag_service_v2.py, rag_service_v3.py, SQLite, ChromaDB 변경 없음.
- 검색 후보를 확보하고 순서를 조정할 뿐, 법령의 적용 요건·정확성·시행일을 판정하지 않음.
- law_search_v3_full.db의 공식 법령명/조문 제목 기반 보강. 조문번호 추측 없음.
- 환경변수 LAW_SEARCH_DB_PATH / LAW_VECTOR_DB_PATH는 최초 import 전에 설정할 것.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass

from . import rag_service_v3 as previous

SEARCH_DB_PATH = previous.SEARCH_DB_PATH
VECTOR_DB_PATH = previous.VECTOR_DB_PATH
EMBEDDING_MODEL_NAME = previous.EMBEDDING_MODEL_NAME
get_search_connection = previous.get_search_connection
get_vector_db = previous.get_vector_db
get_embedding_model = previous.get_embedding_model
get_law_names = previous.get_law_names
extract_explicit_law_refs = previous.extract_explicit_law_refs
exact_search_laws = previous.exact_search_laws
keyword_search_laws = previous.keyword_search_laws
search_laws = previous.search_laws


@dataclass(frozen=True)
class SearchFacet:
    label: str
    law_name: str
    title_prefix: str
    reason: str


def _facets(question: str) -> list[SearchFacet]:
    """명시된 사실관계/쟁점에서만 검색 보강 힌트를 생성한다."""
    flags = previous._signals(question)
    compact = previous.base.compact_text(question)
    facets: list[SearchFacet] = []
    if flags["mixed"] and flags["property_fraud"]:
        facets.append(SearchFacet("형사 사기", "형법", "사기", "criminal_fraud"))
        if "불법행위" in compact:
            facets.append(SearchFacet("민사 불법행위", "민법", "불법행위", "civil_tort"))
    if flags["specific_seafarer"] and flags["wage_settlement"]:
        facets.append(SearchFacet("선원 임금청산", "선원법", "금품청산", "seafarer_wage"))
    return facets


def _facet_candidates(facets: list[SearchFacet], per_facet: int = 15) -> dict[str, list[dict]]:
    """읽기 전용 DB에서 조문 제목과 공식 법령명으로 쟁점별 후보를 별도 확보한다.

    그럴듯한 조문번호나 판례번호를 생성하지 않는다. 없는 조문은 그냥 결과 없음.
    """
    found: dict[str, list[dict]] = {}
    if not facets:
        return found
    with closing(get_search_connection()) as db:
        for facet in facets:
            # 공백이 포함된 '금품 청산' 같은 제목에 대비해 공백 제거 후 검색.
            prefix = previous.base.compact_text(facet.title_prefix)
            rows = db.execute(
                "SELECT * FROM articles WHERE law_name = ? "
                "AND REPLACE(article_title, ' ', '') LIKE ? "
                "ORDER BY CASE WHEN REPLACE(article_title, ' ', '') = ? THEN 0 "
                "WHEN REPLACE(article_title, ' ', '') LIKE ? THEN 1 ELSE 2 END, "
                "LENGTH(article_title), id LIMIT ?",
                (facet.law_name, f"%{prefix}%", prefix, f"{prefix}%", per_facet),
            ).fetchall()
            found[facet.reason] = [previous.base._row_to_source(row, "facet") for row in rows]
    return found


def _boost(question: str, item: dict, facet: SearchFacet) -> float:
    """질문에 있는 명시적 쟁점과 제목이 부합하는 경우에만 가중.

    제목 부합은 법률 적용요건의 검증이 아니다.
    """
    law = item.get("law_name", "")
    title = previous.base.compact_text(item.get("article_title", ""))
    prefix = previous.base.compact_text(facet.title_prefix)
    if law != facet.law_name or not title or prefix not in title:
        return 0.0
    if facet.reason == "civil_tort":
        return 0.89 if title.startswith(prefix) else 0.65
    if facet.reason == "criminal_fraud":
        return 0.30 if title == "사기" else 0.05
    if facet.reason == "seafarer_wage":
        return 0.85 if title == "금품청산" else 0.60
    return 0.0


def hybrid_search_laws(question: str, search_query: str | None = None,
                       category: str = "", k: int = 8) -> list[dict]:
    """v3 기반 검색 결과에 명시된 복수 쟁점/특수 업종 후보를 보강한다.

    일반 질문에는 v3 순위를 그대로 사용하고, 관련 후보가 실제 DB에 있는 경우에만 추가한다.
    """
    if k <= 0 or not (question or "").strip():
        return []
    facets = _facets(question)
    if not facets:
        return previous.hybrid_search_laws(question, search_query, category, k)

    # 원 검색에서 밀린 후보도 보존하기 위해 내부 후보 폭을 넓게 유지한다.
    baseline = previous.hybrid_search_laws(question, search_query, category, k=max(40, k * 5))
    additional = _facet_candidates(facets)
    candidates: dict[tuple, dict] = {}
    for rank, item in enumerate(baseline):
        key = previous._identity(item)
        candidates[key] = {
            "item": dict(item),
            "score": 1.0 / (1.0 + 0.075 * rank),
            "methods": set(item.get("retrieval_methods", [item.get("source_type", "unknown")])),
        }
    for facet in facets:
        for item in additional.get(facet.reason, []):
            key = previous._identity(item)
            if key not in candidates:
                candidates[key] = {"item": dict(item), "score": 0.16,
                                   "methods": set()}
            candidates[key]["methods"].add("facet:" + facet.reason)
    for entry in candidates.values():
        entry["score"] += sum(_boost(question, entry["item"], facet) for facet in facets)
    ordered = sorted(candidates.values(), key=lambda entry: -entry["score"])
    results: list[dict] = []
    for entry in ordered[:k]:
        item = dict(entry["item"])
        item["retrieval_methods"] = sorted(entry["methods"])
        results.append(item)
    return results
