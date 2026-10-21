"""전체 법령 Hybrid RAG v6: 복합 민·형사 질문의 민사 손해배상 후보 보강.

v5에서는 질문 원문에 '불법행위'라는 문자열이 있어야 civil_tort 쟁점별 후보
검색이 작동했다. 이 모듈은 '형사상 사기와 민사상 손해배상'처럼 직접적인 법률
용어가 없는 경우에도 별도의 *후보 검색*을 수행한다. 법률 적용을 단정하지 않는다.

기존 v2~v5, 운영 RAG, SQLite, ChromaDB 원본은 변경하지 않음.
LAW_SEARCH_DB_PATH / LAW_VECTOR_DB_PATH 환경변수는 최초 import 전에 설정할 것.
"""
from __future__ import annotations

from . import rag_service_v4 as v4
from . import rag_service_v5 as v5
from . import rag_search_signals as signals

SEARCH_DB_PATH = v5.SEARCH_DB_PATH
VECTOR_DB_PATH = v5.VECTOR_DB_PATH
EMBEDDING_MODEL_NAME = v5.EMBEDDING_MODEL_NAME
get_search_connection = v5.get_search_connection
get_vector_db = v5.get_vector_db
get_embedding_model = v5.get_embedding_model
get_law_names = v5.get_law_names
extract_explicit_law_refs = v5.extract_explicit_law_refs
exact_search_laws = v5.exact_search_laws
keyword_search_laws = v5.keyword_search_laws
search_laws = v5.search_laws


_needs_civil_damages_facet = signals._needs_civil_damages_facet


def hybrid_search_laws(question: str, search_query: str | None = None,
                       category: str = "", k: int = 8) -> list[dict]:
    """기존 v5 검색에 민사 손해배상 쟁점별 후보 검색을 제한적으로 추가한다.

    조문번호를 하드코딩하지 않고 법령명과 *실제 DB의 공식 조문 제목*으로 후보를
    가져온다. 최종 인용 및 실제 사건 적용 여부는 별도로 판단해야 한다.
    """
    if k <= 0 or not str(question or "").strip():
        return []
    if not _needs_civil_damages_facet(question):
        return v5.hybrid_search_laws(question, search_query=search_query, category=category, k=k)

    facets = v4._facets(question)
    facets.append(v4.SearchFacet("민사 손해배상 근거 후보", "민법", "불법행위", "civil_tort"))
    baseline = v4.previous.hybrid_search_laws(
        question, search_query=search_query, category=category, k=max(40, k * 5)
    )
    extra = v4._facet_candidates(facets)
    candidates: dict[tuple, dict] = {}
    for rank, item in enumerate(baseline):
        key = v4.previous._identity(item)
        candidates[key] = {
            "item": dict(item),
            "score": 1.0 / (1.0 + 0.075 * rank),
            "methods": set(item.get("retrieval_methods") or [item.get("source_type", "unknown")]),
        }
    for facet in facets:
        for item in extra.get(facet.reason, []):
            key = v4.previous._identity(item)
            if key not in candidates:
                candidates[key] = {
                    "item": dict(item), "score": 0.16, "methods": set(),
                }
            candidates[key]["methods"].add("facet:" + facet.reason)

    for entry in candidates.values():
        item = entry["item"]
        for facet in facets:
            if facet.reason == "civil_tort":
                entry["score"] += v5._civil_tort_adjust(question, item)
            else:
                entry["score"] += v4._boost(question, item, facet)
        entry["score"] += v5._offtopic_family_penalty(question, item)

    ordered = sorted(candidates.values(), key=lambda entry: -entry["score"])
    result = []
    for entry in ordered[:k]:
        item = dict(entry["item"])
        item["retrieval_methods"] = sorted(entry["methods"])
        result.append(item)
    return result


def self_test() -> None:
    """외부 법령 DB와 Ollama 없이 후보 보강 및 기존 검색 회귀를 모의 검증."""
    sample = (
        "중고거래 판매자가 돈을 받고 물건을 보내지 않았어요. "
        "형사상 사기와 민사상 손해배상을 알고 싶어요."
    )
    assert _needs_civil_damages_facet(sample)
    assert not _needs_civil_damages_facet("계약을 위반해서 손해배상을 청구하고 싶어요")
    assert not _needs_civil_damages_facet("다른 사람을 속여 돈을 가져가면 어떤 죄인가요?")
    assert not _needs_civil_damages_facet("선원이 퇴직한 뒤 임금을 받지 못했어요")
    assert not _needs_civil_damages_facet(sample + " 민사상 불법행위 규정도 확인해 주세요")

    def item(law: str, number: str, title: str, idx: int) -> dict:
        return {"law_name": law, "article": number, "article_title": title,
                "article_id": idx, "content": "모의 법문", "source_type": "keyword",
                "retrieval_methods": ["keyword"]}

    fraud = item("형법", "제347조", "사기", 1)
    tort = item("민법", "제750조", "불법행위의 내용", 2)
    offset = item("민법", "제496조", "불법행위채권을 수동채권으로 하는 상계의 금지", 3)
    family = item("민법", "제823조", "사기, 강박으로 인한 혼인취소청구권의 소멸", 4)
    secondary = item("형법", "제348조", "준사기", 5)
    baseline_original = v4.previous.hybrid_search_laws
    facet_original = v4._facet_candidates
    delegate_original = v5.hybrid_search_laws
    try:
        v4.previous.hybrid_search_laws = lambda *args, **kwargs: [fraud, family, offset, secondary]
        v4._facet_candidates = lambda facets, **kwargs: {
            "criminal_fraud": [fraud], "civil_tort": [offset, tort]
        }
        output = hybrid_search_laws(sample, category="형사", k=8)
        positions = {(x["law_name"], x["article"]): i for i, x in enumerate(output, 1)}
        assert positions[("민법", "제750조")] <= 3, positions
        assert positions[("형법", "제347조")] <= 3, positions
        assert positions[("민법", "제750조")] < positions[("민법", "제496조")], positions
        assert len({x["article_id"] for x in output}) == len(output)
        assert any("facet:civil_tort" in x["retrieval_methods"] for x in output)
        print("[통과] '민사상 손해배상'을 복합 쟁점으로 인식 / 후보 보강 / 중복 제거")

        sentinel = [item("근로기준법", "제36조", "금품 청산", 8)]
        v5.hybrid_search_laws = lambda *args, **kwargs: sentinel
        assert hybrid_search_laws("퇴사 후 임금을 지급하지 않으면?", category="노동") == sentinel
        assert hybrid_search_laws(sample + " 민사상 불법행위도 알고 싶어요.") == sentinel
        print("[통과] 기존 일반·명시적 불법행위 검색 v5 동작 유지")
    finally:
        v4.previous.hybrid_search_laws = baseline_original
        v4._facet_candidates = facet_original
        v5.hybrid_search_laws = delegate_original


if __name__ == "__main__":
    self_test()
