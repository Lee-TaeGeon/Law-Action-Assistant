"""법령 Hybrid RAG v3 (실험용, 운영 서비스 미연결).

기존 rag_service_v2.py를 보존하고 동일한 SQLite·ChromaDB를 읽어 사용한다.
- 대규모 법령 DB에서 단어 중의성에 의한 오탐을 낮추고 질문 문맥에 맞춰 재정렬.
- keyword_search_laws, hybrid_search_laws를 개선.
- exact_search_laws, search_laws는 v2의 공개 인터페이스 유지.
- 적용 법령, 사실관계, 시행일 또는 법률적 정확성을 확정하는 기능이 아니다.

환경변수 LAW_SEARCH_DB_PATH, LAW_VECTOR_DB_PATH는 이 모듈 첫 import 전에 지정.
"""
from __future__ import annotations

from contextlib import closing
import math
import re

from . import rag_service_v2 as base

# 기존 모듈과 동일한 읽기 전용 설정/공개 API
SEARCH_DB_PATH = base.SEARCH_DB_PATH
VECTOR_DB_PATH = base.VECTOR_DB_PATH
VECTOR_COLLECTION_NAME = base.VECTOR_COLLECTION_NAME
EMBEDDING_MODEL_NAME = base.EMBEDDING_MODEL_NAME
get_search_connection = base.get_search_connection
get_vector_db = base.get_vector_db
get_embedding_model = base.get_embedding_model
get_law_names = base.get_law_names
extract_explicit_law_refs = base.extract_explicit_law_refs
exact_search_laws = base.exact_search_laws
search_laws = base.search_laws  # 별도 의미 검색 인터페이스는 변경하지 않음


def _has(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def _signals(question: str) -> dict[str, bool]:
    text = re.sub(r"\s+", "", question or "")
    has_fraud = _has(text, "사기", "속여", "속이", "기망", "편취")
    has_property = _has(text, "돈", "금전", "재물", "재산", "물건", "대금", "송금", "거래")
    has_contract = _has(text, "계약", "채무", "약정", "이행", "계약위반")
    has_contract_damage = has_contract and _has(text, "위반", "불이행", "손해", "배상", "청구", "지연")
    has_retirement = _has(text, "퇴사", "퇴직", "퇴사후", "퇴직후", "그만두", "사직")
    has_wages = _has(text, "임금", "월급", "급여", "금품", "체불", "미지급")
    mixed = (_has(text, "민사", "반환", "손해배상", "불법행위")
             and _has(text, "형사", "고소", "사기죄", "처벌"))
    return {
        "contract": has_contract_damage,
        "property_fraud": has_fraud and has_property,
        "wage_settlement": has_retirement and has_wages,
        "mixed": mixed,
        "specific_dogup": _has(text, "도급", "하도급", "하수급", "수급인", "건설업"),
        "specific_seafarer": _has(text, "선원", "선박", "승선", "해기사"),
    }


def _ambiguity_penalty(question: str, title: str, signals: dict[str, bool]) -> float:
    """명백히 다른 의미인 부분 문자열 검색 결과에만 패널티. 법령 일괄 배제 없음."""
    if not signals["property_fraud"]:
        return 0.0
    q = base.compact_text(question)
    t = base.compact_text(title)
    # '사기' (詐欺)와 '사기진작'(士氣振作), '분사기', '심사기간' 혼동 방지.
    # 사용자가 해당 물건/표현을 직접 질문하는 경우에는 강제 감점하지 않는다.
    for phrase in ("사기진작", "분사기", "심사기간"):
        if phrase in t and phrase not in q:
            return -0.45
    return 0.0


def _context_bonus(question: str, item: dict, signals: dict[str, bool] | None = None) -> float:
    """추출 단계의 약한 재정렬 힌트. 법률의 실제 적용 가능성 판정이 아니다."""
    s = _signals(question) if signals is None else signals
    law = str(item.get("law_name") or "")
    title = base.compact_text(str(item.get("article_title") or ""))
    q = base.compact_text(question)
    bonus = _ambiguity_penalty(question, title, s)

    # 법령명을 명시적으로 물은 경우 우선하되, 다른 법률을 절대 배제하지 않는다.
    if law and law in question:
        bonus += 0.19

    if s["contract"] and not s["mixed"]:
        if law == "민법":
            bonus += 0.075
            if "채무불이행" in title and "손해배상" in title:
                bonus += 0.15
            elif "손해배상" in title and _has(title, "범위", "방법", "이행지체"):
                bonus += 0.045
    if s["property_fraud"]:
        if law == "형법":
            bonus += 0.075
            if title == "사기":
                bonus += 0.15
            elif title in ("준사기", "컴퓨터등사용사기"):
                bonus += 0.05
        if law == "민법" and s["mixed"] and _has(title, "불법행위", "사기"):
            bonus += 0.09
    if s["wage_settlement"]:
        if law == "근로기준법":
            bonus += 0.085
            if "금품청산" in title:
                bonus += 0.17
            elif "미지급임금" in title:
                bonus += 0.065
            elif title == "임금지급":
                bonus += 0.055
            if "도급" in title and not s["specific_dogup"]:
                bonus -= 0.07
        if law == "선원법" and not s["specific_seafarer"]:
            bonus -= 0.035

    # 특정 업종·계약형태가 질문에 등장하면 관련 특수법의 검색 순위를 보존·회복.
    if s["specific_dogup"] and "도급" in title:
        bonus += 0.10
    if s["specific_seafarer"] and law == "선원법":
        bonus += 0.12
    return bonus


def _identity(item: dict) -> tuple:
    # 번호 없는 법령 본문은 공식 조문번호가 비어 있을 수 있으므로 SQLite ID 우선.
    if item.get("article_id") is not None:
        return ("sqlite", int(item["article_id"]))
    return ("legacy", item.get("law_name"), item.get("article"))


def keyword_search_laws(query: str, k: int = 12) -> list[dict]:
    """v2 후보를 넓게 수집한 후 문맥과 모호어 기준으로 재정렬."""
    if k <= 0 or not (query or "").strip():
        return []
    pool = base.keyword_search_laws(query, k=max(80, k * 6))
    signals = _signals(query)
    ranked = sorted(enumerate(pool), key=lambda pair: -(
        3.0 / (30 + pair[0] + 1) + _context_bonus(query, pair[1], signals)
    ))
    return [dict(item) for _, item in ranked[:k]]


def _nearest_vector_candidates(question: str, k: int = 96) -> list[dict]:
    """MMR 다양성 재정렬 전의 실제 최근접 벡터 후보를 읽기 전용 SQLite로 조회.

    v2의 독립 의미 검색(search_laws)은 유지하고 Hybrid용 후보만 개선.
    """
    if k <= 0 or not question.strip():
        return []
    collection = get_vector_db()
    count = collection.count()
    if count == 0:
        return []
    model = get_embedding_model()
    query_vector = model.encode([question.strip()], convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)[0]
    response = collection.query(query_embeddings=[query_vector.tolist()],
                                n_results=min(count, k), include=["metadatas", "distances"])
    ids = (response.get("ids") or [[]])[0]
    metas = (response.get("metadatas") or [[]])[0]
    if len(ids) != len(metas):
        raise RuntimeError("Chroma 문서 ID와 메타데이터 개수가 다릅니다.")
    parsed_ids = [int(v) for v in ids]
    if len(set(parsed_ids)) != len(parsed_ids):
        raise RuntimeError("Chroma 검색 결과에 중복 문서 ID가 있습니다.")
    if not parsed_ids:
        return []
    placeholders = ",".join("?" for _ in parsed_ids)
    with closing(get_search_connection()) as db:
        rows = {int(row["id"]): row for row in db.execute(
            f"SELECT * FROM articles WHERE id IN ({placeholders})", parsed_ids
        )}
    result = []
    for article_id, meta in zip(parsed_ids, metas):
        row = rows.get(article_id)
        if row is None:
            raise RuntimeError(f"SQLite와 ChromaDB ID 불일치: {article_id}")
        if meta and meta.get("content_hash") != row["content_hash"]:
            raise RuntimeError(f"SQLite와 ChromaDB 원문 해시 불일치: {article_id}")
        result.append(base._row_to_source(row, "vector"))
    return result


def hybrid_search_laws(question: str, search_query: str | None = None,
                       category: str = "", k: int = 8) -> list[dict]:
    """v2 API 호환. Exact + 문맥 재정렬 키워드 + 최근접 벡터 RRF 결합.

    단일 분야 강제 제외 없음. 순위 힌트는 법률 적용 요건 판정에 사용하면 안 됨.
    """
    if k <= 0 or not (question or "").strip():
        return []
    query_for_keywords = (search_query or "").strip()
    query_for_keywords = question if not query_for_keywords else query_for_keywords + ", " + question
    signals = _signals(question)
    channels = (
        ("exact", exact_search_laws(question), 10.0),
        ("keyword", keyword_search_laws(query_for_keywords, k=max(80, k * 8)), 3.0),
        ("vector", _nearest_vector_candidates(question, k=max(96, k * 12)), 1.0),
    )
    candidates: dict[tuple, dict] = {}
    for channel, results, weight in channels:
        for rank, item in enumerate(results, start=1):
            key = _identity(item)
            entry = candidates.setdefault(key, {"item": dict(item), "score": 0.0, "methods": set()})
            entry["score"] += weight / (30 + rank)
            entry["methods"].add(channel)
            if channel == "exact" or (channel == "keyword" and entry["item"].get("source_type") == "vector"):
                entry["item"] = dict(item)
    ranked = sorted(candidates.values(), key=lambda entry: -(
        entry["score"] + _context_bonus(question, entry["item"], signals)
        + base._soft_domain_bonus(entry["item"].get("law_name", ""), category, question)
    ))
    output = []
    for entry in ranked[:k]:
        item = dict(entry["item"])
        item["retrieval_methods"] = sorted(entry["methods"])
        output.append(item)
    return output
