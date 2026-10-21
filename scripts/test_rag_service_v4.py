"""검색 v3/v4 비교 테스트 (운영 DB 수정 없음).

1. python scripts/test_rag_service_v4.py --unit-only
2. python scripts/test_rag_service_v4.py --sqlite <DB> --chroma <DIR>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _example(law: str, article: str, title: str, idx: int) -> dict:
    return {"law_name": law, "article": article, "article_title": title,
            "article_id": idx, "content": "테스트", "source_type": "keyword",
            "retrieval_methods": ["keyword"]}


def unit_test(v3, v4) -> None:
    q_mixed = ("중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. "
               "형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요.")
    q_sailor = "배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다."
    q_general = "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요."
    assert {f.reason for f in v4._facets(q_mixed)} == {"criminal_fraud", "civil_tort"}
    assert {f.reason for f in v4._facets(q_sailor)} == {"seafarer_wage"}
    assert v4._facets(q_general) == []
    fraud = _example("형법", "제347조", "사기", 1)
    fraud2 = _example("형법", "제348조", "준사기", 2)
    civil_irrelevant = _example("민법", "제823조", "사기, 강박으로 인한 혼인취소청구권의 소멸", 3)
    tort = _example("민법", "제750조", "불법행위의 내용", 4)
    wages = _example("근로기준법", "제36조", "금품 청산", 5)
    sailor = _example("선원법", "제55조의2", "금품 청산", 6)
    assert v4._boost(q_mixed, tort, v4.SearchFacet("", "민법", "불법행위", "civil_tort")) > 0
    assert v4._boost(q_sailor, sailor, v4.SearchFacet("", "선원법", "금품청산", "seafarer_wage")) > 0
    assert v4._boost(q_general, wages, v4.SearchFacet("", "선원법", "금품청산", "seafarer_wage")) == 0

    original_base, original_facet = v3.hybrid_search_laws, v4._facet_candidates
    try:
        def mock_base(q, search_query=None, category="", k=8):
            if "중고거래" in q:
                return [fraud, fraud2, civil_irrelevant]
            if "선원" in q:
                return [wages, _example("근로기준법", "제37조", "미지급 임금", 7)]
            return [wages]
        v3.hybrid_search_laws = mock_base
        v4._facet_candidates = lambda facets: {
            "criminal_fraud": [fraud], "civil_tort": [tort], "seafarer_wage": [sailor]
        }
        mixed = v4.hybrid_search_laws(q_mixed, category="형사", k=5)
        assert any(x["article_id"] == tort["article_id"] for x in mixed[:5]), mixed
        assert any(x["article_id"] == fraud["article_id"] for x in mixed[:5]), mixed
        assert all(x["article_id"] != tort["article_id"] or
                   "facet:civil_tort" in x["retrieval_methods"] for x in mixed)
        special = v4.hybrid_search_laws(q_sailor, category="노동", k=3)
        assert special[0]["article_id"] == sailor["article_id"], special
        assert v4.hybrid_search_laws(q_general, k=2)[0]["article_id"] == wages["article_id"]
    finally:
        v3.hybrid_search_laws, v4._facet_candidates = original_base, original_facet
    print("[통과] 혼합 민형사 쟁점별 후보 보강 / 선원 임금조문 우선 / 일반 검색 v3 유지")


def _position(results, law, article) -> str:
    for n, r in enumerate(results, 1):
        if (r.get("law_name"), r.get("article")) == (law, article):
            return str(n)
    return "상위 12 밖"


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG v3/v4 비교")
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--chroma", type=Path)
    parser.add_argument("--unit-only", action="store_true")
    args = parser.parse_args()
    if not args.unit_only:
        if not args.sqlite or not args.chroma or not args.sqlite.is_file() or not args.chroma.is_dir():
            parser.error("실제 비교에는 존재하는 --sqlite 및 --chroma 경로가 필요합니다.")
        os.environ["LAW_SEARCH_DB_PATH"] = str(args.sqlite.resolve())
        os.environ["LAW_VECTOR_DB_PATH"] = str(args.chroma.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.services import rag_service_v3 as v3, rag_service_v4 as v4
    unit_test(v3, v4)
    if args.unit_only:
        return
    tests = [
        ("계약 손해배상", "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사", [("민법", "제390조")]),
        ("형사 사기", "다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?", "형사", [("형법", "제347조")]),
        ("퇴직 임금체불", "퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", "노동", [("근로기준법", "제36조")]),
        ("민형사 복합", "중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. 형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요.", "형사", [("형법", "제347조"), ("민법", "제750조")]),
        ("선원 퇴직 임금", "배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다.", "노동", [("선원법", "제55조의2")]),
    ]
    print("SQLite:", args.sqlite)
    print("ChromaDB:", args.chroma)
    print("주의: 검색 순위 비교만 수행하며 기존 DB·서비스는 수정하지 않습니다.")
    for label, question, category, targets in tests:
        print("\n" + "=" * 60 + "\n[" + label + "] " + question)
        old = v3.hybrid_search_laws(question, category=category, k=12)
        new = v4.hybrid_search_laws(question, category=category, k=12)
        print("[v3 상위 5개]")
        for n, x in enumerate(old[:5], 1):
            print(f"  {n}. {x['law_name']} {x['article']} ({x['article_title']})")
        print("[v4 상위 5개]")
        for n, x in enumerate(new[:5], 1):
            print(f"  {n}. {x['law_name']} {x['article']} ({x['article_title']})")
        for law, article in targets:
            print(f"  {law} {article}: {_position(old,law,article)} → {_position(new,law,article)}")
    print("\n테스트 종료. 실제 법률 적용 여부·법령 시행시점 검증은 별도입니다.")


if __name__ == "__main__":
    main()
