"""검증용 RAG v4/v5 비교: python scripts/test_rag_service_v5.py --unit-only
실제 DB 비교: python scripts/test_rag_service_v5.py --sqlite <sqlite_path> --chroma <chroma_dir>
읽기 전용이며 운영 코드 및 DB를 변경하지 않는다.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def example(law: str, article: str, title: str, idx: int) -> dict:
    return {"law_name": law, "article": article, "article_title": title,
            "article_id": idx, "content": "테스트", "source_type": "keyword",
            "retrieval_methods": ["keyword"]}


def unit_test(v3, v4, v5):
    mixed = ("중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. "
             "형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요.")
    fraud = example("형법", "제347조", "사기", 1)
    general = example("민법", "제750조", "불법행위의 내용", 2)
    offset = example("민법", "제496조", "불법행위채권을 수동채권으로 하는 상계의 금지", 3)
    secondary = example("형법", "제348조", "준사기", 4)
    family = example("민법", "제823조", "사기, 강박으로 인한 혼인취소청구권의 소멸", 5)
    wages = example("근로기준법", "제36조", "금품 청산", 6)
    sailor = example("선원법", "제55조의2", "금품 청산", 7)

    assert v5._civil_tort_adjust(mixed, general) > 0.7
    assert v5._civil_tort_adjust(mixed, offset) < 0.0
    assert v5._civil_tort_adjust(mixed + " 상계 문제도 검토해 주세요", offset) >= 0.0
    assert v5._offtopic_family_penalty(mixed, family) < 0.0

    original_search, original_facets = v3.hybrid_search_laws, v4._facet_candidates
    try:
        def fake_search(q, search_query=None, category="", k=8):
            if "중고거래" in q:
                return [fraud, offset, secondary, family]
            return [wages]
        v3.hybrid_search_laws = fake_search
        v4._facet_candidates = lambda facets: {
            "criminal_fraud": [fraud],
            "civil_tort": [offset, general],
            "seafarer_wage": [sailor],
        }
        result = v5.hybrid_search_laws(mixed, category="형사", k=5)
        ids = [x["article_id"] for x in result]
        assert ids.index(general["article_id"]) < ids.index(offset["article_id"]), result
        assert ids.index(fraud["article_id"]) <= 2 and ids.index(general["article_id"]) <= 2, result
        assert v5.hybrid_search_laws("배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다.", category="노동", k=3)[0]["article_id"] == sailor["article_id"]
        assert v5.hybrid_search_laws("퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", k=3)[0]["article_id"] == wages["article_id"]
    finally:
        v3.hybrid_search_laws, v4._facet_candidates = original_search, original_facets
    print("[통과] 불법행위 일반 규정 우선 / 무관한 상계 감점 / 선원·일반 검색 유지")


def position(results, law, article) -> str:
    return next((str(i) for i, r in enumerate(results, 1)
                 if r.get("law_name") == law and r.get("article") == article), "상위 12 밖")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unit-only", action="store_true")
    p.add_argument("--sqlite", type=Path)
    p.add_argument("--chroma", type=Path)
    args = p.parse_args()
    if not args.unit_only:
        if not args.sqlite or not args.chroma or not args.sqlite.is_file() or not args.chroma.is_dir():
            p.error("실제 비교에는 존재하는 --sqlite 및 --chroma 경로가 필요합니다.")
        os.environ["LAW_SEARCH_DB_PATH"] = str(args.sqlite.resolve())
        os.environ["LAW_VECTOR_DB_PATH"] = str(args.chroma.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.services import rag_service_v3 as v3, rag_service_v4 as v4, rag_service_v5 as v5
    unit_test(v3, v4, v5)
    if args.unit_only:
        return
    tests = [
        ("계약 손해배상", "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사", [("민법", "제390조")]),
        ("형사 사기", "다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?", "형사", [("형법", "제347조")]),
        ("퇴직 임금체불", "퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", "노동", [("근로기준법", "제36조")]),
        ("민형사 복합", "중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. 형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요.", "형사", [("형법", "제347조"),("민법", "제750조"),("민법", "제496조")]),
        ("선원 퇴직 임금", "배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다.", "노동", [("선원법", "제55조의2")]),
    ]
    print("SQLite:", args.sqlite)
    print("ChromaDB:", args.chroma)
    print("주의: 순위 비교만 수행하며 운영 파일 및 DB는 변경하지 않습니다.")
    for label, q, category, targets in tests:
        print("\n" + "=" * 60 + "\n[" + label + "] " + q)
        old = v4.hybrid_search_laws(q, category=category, k=12)
        new = v5.hybrid_search_laws(q, category=category, k=12)
        print("[v4 상위 5개]")
        for n, x in enumerate(old[:5], 1):
            print(f"  {n}. {x['law_name']} {x['article']} ({x['article_title']})")
        print("[v5 상위 5개]")
        for n, x in enumerate(new[:5], 1):
            print(f"  {n}. {x['law_name']} {x['article']} ({x['article_title']})")
        for law, art in targets:
            print(f"  {law} {art}: {position(old, law, art)} → {position(new, law, art)}")
    print("\n※ 검색 순위는 법률 적용 가능성이나 최신성 검증 결과가 아닙니다.")


if __name__ == "__main__":
    main()
