"""법령 Hybrid RAG v2 ↔ v3 비교 (읽기 전용).

전체 DB 예시:
python scripts/test_rag_service_v3.py --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\search_db\\law_search_v3_full.db" --chroma "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\vector_db\\law_db_v4_full_cpu"

먼저 코드 로직만 확인: python scripts/test_rag_service_v3.py --unit-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def unit_tests(v3):
    def it(law, article, title, id):
        return {"law_name": law, "article": article, "article_title": title,
                "article_id": id, "content": "테스트", "source_type": "keyword"}

    question = "다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?"
    fraud = it("형법", "제347조", "사기", 1)
    morale = it("경찰공무원 복무규정", "제16조", "사기진작", 2)
    assert v3._context_bonus(question, fraud) > v3._context_bonus(question, morale)
    question_wages = "퇴사 후 임금을 지급하지 않으면 어떻게 되나요?"
    settlement = it("근로기준법", "제36조", "금품 청산", 3)
    subcontract = it("근로기준법", "제44조", "도급 사업에 대한 임금 지급", 4)
    assert v3._context_bonus(question_wages, settlement) > v3._context_bonus(question_wages, subcontract)
    assert v3._context_bonus(question_wages + " 도급업체입니다", subcontract) > v3._context_bonus(question_wages, subcontract)
    normal = it("민법", "제390조", "채무불이행과 손해배상", 5)
    special = it("해양과학조사법", "제16조", "손해배상", 6)
    question_contract = "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요."
    assert v3._context_bonus(question_contract, normal) > v3._context_bonus(question_contract, special)
    assert v3._identity({"law_name":"A", "article":"", "article_id":20}) != v3._identity({"law_name":"B", "article":"", "article_id":21})
    # 실제 DB를 호출하지 않는 Hybrid 합산 검증. 테스트 도중 원래 함수를 복원한다.
    original = (v3.exact_search_laws, v3.keyword_search_laws, v3._nearest_vector_candidates)
    try:
        v3.exact_search_laws = lambda _q: []
        v3.keyword_search_laws = lambda _q, k=12: [morale, fraud]
        v3._nearest_vector_candidates = lambda _q, k=96: [fraud]
        out = v3.hybrid_search_laws(question, category="형사", k=2)
        assert out[0]["article_id"] == fraud["article_id"], "형법 사기가 맨 앞에 나와야 합니다"
        assert set(out[0]["retrieval_methods"]) == {"keyword", "vector"}
    finally:
        v3.exact_search_laws, v3.keyword_search_laws, v3._nearest_vector_candidates = original
    print("[통과] 법령 문맥 가중치 / 중의어 오탐 / 특수상황 예외 / 내부 문서 ID / Hybrid 합산")


def short(results, limit=5):
    for n, item in enumerate(results[:limit], 1):
        print(f"  {n}. {item['law_name']} {item['article']} ({item['article_title']})")


def positions(results, targets):
    found = {}
    for law, article in targets:
        found[f"{law} {article}"] = next((i for i, r in enumerate(results, 1)
            if (r['law_name'], r['article']) == (law, article)), None)
    return found


def main():
    parser = argparse.ArgumentParser(description="법령 검색 v2/v3 품질 비교 (읽기 전용)")
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--chroma", type=Path)
    parser.add_argument("--unit-only", action="store_true", help="실제 DB/Ollama 없이 문맥 재정렬 검사")
    args = parser.parse_args()
    if not args.unit_only:
        if not args.sqlite or not args.chroma:
            parser.error("--sqlite과 --chroma가 필요합니다.")
        if not args.sqlite.is_file() or not args.chroma.is_dir():
            parser.error("SQLite 파일 또는 ChromaDB 디렉터리가 없습니다.")
        os.environ["LAW_SEARCH_DB_PATH"] = str(args.sqlite.resolve())
        os.environ["LAW_VECTOR_DB_PATH"] = str(args.chroma.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.services import rag_service_v2 as v2, rag_service_v3 as v3
    unit_tests(v3)
    if args.unit_only:
        return
    print("SQLite:", v3.SEARCH_DB_PATH)
    print("ChromaDB:", v3.VECTOR_DB_PATH)
    print("주의: 운영 코드/DB 변경 없음. 이 테스트는 검색 품질의 일부 지표만 확인합니다.")
    cases = [
        ("계약 위반 손해배상", "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사", [("민법","제390조")]),
        ("형사 사기", "다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?", "형사", [("형법","제347조")]),
        ("퇴직 임금체불", "퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", "노동", [("근로기준법","제36조")]),
        ("민형사 복합", "중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. 형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요.", "형사", [("형법","제347조"),("민법","제750조")]),
        ("선원 임금", "배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다.", "노동", [("선원법","제55조의2")]),
    ]
    for title, question, category, targets in cases:
        print("\n" + "="*60 + f"\n[{title}] {question}")
        old = v2.hybrid_search_laws(question, category=category, k=12)
        new = v3.hybrid_search_laws(question, category=category, k=12)
        print("[v2 상위 5개]")
        short(old)
        print("[v3 상위 5개]")
        short(new)
        print("대표 조문 순위 (v2 → v3):")
        left, right = positions(old, targets), positions(new, targets)
        for name in left:
            print(f"  {name}: {left[name] or '상위 12 밖'} → {right[name] or '상위 12 밖'}")
        if title == "형사 사기":
            false_friends = [x for x in new[:5] if any(z in x.get('article_title','') for z in ('사기진작','분사기','심사기간'))]
            print("중의적 단어 상위 5개 혼입:", len(false_friends), "건")
    print("\n비교 완료. 법률상 적용 가능성·사건일 현행법 여부는 별도 검증이 필요합니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
