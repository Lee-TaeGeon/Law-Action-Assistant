"""법령 RAG v2 검색 품질 비교 (읽기 전용).

프로젝트 루트에서 실행:
  python scripts/test_rag_service_v2.py --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\search_db\\law_search_v2_test.db" --chroma "C:\\dev\\Law-Action-Assistant-LawDataV2\\vector_db\\law_db_v3_full"

테스트 DB 6개 법령에 포함된 대표 조문 3개의 순위를 비교한다.
기존 운영 DB, JSON, ChromaDB는 변경하지 않는다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def describe(title, results, target):
    print(f"\n  [{title}]")
    for index, item in enumerate(results[:5], 1):
        print(f"    {index:>2}. {item['law_name']} {item['article']} ({item['article_title']})")
    rank = next((i for i, item in enumerate(results, 1)
                 if (item["law_name"], item["article"]) == target), None)
    print(f"    대표 조문 {target[0]} {target[1]}: {rank if rank is not None else '상위 검색 결과 없음'}위")
    return rank


def main():
    parser = argparse.ArgumentParser(description="법령 Hybrid RAG v2 검색 순위 비교 (읽기 전용)")
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--chroma", required=True, type=Path)
    args = parser.parse_args()
    sqlite_path = args.sqlite.resolve()
    chroma_path = args.chroma.resolve()
    if not sqlite_path.is_file() or not chroma_path.is_dir():
        parser.error("SQLite 파일 또는 ChromaDB 폴더가 없습니다.")

    # 반드시 서비스 모듈을 import하기 전에 환경변수 설정
    os.environ["LAW_SEARCH_DB_PATH"] = str(sqlite_path)
    os.environ["LAW_VECTOR_DB_PATH"] = str(chroma_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.services import rag_service_v2 as rag

    cases = [
        ("계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사", ("민법", "제390조")),
        ("다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?", "형사", ("형법", "제347조")),
        ("퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", "노동", ("근로기준법", "제36조")),
    ]
    print("테스트 SQLite:", rag.SEARCH_DB_PATH)
    print("테스트 ChromaDB:", rag.VECTOR_DB_PATH)
    print("주의: 이 스크립트는 서비스 DB를 변경하거나 운영 서비스에 새 검색 로직을 적용하지 않습니다.")
    successes = 0
    for question, category, target in cases:
        print("\n" + "=" * 65)
        print("질문:", question)
        print("참고 분야:", category)
        keyword = rag.keyword_search_laws(question, k=12)
        vector = rag.search_laws(question, category=category, k=12)
        hybrid = rag.hybrid_search_laws(question, category=category, k=12)
        describe("키워드 단독", keyword, target)
        describe("의미 검색 단독", vector, target)
        rank = describe("새 Hybrid RAG", hybrid, target)
        successes += rank is not None and rank <= 5

    print("\n" + "=" * 65)
    print(f"대표 조문 Top-5 포함: {successes}/{len(cases)}")
    print("※ 대표 조문이 5위 안에 있어도 법률적 적합성·최신성 검증을 통과한 것은 아닙니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[검사 실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
