"""개선된 법령 RAG / LangGraph 연결 테스트 (원본 서비스 변경 없음).

프로젝트 최상위 폴더에서:
python scripts/test_legal_graph_v2.py --sqlite "<test.db>" --chroma "<chroma_dir>"
python scripts/test_legal_graph_v2.py --sqlite "<test.db>" --chroma "<chroma_dir>" --live

기본 모드: 새 그래프 researcher_node 연동 검증 (Ollama 호출 불필요)
--live: 기존 Ollama 모델로 전체 그래프를 호출 (서비스 API 변경 없음)
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

CASES = [
    ("계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사", "민법", "제390조"),
    ("다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?", "형사", "형법", "제347조"),
    ("퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", "노동", "근로기준법", "제36조"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="새 LangGraph + Hybrid RAG 연동 테스트")
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--chroma", required=True, type=Path)
    parser.add_argument("--live", action="store_true", help="Ollama를 통한 실제 LangGraph 전체 테스트")
    parser.add_argument("--question", default=None, help="--live와 함께 테스트할 단일 질문")
    args = parser.parse_args()

    sqlite_path = args.sqlite.expanduser().resolve()
    chroma_path = args.chroma.expanduser().resolve()
    if not sqlite_path.is_file() or not chroma_path.is_dir():
        parser.error("테스트용 SQLite 파일 또는 ChromaDB 폴더가 존재하지 않습니다.")

    # 중요: 서비스/그래프 모듈을 import하기 *전에* DB 경로 설정
    os.environ["LAW_SEARCH_DB_PATH"] = str(sqlite_path)
    os.environ["LAW_VECTOR_DB_PATH"] = str(chroma_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.services import rag_service_v2 as rag
    from backend.graph import legal_graph_v2 as graph_v2

    print("새 SQLite:", rag.SEARCH_DB_PATH)
    print("새 ChromaDB:", rag.VECTOR_DB_PATH)
    print("주의: FastAPI 운영 코드와 DB는 교체하지 않습니다.\n")

    if not args.live:
        # 검색 노드에 해당하는 개선 Hybrid 검색 확인 (Ollama 불필요)
        passed = 0
        for question, category, law_name, article in CASES:
            node_output = graph_v2.researcher_node({
                "question": question,
                "search_query": question,  # 재작성 LLM을 생략한 검색 노드 단독 테스트
                "category": category,
            })
            candidates = node_output["candidate_sources"]
            matches = [
                i for i, item in enumerate(candidates, 1)
                if (item.get("law_name"), item.get("article")) == (law_name, article)
            ]
            print(f"질문: {question}")
            for i, item in enumerate(candidates[:5], 1):
                print(f"  {i}. {item.get('law_name')} {item.get('article')} ({item.get('article_title')})")
            rank = matches[0] if matches else None
            print(f"  대표 조문: {law_name} {article}, 순위: {rank or '없음'}\n")
            if rank is not None and rank <= 5:
                passed += 1
        print(f"대표 조문 Top-5 포함: {passed}/{len(CASES)}")
        print("LangGraph의 researcher_node 연결을 확인했습니다. LLM 전체 실행은 --live로 별도 테스트합니다.")
        return 0 if passed == len(CASES) else 1

    # 새로운 Graph 구현 모듈은 --live일 때만 import (Ollama 호출 가능)
    run_legal_chat = graph_v2.run_legal_chat

    question = args.question or CASES[0][0]
    print("LangGraph 전체 실행 중 (Ollama 사용)...\n질문:", question)
    result = run_legal_chat(question=question, session_id="graph-v2-test-" + uuid.uuid4().hex)
    print("\n분류:", result.get("category", ""))
    print("검색어:", result.get("search_query", ""))
    print("선택한 법령:")
    for source in result.get("sources", []):
        print(f"  - {source.get('law_name')} {source.get('article')} ({source.get('article_title')})")
    print("\n답변:\n", result.get("answer", ""))
    if not result.get("answer"):
        print("[실패] AI 답변이 비어 있습니다.", file=sys.stderr)
        return 1
    print("\n테스트용 LangGraph 실행 완료. 운영 서비스는 변경되지 않았습니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
