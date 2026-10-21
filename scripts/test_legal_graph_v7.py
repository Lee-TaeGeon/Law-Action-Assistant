"""LangGraph v7 + RAG v5 독립 테스트.

실행 (프로젝트 루트 PowerShell):
    python scripts/test_legal_graph_v7.py --unit-only
    python scripts/test_legal_graph_v7.py --sqlite <새_DB> --chroma <새_Chroma>
    python scripts/test_legal_graph_v7.py --sqlite <새_DB> --chroma <새_Chroma> --live

--unit-only는 모의 검색 결과로 연구 노드 연결과 중복/빈문서 처리만 확인한다.
기본 실행은 실제 DB로 연구 노드를 실행하지만 Ollama를 호출하지 않는다.
--live만 Ollama 전체 그래프를 호출한다.
운영 DB 및 서비스 코드는 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4


def unit_test(graph) -> None:
    sample = [
        {"article_id": 11, "law_name": "형법", "article": "제347조", "article_title": "사기", "content": "조문 A"},
        {"article_id": 11, "law_name": "형법", "article": "제347조", "article_title": "사기", "content": "조문 A"},
        {"article_id": 12, "law_name": "관할구역 변경법", "article": "", "article_title": "", "content": "조문번호 없는 공식 본문"},
        {"article_id": 13, "law_name": "민법", "article": "제99조", "article_title": "빈 내용", "content": "   "},
    ]
    old_search = graph._search_candidates
    old_conn = graph.rag.get_search_connection
    old_scope = graph.v3._section_scope
    try:
        graph._search_candidates = lambda q, s, c: sample
        graph.rag.get_search_connection = lambda: sqlite3.connect(":memory:")
        graph.v3._section_scope = lambda db, source: "테스트 범위"
        result = graph.researcher_node({"question": "중고거래 피해", "search_query": "사기", "category": "형사"})
        docs = result["candidate_sources"]
        assert len(docs) == 2, docs
        assert [x["article_id"] for x in docs] == [11, 12], docs
        assert all(x["section_scope"] == "테스트 범위" for x in docs)
        assert graph.researcher_node({"question": ""}) == {"candidate_sources": []}
    finally:
        graph._search_candidates = old_search
        graph.rag.get_search_connection = old_conn
        graph.v3._section_scope = old_scope
    print("[통과] v5 검색 연결 / SQLite 문서 ID 중복 제거 / 번호 없는 법령 후보 보존")


def rank(docs: list[dict], name: str, article: str) -> str:
    return next((str(i) for i, d in enumerate(docs, 1)
                 if d.get("law_name") == name and d.get("article") == article), "상위 16 밖")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--chroma", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.unit_only and args.live:
        parser.error("--unit-only와 --live는 동시에 사용할 수 없습니다.")
    if not args.unit_only:
        if not args.sqlite or not args.sqlite.is_file() or not args.chroma or not args.chroma.is_dir():
            parser.error("--sqlite 및 --chroma에 존재하는 새 테스트 DB 경로를 지정하세요.")
        os.environ["LAW_SEARCH_DB_PATH"] = str(args.sqlite.resolve())
        os.environ["LAW_VECTOR_DB_PATH"] = str(args.chroma.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.graph import legal_graph_v7 as graph

    unit_test(graph)
    if args.unit_only:
        print("단위 검사 완료. Ollama와 실제 DB는 사용하지 않았습니다.")
        return

    cases = [
        ("일반 계약", "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사", [("민법", "제390조")]),
        ("중고거래 민형사", "중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. 형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요.", "형사", [("형법", "제347조"), ("민법", "제750조")]),
        ("선원 임금", "배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다.", "노동", [("선원법", "제55조의2")]),
    ]
    print("SQLite:", args.sqlite)
    print("ChromaDB:", args.chroma)
    print("주의: FastAPI/운영 DB는 변경하지 않습니다.")
    for label, question, category, expected in cases:
        print("\n" + "=" * 64 + "\n", label)
        if args.live:
            state = graph.run_legal_chat(question, "rag-v7-" + uuid4().hex)
            docs = list(state.get("candidate_sources") or [])
            print("AI 분류:", state.get("category"))
            print("검증 상태:", state.get("answer_validation_status"))
            print("선정 조문:", [(s.get("law_name"), s.get("article")) for s in state.get("sources") or []])
            print("답변:\n", state.get("answer"))
        else:
            docs = graph.researcher_node({"question": question, "search_query": "", "category": category})["candidate_sources"]
        print("RAG v5 후보 상위 8건:")
        for i, doc in enumerate(docs[:8], 1):
            print(f"  {i}. {doc.get('law_name')} {doc.get('article')} ({doc.get('article_title')})")
        for name, article in expected:
            print(f"대표 조문 {name} {article}: {rank(docs, name, article)}")
    print("\n검사 종료. 상위 검색 결과가 실제 법률 적용을 입증하지는 않습니다.")


if __name__ == "__main__":
    main()
