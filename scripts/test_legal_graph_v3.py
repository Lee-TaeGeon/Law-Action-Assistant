"""법령 적용범위 회귀 테스트. 운영 DB/서비스 수정 없음.

1) LLM 없는 범위/안전검사 테스트:
 python scripts/test_legal_graph_v3.py --sqlite "<new.db>" --chroma "<new_chroma>"
2) Ollama 포함 전체 테스트:
 python scripts/test_legal_graph_v3.py --sqlite "<new.db>" --chroma "<new_chroma>" --live
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from contextlib import closing
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="법률 조문 적용범위 v3 검증")
    p.add_argument("--sqlite", required=True, type=Path)
    p.add_argument("--chroma", required=True, type=Path)
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    sql_path = args.sqlite.expanduser().resolve()
    chroma_path = args.chroma.expanduser().resolve()
    if not sql_path.is_file() or not chroma_path.is_dir():
        p.error("SQLite 파일과 ChromaDB 폴더 경로를 확인하세요.")
    os.environ["LAW_SEARCH_DB_PATH"] = str(sql_path)
    os.environ["LAW_VECTOR_DB_PATH"] = str(chroma_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.services import rag_service_v2 as rag
    from backend.graph import legal_graph_v3 as graph

    print("SQLite:", rag.SEARCH_DB_PATH)
    print("ChromaDB:", rag.VECTOR_DB_PATH)
    print("운영 파일·DB는 변경하지 않습니다.\n")

    with closing(rag.get_search_connection()) as db:
        rows = db.execute(
            "SELECT * FROM articles WHERE law_name = ? AND article_no IN (?, ?)",
            ("민법", "617", "390"),
        ).fetchall()
        mapped = {r["article_no"]: rag._row_to_source(r, "exact") for r in rows}
        if not {"390", "617"}.issubset(mapped):
            raise RuntimeError("민법 제390조 또는 제617조 데이터가 누락되었습니다.")
        for source in mapped.values():
            source["section_scope"] = graph._section_scope(db, source)

    q = "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요."
    article_390 = mapped["390"]
    article_617 = mapped["617"]
    print("민법 제390조 소속:", article_390["section_scope"])
    print("민법 제617조 소속:", article_617["section_scope"])
    assert "사용대차" in article_617["section_scope"], "법령 체계상 '사용대차' 문맥을 찾지 못했습니다."
    assert graph.scope_eligibility(q, article_390)[0]
    assert not graph.scope_eligibility(q, article_617)[0], "일반 계약 질문에 사용대차 조문이 배제되지 않았습니다."
    assert graph.scope_eligibility("민법 제617조의 의미를 설명해줘", article_617)[0]
    assert graph.scope_eligibility("무상으로 빌린 물건을 함부로 사용했어요", article_617)[0]
    assert graph._has_supported_reference("민법 제390조에 따라 검토할 수 있습니다.", [article_390])[0]
    assert not graph._has_supported_reference("민법 제617조에 따라 6개월 이내 청구합니다.", [article_390])[0]
    assert not graph._has_supported_reference("6개월 이내에 청구해야 합니다.", [article_390])[0]
    assert not graph._has_supported_reference("손해배상 소송을 제기하세요.", [article_390])[0]
    print("[통과] 일반 계약 vs 사용대차 범위 구분, 조문·기간·절차 근거 검사")

    if not args.live:
        print("LLM 호출 없이 회귀 검증 완료. 전체 실행은 --live 옵션으로 별도 확인하세요.")
        return 0

    print("\nOllama 전체 LangGraph 실행 중...", flush=True)
    result = graph.run_legal_chat(q, "v3-scope-test-" + uuid.uuid4().hex)
    print("분류:", result.get("category"))
    print("검색어:", result.get("search_query"))
    print("선정 법령:")
    for s in result.get("sources", []):
        print(" -", s.get("law_name"), s.get("article"), s.get("section_scope"))
    answer = str(result.get("answer", ""))
    print("\n답변:\n", answer)
    if any(s.get("law_name") == "민법" and s.get("article") == "제617조" for s in result.get("sources", [])):
        print("[실패] 일반 계약 질문에 사용대차 제617조가 선택되었습니다.", file=sys.stderr)
        return 1
    if "제617조" in answer or "6개월" in answer or "6월내" in answer or "6월 내" in answer:
        print("[실패] 사용대차 6개월 규정이 일반 계약 답변에 나타났습니다.", file=sys.stderr)
        return 1
    if not answer:
        print("[실패] AI 답변이 비어 있습니다.", file=sys.stderr)
        return 1
    print("\n[통과] 이 사례의 적용범위 오용 재발 여부 검사 통과")
    print("※ 다른 사례에 대한 법적 정확성·현행성은 검증되지 않았습니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
