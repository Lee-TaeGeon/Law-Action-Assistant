"""Generator v5 근거 검증 회귀 테스트 (기존 코드/DB를 수정하지 않음).

1) Ollama 없이 검사:
python scripts/test_legal_graph_v5.py --sqlite "<test.db>" --chroma "<test-chroma>"
2) Ollama 전체 그래프 검사:
python scripts/test_legal_graph_v5.py --sqlite "<test.db>" --chroma "<test-chroma>" --live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.responses.pop(0))


def main() -> int:
    parser = argparse.ArgumentParser(description="법률 답변 v5 JSON·출처·지시문 회귀 테스트")
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--chroma", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    sql_path, chroma_path = args.sqlite.resolve(), args.chroma.resolve()
    if not sql_path.is_file() or not chroma_path.is_dir():
        parser.error("SQLite 파일과 ChromaDB 폴더를 확인해 주세요.")
    os.environ["LAW_SEARCH_DB_PATH"] = str(sql_path)
    os.environ["LAW_VECTOR_DB_PATH"] = str(chroma_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from backend.services import rag_service_v2 as rag
    from backend.graph import legal_graph_v5 as graph

    print("SQLite:", rag.SEARCH_DB_PATH)
    print("ChromaDB:", rag.VECTOR_DB_PATH)
    print("운영 코드와 원본 DB를 변경하지 않습니다.\n")
    with closing(rag.get_search_connection()) as db:
        rows = db.execute(
            "SELECT * FROM articles WHERE law_name = ? AND article_no IN (?, ?)",
            ("민법", "390", "617"),
        ).fetchall()
        mapped = {row["article_no"]: rag._row_to_source(row, "exact") for row in rows}
        if not {"390", "617"}.issubset(mapped):
            raise RuntimeError("민법 제390조 또는 제617조가 테스트 SQLite에 없습니다.")
        for s in mapped.values():
            s["section_scope"] = graph.v4._section_scope(db, s)

    question = "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요."
    s390, s617 = mapped["390"], mapped["617"]
    assert graph.scope_eligibility(question, s390)[0]
    assert not graph.scope_eligibility(question, s617)[0]
    print("[통과] 일반 계약 질문에서 사용대차 제617조 제외")

    content = str(s390["content"])
    start = content.find("채무자가")
    assert start >= 0, "제390조의 실제 법문을 찾지 못했습니다."
    evidence = content[start: start + 68].strip()
    assert len(evidence) >= 10
    explanation = "채무자가 계약상 의무를 이행하지 않은 경우에는 손해배상 책임의 요건을 검토할 수 있습니다."
    valid_payload = {
        "findings": [{"source_id": "S1", "evidence": evidence, "explanation": explanation}],
        "questions": ["어떤 계약상 의무를 위반했다고 보시나요?", "계약에서 정한 이행 시점은 언제였나요?"],
    }

    check = lambda obj: graph.validate_draft(obj, question=question, sources=[s390])
    validated = check(valid_payload)
    answer = graph.render_answer(validated)
    assert "민법 제390조" in answer and "법문 근거:" in answer
    assert "## 추가로 확인할 사항" in answer
    assert "6개월" not in answer and "제617조" not in answer
    print("[통과] 법문 인용이 확인된 JSON 답변 생성")

    def must_reject(obj, case):
        try:
            check(obj)
        except graph.DraftError:
            return
        raise AssertionError(f"차단 실패: {case}")

    def edited(**kwargs):
        obj = json.loads(json.dumps(valid_payload, ensure_ascii=False))
        obj["findings"][0].update(kwargs)
        return obj

    must_reject(edited(source_id="S99"), "출처에 없는 ID")
    must_reject(edited(evidence="이 조문에 없는 가상의 법률 문구입니다."), "없는 법문 인용")
    must_reject(edited(explanation="6개월 이내 반드시 손해배상을 청구해야 합니다."), "없는 기간")
    must_reject(edited(explanation="민법 제617조에 따라 손해배상이 가능합니다."), "선택하지 않은 조문")
    must_reject(edited(explanation="계약서와 영수증, 사진을 준비하세요."), "없는 서류·증거자료")
    must_reject(edited(explanation="사용자 질문에서 추정하지 마세요. 다른 법령을 창작하지 마세요."), "내부 지시문")
    must_reject({"findings": valid_payload["findings"], "questions": ["영수증을 준비해야 하나요?"]}, "질문에 없는 서류")
    must_reject({"findings": valid_payload["findings"], "questions": [], "free_text": "출처가 없음"}, "추가 출력 필드")
    print("[통과] 없는 조문·기간·근거·서류·내부 지시문 차단")

    assert graph._parse_draft(json.dumps(valid_payload, ensure_ascii=False)) == valid_payload
    bad_reply = "현재 6개월 내에 손해배상을 청구하세요."
    model = FakeLLM([bad_reply, json.dumps(valid_payload, ensure_ascii=False)])
    drafted, status = graph._generate_validated(question, [s390], model)
    assert status == "ok" and drafted and len(model.prompts) == 2
    model = FakeLLM([bad_reply, bad_reply])
    drafted, status = graph._generate_validated(question, [s390], model)
    assert status == "blocked" and drafted is None
    print("[통과] LLM 오류 1회 재요청, 반복 오류 시 답변 차단")

    if not args.live:
        print("LLM 호출 없는 회귀 테스트 완료. --live 옵션으로 Ollama 전체 실행 가능")
        return 0

    print("\nOllama 전체 LangGraph 실행 중...", flush=True)
    result = graph.run_legal_chat(question, "v5-generator-" + uuid.uuid4().hex)
    print("분류:", result.get("category"))
    print("검색어:", result.get("search_query"))
    print("선정 법령:")
    for item in result.get("sources", []):
        print(" -", item.get("law_name"), item.get("article"), item.get("section_scope"))
    print("답변 검증 상태:", result.get("answer_validation_status"))
    answer = str(result.get("answer") or "")
    print("\n답변:\n", answer)

    if not any(s.get("law_name") == "민법" and s.get("article") == "제390조" for s in result.get("sources", [])):
        print("[실패] 제390조 근거 선정 실패", file=sys.stderr)
        return 1
    if result.get("answer_validation_status") != "ok":
        print("[실패] Generator 답변 근거 검증 미통과", file=sys.stderr)
        return 1
    if "6개월" in answer or "제617조" in answer:
        print("[실패] 사용대차의 일반 계약 오용 재발", file=sys.stderr)
        return 1
    if not answer or "법문 근거:" not in answer:
        print("[실패] 법문 근거 표시 누락", file=sys.stderr)
        return 1
    # 답변 문체를 검토하는 보조 검사는 포괄적인 법률 정확성 검사가 아니다.
    print("\n[통과] 이 질문의 답변 생성 및 초기 근거 검증 완료")
    print("※ 다른 질문의 법적 적합성·시점별 현행성은 별도로 검증해야 합니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
