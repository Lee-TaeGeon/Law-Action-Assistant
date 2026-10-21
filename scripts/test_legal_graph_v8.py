"""LangGraph v8 인용 검증형 Generator v6 회귀 테스트.

프로젝트 루트에서:
  python scripts/test_legal_graph_v8.py --unit-only
  python scripts/test_legal_graph_v8.py --sqlite <FULL_SQLITE> --chroma <FULL_CHROMA> --live

기존 운영 코드·DB를 쓰거나 수정하지 않음. --live만 Ollama를 호출함.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


class FakeLLM:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=next(self.answers))


def unit_test(graph):
    civil = {
        "law_name": "민법", "article": "제394조", "article_title": "손해배상의 방법",
        "content": "<민법>\n제394조 (손해배상의 방법)\n다른 의사표시가 없으면 손해는 금전으로 배상한다.",
    }
    marine = {
        "law_name": "선원법", "article": "제55조의3", "article_title": "미지급 임금에 대한 지연이자",
        "content": "<선원법>\n제55조의3 (미지급 임금에 대한 지연이자)\n"
                   "선박소유자는 연 100분의 40 이내의 범위에서 대통령령으로 정하는 이율에 따른 지연이자를 지급하여야 한다.",
    }
    sources = [civil, marine]
    good = {"findings": [
        {"source_id": "S1", "evidence": "다른 의사표시가 없으면 손해는 금전으로 배상한다."},
        {"source_id": "S2", "evidence": "연 100분의 40 이내의 범위에서 대통령령으로 정하는 이율에 따른 지연이자를 지급하여야 한다."},
    ]}
    validated = graph.validate_evidence_draft(good, sources)
    answer = graph.render_evidence_answer(validated, "선원이 퇴직한 뒤 임금을 받지 못했습니다.")
    assert "법문 원문" in answer and "민법 제394조" in answer
    assert "대통령령으로 정하는 이율" in answer
    assert "금전배상만 인정" not in answer
    assert "연 40% 지연이자가 적용" not in answer
    assert "제한된 테스트 법령" not in answer
    assert "퇴직일" in answer
    print("[통과] 검증된 법문만 표시 / 과장된 해석·기한·이율 단정 방지")

    def must_fail(payload):
        try:
            graph.validate_evidence_draft(payload, sources)
        except graph.EvidenceError:
            return
        raise AssertionError(f"차단 실패: {payload}")

    must_fail({"findings": [dict(good["findings"][0], explanation="금전배상만 인정") ]})
    must_fail({"findings": [{"source_id": "S1", "evidence": "이 법은 금전배상만을 허용한다."}]})
    must_fail({"findings": [{"source_id": "S99", "evidence": good["findings"][0]["evidence"]}]})
    must_fail({"findings": [good["findings"][0], good["findings"][0]]})
    must_fail({"findings": [good["findings"][0]], "questions": ["법률상 최대 이율은 40%인가요?"]})
    must_fail({"findings": [{"source_id": "S1", "evidence": "제394조 (손해배상의 방법)"}]})
    try:
        graph.parse_evidence_draft('```json\n' + json.dumps(good, ensure_ascii=False) + '\n```')
    except graph.EvidenceError:
        pass
    else:
        raise AssertionError("코드펜스 차단 실패")
    print("[통과] 임의 설명·왜곡 인용·조문 헤더·없는 출처·중복·자유 질문 차단")

    good_raw = json.dumps(good, ensure_ascii=False)
    model = FakeLLM(['{"findings":[]}', good_raw])
    verified, status = graph.select_verified_evidence("퇴직 임금", sources, model)
    assert status == "ok" and model.calls == 2 and len(verified) == 2
    model = FakeLLM(['{"findings":[]}', '{"findings":[]}'])
    verified, status = graph.select_verified_evidence("퇴직 임금", sources, model)
    assert status == "blocked" and verified is None
    assert graph._facts_to_confirm("중고거래 사기 피해")[0].startswith("거래 당시")
    assert graph._facts_to_confirm("계약 위반 손해배상")[0].startswith("어떤 의무")
    print("[통과] 잘못된 LLM 출력 1회 재시도·반복 실패 시 차단·사실확인 질문")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--chroma", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.unit_only:
        if not args.sqlite or not args.sqlite.is_file() or not args.chroma or not args.chroma.is_dir():
            parser.error("--sqlite 및 --chroma에 기존 전체 테스트 DB 경로를 지정하세요.")
        os.environ["LAW_SEARCH_DB_PATH"] = str(args.sqlite.resolve())
        os.environ["LAW_VECTOR_DB_PATH"] = str(args.chroma.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.graph import legal_graph_v8 as graph

    unit_test(graph)
    if args.unit_only:
        print("단위 검사 완료. 실제 법령 DB와 Ollama는 호출하지 않았습니다.")
        return
    if not args.live:
        print("현재는 단위 검사만 실행했습니다. --live 옵션으로 Ollama 전체 그래프 검사가 가능합니다.")
        return

    for label, question in [
        ("일반 계약", "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요."),
        ("중고거래 민형사", "중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. 형사상 사기죄와 민사상 불법행위 손해배상을 알고 싶어요."),
        ("선원 임금", "배에서 일하는 선원이 퇴직한 뒤 임금을 받지 못했습니다."),
    ]:
        print("\n===", label, "===", flush=True)
        result = graph.run_legal_chat(question, "generator-v8-" + uuid4().hex)
        print("검증 상태:", result.get("answer_validation_status"))
        print("선정 법령:", [(s.get("law_name"), s.get("article")) for s in (result.get("sources") or [])])
        print("답변:\n", result.get("answer"))
        status = result.get("answer_validation_status")
        if status != "ok":
            raise RuntimeError(f"{label}: 인용 검증 실패. 로그를 확인하세요 (status={status})")
        answer = str(result.get("answer") or "")
        if "법문 원문:" not in answer or "제한된 테스트 법령" in answer:
            raise RuntimeError(f"{label}: 검증된 원문 출력 누락 또는 예전 안내문 재발")
    print("\n[통과] 3개 대표 사례의 인용·표시 검증. 법률적 적용 정확성은 별도 검토 필요")


if __name__ == "__main__":
    main()
