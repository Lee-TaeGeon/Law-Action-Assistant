"""법률 AI v5 다중 사건 회귀 테스트 — 기존 서비스·데이터를 변경하지 않음.

프로젝트 위치: scripts/test_legal_graph_v5_cases.py

예시:
    python scripts/test_legal_graph_v5_cases.py --sqlite "<테스트 SQLite 경로>" --chroma "<테스트 ChromaDB 폴더>"
    python scripts/test_legal_graph_v5_cases.py --sqlite "<...>" --chroma "<...>" --case fraud --live
    python scripts/test_legal_graph_v5_cases.py --list-cases

기본 실행은 Ollama 없이 RAG 후보 검색·적용범위 검사만 수행한다.
--live는 선택한 질문 하나의 LangGraph 전체 흐름을 테스트한다.
이 검사는 법적 타당성·시점별 현행성·실제 상담 적합성을 보증하지 않는다.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    key: str
    title: str
    question: str
    category: str
    search_terms: str
    expected: tuple[tuple[str, str], ...] = ()
    forbidden: tuple[tuple[str, str], ...] = ()
    # 복합 사건과 현재 인덱스에 없는 법령은 탐색적 테스트. 발견한 문제를 숨기지 않고 경고한다.
    diagnostic: bool = False


CASES: tuple[Case, ...] = (
    Case(
        "contract", "일반 계약 위반과 손해배상",
        "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.", "민사",
        "채무불이행, 손해배상, 계약 위반, 손해배상의 범위",
        expected=(("민법", "제390조"),),
        forbidden=(("민법", "제617조"),),
    ),
    Case(
        "fraud", "형사 사기 피해",
        "다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?", "형사",
        "사기, 기망, 재물, 재산상 이익",
        expected=(("형법", "제347조"),),
    ),
    Case(
        "wages", "퇴사 후 임금 미지급",
        "퇴사 후 임금을 지급하지 않으면 어떻게 되나요?", "노동",
        "금품 청산, 퇴직, 임금 지급, 미지급 임금",
        expected=(("근로기준법", "제36조"),),
    ),
    Case(
        "free_loan", "사용대차 특수 규정",
        "친구에게 물건을 무상으로 빌려줬는데 약속과 다르게 사용해서 손해가 났어요. "
        "민법 제617조에 따른 청구기간을 설명해 주세요.", "민사",
        "사용대차, 차용물, 손해배상, 손해배상 청구 기간",
        expected=(("민법", "제617조"),),
    ),
    Case(
        "mixed", "민사·형사 복합 중고거래 분쟁",
        "중고거래 판매자가 처음부터 속여 돈을 받고 물건을 보내지 않았어요. "
        "형사상 사기죄와 민사상 불법행위 손해배상을 함께 알고 싶어요.", "형사",
        "사기, 기망, 불법행위, 손해배상, 재산상 이익",
        expected=(("형법", "제347조"), ("민법", "제750조")),
        diagnostic=True,
    ),
    Case(
        "missing", "현재 테스트 DB에 없는 법령",
        "의료법 제27조의 무면허 의료행위 관련 규정을 알려 주세요.", "기타",
        "의료법, 무면허 의료행위",
        forbidden=(("의료법", "제27조"),),
        diagnostic=True,
    ),
)

CASE_BY_KEY = {case.key: case for case in CASES}


def article_key(item: dict[str, Any]) -> tuple[str, str]:
    """출처 객체를 (법령명, '제N조[의N]') 식별자로 정규화한다."""
    law = str(item.get("law_name") or "").strip()
    article = re.sub(r"\s+", "", str(item.get("article") or ""))
    match = re.fullmatch(r"제?(\d+)조?(?:의(\d+))?", article)
    if match:
        article = f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else "")
    return law, article


def has_ref(items: list[dict[str, Any]], target: tuple[str, str]) -> bool:
    return any(article_key(item) == target for item in items)


def check_sources(case: Case, items: list[dict[str, Any]], *, top: int | None = None) -> list[str]:
    """현재 후보를 평가한다. 다중 법령 지원은 아직 탐색적 지표다."""
    selected = items if top is None else items[:top]
    issues: list[str] = []
    for target in case.expected:
        if not has_ref(selected, target):
            issues.append(f"필요한 대표 조문 미포함: {target[0]} {target[1]}")
    for target in case.forbidden:
        if has_ref(selected, target):
            issues.append(f"적용범위가 맞지 않거나 인덱스에 없는 조문 등장: {target[0]} {target[1]}")
    return issues


def check_live(case: Case, state: dict[str, Any]) -> list[str]:
    issues = check_sources(case, list(state.get("sources") or []))
    status = str(state.get("answer_validation_status") or "")
    answer = str(state.get("answer") or "")
    if case.key == "missing":
        # 테스트 DB의 법령 누락 상황에서는 무리하게 확정적 안내를 하지 않는지가 핵심.
        if "의료법 제27조" in answer:
            issues.append("색인되지 않은 의료법 제27조를 답변에 포함")
        if status == "ok":
            issues.append("색인되지 않은 법령 질문에 다른 출처로 답변 생성: 수동 검토 필요")
    else:
        if status != "ok":
            issues.append(f"답변 검증 상태가 ok가 아님: {status or '미기록'}")
        if status == "ok" and "법문 근거:" not in answer:
            issues.append("검증 완료 상태인데 법문 근거가 없음")
    if case.key == "contract":
        if any(term in answer for term in ("제617조", "6개월", "6월 내", "6월내")):
            issues.append("일반 계약에 사용대차 제617조/6개월 기간이 사용됨")
    return issues


def self_test() -> None:
    contract = CASE_BY_KEY["contract"]
    assert article_key({"law_name": "민법", "article": " 390 "}) == ("민법", "제390조")
    assert article_key({"law_name": "민법", "article": "제392조의2"}) == ("민법", "제392조의2")
    assert not check_sources(contract, [{"law_name": "민법", "article": "제390조"}])
    assert check_sources(contract, [{"law_name": "민법", "article": "제617조"}])
    assert check_sources(CASE_BY_KEY["mixed"], [{"law_name": "형법", "article": "제347조"}])
    good = {"sources": [{"law_name": "민법", "article": "제390조"}],
            "answer_validation_status": "ok", "answer": "법문 근거: 조문 인용"}
    assert not check_live(contract, good)
    bad = dict(good, answer="법문 근거: 6개월 안에 청구하세요")
    assert check_live(contract, bad)
    print("[통과] 테스트 평가기 자체 검증 — 실제 검색/LLM 검증과 별도")


def print_items(items: list[dict[str, Any]], limit: int = 5) -> None:
    for rank, item in enumerate(items[:limit], 1):
        print(f"  {rank}. {item.get('law_name', '')} {item.get('article', '')} "
              f"({item.get('article_title', '')})")


def search_case(case: Case, rag: Any, v5: Any) -> list[str]:
    # 이 사전 검사는 LLM이 만든 재작성 검색어 대신 고정 키워드를 사용한다.
    items = rag.hybrid_search_laws(
        question=case.question, search_query=case.search_terms,
        category=case.category, k=20,
    )
    with closing(rag.get_search_connection()) as db:
        scoped_items: list[dict[str, Any]] = []
        for item in items:
            enriched = dict(item)
            enriched["section_scope"] = v5.v4._section_scope(db, enriched)
            allowed, _ = v5.scope_eligibility(case.question, enriched)
            if allowed:
                scoped_items.append(enriched)
    print("[Hybrid RAG 상위 5개 — 고정 검색어 사용]")
    print_items(scoped_items)
    issues = check_sources(case, scoped_items, top=5)
    return issues


def run_live(case: Case, graph: Any) -> list[str]:
    print("\n[Ollama 전체 LangGraph 실행]", flush=True)
    state = graph.run_legal_chat(case.question, "v5-multicase-" + uuid.uuid4().hex)
    print("LLM 분류:", state.get("category"))
    print("LLM 검색어:", state.get("search_query"))
    print("선택한 조문:")
    print_items(list(state.get("sources") or []), limit=10)
    print("Reranker 상태:", state.get("reranker_status"))
    print("Generator 검증 상태:", state.get("answer_validation_status"))
    print("답변:\n", state.get("answer"))
    return check_live(case, state)


def main() -> int:
    parser = argparse.ArgumentParser(description="법률 AI v5 다중 사건 RAG·Ollama 회귀 테스트")
    parser.add_argument("--sqlite", type=Path, help="신규 테스트 SQLite 경로")
    parser.add_argument("--chroma", type=Path, help="신규 테스트 ChromaDB 경로")
    parser.add_argument("--case", choices=sorted(CASE_BY_KEY), help="특정 사례만 시험")
    parser.add_argument("--live", action="store_true", help="선택 사례 1개 Ollama 전체 실행")
    parser.add_argument("--list-cases", action="store_true", help="테스트 사건 목록만 출력")
    parser.add_argument("--self-test", action="store_true", help="외부 라이브러리 없이 검사 로직 검증")
    args = parser.parse_args()

    if args.list_cases:
        for case in CASES:
            kind = "탐색적" if case.diagnostic else "대표 회귀"
            print(f"{case.key:<10} {case.title} ({kind})")
        return 0
    if args.self_test:
        self_test()
        return 0
    if not args.sqlite or not args.chroma:
        parser.error("--sqlite와 --chroma 경로를 지정해 주세요.")
    if args.live and not args.case:
        parser.error("--live 실행 시 --case 한 개를 지정해 주세요. 한 번에 Ollama 호출이 몰리는 것을 방지합니다.")

    sql_path = args.sqlite.expanduser().resolve()
    chroma_path = args.chroma.expanduser().resolve()
    if not sql_path.is_file() or not chroma_path.is_dir():
        parser.error("새 테스트 SQLite 또는 ChromaDB 경로가 존재하지 않습니다.")
    # 환경변수는 모델/서비스가 import되기 전에 설정해야 한다.
    os.environ["LAW_SEARCH_DB_PATH"] = str(sql_path)
    os.environ["LAW_VECTOR_DB_PATH"] = str(chroma_path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from backend.services import rag_service_v2 as rag
    from backend.graph import legal_graph_v5 as graph

    print("SQLite:", rag.SEARCH_DB_PATH)
    print("ChromaDB:", rag.VECTOR_DB_PATH)
    print("주의: 운영용 main.py·기존 DB를 변경하지 않습니다.")
    print("주의: 6개 법령으로 검증하는 테스트이며 법률적 정확도를 보장하지 않습니다.")

    cases = (CASE_BY_KEY[args.case],) if args.case else CASES
    hard_failures = 0
    warnings = 0
    for case in cases:
        print("\n" + "=" * 67)
        print(f"[{case.key}] {case.title}")
        print("질문:", case.question)
        try:
            if case.key == "missing":
                # 인덱스에 없는 법령을 수동으로 없다고 가정하지 않는다.
                with closing(rag.get_search_connection()) as db:
                    found = db.execute("SELECT 1 FROM laws WHERE law_name=? LIMIT 1", ("의료법",)).fetchone()
                if found:
                    print("[주의] 의료법이 실제 DB에 있습니다. 이 사례는 누락 테스트가 아니므로 건너뜁니다.")
                    warnings += 1
                    continue
            issues = run_live(case, graph) if args.live else search_case(case, rag, graph)
        except Exception as exc:
            print(f"[실패] 실행 중 예외: {type(exc).__name__}: {exc}", file=sys.stderr)
            if case.diagnostic:
                warnings += 1
            else:
                hard_failures += 1
            continue

        if not issues:
            print("[통과] 기준 충족 (법적 정확성 검증은 아님)")
        else:
            kind = "[주의] 추가 개선 필요" if case.diagnostic else "[실패] 회귀 기준 미충족"
            print(kind)
            for issue in issues:
                print(" -", issue)
            if case.diagnostic:
                warnings += 1
            else:
                hard_failures += 1

    print("\n" + "=" * 67)
    print(f"검사 완료: 필수 기준 실패 {hard_failures}건 / 탐색적 주의 {warnings}건")
    print("※ 검색 상위 결과와 LLM 출력은 법적 타당성·시점별 적용법령의 확인이 아닙니다.")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
