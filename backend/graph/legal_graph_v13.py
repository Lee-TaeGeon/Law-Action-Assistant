"""격리 시험용 LangGraph v13: v12 + 사업자 요건 확인되지 않은 거래기록 의무 근거 보류.

저장 위치: backend/graph/legal_graph_v13.py

- v11의 사용자 진술 문맥·명시 조문 검사·RAG v6·v9 인용 검증/복구 유지.
- v4 Reranker가 선택하기 전/후에 '명백히 다른 특수 법률쟁점' 일부를 검토.
- 법령을 명시하여 질문한 경우 무조건 배제하지 않음.
- 이 가드는 모든 법률의 적용범위를 판정하거나 최신성을 보증하지 않음.
- 운영 main.py, legal_graph.py, SQLite, ChromaDB 및 이전 테스트 버전 변경 없음.
- DB 환경변수는 최초 import 전에 설정.
"""
from __future__ import annotations

from typing import TypeAlias


from backend.graph import legal_graph_v4 as v4
from backend.graph import legal_graph_v10 as v10
from backend.graph import legal_graph_v11 as v11
from backend.graph import legal_reranker as reranker_module
from backend.graph import legal_graph_builder as graph_builder
from backend.graph import legal_scope_guard as scope_guard

AgentState: TypeAlias = v11.AgentState


_compact = scope_guard._compact
_has_any = scope_guard._has_any
_out_of_scope_reason = scope_guard._out_of_scope_reason

reranker_node = reranker_module.reranker_node


workflow, legal_graph = graph_builder.build_legal_graph(
    AgentState,
    reranker_node,
)


def run_legal_chat(question: str, session_id: str) -> dict:
    if not str(question or "").strip() or not str(session_id or "").strip():
        raise ValueError("질문과 session_id가 필요합니다.")
    if len(question) > v11.MAX_CURRENT_CHARS:
        raise ValueError(f"질문은 {v11.MAX_CURRENT_CHARS}자 이하여야 합니다.")
    return legal_graph.invoke(
        {"incoming_question": question.strip()},
        config={"configurable": {"thread_id": session_id}},
    )


def self_test() -> None:
    """Ollama·SQLite·ChromaDB 호출 없이 필터/선정/노드 연결 확인."""
    from types import SimpleNamespace

    case = "중고거래 판매자가 돈을 받고 물건을 보내지 않았어요. 형사상 사기와 민사상 손해배상을 알고 싶어요."
    followup = "판매자가 물건을 보내겠다고 여러 번 약속했는데 아직 보내지 않았어요. 어떤 사실을 추가로 확인하나요?"
    state = {"incoming_question": followup, "question": v11.contextual_question([case], followup), "category": "형사"}

    def source(law, article, title):
        return {"law_name": law, "article": article, "article_title": title,
                "section_scope": "", "content": "실제 근거 판단이 필요한 조문 본문"}

    fraud = source("형법", "제347조", "사기")
    tort = source("민법", "제750조", "불법행위의 내용")
    family = source("민법", "제854조", "사기, 강박으로 인한 승인의 취소")
    marriage = source("민법", "제823조", "사기, 강박으로 인한 혼인취소청구권의 소멸")
    offset = source("민법", "제496조", "불법행위채권을 수동채권으로 하는 상계의 금지")
    wartime = source("형법", "제103조", "전시군수계약불이행")
    commerce = source("전자상거래 등에서의 소비자보호에 관한 법률", "제6조", "거래기록의 보존 등")
    assert _out_of_scope_reason(state, commerce) is not None
    assert _out_of_scope_reason({
        "incoming_question": "판매자는 사업자입니다. 전자상거래에서 구매한 물품을 안 보내요",
        "question": "판매자는 사업자입니다. 전자상거래에서 구매한 물품을 안 보내요",
    }, commerce) is None
    assert _out_of_scope_reason({
        "incoming_question": "사업자 여부는 모릅니다. 중고거래 후 배송이 안 됩니다",
        "question": "사업자 여부는 모릅니다. 중고거래 후 배송이 안 됩니다",
    }, commerce) is not None
    assert _out_of_scope_reason({
        "incoming_question": "개인 판매자이고 사업자가 아닙니다",
        "question": "개인 판매자이고 사업자가 아닙니다",
    }, commerce) is not None
    assert _out_of_scope_reason({
        "incoming_question": "전자상거래 등에서의 소비자보호에 관한 법률 제6조를 설명해 주세요",
        "question": "전자상거래 등에서의 소비자보호에 관한 법률 제6조를 설명해 주세요",
    }, commerce) is None
    assert _out_of_scope_reason({
        "incoming_question": "전자상거래법 제6조의 내용을 알려 주세요",
        "question": "전자상거래법 제6조의 내용을 알려 주세요",
    }, commerce) is None
    # 제6조의2를 물은 경우 제6조 직접 질의로 오인하지 않는다.
    assert _out_of_scope_reason({
        "incoming_question": "전자상거래 등에서의 소비자보호에 관한 법률 제6조의2는?",
        "question": "전자상거래 등에서의 소비자보호에 관한 법률 제6조의2는?",
    }, commerce) is not None
    print("[통과] 전자상거래법 제6조 사업자 여부 미확인·개인 간 거래 보류 / 확인된 사업자·명시 조문 질의 유지")
    kept = [s for s in (fraud, tort, family, marriage, offset, wartime) if not _out_of_scope_reason(state, s)]
    assert kept == [fraud, tort], kept
    print("[통과] 중고거래 후속 질문에서 일반 사기·불법행위 후보 유지, 무관한 특수조문 보류")

    assert _out_of_scope_reason({"incoming_question": "민법 제854조를 설명해 줘"}, family) is None
    assert _out_of_scope_reason({"incoming_question": "혼인 취소는 언제 가능한가요?"}, marriage) is None
    assert _out_of_scope_reason({"incoming_question": "전시군수계약불이행은 어떤 죄인가요?"}, wartime) is None
    assert _out_of_scope_reason({"incoming_question": "상계가 가능한가요?"}, offset) is None
    print("[통과] 명시적 조문 질문·실제 특수 사실관계에서는 관련 후보 보존")

    orig = v4.original.get_fast_llm
    try:
        v4.original.get_fast_llm = lambda: SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="C00,C01")
        )
        result = reranker_node({**state, "candidate_sources": [fraud, family, tort, offset]})
        assert result["reranker_status"] == "ok" and result["sources"] == [fraud, tort], result
        none = reranker_node({**state, "candidate_sources": [family, marriage]})
        assert none == {"sources": [], "reranker_status": "none"}, none
    finally:
        v4.original.get_fast_llm = orig
    print("[통과] 기존 v4 Reranker 연결·조문 선정/무근거 응답 처리")

    expected = {
        "prepare_turn": v11.prepare_turn_node,
        "researcher": v10.researcher_node,
        "reranker": reranker_node,
        "generator": v11.generator_node,
        "finish_turn": v11.finish_turn_node,
    }
    for name, func in expected.items():
        spec = workflow.nodes.get(name)
        assert spec is not None, name
        actual = getattr(getattr(spec, "runnable", None), "func", None)
        assert actual is func, f"노드 연결 불일치: {name}"
    print("[통과] v11 대화 문맥·RAG v6·v9 인용 검증 복구 노드 유지")
    print("단위 테스트 완료. 실제 DB·Ollama는 호출하지 않았습니다.")


if __name__ == "__main__":
    self_test()
