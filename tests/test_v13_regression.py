import pytest

from backend.graph import legal_graph_v13 as v13
from backend.graph import legal_graph_v11 as v11
from backend.graph import legal_graph_v10 as v10


def make_source(law, article, title=""):
    return {
        "law_name": law,
        "article": article,
        "article_title": title,
        "section_scope": "",
        "content": "테스트용 법령 본문",
    }


def make_state(question):
    return {
        "incoming_question": question,
        "question": question,
    }


def test_unknown_business_scope():
    source = make_source(
        "전자상거래 등에서의 소비자보호에 관한 법률",
        "제6조",
        "거래기록의 보존 등",
    )

    state = make_state(
        "중고거래 판매자가 사업자인지 모르겠습니다."
    )

    assert v13._out_of_scope_reason(state, source) is not None


def test_confirmed_business_scope():
    source = make_source(
        "전자상거래 등에서의 소비자보호에 관한 법률",
        "제6조",
        "거래기록의 보존 등",
    )

    state = make_state(
        "판매자는 사업자입니다. 전자상거래에서 물품을 구매했습니다."
    )

    assert v13._out_of_scope_reason(state, source) is None


def test_explicit_article_reference():
    source = make_source(
        "전자상거래 등에서의 소비자보호에 관한 법률",
        "제6조",
        "거래기록의 보존 등",
    )

    state = make_state(
        "전자상거래 등에서의 소비자보호에 관한 법률 제6조를 설명해 주세요."
    )

    assert v13._out_of_scope_reason(state, source) is None


def test_irrelevant_family_article():
    source = make_source(
        "민법",
        "제823조",
        "사기, 강박으로 인한 혼인취소청구권의 소멸",
    )

    state = make_state(
        "중고거래 판매자가 사기를 쳤어요. 돈을 돌려받고 싶어요."
    )

    assert v13._out_of_scope_reason(state, source) is not None


def test_relevant_fraud_article():
    source = make_source(
        "형법",
        "제347조",
        "사기",
    )

    state = make_state(
        "중고거래 판매자가 돈을 받고 물건을 보내지 않았어요."
    )

    assert v13._out_of_scope_reason(state, source) is None


def test_v13_node_connections():
    expected = {
        "prepare_turn": v11.prepare_turn_node,
        "reference_guard": v11.reference_guard_node,
        "researcher": v10.researcher_node,
        "reranker": v13.reranker_node,
        "generator": v11.generator_node,
        "finish_turn": v11.finish_turn_node,
    }

    for name, function in expected.items():
        spec = v13.workflow.nodes.get(name)

        assert spec is not None, name

        actual = getattr(
            getattr(spec, "runnable", None),
            "func",
            None,
        )

        assert actual is function, name


def test_invalid_chat_input():
    with pytest.raises(ValueError):
        v13.run_legal_chat("", "test-session")

    with pytest.raises(ValueError):
        v13.run_legal_chat("법률 질문", "")