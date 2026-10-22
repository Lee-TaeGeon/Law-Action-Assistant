
"""LangGraph v13 질문 의도 상태 연결 회귀 테스트."""

from backend.graph import legal_graph_v11 as v11
from backend.graph import legal_graph_v13 as v13


def test_v13_state_supports_question_intents():
    """v13이 사용하는 상태에 의도 분석 필드가 존재한다."""

    assert v13.AgentState is v11.AgentState
    assert "question_intents" in v11.AgentState.__annotations__
    assert "primary_intent" in v11.AgentState.__annotations__


def test_prepare_turn_detects_multiple_intents():
    """한 질문에 증거 확보와 신고 요청이 함께 있으면 모두 보존한다."""

    question = (
        "임금체불 증거를 확보하고 "
        "신고하는 방법도 알려주세요."
    )

    result = v11.prepare_turn_node({
        "incoming_question": question,
        "user_turn_history": [],
    })

    assert result["question_intents"] == [
        "evidence_collection",
        "procedure",
    ]
    assert result["primary_intent"] == "evidence_collection"
    assert result["question"] == question


def test_followup_intent_is_recalculated():
    """이전 질문의 의도가 새로운 질문에 남지 않아야 한다."""

    result = v11.prepare_turn_node({
        "incoming_question": "고소 절차를 알려주세요.",
        "user_turn_history": [
            "중고거래 증거를 확보하고 싶습니다."
        ],
        "question_intents": ["evidence_collection"],
        "primary_intent": "evidence_collection",
    })

    assert result["question_intents"] == ["procedure"]
    assert result["primary_intent"] == "procedure"
    assert result["followup_context_used"] is True


def test_new_case_resets_history_but_detects_intent():
    """새 사건에서는 과거 문맥을 제거하고 현재 의도만 분석한다."""

    result = v11.prepare_turn_node({
        "incoming_question": (
            "새 사건: 임금체불을 입증하려면 "
            "어떤 자료를 모아야 하나요?"
        ),
        "user_turn_history": [
            "중고거래 사기 신고 절차를 알려주세요."
        ],
        "question_intents": ["procedure"],
        "primary_intent": "procedure",
    })

    assert result["user_turn_history"] == []
    assert result["followup_context_used"] is False
    assert result["question_intents"] == ["evidence_collection"]
    assert result["primary_intent"] == "evidence_collection"


def test_unclear_followup_does_not_inherit_old_intent():
    """현재 질문이 불명확하면 이전 의도를 자동 승계하지 않는다."""

    result = v11.prepare_turn_node({
        "incoming_question": "그럼 어떻게 해야 할까요?",
        "user_turn_history": [
            "중고거래 증거를 확보하려면 어떻게 하나요?"
        ],
        "question_intents": ["evidence_collection"],
        "primary_intent": "evidence_collection",
    })

    assert result["question_intents"] == ["other"]
    assert result["primary_intent"] == "other"
