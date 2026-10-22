
"""증거 확보 질문에 대한 답변 품질 회귀 테스트."""

from backend.graph import legal_graph_v11 as v11


BASE_ANSWER = (
    "## 확인된 법령 원문\n"
    "- **형법 제347조 (사기)**\n"
    "  - 법문 원문: 테스트용 검증 인용문\n\n"
    "## 사건 확인에 필요한 사실\n"
    "1. 거래 당시 어떤 약속을 했나요?\n\n"
    "## 검증 범위와 주의사항\n"
    "법령 원문 인용 여부만 검증했습니다."
)


def fake_verified_generator(state):
    return {
        "answer": BASE_ANSWER,
        "answer_validation_status": "ok",
        "evidence_selection_mode": "llm_verified",
        "sources": [],
    }


def test_evidence_question_provides_checklist(monkeypatch):
    monkeypatch.setattr(
        v11.v9, "generator_node", fake_verified_generator
    )

    question = (
        "중고거래 판매자가 환불을 미루고 있습니다. "
        "어떤 증거를 확보해야 하나요?"
    )

    result = v11.generator_node({
        "question": question,
        "incoming_question": question,
        "user_turn_history": [],
    })

    answer = result["answer"]

    assert "증거 확보" in answer
    assert "판매 게시글" in answer
    assert "대화" in answer
    assert "결제 내역" in answer
    assert "배송" in answer
    assert "환불 요청" in answer

    # 기존 검증된 법령 내용은 보존
    assert BASE_ANSWER in answer
    assert result["answer_validation_status"] == "ok"
    assert result["evidence_selection_mode"] == "llm_verified"


def test_evidence_followup_uses_previous_context(monkeypatch):
    monkeypatch.setattr(
        v11.v9, "generator_node", fake_verified_generator
    )

    current = (
        "방금 말씀드린 사건에서 판매자가 환불을 "
        "미루고 있습니다. 어떤 증거를 확보해야 하나요?"
    )

    result = v11.generator_node({
        "question": current,
        "incoming_question": current,
        "user_turn_history": [
            "중고거래 판매자에게 돈을 보냈지만 물건을 받지 못했습니다."
        ],
    })

    answer = result["answer"]

    assert "판매 게시글" in answer
    assert "결제 내역" in answer
    assert "환불 요청" in answer
    assert BASE_ANSWER in answer


def test_unrelated_question_does_not_get_checklist(monkeypatch):
    monkeypatch.setattr(
        v11.v9, "generator_node", fake_verified_generator
    )

    question = "형법 제347조의 사기죄 성립 요건은 무엇인가요?"

    result = v11.generator_node({
        "question": question,
        "incoming_question": question,
        "user_turn_history": [],
    })

    assert result["answer"] == BASE_ANSWER


def test_blocked_answer_does_not_get_guidance(monkeypatch):
    def fake_blocked_generator(state):
        return {
            "answer": "법령 원문 검증에 실패했습니다.",
            "answer_validation_status": "blocked",
            "evidence_selection_mode": "none",
        }

    monkeypatch.setattr(
        v11.v9, "generator_node", fake_blocked_generator
    )

    question = "중고거래 사기 증거를 확보하려면?"

    result = v11.generator_node({
        "question": question,
        "incoming_question": question,
        "user_turn_history": [],
    })

    assert result["answer_validation_status"] == "blocked"
    assert result["answer"] == "법령 원문 검증에 실패했습니다."
