
"""여러 법률 질문 의도를 동시에 분석하는 기능의 회귀 테스트."""

import pytest

from backend.graph import legal_question_intent as intent


@pytest.mark.parametrize(
    ("question", "expected_intents", "expected_primary"),
    [
        (
            "임금체불 증거를 확보하고 신고하는 방법도 알려주세요.",
            ["evidence_collection", "procedure"],
            "evidence_collection",
        ),
        (
            "사기죄의 성립 요건과 고소 절차를 설명해 주세요.",
            ["legal_explanation", "procedure"],
            "legal_explanation",
        ),
        (
            "전세보증금 반환 내용증명 초안을 작성해 주세요.",
            ["document_draft"],
            "document_draft",
        ),
        (
            "교통사고 이후 어떤 증거를 남겨야 하나요?",
            ["evidence_collection"],
            "evidence_collection",
        ),
        (
            "양육비 미지급을 입증하려면 어떤 자료가 필요한가요?",
            ["evidence_collection"],
            "evidence_collection",
        ),
        (
            "형사소송법상 증거능력의 법적 요건을 설명해줘.",
            ["legal_explanation"],
            "legal_explanation",
        ),
        (
            "교통사고가 났는데 어떻게 해야 하나요?",
            ["other"],
            "other",
        ),
    ],
)
def test_analyze_question_intents(
    question,
    expected_intents,
    expected_primary,
):
    result = intent.analyze_question_intents(question)

    assert result["intents"] == expected_intents
    assert result["primary_intent"] == expected_primary


def test_existing_single_intent_classifier_preserved():
    """기존 함수와의 하위 호환성을 유지한다."""

    assert intent.classify_question_intent(
        "임금체불을 입증하려면 어떤 자료를 모아야 하나요?"
    ) == "evidence_collection"
