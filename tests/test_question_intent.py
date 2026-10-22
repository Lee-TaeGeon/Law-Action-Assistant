
"""전체 법률 분야에 공통 적용할 질문 의도 분류 테스트."""

import pytest

from backend.graph.legal_question_intent import classify_question_intent


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # 증거 확보 - 다양한 법률 분야
        (
            "중고거래 사기를 당했는데 어떤 증거를 확보해야 하나요?",
            "evidence_collection",
        ),
        (
            "임금체불을 입증하려면 어떤 자료를 모아야 하나요?",
            "evidence_collection",
        ),
        (
            "전세보증금을 돌려받지 못했습니다. 어떤 기록을 보관해야 하나요?",
            "evidence_collection",
        ),
        (
            "교통사고 이후 어떤 증거를 남겨야 하나요?",
            "evidence_collection",
        ),
        (
            "직장 내 괴롭힘을 당했습니다. 어떤 기록을 남겨야 하나요?",
            "evidence_collection",
        ),
        (
            "양육비 미지급을 입증할 자료는 무엇이 필요한가요?",
            "evidence_collection",
        ),

        # 법령 설명
        (
            "형법 제347조의 사기죄 성립 요건을 설명해줘.",
            "legal_explanation",
        ),
        (
            "민법 제750조는 어떤 경우에 적용되나요?",
            "legal_explanation",
        ),
        (
            "형사소송법상 증거능력의 법적 요건을 설명해줘.",
            "legal_explanation",
        ),

        # 신고 및 법적 절차
        (
            "사기 피해를 경찰에 신고하는 절차를 알려줘.",
            "procedure",
        ),
        (
            "임금체불 진정은 어디에 어떻게 제기하나요?",
            "procedure",
        ),
        (
            "보증금 반환소송을 시작하는 절차가 궁금해요.",
            "procedure",
        ),

        # 법률 문서 작성
        (
            "보증금 반환을 요구하는 내용증명 초안을 작성해줘.",
            "document_draft",
        ),
        (
            "사기 피해 고소장 초안을 만들어줘.",
            "document_draft",
        ),

        # 사건 분석
        (
            "이 사건이 민사인지 형사인지 분석해줘.",
            "case_analysis",
        ),
        (
            "판매자가 돈만 받고 물건을 안 보냈습니다. 법적으로 어떻게 평가할 수 있나요?",
            "case_analysis",
        ),
    ],
)
def test_classify_question_intent(question, expected):
    assert classify_question_intent(question) == expected
