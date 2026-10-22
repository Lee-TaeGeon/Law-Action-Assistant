
"""법률 분야에 독립적인 질문 의도 분류 모듈.

단일 질문 의도를 규칙 기반으로 분류한다.
법률적 판단이나 관련 법령의 적용 여부는 결정하지 않는다.
"""

from __future__ import annotations

import re


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_question_intent(question: str) -> str:
    """사용자의 법률 질문에서 주된 요청 유형을 분류한다."""

    text = re.sub(r"\s+", " ", str(question or "")).strip()

    if not text:
        return "other"

    # 1. 법률 문서 초안 작성
    document_terms = (
        "내용증명", "고소장", "소장", "진정서",
        "신청서", "탄원서", "합의서",
    )
    writing_terms = (
        "초안", "작성", "써줘", "만들어",
        "양식", "문구",
    )

    if (
        _has_any(text, document_terms)
        and _has_any(text, writing_terms)
    ):
        return "document_draft"

    # 2. 법령 설명 및 법적 요건
    legal_terms = (
        "형법", "민법", "형사소송법", "민사소송법",
        "법률", "법령", "법적", "조문", "조항",
        "성립 요건", "증거능력",
    )
    explanation_terms = (
        "설명", "적용되", "해석", "의미",
        "성립 요건", "법적 요건",
    )

    if (
        _has_any(text, legal_terms)
        and _has_any(text, explanation_terms)
    ):
        return "legal_explanation"

    # 3. 신고, 고소, 소송 등 절차
    procedure_terms = (
        "절차", "신고하는", "신고 방법",
        "고소하려면", "진정", "제기하",
        "접수", "신청하는 방법",
        "소송을 시작",
    )

    if _has_any(text, procedure_terms):
        return "procedure"

    # 4. 증거 수집 및 기록 보존
    evidence_terms = (
        "증거", "자료", "기록", "입증",
    )
    evidence_actions = (
        "확보", "모으", "모아", "보관",
        "보존", "남기", "남겨", "수집",
        "준비", "필요", "챙기", "입증",
    )

    if (
        _has_any(text, evidence_terms)
        and _has_any(text, evidence_actions)
    ):
        return "evidence_collection"

    # 5. 사건의 법률적 평가 및 분석
    analysis_terms = (
        "분석", "평가", "법적으로 어떻게",
        "민사인지 형사인지", "판단",
    )

    if _has_any(text, analysis_terms):
        return "case_analysis"

    # 의도를 확정하기 어려운 질문
    return "other"
