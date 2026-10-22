
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



def analyze_question_intents(question: str) -> dict:
    """질문에 포함된 여러 요청 의도를 분석한다.

    primary_intent는 가장 먼저 나타난 명시적 요청을 기준으로 한다.
    법령의 적용 여부나 법률적 판단은 수행하지 않는다.
    """

    text = re.sub(r"\s+", " ", str(question or "")).strip()

    if not text:
        return {
            "intents": ["other"],
            "primary_intent": "other",
        }

    matches: list[tuple[int, str]] = []

    def add_match(intent_name: str, pattern: str) -> None:
        found = re.search(pattern, text)
        if found:
            matches.append((found.start(), intent_name))

    # 1. 문서 초안 작성
    document = re.search(
        r"내용증명|고소장|소장|진정서|신청서|탄원서|합의서",
        text,
    )
    writing = re.search(
        r"초안|작성|써줘|만들어|양식|문구",
        text,
    )

    if document and writing:
        matches.append(
            (document.start(), "document_draft")
        )

    # 2. 법령 및 법적 요건 설명
    add_match(
        "legal_explanation",
        (
            r"성립\s*요건|법적\s*요건|증거능력|"
            r"조문|조항|법령?\s*설명|해석|"
            r"적용되|"
            r"(?:형법|민법)\s*제\d+조"
        ),
    )

    # 3. 증거 확보 및 기록 보존
    evidence = re.search(
        r"증거(?!능력)|자료|기록|입증",
        text,
    )
    evidence_action = re.search(
        r"확보|모으|모아|보관|보존|남기|남겨|"
        r"수집|준비|필요|챙기|입증",
        text,
    )

    if evidence and evidence_action:
        matches.append(
            (evidence.start(), "evidence_collection")
        )

    # 4. 신고·고소·소송 절차
    add_match(
        "procedure",
        (
            r"신고(?:하는|할|하|방법|절차)?|"
            r"고소\s*절차|고소하|"
            r"진정(?:을|은|하는|제기)|"
            r"소송\s*(?:절차|시작|제기)|"
            r"제기하|접수|"
            r"신청(?:하는)?\s*(?:방법|절차)|절차"
        ),
    )

    # 5. 사건 분석 및 평가
    add_match(
        "case_analysis",
        r"분석|평가|법적으로 어떻게|민사인지 형사인지|판단",
    )

    if not matches:
        fallback = classify_question_intent(text)
        return {
            "intents": [fallback],
            "primary_intent": fallback,
        }

    # 질문에서 먼저 등장한 요청부터 정렬
    matches.sort(key=lambda item: item[0])

    intents = list(dict.fromkeys(
        intent_name for _, intent_name in matches
    ))

    return {
        "intents": intents,
        "primary_intent": intents[0],
    }
