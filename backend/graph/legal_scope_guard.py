"""Legal source scope rules extracted from v13."""
from __future__ import annotations

import re
from typing import Any, TypeAlias

from backend.graph import legal_graph_v3 as v3
from backend.graph import legal_graph_v11 as v11

AgentState: TypeAlias = v11.AgentState



def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(_compact(word) in text for word in words)


def _out_of_scope_reason(state: AgentState, source: dict) -> str | None:
    """명시적 적용대상이 불일치하는 일부 후보만 제한적으로 거른다.

    법령 해석을 자동으로 검증하는 함수가 아니다. 특히 명시 법령 질문은
    배제하지 않고 후속 단계가 적용 여부를 검토하도록 유지한다.
    """
    current = str(state.get("incoming_question") or state.get("question") or "")
    context = str(state.get("question") or current)
    # 사용자가 특정 법령과 조문을 직접 질문했다면 그 조문을 설명할 기회를 남긴다.
    # v3._explicit_ref()는 법령명 공백을 정규화하지 않아서
    # "전자상거래 등에서의 소비자보호에 관한 법률 제6조" 같은 질문을 놓친다.
    law_ref = _compact(source.get("law_name"))
    article_ref = _compact(source.get("article"))
    if law_ref and article_ref and re.search(
        re.escape(law_ref + article_ref) + r"(?!의\d)", _compact(current)
    ):
        return None
    if v3._explicit_ref(current, source):
        return None

    eligible, reason = v3.scope_eligibility(context, source)
    if not eligible:
        return reason or "특수 적용범위 확인되지 않음"

    law = _compact(source.get("law_name"))
    article = _compact(source.get("article"))
    title = _compact(source.get("article_title"))
    q = _compact(context)

    # 전자상거래법 제6조는 '사업자'의 거래기록 보존 의무를 규정한다.
    # 개인 간 중고거래 또는 판매자 사업자 여부가 불명확한 상담에서 이를
    # '그 판매자의 의무' 근거로 단정하지 않도록 후보를 보류한다.
    # 전체 전자상거래법의 적용을 판단하는 로직이 아니며, 특정 조문 질의는
    # 이 함수 처음의 _explicit_ref 예외로 설명 대상으로 유지된다.
    if law == "전자상거래등에서의소비자보호에관한법률" and article == "제6조":
        if "전자상거래법제6조" in _compact(current):
            return None
        explicit_private = _has_any(q, (
            "개인간거래", "개인판매자", "개인대개인", "사업자가아닌",
            "사업자가아님", "사업자아님", "사업자등록하지않은",
            "사업자등록안한", "개인끼리", "개인사이거래",
        ))
        unknown_business = _has_any(q, (
            "사업자인지", "사업자여부", "사업자인지모르", "사업자인지확인",
            "사업자일지", "사업자일수도", "사업자일수있",
        ))
        explicit_seller_business = _has_any(q, (
            "판매자가사업자입니다", "판매자는사업자입니다", "판매자는사업자이고",
            "판매자가사업자로등록", "판매자는통신판매업자",
            "판매자가통신판매업자", "상대방은사업자입니다",
            "사업자인판매자", "통신판매업자에게", "쇼핑몰사업자",
            "온라인쇼핑몰에서구매", "사업자등록된판매자",
        ))
        if explicit_private or unknown_business or not explicit_seller_business:
            return "제6조의 사업자 관련 적용대상 사실관계가 확인되지 않음"

    # 추상적인 '사기' 또는 '승인'이라는 단어가 법률상 서로 다른 분야의
    # 조문을 동일 사건의 직접 근거로 끌어올리는 문제를 제한적으로 방지.
    if law == "민법":
        if "혼인" in title and not _has_any(q, ("혼인", "결혼", "배우자", "부부")):
            return "혼인 관계의 사실관계가 확인되지 않음"
        if "이혼" in title and not _has_any(q, ("이혼", "혼인", "배우자", "부부")):
            return "이혼 관련 사실관계가 확인되지 않음"
        if "약혼" in title and not _has_any(q, ("약혼", "혼약")):
            return "약혼 관련 사실관계가 확인되지 않음"
        if "상속" in title and not _has_any(q, ("상속", "유산", "피상속", "상속인")):
            return "상속 관련 사실관계가 확인되지 않음"
        if "상계" in title and not _has_any(q, ("상계", "채권채무를서로", "서로공제")):
            return "상계 쟁점이 확인되지 않음"
        # 이번에 재현된 '민법 제854조' 혼입: 본문의 '승인'은 일반적인
        # 중고거래 배송 약속과 같은 의미가 아니다. 특정 신분관계나
        # 이 조문 자체에 대한 질문인 경우에는 배제하지 않는다.
        if article == "제854조" and "승인의취소" in title and not _has_any(
            q, ("승인", "인지", "친생", "양육", "친권", "가족관계", "제854조")
        ):
            return "승인 취소에 관한 별도 법률관계가 질문에 확인되지 않음"

    if law == "형법" and title.startswith("전시") and not _has_any(
        q, ("전시", "전쟁", "전시군수", "전시공수")
    ):
        return "전시 상황이 확인되지 않음"

    return None
