
"""질문 의도에 따른 일반적인 행동 안내.

법령 인용 검증과 별도로 동작하며,
법적 의무나 사건의 법률적 판단을 생성하지 않는다.
"""


def build_evidence_guidance(
    question: str,
    previous_turns: list[str] | None = None,
) -> str:
    """중고거래 증거 확보 질문에 한해 안내문을 반환한다."""

    current = str(question or "").strip()
    history = previous_turns or []
    context = " ".join(
        [current, *(str(turn) for turn in history[-3:])]
    )

    # 증거·자료 확보를 요청하는 질문인지 확인
    evidence_terms = ("증거", "자료")
    action_terms = (
        "확보", "모으", "모아", "보존", "보관",
        "수집", "준비", "필요", "챙기", "남기",
    )

    if not any(term in current for term in evidence_terms):
        return ""

    if not any(term in current for term in action_terms):
        return ""

    # 처음에는 중고거래 관련 사건에만 적용
    marketplace_terms = (
        "중고거래", "중고 거래",
        "판매자", "환불", "배송",
        "물건을 받지",
    )

    if not any(term in context for term in marketplace_terms):
        return ""

    return (
        "## 증거 확보 안내 (일반적인 기록 보존 방법)\n"
        "다음은 중고거래 분쟁에서 사실관계를 정리하기 위한 "
        "일반적인 자료 보존 안내입니다. "
        "법령 원문에 직접 규정된 의무나 범죄 성립 판단은 아닙니다.\n\n"

        "- **판매 게시글**: 상품 설명, 가격, 판매자 표시, "
        "게시 날짜가 확인되도록 보존하세요.\n"

        "- **대화 내역**: 판매자와 주고받은 채팅, 문자, "
        "배송·환불 약속을 시간순으로 저장하세요.\n"

        "- **결제 내역**: 계좌이체 또는 결제 영수증에서 "
        "거래 일시, 금액, 지급 대상을 확인할 수 있는 "
        "자료를 보관하세요.\n"

        "- **배송 관련 기록**: 발송 약속, 운송장 번호, "
        "배송 조회 내역이 있다면 보존하세요.\n"

        "- **환불 요청 기록**: 환불 요청 날짜, 판매자의 답변, "
        "환불 약속 및 이후 진행 상황을 정리하세요.\n\n"

        "가능하면 날짜와 대화 맥락이 보이도록 원본 기록을 보존하세요. "
        "다른 사람에게 자료를 공유할 때는 불필요한 개인정보를 "
        "가리는 것이 좋습니다."
    )
