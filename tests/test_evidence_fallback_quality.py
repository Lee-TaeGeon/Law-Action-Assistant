
"""법령 원문 자동 복구의 품질 회귀 테스트.

실제 법령 DB, ChromaDB, Ollama를 호출하지 않는다.
가상의 법령 데이터를 이용해 인용문 추출 동작만 검사한다.
"""

from backend.graph import legal_graph_v9 as v9


def test_fallback_prefers_provision_over_article_heading():
    """실질적인 본문이 있는데 조문 제목만 인용해서는 안 된다."""

    heading = "제10조(문서 보관에 관한 일반 원칙)"

    provision = (
        "① 관리자는 확인에 필요한 자료를 "
        "지정된 장소에 보관해야 한다."
    )

    source = {
        "law_name": "가상테스트법",
        "article": "제10조",
        "article_title": "문서 보관에 관한 일반 원칙",
        "content": f"{heading}\n{provision}",
    }

    verified = v9.deterministic_verified_evidence(
        sources=[source],
        max_quotes=1,
    )

    assert len(verified) == 1

    selected = verified[0]["evidence"]

    # 조문 제목만 인용하는 기존 오류를 방지한다.
    assert selected.strip() != heading

    # 실제 규정 내용이 포함된 구절을 선택해야 한다.
    assert "보관해야 한다" in selected
