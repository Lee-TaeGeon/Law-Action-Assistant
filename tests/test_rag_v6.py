from backend.services import rag_service_v6 as v6


SAMPLE = (
    "중고거래 판매자가 돈을 받고 물건을 보내지 않았어요. "
    "형사상 사기와 민사상 손해배상을 알고 싶어요."
)


def make_source(law, article, title, article_id):
    return {
        "law_name": law,
        "article": article,
        "article_title": title,
        "article_id": article_id,
        "content": "테스트용 법령 본문",
        "source_type": "keyword",
        "retrieval_methods": ["keyword"],
    }


def test_mixed_case_detection():
    assert v6._needs_civil_damages_facet(SAMPLE)

    assert not v6._needs_civil_damages_facet(
        "계약을 위반해서 손해배상을 청구하고 싶어요"
    )

    assert not v6._needs_civil_damages_facet(
        "다른 사람을 속여 돈을 가져가면 어떤 죄인가요?"
    )

    assert not v6._needs_civil_damages_facet(
        "퇴사 후 임금을 받지 못했어요"
    )


def test_existing_v5_search_preserved(monkeypatch):
    expected = [
        make_source(
            "근로기준법", "제36조", "금품 청산", 10
        )
    ]

    calls = []

    def fake_search(
        question, search_query=None, category="", k=8
    ):
        calls.append(
            (question, search_query, category, k)
        )
        return expected

    monkeypatch.setattr(
        v6.v5,
        "hybrid_search_laws",
        fake_search,
    )

    question = "퇴사 후 임금을 받지 못했습니다."

    result = v6.hybrid_search_laws(
        question,
        search_query="임금 체불",
        category="노동",
        k=8,
    )

    assert result == expected
    assert calls == [
        (question, "임금 체불", "노동", 8)
    ]


def test_civil_damages_candidate_boost(monkeypatch):
    fraud = make_source(
        "형법", "제347조", "사기", 1
    )
    tort = make_source(
        "민법", "제750조", "불법행위의 내용", 2
    )
    offset = make_source(
        "민법", "제496조",
        "불법행위채권을 수동채권으로 하는 상계의 금지",
        3,
    )
    family = make_source(
        "민법", "제823조",
        "사기, 강박으로 인한 혼인취소청구권의 소멸",
        4,
    )

    def fake_baseline(*args, **kwargs):
        return [fraud, family, offset, dict(fraud)]

    def fake_facets(facets, **kwargs):
        return {
            "criminal_fraud": [fraud],
            "civil_tort": [offset, tort],
        }

    monkeypatch.setattr(
        v6.v4.previous,
        "hybrid_search_laws",
        fake_baseline,
    )

    monkeypatch.setattr(
        v6.v4,
        "_facet_candidates",
        fake_facets,
    )

    result = v6.hybrid_search_laws(
        SAMPLE, category="형사", k=8
    )

    positions = {
        (item["law_name"], item["article"]): index
        for index, item in enumerate(result, 1)
    }

    assert positions[("형법", "제347조")] <= 3
    assert positions[("민법", "제750조")] <= 3

    assert (
        positions[("민법", "제750조")]
        < positions[("민법", "제496조")]
    )

    assert len({
        item["article_id"] for item in result
    }) == len(result)

    assert any(
        "facet:civil_tort" in item["retrieval_methods"]
        for item in result
    )


def test_explicit_tort_uses_existing_search(monkeypatch):
    expected = [
        make_source(
            "민법", "제750조", "불법행위의 내용", 2
        )
    ]

    monkeypatch.setattr(
        v6.v5,
        "hybrid_search_laws",
        lambda *args, **kwargs: expected,
    )

    question = SAMPLE + " 민사상 불법행위 규정도 확인해주세요."

    assert not v6._needs_civil_damages_facet(question)

    result = v6.hybrid_search_laws(question)

    assert result == expected


def test_invalid_search_inputs(monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError(
            "Search must not be called"
        )

    monkeypatch.setattr(
        v6.v5,
        "hybrid_search_laws",
        unexpected_search,
    )

    monkeypatch.setattr(
        v6.v4.previous,
        "hybrid_search_laws",
        unexpected_search,
    )

    assert v6.hybrid_search_laws("", k=8) == []
    assert v6.hybrid_search_laws(" ", k=8) == []
    assert v6.hybrid_search_laws(SAMPLE, k=0) == []
    assert v6.hybrid_search_laws(SAMPLE, k=-1) == []