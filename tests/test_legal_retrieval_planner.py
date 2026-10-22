
import json
from types import SimpleNamespace

import pytest

from backend.graph.legal_retrieval_planner import (
    RetrievalPlanningError,
    generate_retrieval_plan,
)
from backend.schemas.legal_case_analysis import LegalCaseAnalysis


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        content = (
            self.response
            if isinstance(self.response, str)
            else json.dumps(self.response, ensure_ascii=False)
        )
        return SimpleNamespace(content=content)


def sample_analysis():
    return LegalCaseAnalysis.model_validate({
        "user_stated_facts": ["거래 대금을 송금했습니다."],
        "user_requests": [
            "형사상 책임을 확인하고 싶습니다.",
            "돈을 돌려받는 방법을 알고 싶습니다.",
        ],
        "possible_legal_issues": [],
        "missing_facts": [],
    })


def search_item(index):
    return {
        "request_index": index,
        "searches": [
            {
                "source_type": "law",
                "query": "관련 법적 근거",
            }
        ],
        "clarification_questions": [],
    }


def test_valid_plan():
    analysis = sample_analysis()
    llm = FakeLLM({
        "items": [
            search_item(1),
            search_item(2),
        ]
    })

    plan = generate_retrieval_plan(analysis, llm)

    assert len(plan.items) == 2
    assert [x.request_index for x in plan.items] == [1, 2]


@pytest.mark.parametrize(
    "items",
    [
        [search_item(1)],
        [search_item(1), search_item(1)],
        [search_item(1), search_item(3)],
        [
            {
                "request_index": 1,
                "searches": [],
                "clarification_questions": [],
            },
            search_item(2),
        ],
        [
            {
                "request_index": 1,
                "searches": [
                    {
                        "source_type": "unknown",
                        "query": "검색 질문",
                    }
                ],
                "clarification_questions": [],
            },
            search_item(2),
        ],
    ],
)
def test_invalid_plans_are_rejected(items):
    analysis = sample_analysis()
    llm = FakeLLM({"items": items})

    with pytest.raises(RetrievalPlanningError):
        generate_retrieval_plan(analysis, llm)


def test_invalid_json_is_rejected():
    analysis = sample_analysis()
    llm = FakeLLM("{invalid json")

    with pytest.raises(RetrievalPlanningError):
        generate_retrieval_plan(analysis, llm)


def test_non_string_response_is_rejected():
    analysis = sample_analysis()

    class InvalidLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content=["not", "a", "string"])

    with pytest.raises(RetrievalPlanningError):
        generate_retrieval_plan(analysis, InvalidLLM())
