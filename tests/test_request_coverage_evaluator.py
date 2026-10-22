
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evaluation.evaluate_request_coverage import (
    evaluate_request_coverage,
)


class FakeLLM:
    def __init__(self, judgments):
        self.judgments = judgments

    def invoke(self, prompt):
        return SimpleNamespace(
            content=json.dumps(
                {"judgments": self.judgments},
                ensure_ascii=False,
            )
        )


@pytest.mark.parametrize(
    "status",
    ["covered", "partial", "missing"],
)
def test_coverage_status_is_preserved(status):
    llm = FakeLLM([
        {
            "expected_index": 1,
            "status": status,
            "reason": "테스트 판정",
        }
    ])

    report = evaluate_request_coverage(
        ["사용자의 원래 요청"],
        ["분석기가 추출한 요청"],
        llm,
    )

    assert report.judgments[0].status == status
    assert report.judgments[0].expected_index == 1


def test_multiple_requests():
    llm = FakeLLM([
        {
            "expected_index": 1,
            "status": "covered",
            "reason": "첫 번째 요청 반영",
        },
        {
            "expected_index": 2,
            "status": "missing",
            "reason": "두 번째 요청 누락",
        },
    ])

    report = evaluate_request_coverage(
        ["첫 번째 요청", "두 번째 요청"],
        ["첫 번째 요청"],
        llm,
    )

    assert len(report.judgments) == 2
    assert report.judgments[0].status == "covered"
    assert report.judgments[1].status == "missing"


@pytest.mark.parametrize(
    "indices",
    [
        [1, 1],
        [1],
        [2, 3],
    ],
)
def test_invalid_request_indices_are_rejected(indices):
    llm = FakeLLM([
        {
            "expected_index": index,
            "status": "covered",
            "reason": "테스트",
        }
        for index in indices
    ])

    with pytest.raises(ValueError):
        evaluate_request_coverage(
            ["첫 번째 요청", "두 번째 요청"],
            ["첫 번째 요청"],
            llm,
        )


def test_invalid_status_is_rejected():
    llm = FakeLLM([
        {
            "expected_index": 1,
            "status": "unknown",
            "reason": "잘못된 상태",
        }
    ])

    with pytest.raises(ValidationError):
        evaluate_request_coverage(
            ["원래 요청"],
            ["추출 요청"],
            llm,
        )
