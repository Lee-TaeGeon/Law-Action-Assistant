
import pytest

from backend.graph.legal_graph_v4 import parse_reranker_choice


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("C12,C00", ([12, 0], "ok")),
        ("C12\nC00\nC08\nC11", ([12, 0, 8, 11], "ok")),
        ("C12\nC00,C08", ([12, 0, 8], "ok")),
        ("C12,C45", ([], "invalid")),
        ("C12,ABC", ([], "invalid")),
        ("C12,C12", ([12], "ok")),
        ("없음", ([], "none")),
    ],
)
def test_reranker_candidate_id_parsing(reply, expected):
    candidates = [
        {"article_id": index}
        for index in range(16)
    ]

    result = parse_reranker_choice(
        reply,
        candidates,
    )

    assert result == expected
