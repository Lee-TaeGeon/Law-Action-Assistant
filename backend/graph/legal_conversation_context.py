"""Conversation context helpers extracted from v11."""
from __future__ import annotations

import re
from typing import Any



MAX_PREVIOUS_TURNS = 3


MAX_TURN_CHARS = 600


MAX_CURRENT_CHARS = 2000


RESET_PATTERN = re.compile(
    r"^\s*(?:새\s*(?:사건|상담|질문)|다른\s*사건|별도\s*사건|"
    r"이전\s*사건과\s*(?:무관|별개))\b"
)


def _clean_turn(value: Any, max_chars: int = MAX_TURN_CHARS) -> str:
    """과거 진술에 과도한 길이/개행이 들어오지 않게 제한한다."""
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_chars]


def _history(state: dict) -> list[str]:
    raw = state.get("user_turn_history")
    if not isinstance(raw, list):
        return []
    result = [_clean_turn(t) for t in raw[-MAX_PREVIOUS_TURNS:]]
    return [t for t in result if t]


def _is_new_case(text: str) -> bool:
    return bool(RESET_PATTERN.match(text))


def contextual_question(previous: list[str], current: str) -> str:
    """이전 진술을 사용자 진술로만 표시한다. AI 답변을 사건 사실로 합치지 않는다."""
    if not previous:
        return current
    parts = [
        "[이전 대화의 사용자 진술 - 사실 확인 전이며 이번 질문의 문맥 참고용]",
    ]
    parts.extend(f"{idx}. {item}" for idx, item in enumerate(previous, 1))
    parts.extend(["", "[이번 사용자 질문 또는 추가 진술]", current])
    return "\n".join(parts)
