"""Hybrid RAG search intent and facet signals."""
from __future__ import annotations

from . import rag_service_v4 as v4


def _needs_civil_damages_facet(question: str) -> bool:
    """형사 재산피해 + 명시적 민사 손해배상 요청인지 제한적으로 확인한다.

    채무불이행과 불법행위는 별도 법률상 요건이 필요하다. 여기서는 조문을
    답으로 확정하는 것이 아니라, 관련 법령의 검색 후보를 확보하는 것뿐이다.
    """
    flags = v4.previous._signals(question)
    q = v4.previous.base.compact_text(question)
    return bool(
        flags["mixed"]
        and flags["property_fraud"]
        and "민사" in q
        and ("손해배상" in q or "불법행위" in q)
        and "불법행위" not in q
    )

