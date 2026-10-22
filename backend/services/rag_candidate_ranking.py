"""Hybrid RAG candidate merging and ranking."""
from __future__ import annotations

from . import rag_service_v4 as v4
from . import rag_service_v5 as v5


def rank_candidates(question: str, facets, baseline, extra, k: int) -> list[dict]:
    candidates: dict[tuple, dict] = {}
    for rank, item in enumerate(baseline):
        key = v4.previous._identity(item)
        candidates[key] = {
            "item": dict(item),
            "score": 1.0 / (1.0 + 0.075 * rank),
            "methods": set(item.get("retrieval_methods") or [item.get("source_type", "unknown")]),
        }
    for facet in facets:
        for item in extra.get(facet.reason, []):
            key = v4.previous._identity(item)
            if key not in candidates:
                candidates[key] = {
                    "item": dict(item), "score": 0.16, "methods": set(),
                }
            candidates[key]["methods"].add("facet:" + facet.reason)

    for entry in candidates.values():
        item = entry["item"]
        for facet in facets:
            if facet.reason == "civil_tort":
                entry["score"] += v5._civil_tort_adjust(question, item)
            else:
                entry["score"] += v4._boost(question, item, facet)
        entry["score"] += v5._offtopic_family_penalty(question, item)

    ordered = sorted(candidates.values(), key=lambda entry: -entry["score"])
    result = []
    for entry in ordered[:k]:
        item = dict(entry["item"])
        item["retrieval_methods"] = sorted(entry["methods"])
        result.append(item)
    return result

