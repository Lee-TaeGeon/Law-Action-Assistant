"""Shared v13 API validation and source filtering."""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend.schemas.chat import LawSource


def _validate_test_data_paths() -> tuple[Path, Path, dict]:
    """Fail closed rather than silently falling back to old production databases."""
    sqlite_value = os.environ.get("LAW_SEARCH_DB_PATH", "").strip()
    vector_value = os.environ.get("LAW_VECTOR_DB_PATH", "").strip()
    if not sqlite_value or not vector_value:
        raise RuntimeError(
            "테스트 서버는 LAW_SEARCH_DB_PATH 및 LAW_VECTOR_DB_PATH를 반드시 지정해야 합니다."
        )

    sqlite_path = Path(sqlite_value).expanduser().resolve()
    vector_path = Path(vector_value).expanduser().resolve()
    if not sqlite_path.is_file() or not vector_path.is_dir():
        raise RuntimeError("테스트 SQLite 파일 또는 ChromaDB 폴더가 존재하지 않습니다.")
    if sqlite_path.name.casefold() == "law_search.db" or vector_path.name.casefold() in {
        "law_db_optimized", "law_db_full"
    }:
        raise RuntimeError("기존 운영 DB 경로는 테스트 서버에 연결할 수 없습니다.")

    marker = vector_path / "_BUILD_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError("ChromaDB 완료 마커가 없어 미완성 DB 연결을 차단했습니다.")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("ChromaDB 완료 마커를 읽을 수 없습니다.") from exc
    source_value = manifest.get("source_sqlite")
    if not isinstance(source_value, str) or not source_value.strip():
        raise RuntimeError("ChromaDB 완료 기록에 원본 SQLite 경로가 없습니다.")
    if (
        manifest.get("status") != "complete"
        or manifest.get("partial_test") is not False
        or manifest.get("collection") != "law_articles"
        or manifest.get("embedding_model") != "jhgan/ko-sroberta-multitask"
        or Path(source_value).resolve() != sqlite_path
        or not isinstance(manifest.get("source_article_count"), int)
        or not isinstance(manifest.get("vector_count"), int)
        or manifest["source_article_count"] != manifest["vector_count"]
        or manifest["vector_count"] < 1
    ):
        raise RuntimeError("SQLite와 ChromaDB 완료 기록이 일치하지 않습니다.")
    return sqlite_path, vector_path, manifest



def _quoted_sources(result: dict) -> list[LawSource]:
    """Only show sources whose headings actually appear in the verified response.

    Exact matches are conservative: if a heading cannot be matched, omit its card
    rather than imply that an unquoted source was verified in the answer.
    """
    if result.get("answer_validation_status") != "ok":
        return []
    answer_lines = {line.strip() for line in str(result.get("answer") or "").splitlines()}
    output: list[LawSource] = []
    seen: set[tuple[str, str]] = set()
    for source in result.get("sources") or []:
        if not isinstance(source, dict):
            continue
        law_name = str(source.get("law_name") or "").strip()
        article = str(source.get("article") or "").strip()
        title = str(source.get("article_title") or "").strip()
        content = str(source.get("content") or "")
        if not law_name or not content:
            continue
        heading = f"{law_name} {article}" if article else f"{law_name} (조문번호 없는 법령 본문)"
        if title:
            heading += f" ({title})"
        if f"- **{heading}**" not in answer_lines:
            continue
        key = (law_name, article)
        if key in seen:
            continue
        seen.add(key)
        output.append(LawSource(law_name=law_name, article=article or None, content=content))
    return output

