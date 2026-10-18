"""법령 수집기 v2/v3 JSON을 검색용 SQLite 데이터베이스로 안전하게 변환.

목적:
  - 국가법령정보센터에서 수집한 *새 구조*의 JSON만 사용한다.
  - 조문번호/가지번호를 공식값 그대로 보존한다. 번호를 추측해 생성하지 않는다.
  - 장·절 등 비조문 노드를 별도 테이블로 저장한다.
  - 이전 law_search.db를 읽거나 수정하지 않고, 지정한 새 경로로만 DB를 생성한다.
  - 기존 rag_service.py가 기대하는 laws/articles/articles_fts 테이블을 유지한다.

테스트 실행 (PowerShell, 프로젝트 루트):
  python scripts/build_law_search_db_v2.py `
    --source "C:\\dev\\Law-Action-Assistant-LawDataV2\\structured_json" `
    --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\search_db\\law_search_v2_test.db"

주의:
  이 파일은 *수집된 법령의 현재 스냅샷 1개씩* DB로 만든다.
  자동 동기화/시행일별 버전 검색/ChromaDB 갱신 기능은 아직 구현하지 않는다.
  운영 중인 기존 law_search.db에 이 DB를 곧바로 덮어씌우지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def require_mapping(obj: Any, label: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{label}: JSON 객체(dict)가 아닙니다.")
    return obj


def normalize_official_number(raw_no: Any, raw_branch: Any, label: str) -> tuple[str, str, str]:
    number = clean(raw_no)
    branch = clean(raw_branch)
    if not re.fullmatch(r"[0-9]+", number):
        raise ValueError(f"{label}: 비정상 조문번호: {number!r}")
    if branch and not re.fullmatch(r"[0-9]+", branch):
        raise ValueError(f"{label}: 비정상 조문가지번호: {branch!r}")
    number = str(int(number))
    branch = str(int(branch)) if branch and int(branch) > 0 else ""
    full = number + (f"의{branch}" if branch else "")
    display = f"제{number}조" + (f"의{branch}" if branch else "")
    return full, branch, display


def paragraph_lines(paragraphs: Any, label: str) -> list[str]:
    if not isinstance(paragraphs, list):
        raise ValueError(f"{label}: paragraphs는 리스트여야 합니다.")
    lines: list[str] = []
    for p_idx, para in enumerate(paragraphs, 1):
        para = require_mapping(para, f"{label} 항 {p_idx}")
        text = clean(para.get("para_content"))
        if text:
            lines.append(text)
        items = para.get("items", [])
        if not isinstance(items, list):
            raise ValueError(f"{label}: items가 리스트가 아닙니다.")
        for item in items:
            item = require_mapping(item, f"{label} 호")
            text = clean(item.get("item_content"))
            if text:
                lines.append(text)
            sub_items = item.get("sub_items", [])
            if not isinstance(sub_items, list):
                raise ValueError(f"{label}: sub_items가 리스트가 아닙니다.")
            for sub in sub_items:
                sub = require_mapping(sub, f"{label} 목")
                text = clean(sub.get("sub_content"))
                if text:
                    lines.append(text)
    return lines


def get_law_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError("--source 파일은 .json이어야 합니다.")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"법령 JSON 파일/폴더를 찾지 못했습니다: {path}")
    # 수집기 v3의 법령별 파일만 사용. 다른 생성물은 자동으로 포함하지 않는다.
    files = sorted(path.glob("law_*.json"))
    if not files:
        raise ValueError(f"law_*.json 파일이 없습니다: {path}")
    return files


def load_law_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            law = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: JSON 문법 오류: {exc}") from exc
    law = require_mapping(law, path.name)
    required = ("law_id", "law_name", "collected_at_utc", "api_target", "effective_date", "data")
    missing = [k for k in required if k not in law]
    if missing:
        raise ValueError(f"{path.name}: 필수 메타데이터 누락 {missing}. 옛날 JSON은 사용하지 마세요.")
    if not clean(law.get("law_id")) or not clean(law.get("law_name")):
        raise ValueError(f"{path.name}: 법령 ID 또는 이름이 비어 있습니다.")
    if not clean(law.get("collected_at_utc")):
        raise ValueError(f"{path.name}: 수집 날짜가 비어 있습니다.")
    if not clean(law.get("api_target")):
        raise ValueError(f"{path.name}: API target이 비어 있습니다.")
    if not clean(law.get("effective_date")):
        raise ValueError(f"{path.name}: 시행일이 비어 있습니다. 별도 검증이 필요합니다.")
    if not isinstance(law["data"], list) or not law["data"]:
        raise ValueError(f"{path.name}: 실제 조문 리스트가 없거나 비어 있습니다.")
    if "non_article_nodes" in law and not isinstance(law["non_article_nodes"], list):
        raise ValueError(f"{path.name}: non_article_nodes가 리스트가 아닙니다.")
    return law


def schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    PRAGMA foreign_keys = ON;
    CREATE TABLE laws (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        law_name TEXT NOT NULL UNIQUE,
        source_law_id TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL DEFAULT '',
        api_target TEXT NOT NULL,
        collected_at_utc TEXT NOT NULL,
        promulgation_date TEXT NOT NULL DEFAULT '',
        promulgation_number TEXT NOT NULL DEFAULT '',
        effective_date TEXT NOT NULL,
        revision_type TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        law_id INTEGER NOT NULL REFERENCES laws(id),
        law_name TEXT NOT NULL,
        source_order INTEGER NOT NULL,
        source_article_no TEXT NOT NULL,
        article_no TEXT NOT NULL,
        article_display TEXT NOT NULL,
        article_title TEXT NOT NULL DEFAULT '',
        body TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        normalization_method TEXT NOT NULL DEFAULT 'official',
        article_branch_no TEXT NOT NULL DEFAULT '',
        article_body_raw TEXT NOT NULL DEFAULT '',
        article_effective_date TEXT NOT NULL DEFAULT '',
        article_status TEXT NOT NULL DEFAULT '조문',
        article_revision_type TEXT NOT NULL DEFAULT '',
        article_changed TEXT NOT NULL DEFAULT '',
        paragraphs_json TEXT NOT NULL DEFAULT '[]',
        UNIQUE(law_id, article_no)
    );
    CREATE TABLE law_sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        law_id INTEGER NOT NULL REFERENCES laws(id),
        source_order INTEGER NOT NULL,
        node_type TEXT NOT NULL,
        node_number TEXT NOT NULL DEFAULT '',
        node_branch_number TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL DEFAULT '',
        effective_date TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX idx_articles_law_name ON articles(law_name);
    CREATE INDEX idx_articles_law_article ON articles(law_name, article_no);
    CREATE INDEX idx_articles_source_article ON articles(law_name, source_article_no);
    CREATE INDEX idx_articles_title ON articles(article_title);
    CREATE INDEX idx_articles_hash ON articles(content_hash);
    CREATE INDEX idx_sections_law_order ON law_sections(law_id, source_order);
    CREATE VIRTUAL TABLE articles_fts USING fts5(
        law_name, article_no, article_title, content,
        content='articles', content_rowid='id', tokenize='unicode61'
    );
    """)


def insert_law(conn: sqlite3.Connection, law: dict, filename: str) -> tuple[int, int]:
    name = clean(law["law_name"])
    source_id = clean(law["law_id"])
    cursor = conn.execute("""
        INSERT INTO laws
        (law_name, source_law_id, source, api_target, collected_at_utc,
         promulgation_date, promulgation_number, effective_date, revision_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
        name, source_id, clean(law.get("source")), clean(law["api_target"]),
        clean(law["collected_at_utc"]), clean(law.get("promulgation_date")),
        clean(law.get("promulgation_number")), clean(law["effective_date"]),
        clean(law.get("revision_type")),
    ))
    law_id = cursor.lastrowid
    assert law_id is not None

    for index, node in enumerate(law.get("non_article_nodes", []), 1):
        node = require_mapping(node, f"{filename} 비조문 노드 {index}")
        conn.execute("""
            INSERT INTO law_sections
            (law_id, source_order, node_type, node_number, node_branch_number, text, effective_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
            law_id, int(node.get("source_order") or index), clean(node.get("node_type")),
            clean(node.get("node_number")), clean(node.get("node_branch_number")),
            clean(node.get("text")), clean(node.get("effective_date")),
        ))

    count = 0
    seen: set[str] = set()
    for index, article in enumerate(law["data"], 1):
        article = require_mapping(article, f"{filename} 조문 {index}")
        label = f"{filename} 조문 {index}"
        for field in ("article_no", "article_branch_no", "article_display", "article_body", "paragraphs"):
            if field not in article:
                raise ValueError(f"{label}: {field} 누락. 새 수집기의 JSON이 아닙니다.")
        status = clean(article.get("article_status", "조문"))
        if status != "조문":
            raise ValueError(f"{label}: 조문이 아닌 노드가 data에 포함됐습니다: {status}")
        no, branch, display = normalize_official_number(
            article["article_no"], article["article_branch_no"], label
        )
        if clean(article["article_display"]) != display:
            raise ValueError(f"{label}: 공식 조문 표시 불일치: {article['article_display']!r} != {display!r}")
        if no in seen:
            raise ValueError(f"{filename}: 실제 조문 중복 {display}. 자동으로 새 번호를 만들지 않습니다.")
        seen.add(no)
        raw_body = clean(article["article_body"])
        sections = paragraph_lines(article["paragraphs"], label)
        body = "\n".join(p for p in [raw_body, *sections] if p)
        if not body:
            raise ValueError(f"{label}: 제목을 제외한 조문 본문이 비었습니다. XML을 확인하세요.")
        title = clean(article.get("article_title"))
        heading = display + (f" ({title})" if title else "")
        content = f"<{name}>\n{heading}\n{body}"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        conn.execute("""
            INSERT INTO articles
            (law_id, law_name, source_order, source_article_no, article_no,
             article_display, article_title, body, content, content_hash,
             normalization_method, article_branch_no, article_body_raw,
             article_effective_date, article_status, article_revision_type,
             article_changed, paragraphs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            law_id, name, int(article.get("source_order") or index), clean(article["article_no"]),
            no, display, title, body, content, digest, "official",
            branch, raw_body, clean(article.get("article_effective_date")),
            status, clean(article.get("article_revision_type")),
            clean(article.get("article_changed")),
            json.dumps(article["paragraphs"], ensure_ascii=False, separators=(",", ":")),
        ))
        count += 1
    return count, len(law.get("non_article_nodes", []))


def build(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    inputs = get_law_files(source)
    if output.name.lower() == "law_search.db":
        raise ValueError("안전을 위해 기존 서비스 DB 이름 law_search.db로는 생성하지 않습니다. 테스트용 새 이름을 사용하세요.")
    if output.exists():
        raise FileExistsError(f"출력 파일이 이미 존재해 중단합니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # 같은 디렉터리에 임시 DB 생성. 완료 및 검증 전에는 결과 파일을 만들지 않는다.
    fd, temp_name = tempfile.mkstemp(prefix=".law_search_v2_", suffix=".building", dir=output.parent)
    os.close(fd)
    temp_db = Path(temp_name)
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(temp_db)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        schema(conn)
        law_count = article_count = section_count = 0
        for path in inputs:
            law = load_law_json(path)
            try:
                with conn:
                    articles, sections = insert_law(conn, law, path.name)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"{path.name}: ID/이름/조문 식별자 중복 또는 FK 오류: {exc}") from exc
            law_count += 1
            article_count += articles
            section_count += sections
            print(f"[{law_count}/{len(inputs)}] {law['law_name']} : 조문 {articles}건, 장·절 노드 {sections}건")

        with conn:
            conn.execute("INSERT INTO articles_fts(articles_fts) VALUES ('rebuild')")
            conn.execute("INSERT INTO articles_fts(articles_fts) VALUES ('integrity-check')")
            conn.execute("ANALYZE")
        check = conn.execute("PRAGMA integrity_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError(f"SQLite integrity_check 실패: {check}")
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise RuntimeError("SQLite foreign_key_check 실패")
        if conn.execute("SELECT count(*) FROM articles").fetchone()[0] != article_count:
            raise RuntimeError("조문 건수 대조 실패")
        if conn.execute("SELECT count(*) FROM articles_fts").fetchone()[0] != article_count:
            raise RuntimeError("FTS 건수 대조 실패")
        conn.close()
        conn = None
        # 하드링크는 목적지가 이미 존재하면 실패: 기존 DB 덮어쓰기 방지.
        # 임시 파일과 결과 파일은 같은 폴더여서 동일 파일시스템에 위치한다.
        os.link(temp_db, output)
        temp_db.unlink()
        print("\nDB 생성 및 검증 완료")
        print(f"법령 {law_count:,}건 / 실제 조문 {article_count:,}건 / 장·절 노드 {section_count:,}건")
        print(f"새 DB: {output}")
        print("기존 law_search.db와 ChromaDB는 변경하지 않았습니다.")
        print("주의: 신규 DB의 조문 ID는 이전 ChromaDB ID와 연결되지 않습니다.")
    finally:
        if conn is not None:
            conn.close()
        if temp_db.exists():
            temp_db.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="안전한 법령 검색 SQLite DB v2 생성 (기존 DB 변경 금지)")
    parser.add_argument("--source", type=Path, required=True, help="law_v3.py의 structured_json 폴더 또는 개별 law_ID.json")
    parser.add_argument("--output", type=Path, required=True, help="기존 DB와 다른 테스트 DB 파일 경로")
    args = parser.parse_args()
    try:
        build(args.source, args.output)
    except (OSError, ValueError, sqlite3.DatabaseError, RuntimeError) as exc:
        parser.exit(1, f"오류: {exc}\n기존 서비스 DB와 원본 JSON은 수정하지 않았습니다.\n")


if __name__ == "__main__":
    main()
