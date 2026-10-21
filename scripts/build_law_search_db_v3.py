"""법령 수집기 v4/v5 JSON -> 검색 SQLite v3 (운영 DB를 건드리지 않음).

번호가 없는 단문 법령은 공식 조문번호를 만들어내지 않고 articles에
'검색 문서' 한 행으로 수록한다. 별도의 unnumbered_provisions에도 원문을 보존한다.

예시 (프로젝트 루트 PowerShell):
  python scripts/build_law_search_db_v3.py --source "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\structured_json" --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\search_db\\law_search_v3_smoke.db" --law-id 014378 --law-id 010719
  python scripts/build_law_search_db_v3.py --source "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\structured_json" --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\search_db\\law_search_v3_full.db"

출력 파일이 이미 있으면 실패한다. 원본 데이터/기존 DB는 읽기 전용으로만 접근한다.
API target=law 스냅샷용으로, 시행 시점별 법령 선택이나 자동 갱신은 구현하지 않는다.
ChromaDB는 이 파일로 생성되지 않는다. 새 SQLite의 article ID로 새로 생성해야 한다.
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
from typing import Any


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def obj(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: JSON 객체가 아닙니다")
    return value


def official_number(raw_no: Any, raw_branch: Any, label: str) -> tuple[str, str, str]:
    number, branch = clean(raw_no), clean(raw_branch)
    if not re.fullmatch(r"[0-9]+", number) or (branch and not re.fullmatch(r"[0-9]+", branch)):
        raise ValueError(f"{label}: 공식 조문/가지번호가 비정상입니다: {number!r}, {branch!r}")
    number = str(int(number))
    branch = str(int(branch)) if branch and int(branch) else ""
    no = number + (f"의{branch}" if branch else "")
    return no, branch, "제" + number + "조" + (f"의{branch}" if branch else "")


def paragraph_lines(paragraphs: Any, label: str) -> list[str]:
    if not isinstance(paragraphs, list):
        raise ValueError(f"{label}: paragraphs가 리스트가 아닙니다")
    lines = []
    for p in paragraphs:
        p = obj(p, label + " 항")
        if clean(p.get("para_content")):
            lines.append(clean(p["para_content"]))
        items = p.get("items", [])
        if not isinstance(items, list):
            raise ValueError(f"{label}: items가 리스트가 아닙니다")
        for item in items:
            item = obj(item, label + " 호")
            if clean(item.get("item_content")):
                lines.append(clean(item["item_content"]))
            subs = item.get("sub_items", [])
            if not isinstance(subs, list):
                raise ValueError(f"{label}: sub_items가 리스트가 아닙니다")
            for sub in subs:
                sub = obj(sub, label + " 목")
                if clean(sub.get("sub_content")):
                    lines.append(clean(sub["sub_content"]))
    return lines


def collect_inputs(source: Path, law_ids: list[str]) -> tuple[list[Path], bool]:
    if source.is_file():
        if source.suffix.casefold() != ".json" or law_ids:
            raise ValueError("개별 JSON 파일에는 --law-id를 지정하지 마세요")
        return [source], False
    if not source.is_dir():
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {source}")
    if law_ids:
        ids = list(dict.fromkeys(law_ids))
        if any(not re.fullmatch(r"[0-9]+", i) for i in ids):
            raise ValueError("--law-id는 숫자여야 합니다")
        paths = [source / f"law_{i}.json" for i in ids]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"시험용 JSON이 없습니다: {missing[:3]}")
        return paths, False
    paths = sorted(source.glob("law_*.json"))
    if not paths:
        raise ValueError("입력 디렉터리에 law_*.json이 없습니다")
    # 전체 모드에서 목록 스냅샷이 있으면 파일 ID 집합을 대조한다.
    catalog_path = source.parent / "catalog" / "law_ids.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
        ids = catalog.get("law_ids")
        if (catalog.get("api_target") != "law" or not isinstance(ids, list)
                or len(ids) != catalog.get("total_count") or len(set(ids)) != len(ids)):
            raise ValueError("전체 법령 목록 스냅샷이 비정상입니다")
        expected = {f"law_{law_id}.json" for law_id in ids}
        actual = {p.name for p in paths}
        if actual != expected:
            raise ValueError(f"법령 목록/JSON 불일치: 누락 {len(expected-actual)}, 목록 외 {len(actual-expected)}")
        print(f"[확인] 전체 법령 목록과 JSON 파일 {len(paths):,}개 일치")
    else:
        print("[주의] 목록 스냅샷이 없어 전체 법령 완전성 대조를 수행하지 않았습니다")
    return paths, True


def load_law(path: Path, verify_xml: bool) -> dict:
    law = obj(json.loads(path.read_text(encoding="utf-8-sig")), path.name)
    required = ["law_id", "law_name", "collected_at_utc", "api_target", "effective_date", "data"]
    missing = [field for field in required if field not in law]
    if missing:
        raise ValueError(f"{path.name}: 필수 필드 누락: {missing}")
    if not clean(law["law_id"]).isdigit() or path.name != f"law_{law['law_id']}.json":
        raise ValueError(f"{path.name}: 파일명과 법령 ID가 일치하지 않습니다")
    if not all(clean(law[key]) for key in ("law_name", "collected_at_utc", "api_target", "effective_date")):
        raise ValueError(f"{path.name}: 법령 메타데이터에 빈 값이 있습니다")
    if law["api_target"] != "law":
        raise ValueError(f"{path.name}: 법령 API target=law가 아닙니다")
    if not isinstance(law["data"], list):
        raise ValueError(f"{path.name}: data가 리스트가 아닙니다")
    if not isinstance(law.get("non_article_nodes", []), list):
        raise ValueError(f"{path.name}: non_article_nodes가 리스트가 아닙니다")
    if verify_xml:
        sha = clean(law.get("source_xml_sha256"))
        xml_path = path.parent.parent / "raw_xml" / (path.stem + ".xml")
        if not re.fullmatch(r"[a-f0-9]{64}", sha) or not xml_path.is_file():
            raise ValueError(f"{path.name}: XML 또는 SHA-256 메타데이터 누락")
        with xml_path.open("rb") as f:
            raw_hash = hashlib.file_digest(f, "sha256").hexdigest()
        if raw_hash != sha:
            raise ValueError(f"{path.name}: XML/JSON SHA-256 불일치")
    structure = law.get("document_structure", "numbered_articles" if law["data"] else "")
    if structure not in {"numbered_articles", "unnumbered_body"}:
        raise ValueError(f"{path.name}: 지원하지 않는 법령 구조 {structure!r}")
    if structure == "numbered_articles" and not law["data"]:
        raise ValueError(f"{path.name}: 실제 조문이 비어 있습니다")
    if structure == "unnumbered_body":
        if law["data"]:
            raise ValueError(f"{path.name}: 번호 없는 법령에 번호 있는 조문이 존재합니다")
        nodes = law.get("non_article_nodes", [])
        provisions = law.get("unnumbered_provisions")
        if (not isinstance(provisions, list) or len(provisions) != 1
                or len(nodes) != 1 or nodes[0] != provisions[0]):
            raise ValueError(f"{path.name}: 번호 없는 법령의 원문 구조가 비정상입니다")
        item = obj(provisions[0], path.name + " 본문")
        if (clean(item.get("node_type")) != "전문"
                or clean(item.get("node_number")) not in ("", "0")
                or clean(item.get("node_branch_number")) not in ("", "0")
                or len(clean(item.get("text"))) < 8):
            raise ValueError(f"{path.name}: 공식 조문번호 없는 본문으로 인정할 수 없습니다")
    return law


def schema(db: sqlite3.Connection) -> None:
    db.executescript("""
    PRAGMA foreign_keys=ON;
    CREATE TABLE laws (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        law_name TEXT NOT NULL,
        source_law_id TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL DEFAULT '',
        api_target TEXT NOT NULL,
        collected_at_utc TEXT NOT NULL,
        promulgation_date TEXT NOT NULL DEFAULT '',
        promulgation_number TEXT NOT NULL DEFAULT '',
        effective_date TEXT NOT NULL,
        revision_type TEXT NOT NULL DEFAULT '',
        document_structure TEXT NOT NULL DEFAULT 'numbered_articles'
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
    CREATE TABLE unnumbered_provisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        law_id INTEGER NOT NULL REFERENCES laws(id),
        source_order INTEGER NOT NULL,
        node_type TEXT NOT NULL,
        text TEXT NOT NULL,
        effective_date TEXT NOT NULL DEFAULT '',
        article_row_id INTEGER NOT NULL UNIQUE REFERENCES articles(id),
        UNIQUE(law_id, source_order)
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


def insert_article(db: sqlite3.Connection, values: tuple) -> int:
    cur = db.execute("""
        INSERT INTO articles (
            law_id, law_name, source_order, source_article_no, article_no,
            article_display, article_title, body, content, content_hash,
            normalization_method, article_branch_no, article_body_raw,
            article_effective_date, article_status, article_revision_type,
            article_changed, paragraphs_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, values)
    return int(cur.lastrowid)


def insert_law(db: sqlite3.Connection, law: dict, filename: str) -> tuple[int, int, int]:
    name = clean(law["law_name"])
    structure = law.get("document_structure", "numbered_articles")
    cursor = db.execute("""
        INSERT INTO laws
          (law_name,source_law_id,source,api_target,collected_at_utc,
           promulgation_date,promulgation_number,effective_date,revision_type,document_structure)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        name, clean(law["law_id"]), clean(law.get("source")), law["api_target"],
        law["collected_at_utc"], clean(law.get("promulgation_date")),
        clean(law.get("promulgation_number")), law["effective_date"],
        clean(law.get("revision_type")), structure,
    ))
    law_id = int(cursor.lastrowid)
    section_count = 0
    if structure == "numbered_articles":
        for idx, node in enumerate(law.get("non_article_nodes", []), 1):
            node = obj(node, f"{filename} 비조문 노드 {idx}")
            db.execute("""
                INSERT INTO law_sections
                (law_id,source_order,node_type,node_number,node_branch_number,text,effective_date)
                VALUES (?,?,?,?,?,?,?)
            """, (
                law_id, int(node.get("source_order") or idx), clean(node.get("node_type")),
                clean(node.get("node_number")), clean(node.get("node_branch_number")),
                clean(node.get("text")), clean(node.get("effective_date")),
            ))
            section_count += 1
        article_count = 0
        seen = set()
        for idx, item in enumerate(law["data"], 1):
            item = obj(item, f"{filename} 조문 {idx}")
            label = f"{filename} 조문 {idx}"
            for field in ("article_no", "article_branch_no", "article_display", "article_body", "paragraphs"):
                if field not in item:
                    raise ValueError(f"{label}: 필수 필드 {field} 없음")
            if clean(item.get("article_status", "조문")) != "조문":
                raise ValueError(f"{label}: data에 실제 조문 아닌 항목 포함")
            number, branch, display = official_number(item["article_no"], item["article_branch_no"], label)
            if clean(item["article_display"]) != display or number in seen:
                raise ValueError(f"{label}: 공식 조문번호 표시 불일치 또는 중복: {display}")
            seen.add(number)
            raw_body = clean(item["article_body"])
            body = "\n".join(p for p in [raw_body, *paragraph_lines(item["paragraphs"], label)] if p)
            if not body:
                raise ValueError(f"{label}: 본문이 없습니다. 원본 XML 수동 점검 필요")
            title = clean(item.get("article_title"))
            heading = display + (f" ({title})" if title else "")
            content = f"<{name}>\n{heading}\n{body}"
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            insert_article(db, (
                law_id, name, int(item.get("source_order") or idx), clean(item["article_no"]), number,
                display, title, body, content, digest, "official", branch, raw_body,
                clean(item.get("article_effective_date")), "조문", clean(item.get("article_revision_type")),
                clean(item.get("article_changed")), json.dumps(item["paragraphs"], ensure_ascii=False, separators=(",", ":")),
            ))
            article_count += 1
        return article_count, section_count, 0

    # 번호 없는 법령도 검색 및 벡터 생성을 위해 articles에 '본문 문서' 1행을 둔다.
    # articles.id는 내부 행 식별자일 뿐, 법률 조문번호가 아니다.
    provision = law["unnumbered_provisions"][0]
    body = clean(provision["text"])
    content = f"<{name}>\n{body}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    row_id = insert_article(db, (
        law_id, name, int(provision.get("source_order") or 1), "", "", "", "",
        body, content, digest, "unnumbered_body", "", body,
        clean(provision.get("effective_date")), "전문", "", "", "[]",
    ))
    db.execute("""
        INSERT INTO unnumbered_provisions
        (law_id, source_order, node_type, text, effective_date, article_row_id)
        VALUES (?,?,?,?,?,?)
    """, (
        law_id, int(provision.get("source_order") or 1), "전문", body,
        clean(provision.get("effective_date")), row_id,
    ))
    return 0, 0, 1


def build(source: Path, output: Path, law_ids: list[str], verify_xml: bool) -> None:
    source, output = source.resolve(), output.resolve()
    if output.name.casefold() == "law_search.db":
        raise ValueError("운영 DB명 law_search.db로 생성할 수 없습니다")
    if output.exists():
        raise FileExistsError(f"이미 존재하는 DB는 덮어쓰지 않습니다: {output}")
    if any("backup" in p.casefold() for p in output.parts):
        raise ValueError("백업 경로를 출력 폴더로 사용할 수 없습니다")
    if (output == source or source in output.parents):
        raise ValueError("입력 JSON 폴더 안에 DB를 만들지 않습니다")
    # 레포지토리 바깥에서만 새 DB를 생성한다.
    project_root = Path(__file__).resolve().parents[1]
    if output == project_root or project_root in output.parents:
        raise ValueError("새 DB는 Git 프로젝트 외부 경로에 저장하세요")
    paths, full_mode = collect_inputs(source, law_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".law_search_v3_", suffix=".building", dir=output.parent)
    os.close(fd)
    temp_db = Path(temp_name)
    db: sqlite3.Connection | None = None
    try:
        db = sqlite3.connect(temp_db)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        schema(db)
        total_laws = total_articles = total_sections = total_unnumbered = 0
        # 개별 문서 단위 커밋; 실패 시 전체 임시 DB 폐기.
        for i, path in enumerate(paths, 1):
            law = load_law(path, verify_xml)
            try:
                with db:
                    articles, sections, unnumbered = insert_law(db, law, path.name)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"{path.name}: 법령ID/조문/참조 무결성 오류: {exc}") from exc
            total_laws += 1
            total_articles += articles
            total_sections += sections
            total_unnumbered += unnumbered
            if i <= 10 or i % 100 == 0 or i == len(paths):
                print(f"[{i:,}/{len(paths):,}] {law['law_name']} | 조문 {articles} | 번호 없음 {unnumbered}", flush=True)
        with db:
            db.execute("INSERT INTO articles_fts(articles_fts) VALUES ('rebuild')")
            db.execute("INSERT INTO articles_fts(articles_fts) VALUES ('integrity-check')")
            db.execute("ANALYZE")
        expected_rows = total_articles + total_unnumbered
        counts = {
            "laws": total_laws,
            "articles": expected_rows,
            "law_sections": total_sections,
            "unnumbered_provisions": total_unnumbered,
            "articles_fts": expected_rows,
        }
        for table, expected in counts.items():
            n = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if n != expected:
                raise RuntimeError(f"테이블 {table} 건수 검증 실패: {n} != {expected}")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity_check 실패")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("SQLite foreign_key_check 실패")
        if db.execute("SELECT COUNT(*) FROM articles WHERE normalization_method='unnumbered_body' AND (article_no<>'' OR article_display<>'')").fetchone()[0]:
            raise RuntimeError("번호 없는 법령에 임의 조문번호가 발견됐습니다")
        if (db.execute("SELECT COUNT(*) FROM articles WHERE normalization_method='unnumbered_body'").fetchone()[0] != total_unnumbered):
            raise RuntimeError("번호 없는 검색 문서 건수가 원문 테이블과 다릅니다")
        db.close()
        db = None
        # 동일 파일시스템 하드링크를 이용한 no-clobber 결과 게시.
        os.link(temp_db, output)
        print("\n[성공] 신규 SQLite 생성 및 검사 완료")
        print(f"법령 {total_laws:,}개 / 공식 번호 있는 조문 {total_articles:,}개 / 번호 없는 본문 {total_unnumbered:,}개")
        print(f"검색 행 및 FTS {expected_rows:,}건 / 장·절 노드 {total_sections:,}건")
        print(f"출력: {output}")
        print("기존 서비스 DB/ChromaDB는 변경하지 않았습니다.")
        print("주의: ChromaDB는 이 새 SQLite의 articles.id 기준으로 별도 구축해야 합니다.")
        if not full_mode:
            print("주의: 이 DB는 일부 법령을 사용한 시험용입니다. 운영 RAG에 연결하지 마세요.")
    finally:
        if db is not None:
            db.close()
        temp_db.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="번호 없는 법령 지원: 안전한 검색 SQLite v3")
    parser.add_argument("--source", required=True, type=Path, help="수집 v5 structured_json 폴더 또는 law_ID.json")
    parser.add_argument("--output", required=True, type=Path, help="기존 DB와 다른 외부 새 SQLite 파일")
    parser.add_argument("--law-id", action="append", default=[], help="시험용 법령 ID. 반복 지정 가능")
    parser.add_argument("--no-verify-xml", action="store_true", help="원본 XML SHA-256 확인 생략 (검증용으로 권장하지 않음)")
    args = parser.parse_args()
    try:
        build(args.source, args.output, args.law_id, not args.no_verify_xml)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.DatabaseError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"오류: {exc}\n기존 DB와 원본 데이터는 변경하지 않았습니다.\n")


if __name__ == "__main__":
    main()
