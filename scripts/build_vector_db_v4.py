"""대규모 SQLite v3 -> ChromaDB 증분/재개 빌더 (기존 DB 보호).

특징
- 기존 law_search.db, law_db_optimized, law_db_full 및 백업 경로는 변경하지 않는다.
- SQLite를 읽기 전용으로 열어 articles.id와 원문 SHA256을 기준으로 인덱싱한다.
- 출력 폴더는 새로 생성하며, 중단된 *동일 빌드*에만 --resume으로 재접속한다.
- 배치마다 기존 Chroma ID와 SHA256을 확인하고 빠진 문서만 임베딩한다.
- 마지막에 전체 ID/원문 해시와 총 건수를 확인한 후 완료 마커를 생성한다.
- 법령 버전의 역사적 효력 판정, 자동 갱신, 운영 DB 전환은 수행하지 않는다.

PowerShell 예시 (프로젝트 루트):
  python scripts/build_vector_db_v4.py --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\search_db\\law_search_v3_full.db" --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\vector_db\\law_db_v4_smoke" --limit 20

  # 대규모 구축은 소규모 테스트 후 별도 폴더에서 시작:
  python scripts/build_vector_db_v4.py --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\search_db\\law_search_v3_full.db" --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\vector_db\\law_db_v4_full"

  # 위 대규모 구축이 중간에 중단되었다면 정확히 같은 경로에 대해:
  python scripts/build_vector_db_v4.py --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\search_db\\law_search_v3_full.db" --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\vector_db\\law_db_v4_full" --resume

주의: --limit로 만든 DB는 검색 전체 데이터가 아니므로 운영 서비스에 연결하지 말 것.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


COLLECTION_NAME = "law_articles"
DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_BATCH_SIZE = 32
BUILD_VERSION = 4
IN_PROGRESS = "_BUILD_IN_PROGRESS.json"
COMPLETE = "_BUILD_COMPLETE.json"


def fingerprint_sqlite(path: Path) -> dict[str, int | str]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_paths(sqlite_path: Path, output_dir: Path, resume: bool) -> tuple[Path, Path]:
    sqlite_path = sqlite_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"입력 SQLite 파일이 없습니다: {sqlite_path}")
    if sqlite_path.name.casefold() == "law_search.db":
        raise ValueError("기존 운영 law_search.db는 입력으로 사용할 수 없습니다")
    if any(part.casefold() in {"law_db_optimized", "law_db_full", ".git"}
           or "backup" in part.casefold() for part in output_dir.parts):
        raise ValueError("기존 ChromaDB, Git 또는 백업 경로는 출력으로 사용할 수 없습니다")
    if output_dir == sqlite_path or output_dir in sqlite_path.parents or sqlite_path in output_dir.parents:
        raise ValueError("입력 SQLite와 출력 디렉터리의 경로가 중첩됩니다")
    if resume:
        if not output_dir.is_dir():
            raise ValueError("--resume을 지정했지만 기존 미완성 출력 폴더가 없습니다")
    elif output_dir.exists():
        raise FileExistsError(f"출력 경로가 이미 존재합니다. 덮어쓰지 않습니다: {output_dir}")
    return sqlite_path, output_dir


def open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        articles = {r[1] for r in conn.execute("PRAGMA table_info(articles)")}
        laws = {r[1] for r in conn.execute("PRAGMA table_info(laws)")}
        required_a = {"id", "law_id", "law_name", "article_no", "article_display", "article_title",
                      "article_effective_date", "normalization_method", "content", "content_hash"}
        required_l = {"id", "source_law_id", "effective_date"}
        if required_a - articles or required_l - laws:
            raise ValueError(f"SQLite v3 스키마 불일치: articles={sorted(required_a - articles)}, "
                             f"laws={sorted(required_l - laws)}")
        integrity = conn.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError(f"SQLite quick_check 실패: {integrity}")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SQLite 외래 키 검증 실패")
        numbered = int(conn.execute("SELECT COUNT(*) FROM articles WHERE normalization_method!='unnumbered_body'").fetchone()[0])
        unnumbered = int(conn.execute("SELECT COUNT(*) FROM articles WHERE normalization_method='unnumbered_body'").fetchone()[0])
        if not numbered + unnumbered:
            raise ValueError("SQLite articles 테이블이 비어 있습니다")
        if conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='unnumbered_provisions'").fetchone()[0]:
            reference_count = int(conn.execute("SELECT COUNT(*) FROM unnumbered_provisions").fetchone()[0])
            if reference_count != unnumbered:
                raise ValueError(f"번호 없는 원문 참조 건수 불일치: {unnumbered} != {reference_count}")
    except Exception:
        conn.close()
        raise
    return conn


def iter_batches(conn: sqlite3.Connection, batch_size: int, limit: int | None) -> Iterator[list[sqlite3.Row]]:
    query = """
        SELECT a.id, a.law_id, a.law_name, a.article_no, a.article_display,
               a.article_title, a.article_effective_date, a.normalization_method,
               a.content, a.content_hash, l.effective_date AS law_effective_date,
               l.source_law_id AS official_law_id
          FROM articles a JOIN laws l ON a.law_id = l.id
         ORDER BY a.id
    """
    cursor = conn.execute(query + (" LIMIT ?" if limit is not None else ""),
                          (limit,) if limit is not None else ())
    while rows := cursor.fetchmany(batch_size):
        yield rows


def create_client_collection(output: Path, model_name: str, resume: bool):
    import chromadb
    client = chromadb.PersistentClient(path=str(output))
    if resume:
        col = client.get_collection(name=COLLECTION_NAME)
        meta = col.metadata or {}
        if (meta.get("embedding_model") != model_name or meta.get("hnsw:space") != "l2"
                or meta.get("schema") != "sqlite-v3" or meta.get("embedding_normalized") != "true"):
            raise ValueError("기존 Chroma 컬렉션의 모델/스키마 설정이 다릅니다. 재개할 수 없습니다")
    else:
        col = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2", "embedding_model": model_name,
                      "embedding_normalized": "true", "schema": "sqlite-v3"},
        )
    return client, col


def load_model(name: str, device: str | None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device) if device else SentenceTransformer(name)


def atomic_json(path: Path, payload: dict[str, Any]):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare_output(output: Path, sqlite_fingerprint: dict[str, Any], model_name: str,
                   limit: int | None, batch_size: int, resume: bool) -> None:
    if resume:
        if (output / COMPLETE).exists():
            raise ValueError("이미 완성된 벡터 DB입니다. --resume으로 변경하지 않습니다")
        marker = output / IN_PROGRESS
        if not marker.is_file():
            raise ValueError("이 폴더에는 v4의 생성 진행 기록이 없습니다. 임의의 ChromaDB에 연결하지 않습니다")
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if (manifest.get("build_version") != BUILD_VERSION
                or manifest.get("sqlite_fingerprint") != sqlite_fingerprint
                or manifest.get("embedding_model") != model_name
                or manifest.get("limit") != limit
                or manifest.get("collection") != COLLECTION_NAME):
            raise ValueError("기존 진행 기록과 입력 SQLite/모델/limit가 다릅니다. 재개 중단")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    atomic_json(output / IN_PROGRESS, {
        "build_version": BUILD_VERSION, "status": "in_progress",
        "sqlite_fingerprint": sqlite_fingerprint,
        "embedding_model": model_name, "collection": COLLECTION_NAME,
        "limit": limit, "batch_size_initial": batch_size,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def metadata_for(row: sqlite3.Row, digest: str) -> dict[str, Any]:
    doc_type = str(row["normalization_method"] or "")
    if doc_type not in {"official", "unnumbered_body"}:
        raise ValueError(f"알 수 없는 법령 문서 유형: 조문 ID={row['id']}, {doc_type!r}")
    if doc_type == "unnumbered_body" and (row["article_no"] or row["article_display"]):
        raise ValueError(f"번호 없는 법령에 공식 조문번호가 기록됨: {row['id']}")
    if doc_type != "unnumbered_body" and not row["article_display"]:
        raise ValueError(f"공식 조문번호가 누락된 조문: {row['id']}")
    return {
        "sqlite_article_id": int(row["id"]),
        "sqlite_law_id": int(row["law_id"]),
        "official_law_id": str(row["official_law_id"]),
        "law_name": str(row["law_name"]),
        "article_no": str(row["article_no"] or ""),
        "article_display": str(row["article_display"] or ""),
        "article_title": str(row["article_title"] or ""),
        "law_effective_date": str(row["law_effective_date"] or ""),
        "article_effective_date": str(row["article_effective_date"] or ""),
        "document_type": doc_type,
        "content_hash": digest,
    }


def inspect_existing(col: Any, expected_meta: dict[str, dict[str, Any]]) -> set[str]:
    """SQLite 배치에 해당하는 Chroma 데이터 검증. 누락된 ID만 반환한다."""
    ids = list(expected_meta)
    stored = col.get(ids=ids, include=["metadatas", "documents"])
    old_ids = [str(x) for x in (stored.get("ids") or [])]
    old_metas = stored.get("metadatas") or []
    old_docs = stored.get("documents") or []
    if len(old_ids) != len(old_metas) or len(old_ids) != len(old_docs) or len(old_ids) != len(set(old_ids)):
        raise RuntimeError("Chroma 배치 ID/메타데이터 응답이 일치하지 않습니다")
    for id_, meta, doc in zip(old_ids, old_metas, old_docs):
        if id_ not in expected_meta:
            raise RuntimeError(f"예상하지 못한 Chroma 문서: {id_}")
        if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in expected_meta[id_].items()):
            raise RuntimeError(f"기존 Chroma 문서/원문 해시/메타데이터 불일치: {id_}")
        if not isinstance(doc, str) or hash_text(doc) != expected_meta[id_]["content_hash"]:
            raise RuntimeError(f"Chroma에 저장된 실제 원문이 SQLite와 불일치: {id_}")
    return set(ids) - set(old_ids)


def encode(model: Any, docs: list[str], batch_size: int):
    import numpy as np
    array = np.asarray(model.encode(docs, batch_size=batch_size, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(docs) or array.shape[1] <= 0:
        raise RuntimeError("임베딩 출력 차원 오류")
    if not np.isfinite(array).all():
        raise RuntimeError("임베딩에 NaN/Inf가 있습니다")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        raise RuntimeError("임베딩 벡터가 정규화되지 않았습니다")
    return array


def build(sqlite_path: Path, output: Path, *, model_name: str = DEFAULT_MODEL,
          batch_size: int = DEFAULT_BATCH_SIZE, limit: int | None = None,
          device: str | None = None, resume: bool = False) -> None:
    if not 1 <= batch_size <= 128:
        raise ValueError("--batch-size는 1~128이어야 합니다")
    if limit is not None and limit < 1:
        raise ValueError("--limit는 1 이상이어야 합니다")
    sqlite_path, output = validate_paths(sqlite_path, output, resume)
    source_fp = fingerprint_sqlite(sqlite_path)
    conn = open_sqlite(sqlite_path)
    output_prepared = False
    complete = False
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
        expected = min(total, limit) if limit is not None else total
        unnumbered = int(conn.execute("SELECT COUNT(*) FROM articles WHERE normalization_method='unnumbered_body'").fetchone()[0])
        print(f"SQLite 전체 검색 문서: {total:,}건 (번호 없는 본문 {unnumbered:,}건)")
        print(f"이번 임베딩 대상: {expected:,}건 / 배치 크기: {batch_size}")
        if limit is not None:
            print("주의: --limit DB는 일부 문서만 포함하므로 운영 서비스에 연결하지 마세요")
        # 값이 다르면 ChromaDB를 열거나 수정하지 않고 중단.
        if resume:
            prepare_output(output, source_fp, model_name, limit, batch_size, True)
        print(f"임베딩 모델 로딩: {model_name}" + (f" / 장치: {device}" if device else ""))
        model = load_model(model_name, device)
        prepare_output(output, source_fp, model_name, limit, batch_size, resume)
        output_prepared = True
        _client, collection = create_client_collection(output, model_name, resume)
        # ChromaDB 메타데이터에 없는 문서는 재임베딩, 기존 문서는 해시 검증 후 생략.
        processed = 0
        skipped = 0
        created = 0
        embed_dim = None
        for rows in iter_batches(conn, batch_size, limit):
            ids = [str(row["id"]) for row in rows]
            docs = [str(row["content"]) for row in rows]
            expected_meta = {}
            for row, text in zip(rows, docs):
                if not text.strip():
                    raise ValueError(f"빈 조문 본문: {row['id']}")
                digest = hash_text(text)
                if digest != row["content_hash"]:
                    raise ValueError(f"SQLite 원문 SHA256 불일치: {row['id']}")
                expected_meta[str(row["id"])] = metadata_for(row, digest)
            missing = inspect_existing(collection, expected_meta)
            skipped += len(ids) - len(missing)
            if missing:
                new_ids = [i for i in ids if i in missing]
                new_docs = [docs[idx] for idx, i in enumerate(ids) if i in missing]
                vecs = encode(model, new_docs, batch_size)
                if embed_dim is None:
                    embed_dim = int(vecs.shape[1])
                elif embed_dim != vecs.shape[1]:
                    raise RuntimeError("배치 간 임베딩 차원 불일치")
                collection.add(ids=new_ids, documents=new_docs, embeddings=vecs.tolist(),
                               metadatas=[expected_meta[i] for i in new_ids])
                created += len(new_ids)
            processed += len(ids)
            if processed % (batch_size * 10) == 0 or processed == expected:
                print(f"진행: {processed:,}/{expected:,} / 신규 {created:,} / 기존 검증 {skipped:,}", flush=True)
        if processed != expected:
            raise RuntimeError(f"SQLite 처리 건수 불일치: {processed} != {expected}")
        if fingerprint_sqlite(sqlite_path) != source_fp:
            raise RuntimeError("빌드 중 입력 SQLite 파일의 크기 또는 수정 시각이 변경됐습니다")
        if collection.count() != expected:
            raise RuntimeError(f"Chroma 문서 개수 불일치: {collection.count()} != {expected}")
        # 전수 재조회: 완료 직전 Chroma ID·메타데이터·본문 해시를 다시 확인.
        verified = 0
        for rows in iter_batches(conn, batch_size, limit):
            expected_meta = {}
            for row in rows:
                text = str(row["content"])
                digest = hash_text(text)
                if digest != row["content_hash"]:
                    raise RuntimeError(f"최종 검증 중 SQLite 해시 불일치: {row['id']}")
                expected_meta[str(row["id"])] = metadata_for(row, digest)
            if inspect_existing(collection, expected_meta):
                raise RuntimeError("최종 검증 실패: 일부 SQLite 문서 ID가 Chroma에 없습니다")
            verified += len(rows)
        if verified != expected or collection.count() != expected:
            raise RuntimeError("전체 ID 전수 검증 또는 문서 개수 검증 실패")
        if fingerprint_sqlite(sqlite_path) != source_fp:
            raise RuntimeError("최종 검증 중 입력 SQLite가 변경됐습니다")
        manifest = {
            "status": "complete", "build_version": BUILD_VERSION,
            "source_sqlite": str(sqlite_path), "source_sqlite_fingerprint": source_fp,
            "source_article_count": total, "vector_count": expected,
            "partial_test": limit is not None, "collection": COLLECTION_NAME,
            "embedding_model": model_name,
            "embedding_dimension": embed_dim or getattr(model, "get_sentence_embedding_dimension", lambda: None)(),
            "unnumbered_source_docs": unnumbered, "created_this_run": created,
            "verified_existing_this_run": skipped,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(output / COMPLETE, manifest)
        (output / IN_PROGRESS).unlink()
        complete = True
        print("\n[성공] ChromaDB 구축 및 전체 SQLite ID·원문 해시 검증 완료")
        print(f"문서 {expected:,}건 / 신규 {created:,}건 / 기존 검증 {skipped:,}건")
        print(f"새 ChromaDB: {output}")
        print("기존 운영 DB와 원본 SQLite는 수정하지 않았습니다")
    finally:
        conn.close()
        if output_prepared and not complete:
            print(f"주의: 미완성 DB 보존됨: {output}", file=sys.stderr)
            print("같은 입력/출력/--limit/--model로 --resume을 붙여 재실행할 수 있습니다.", file=sys.stderr)
            print(f"{COMPLETE}가 생기기 전까지 서비스에 연결하지 마세요.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="대용량 법령 SQLite v3 → 중단 후 재개 가능한 ChromaDB v4")
    parser.add_argument("--sqlite", type=Path, required=True, help="새로 생성한 law_search_v3_full.db")
    parser.add_argument("--output", type=Path, required=True, help="기존에 없는 새 벡터 DB 폴더")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="임베딩 장치(미지정 시 자동 선택)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="앞 N개 문서만 테스트(운영 연결 금지)")
    parser.add_argument("--resume", action="store_true", help="같은 입력으로 중단된 v4 DB에 이어서 저장")
    args = parser.parse_args()
    try:
        build(args.sqlite, args.output, model_name=args.model, batch_size=args.batch_size,
              limit=args.limit, device=args.device, resume=args.resume)
    except (OSError, ValueError, sqlite3.DatabaseError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        parser.exit(1, f"오류: {exc}\n기존 운영 DB와 SQLite 원본은 변경하지 않았습니다.\n")


if __name__ == "__main__":
    main()
