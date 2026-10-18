"""새로운 SQLite 법령 DB에서 안전하게 ChromaDB를 생성하는 독립 실행 도구.

- 개선된 build_law_search_db_v2.py의 SQLite 파일만 입력으로 사용.
- 기존 law_db_full, law_db_optimized, law_search.db는 수정하지 않음.
- 기존 임베딩은 재사용하지 않음 (원문 내용/식별자 변경 가능성).
- 출력 경로가 있으면 즉시 중단. 전용 출력 폴더를 새로 생성하고 검증 완료 마커가 있을 때만 결과를 인정.
- SQLite articles.id 를 Chroma 문서 ID로 사용하여 정확히 매핑.

Windows PowerShell 예시 (프로젝트 루트에서 실행):
  python scripts/build_vector_db_v3.py `
    --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\search_db\\law_search_v2_test.db" `
    --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\vector_db\\law_db_v2_smoke" `
    --limit 20

전체 1,840개 테스트 DB 임베딩 (다른 새 경로 사용):
  python scripts/build_vector_db_v3.py `
    --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\search_db\\law_search_v2_test.db" `
    --output "C:\\dev\\Law-Action-Assistant-LawDataV2\\vector_db\\law_db_v2_test"

주의: 이 도구는 법령 시행 시점 검증/자동 업데이트/서비스 DB 교체를 수행하지 않음.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COLLECTION_NAME = "law_articles"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_BATCH_SIZE = 32


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_paths(sqlite_path: Path, output_dir: Path) -> tuple[Path, Path]:
    sqlite_path = sqlite_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite 파일이 없습니다: {sqlite_path}")
    if sqlite_path.name.lower() == "law_search.db":
        raise ValueError("기존 운영용 law_search.db를 입력으로 사용할 수 없습니다. 새 테스트 DB를 지정하세요.")

    # 중요한 기존 벡터 디렉터리와 그 하위 경로는 출력으로 지정하지 못하게 한다.
    protected_names = {"law_db_optimized", "law_db_full", ".git"}
    if any(part.lower() in protected_names for part in output_dir.parts):
        raise ValueError("기존 ChromaDB 또는 .git 내부 경로는 출력으로 사용할 수 없습니다.")
    if output_dir.exists():
        raise FileExistsError(f"출력 폴더가 이미 있으므로 덮어쓰지 않습니다: {output_dir}")
    if output_dir == sqlite_path.parent or output_dir == sqlite_path:
        raise ValueError("출력 폴더가 SQLite DB 위치와 충돌합니다.")
    return sqlite_path, output_dir


def open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    # SQLite URI의 경로 인코딩은 pathlib.as_uri() 사용
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
        required = {"id", "law_id", "law_name", "article_no", "article_display", "article_title",
                    "content", "content_hash", "article_effective_date"}
        if not required.issubset(columns):
            raise ValueError("새 법령 SQLite v2의 articles 스키마가 아닙니다. 부족한 컬럼: "
                             + ", ".join(sorted(required - columns)))
        law_columns = {row[1] for row in conn.execute("PRAGMA table_info(laws)")}
        if "effective_date" not in law_columns:
            raise ValueError("새 법령 SQLite v2의 laws.effective_date가 없습니다.")
        check = conn.execute("PRAGMA integrity_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError(f"SQLite 무결성 검사 실패: {check}")
    except Exception:
        conn.close()
        raise
    return conn


def count_articles(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0])


def iter_article_batches(conn: sqlite3.Connection, batch_size: int, limit: int | None):
    query = """
        SELECT a.id, a.law_id, a.law_name, a.article_no, a.article_display,
               a.article_title, a.content, a.content_hash, a.article_effective_date,
               l.effective_date AS law_effective_date, l.source_law_id AS official_law_id
        FROM articles a JOIN laws l ON a.law_id = l.id
        ORDER BY a.id
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    cursor = conn.execute(query, params)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def create_collection(output_dir: Path, model_name: str):
    import chromadb  # 실제 실행 시에만 라이브러리 로드

    client = chromadb.PersistentClient(path=str(output_dir))
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2", "embedding_model": model_name,
                  "embedding_normalized": "true", "schema": "sqlite-v2"},
    )
    return client, collection


def load_model(model_name: str, device: str | None):
    from sentence_transformers import SentenceTransformer
    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def verify_collection(collection: Any, expected_hashes: dict[str, str], batch_size: int = 500) -> None:
    if collection.count() != len(expected_hashes):
        raise RuntimeError("벡터 개수가 SQLite 조문 수와 일치하지 않습니다.")
    seen: set[str] = set()
    for offset in range(0, len(expected_hashes), batch_size):
        result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        if len(ids) != len(metadatas):
            raise RuntimeError("Chroma ID와 메타데이터 개수가 다릅니다.")
        for article_id, metadata in zip(ids, metadatas):
            article_id = str(article_id)
            if article_id in seen or article_id not in expected_hashes:
                raise RuntimeError(f"Chroma ID 중복 또는 예상하지 못한 ID: {article_id}")
            if not metadata or metadata.get("content_hash") != expected_hashes[article_id]:
                raise RuntimeError(f"Chroma 본문 해시 불일치: {article_id}")
            seen.add(article_id)
    if seen != set(expected_hashes):
        raise RuntimeError("Chroma와 SQLite의 조문 ID 집합이 다릅니다.")


def build(sqlite_path: Path, output_dir: Path, *, model_name: str = EMBEDDING_MODEL,
          batch_size: int = DEFAULT_BATCH_SIZE, limit: int | None = None,
          device: str | None = None) -> None:
    if batch_size < 1 or batch_size > 128:
        raise ValueError("--batch-size는 1~128 사이여야 합니다.")
    if limit is not None and limit < 1:
        raise ValueError("--limit는 1 이상이어야 합니다.")
    sqlite_path, output_dir = validate_paths(sqlite_path, output_dir)
    conn = open_sqlite_readonly(sqlite_path)
    created_output = False
    complete = False
    try:
        total = count_articles(conn)
        expected = min(total, limit) if limit else total
        if expected < 1:
            raise ValueError("SQLite에 저장된 조문이 없습니다.")
        print(f"SQLite 전체 조문: {total:,}건 / 이번 임베딩 대상: {expected:,}건")
        if limit is not None:
            print("주의: --limit가 지정된 일부 조문만의 테스트용 ChromaDB입니다. 실제 RAG에 연결하면 안 됩니다.")

        # 파일 쓰기 전, 모델 다운로드/로딩부터 검사
        print(f"임베딩 모델 준비: {model_name}")
        model = load_model(model_name, device)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Windows에서는 Chroma의 chroma.sqlite3 파일 핸들이 열린 동안
        # 디렉터리 rename / rmtree를 시도하면 WinError 32가 발생할 수 있다.
        # 따라서 임시 디렉터리 이름 변경 대신, 새 출력 디렉터리에서 직접 구축하고
        # _BUILD_COMPLETE.json 파일이 존재할 때만 완성된 결과로 인정한다.
        output_dir.mkdir(exist_ok=False)
        created_output = True
        (output_dir / "_BUILD_IN_PROGRESS.txt").write_text(
            "생성 진행 중입니다. 완료 마커가 없으면 RAG에 연결하지 마세요.\n",
            encoding="utf-8",
        )
        _, collection = create_collection(output_dir, model_name)
        expected_hashes: dict[str, str] = {}
        processed = 0
        embedding_dim: int | None = None
        import numpy as np

        for rows in iter_article_batches(conn, batch_size, limit):
            ids: list[str] = []
            docs: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for row in rows:
                article_id = str(row["id"])
                content = row["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"조문 ID {article_id}: 본문이 비어 있습니다.")
                digest = sha256_text(content)
                if digest != row["content_hash"]:
                    raise ValueError(f"조문 ID {article_id}: SQLite content_hash 불일치")
                if article_id in expected_hashes:
                    raise ValueError(f"중복된 조문 ID: {article_id}")
                expected_hashes[article_id] = digest
                ids.append(article_id)
                docs.append(content)
                metadatas.append({
                    "sqlite_article_id": int(row["id"]),
                    "sqlite_law_id": int(row["law_id"]),
                    "official_law_id": str(row["official_law_id"]),
                    "law_name": str(row["law_name"]),
                    "article_no": str(row["article_no"]),
                    "article_display": str(row["article_display"]),
                    "article_title": str(row["article_title"] or ""),
                    "law_effective_date": str(row["law_effective_date"] or ""),
                    "article_effective_date": str(row["article_effective_date"] or ""),
                    "content_hash": digest,
                })
            vectors = np.asarray(model.encode(
                docs, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(ids):
                raise RuntimeError("임베딩 모델 출력 크기가 입력 조문 수와 일치하지 않습니다.")
            if vectors.shape[1] < 1 or not np.isfinite(vectors).all():
                raise RuntimeError("잘못된 임베딩 벡터가 생성됐습니다.")
            if embedding_dim is None:
                embedding_dim = int(vectors.shape[1])
            elif vectors.shape[1] != embedding_dim:
                raise RuntimeError("임베딩 차원이 일치하지 않습니다.")
            collection.add(ids=ids, documents=docs, embeddings=vectors.tolist(), metadatas=metadatas)
            processed += len(ids)
            print(f"임베딩 생성: {processed:,}/{expected:,}건", flush=True)

        if processed != expected:
            raise RuntimeError(f"SQLite 조회 개수가 기대와 다릅니다: {processed} != {expected}")
        verify_collection(collection, expected_hashes)
        # 모든 조문 ID와 본문 해시가 일치한 뒤에만 완료 마커를 기록한다.
        # 이 파일은 다른 프로세스의 배포/검증 단계에서 신뢰할 수 있는 완료 신호다.
        manifest = {
            "status": "complete",
            "source_sqlite": str(sqlite_path),
            "source_article_count": total,
            "vector_count": processed,
            "partial_test": limit is not None,
            "collection": COLLECTION_NAME,
            "embedding_model": model_name,
            "embedding_dimension": embedding_dim,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "_BUILD_COMPLETE.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "_BUILD_IN_PROGRESS.txt").unlink()
        complete = True
        print("\n새 ChromaDB 생성 및 조문 ID 검증 완료")
        print(f"조문 벡터: {processed:,}개 / 임베딩 차원: {embedding_dim}")
        print(f"컬렉션: {COLLECTION_NAME} / 모델: {model_name}")
        print(f"출력 경로: {output_dir}")
        print("기존 law_search.db 및 law_db_optimized는 변경하지 않았습니다.")
        print("주의: 새 DB는 아직 서비스와 연결되지 않았습니다.")
    finally:
        conn.close()
        # 실패한 폴더는 일부 벡터가 들어있을 수 있어 자동 삭제하지 않는다.
        # 특히 Windows에서 Chroma 파일이 열려 있을 때 삭제하면 WinError 32가
        # 발생하여 원래 실패 원인을 가릴 수 있다.
        if created_output and not complete:
            print(
                f"주의: 미완성 출력 폴더가 남아 있습니다: {output_dir}\n"
                "완료 마커(_BUILD_COMPLETE.json)가 없으면 서비스에 연결하지 마세요.\n"
                "원인 수정 후 새로운 출력 폴더명을 사용해 재시도하세요.",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="새 SQLite 기반 안전한 ChromaDB 생성 (기존 DB 삭제 금지)")
    parser.add_argument("--sqlite", required=True, type=Path, help="build_law_search_db_v2.py의 새 SQLite 경로")
    parser.add_argument("--output", required=True, type=Path, help="존재하지 않는 새 출력 폴더 경로")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="임베딩 모델 (서비스 RAG와 같아야 함)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="앞 N개 조문으로 소규모 테스트. 실제 서비스 사용 불가")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="임베딩 장치 (기본 자동선택)")
    args = parser.parse_args()
    try:
        build(args.sqlite, args.output, model_name=args.model,
              batch_size=args.batch_size, limit=args.limit, device=args.device)
    except (OSError, ValueError, sqlite3.Error, RuntimeError, ImportError) as exc:
        parser.exit(1, f"오류: {exc}\n기존 데이터베이스 및 원본 데이터는 변경하지 않았습니다.\n")


if __name__ == "__main__":
    main()
