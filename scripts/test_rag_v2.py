"""개선된 법률 SQLite + ChromaDB의 검색 연결 상태를 읽기 전용으로 점검한다.

사용 예 (Windows PowerShell, 프로젝트 최상위 폴더):
    python scripts/test_rag_v2.py --sqlite "C:\\dev\\Law-Action-Assistant-LawDataV2\\search_db\\law_search_v2_test.db" --chroma "C:\\dev\\Law-Action-Assistant-LawDataV2\\vector_db\\law_db_v3_full"

기존 law_search.db, law_db_optimized/는 수정하거나 삭제하지 않는다.
현재 테스트 DB에 들어 있는 6개 법령만 검색하므로 검색 결과를 전체 법률 검색 성능으로 해석하지 말 것.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

COLLECTION = "law_articles"
MODEL_NAME = "jhgan/ko-sroberta-multitask"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def connect_readonly(path: Path) -> sqlite3.Connection:
    # pathlib.as_uri()는 Windows 드라이브 경로도 안전하게 file: URI로 변환한다.
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def print_exact(conn: sqlite3.Connection) -> None:
    print("\n=== 1. 조문번호 정확 검색 (SQLite) ===")
    for name, number in [("민법", "390"), ("형법", "347"), ("근로기준법", "36")]:
        row = conn.execute(
            "SELECT id, law_name, article_display, article_title, content "
            "FROM articles WHERE law_name = ? AND article_no = ? LIMIT 1",
            (name, number),
        ).fetchone()
        if row:
            print(f"[확인] ID={row['id']}, {row['law_name']} {row['article_display']} ({row['article_title']})")
            print("       내용:", " ".join(row["content"].split())[:125])
        else:
            print(f"[조회 결과 없음] {name} 제{number}조 — DB의 법령 버전과 수집 범위를 확인하세요.")


def print_keyword(conn: sqlite3.Connection) -> None:
    print("\n=== 2. 키워드 검색 (SQLite FTS5) ===")
    # 검색어를 작은따옴표 SQL 매개변수 + FTS 큰따옴표로 안전하게 감싼다.
    for keyword in ["손해배상", "사기", "임금"]:
        try:
            rows = conn.execute(
                "SELECT a.id, a.law_name, a.article_display, a.article_title "
                "FROM articles_fts f JOIN articles a ON a.id = f.rowid "
                "WHERE articles_fts MATCH ? ORDER BY bm25(articles_fts) LIMIT 3",
                (f'"{keyword}"',),
            ).fetchall()
            print(f"검색어 '{keyword}': {len(rows)}개 (최대 3개 표시)")
            for row in rows:
                print(f"  - ID={row['id']} {row['law_name']} {row['article_display']} ({row['article_title']})")
        except sqlite3.OperationalError as exc:
            print(f"[FTS 오류] {keyword}: {exc}")


def verify_vector_ids(conn: sqlite3.Connection, collection, expected_count: int) -> None:
    print("\n=== 3. SQLite ↔ ChromaDB ID·원문 해시 검사 ===")
    source = {
        str(row["id"]): row["content_hash"]
        for row in conn.execute("SELECT id, content_hash FROM articles")
    }
    if len(source) != expected_count:
        raise RuntimeError("SQLite 조문 ID가 중복되거나 개수가 변했습니다.")
    if collection.count() != expected_count:
        raise RuntimeError(f"벡터 개수가 다릅니다: SQLite {expected_count}, ChromaDB {collection.count()}")
    found = set()
    for offset in range(0, expected_count, 300):
        result = collection.get(limit=300, offset=offset, include=["metadatas"])
        ids = result.get("ids") or []
        metas = result.get("metadatas") or []
        if len(ids) != len(metas):
            raise RuntimeError("벡터 ID와 메타데이터 건수가 다릅니다.")
        for item_id, meta in zip(ids, metas):
            if item_id not in source or item_id in found:
                raise RuntimeError(f"예상하지 못한 벡터 ID 또는 중복 ID: {item_id}")
            if not meta or meta.get("content_hash") != source[item_id]:
                raise RuntimeError(f"SQLite와 벡터 원문 해시가 다릅니다: {item_id}")
            found.add(item_id)
    if found != set(source):
        raise RuntimeError("SQLite와 ChromaDB에 들어 있는 조문 ID 집합이 다릅니다.")
    print(f"[통과] 전체 {expected_count:,}건의 ID와 원문 해시가 정확히 일치합니다.")


def print_semantic(conn: sqlite3.Connection, collection, model_name: str, queries: list[str]) -> None:
    print("\n=== 4. 자연어 의미 검색 (ChromaDB → SQLite) ===")
    from sentence_transformers import SentenceTransformer

    print("임베딩 모델 로딩:", model_name)
    model = SentenceTransformer(model_name)
    for question in queries:
        vector = model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()
        result = collection.query(query_embeddings=[vector], n_results=5, include=["distances", "metadatas"])
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        print(f"\n질문: {question}")
        if not ids:
            print("  [검색 결과 없음]")
        for rank, item_id in enumerate(ids, start=1):
            row = conn.execute(
                "SELECT id, law_name, article_display, article_title, content FROM articles WHERE id = ?",
                (int(item_id),),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"벡터 검색 결과가 SQLite에 없습니다: ID={item_id}")
            distance = float(distances[rank - 1]) if rank <= len(distances) else float("nan")
            print(f"  {rank}. {row['law_name']} {row['article_display']} ({row['article_title']}) | ID={item_id} | L2거리={distance:.4f}")
            print("     ", " ".join(row["content"].split())[:160])


def run(sqlite_path: Path, chroma_path: Path, queries: list[str]) -> None:
    sqlite_path = sqlite_path.expanduser().resolve()
    chroma_path = chroma_path.expanduser().resolve()
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite 파일 없음: {sqlite_path}")
    if not chroma_path.is_dir():
        raise FileNotFoundError(f"ChromaDB 폴더 없음: {chroma_path}")
    manifest_path = chroma_path / "_BUILD_COMPLETE.json"
    if not manifest_path.is_file():
        raise RuntimeError("완료 표시(_BUILD_COMPLETE.json)가 없는 ChromaDB는 검사하지 않습니다.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("partial_test"):
        raise RuntimeError("완료되지 않았거나 일부 조문만 임베딩한 테스트 DB입니다.")
    if Path(str(manifest.get("source_sqlite", ""))).resolve() != sqlite_path:
        raise RuntimeError("ChromaDB가 지금 지정한 SQLite로 생성된 것이 아닙니다.")
    if manifest.get("collection") != COLLECTION:
        raise RuntimeError("ChromaDB 컬렉션명이 다릅니다.")

    import chromadb

    conn = connect_readonly(sqlite_path)
    try:
        count = int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
        law_count = int(conn.execute("SELECT COUNT(*) FROM laws").fetchone()[0])
        print(f"입력 SQLite: {sqlite_path}\n입력 ChromaDB: {chroma_path}")
        print(f"법령 {law_count}개, 조문 {count:,}개")
        if count != manifest.get("vector_count") or count != manifest.get("source_article_count"):
            raise RuntimeError("빌드 완료 기록과 SQLite 조문 개수가 일치하지 않습니다.")
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(COLLECTION)
        verify_vector_ids(conn, collection, count)
        print_exact(conn)
        print_keyword(conn)
        print_semantic(conn, collection, manifest.get("embedding_model") or MODEL_NAME, queries)
        print("\n검사 완료. 원본 DB를 삭제하거나 재생성하지 않았습니다.")
        print("참고: 검색 결과의 법률적 적합성은 별도 평가가 필요합니다. 현재 6개 법령만 포함됩니다.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="법률 SQLite와 ChromaDB 연결 및 검색 검증 (읽기 전용)")
    parser.add_argument("--sqlite", type=Path, required=True, help="새 SQLite 테스트 DB 파일")
    parser.add_argument("--chroma", type=Path, required=True, help="새 ChromaDB 전체 벡터 폴더")
    parser.add_argument(
        "--query", action="append", dest="queries",
        help="자연어 검색 테스트 질문 (여러 번 사용 가능)",
    )
    args = parser.parse_args()
    questions = args.queries or [
        "계약을 위반한 상대방에게 손해배상을 청구하고 싶어요.",
        "다른 사람을 속여 돈을 가져간 경우 어떤 법이 적용되나요?",
        "퇴사 후 임금을 지급하지 않으면 어떻게 되나요?",
    ]
    try:
        run(args.sqlite, args.chroma, questions)
    except Exception as exc:
        print(f"\n검사 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
