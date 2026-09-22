import re
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OLD_CHROMA_PATH = BASE_DIR / "law_db_full"
NEW_CHROMA_PATH = BASE_DIR / "law_db_optimized"
SQLITE_PATH = BASE_DIR / "law_search.db"

OLD_COLLECTION_NAME = "langchain"
NEW_COLLECTION_NAME = "law_articles"

MODEL_NAME = "jhgan/ko-sroberta-multitask"

READ_BATCH_SIZE = 1000
WRITE_BATCH_SIZE = 1000
EMBED_BATCH_SIZE = 64


# ============================================================
# 공통 함수
# ============================================================

def compact(text):
    if not text:
        return ""

    return re.sub(
        r"\s+",
        "",
        str(text),
    )


def normalize_article_no(value):
    if value is None:
        return ""

    text = compact(value)

    match = re.fullmatch(
        r"제?(\d+)조?(?:의(\d+))?",
        text,
    )

    if not match:
        return text

    base = match.group(1)
    sub = match.group(2)

    if sub:
        return f"{base}의{sub}"

    return base


def parse_old_document(document):
    """
    기존 Chroma 문서에서 조문 제목과 본문을 분리한다.
    """

    document = (
        document or ""
    ).strip()

    lines = document.splitlines()

    article_title = ""
    article_line_index = None

    for index, line in enumerate(lines[:3]):
        match = re.match(
            r"^\s*제\s*\d+\s*조"
            r"(?:의\s*\d+)?"
            r"\s*\((.*)\)\s*$",
            line,
        )

        if match:
            article_title = (
                match.group(1)
                .strip()
            )

            article_line_index = index
            break

    if article_line_index is None:
        body = "\n".join(lines).strip()
    else:
        body = "\n".join(
            lines[
                article_line_index + 1:
            ]
        ).strip()

    return (
        article_title,
        body,
    )


# ============================================================
# SQLite 연결
# ============================================================

print("SQLite 연결...")

db = sqlite3.connect(
    f"file:{SQLITE_PATH}?mode=ro",
    uri=True,
)

db.row_factory = sqlite3.Row


# ============================================================
# SQLite 조문 인덱스 생성
# ============================================================

print("SQLite 조문 인덱스 생성...")

rows = db.execute(
    """
    SELECT
        id,
        law_name,
        source_article_no,
        article_title
    FROM articles
    ORDER BY id
    """
).fetchall()

sqlite_count = len(rows)

article_index = defaultdict(list)

all_article_ids = set()

for row in rows:
    article_id = int(row["id"])

    all_article_ids.add(
        article_id
    )

    key = (
        row["law_name"],
        normalize_article_no(
            row["source_article_no"]
        ),
    )

    article_index[key].append(
        (
            article_id,
            compact(
                row["article_title"]
            ),
        )
    )

print(
    "SQLite 조문 수:",
    f"{sqlite_count:,}",
)


# ============================================================
# 기존 Chroma 열기
# ============================================================

print("기존 Chroma 연결...")

old_client = chromadb.PersistentClient(
    path=str(
        OLD_CHROMA_PATH
    )
)

old_collection = (
    old_client.get_collection(
        OLD_COLLECTION_NAME
    )
)

old_count = (
    old_collection.count()
)

print(
    "기존 벡터 수:",
    f"{old_count:,}",
)


# ============================================================
# 새 DB 초기화
# ============================================================

if NEW_CHROMA_PATH.exists():
    print()
    print(
        "기존 law_db_optimized 폴더가 존재합니다."
    )
    print(
        "새로 만들기 위해 삭제합니다."
    )

    shutil.rmtree(
        NEW_CHROMA_PATH
    )


print()
print(
    "새 Chroma DB 생성..."
)

new_client = (
    chromadb.PersistentClient(
        path=str(
            NEW_CHROMA_PATH
        )
    )
)


# 기존 DB와 동일하게 L2 거리 사용
new_collection = (
    new_client.create_collection(
        name=NEW_COLLECTION_NAME,
        metadata={
            "hnsw:space": "l2",
        },
    )
)


# ============================================================
# 본문 비교 함수
# ============================================================

def compare_body(
    candidate_ids,
    old_body,
):
    if not candidate_ids:
        return []

    placeholders = ",".join(
        "?"
        for _ in candidate_ids
    )

    sql = f"""
        SELECT
            id,
            body
        FROM articles
        WHERE id IN (
            {placeholders}
        )
    """

    body_rows = db.execute(
        sql,
        candidate_ids,
    ).fetchall()

    old_compact = compact(
        old_body
    )

    matched = []

    for row in body_rows:
        if (
            compact(
                row["body"]
            )
            == old_compact
        ):
            matched.append(
                int(
                    row["id"]
                )
            )

    return matched


# ============================================================
# 기존 Vector → SQLite article_id 매핑
# ============================================================

def find_sqlite_article_id(
    metadata,
    document,
):
    law_name = (
        metadata.get(
            "law_name",
            ""
        )
        or ""
    ).strip()

    source_article_no = (
        normalize_article_no(
            metadata.get(
                "article_no"
            )
        )
    )

    (
        old_title,
        old_body,
    ) = parse_old_document(
        document
    )

    normalized_title = (
        compact(
            old_title
        )
    )

    # 빈 placeholder
    if (
        not normalized_title
        and not compact(
            old_body
        )
    ):
        return None

    key = (
        law_name,
        source_article_no,
    )

    candidates = (
        article_index.get(
            key,
            [],
        )
    )

    if not candidates:
        return None

    # 제목 일치
    title_matches = [
        article_id
        for (
            article_id,
            sqlite_title,
        )
        in candidates
        if (
            normalized_title
            and sqlite_title
            == normalized_title
        )
    ]

    if len(title_matches) == 1:
        return title_matches[0]

    # 제목만으로 구별 안 되면 본문 비교
    if title_matches:
        body_candidate_ids = (
            title_matches
        )
    else:
        body_candidate_ids = [
            article_id
            for (
                article_id,
                _
            )
            in candidates
        ]

    body_matches = (
        compare_body(
            body_candidate_ids,
            old_body,
        )
    )

    if len(body_matches) == 1:
        return body_matches[0]

    return None


# ============================================================
# 새 DB 쓰기 버퍼
# ============================================================

pending_ids = []
pending_embeddings = []


def flush_pending():
    if not pending_ids:
        return

    new_collection.add(
        ids=[
            str(article_id)
            for article_id
            in pending_ids
        ],
        embeddings=[
            embedding.tolist()
            if isinstance(
                embedding,
                np.ndarray,
            )
            else embedding
            for embedding
            in pending_embeddings
        ],
    )

    pending_ids.clear()
    pending_embeddings.clear()


# ============================================================
# 기존 벡터 재사용
# ============================================================

print()
print(
    "=" * 70
)

print(
    "기존 벡터 재사용 시작"
)

print(
    "=" * 70
)


mapped_article_ids = set()

placeholder_count = 0
mapping_failed_count = 0
duplicate_count = 0

processed = 0


for offset in range(
    0,
    old_count,
    READ_BATCH_SIZE,
):
    result = old_collection.get(
        limit=READ_BATCH_SIZE,
        offset=offset,
        include=[
            "embeddings",
            "documents",
            "metadatas",
        ],
    )

    embeddings = (
        result["embeddings"]
    )

    documents = (
        result["documents"]
    )

    metadatas = (
        result["metadatas"]
    )

    for (
        embedding,
        document,
        metadata,
    ) in zip(
        embeddings,
        documents,
        metadatas,
    ):
        processed += 1

        (
            old_title,
            old_body,
        ) = parse_old_document(
            document
        )

        if (
            not compact(
                old_title
            )
            and not compact(
                old_body
            )
        ):
            placeholder_count += 1
            continue

        article_id = (
            find_sqlite_article_id(
                metadata,
                document,
            )
        )

        if article_id is None:
            mapping_failed_count += 1
            continue

        # 동일 SQLite 조문에 여러 기존 벡터가
        # 매핑되면 첫 번째만 사용
        if (
            article_id
            in mapped_article_ids
        ):
            duplicate_count += 1
            continue

        mapped_article_ids.add(
            article_id
        )

        pending_ids.append(
            article_id
        )

        pending_embeddings.append(
            np.asarray(
                embedding,
                dtype=np.float32,
            )
        )

        if (
            len(pending_ids)
            >= WRITE_BATCH_SIZE
        ):
            flush_pending()

    print(
        f"{processed:,} / "
        f"{old_count:,} 처리"
        f" | 재사용 {len(mapped_article_ids):,}"
    )


flush_pending()


# ============================================================
# 기존 벡터가 없는 SQLite 조문 확인
# ============================================================

missing_article_ids = sorted(
    all_article_ids
    - mapped_article_ids
)

print()
print(
    "기존 벡터 재사용 완료"
)

print(
    "재사용:",
    f"{len(mapped_article_ids):,}",
)

print(
    "Placeholder 제외:",
    f"{placeholder_count:,}",
)

print(
    "매핑 실패:",
    f"{mapping_failed_count:,}",
)

print(
    "중복 벡터 제외:",
    f"{duplicate_count:,}",
)

print(
    "새 임베딩 필요:",
    f"{len(missing_article_ids):,}",
)


# ============================================================
# 누락 조문 새 임베딩
# ============================================================

if missing_article_ids:

    print()
    print(
        "=" * 70
    )

    print(
        "누락 조문 새 임베딩 시작"
    )

    print(
        "=" * 70
    )

    print(
        "모델:",
        MODEL_NAME,
    )

    model = (
        SentenceTransformer(
            MODEL_NAME
        )
    )

    completed = 0

    for start in range(
        0,
        len(missing_article_ids),
        EMBED_BATCH_SIZE,
    ):
        batch_ids = (
            missing_article_ids[
                start:
                start
                + EMBED_BATCH_SIZE
            ]
        )

        placeholders = ",".join(
            "?"
            for _ in batch_ids
        )

        batch_rows = db.execute(
            f"""
            SELECT
                id,
                content
            FROM articles
            WHERE id IN (
                {placeholders}
            )
            ORDER BY id
            """,
            batch_ids,
        ).fetchall()

        texts = []

        actual_ids = []

        for row in batch_rows:
            text = (
                row["content"]
                or ""
            ).strip()

            if not text:
                continue

            actual_ids.append(
                int(
                    row["id"]
                )
            )

            texts.append(
                text
            )

        if not texts:
            continue

        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        new_collection.add(
            ids=[
                str(article_id)
                for article_id
                in actual_ids
            ],
            embeddings=[
                vector.tolist()
                for vector
                in vectors
            ],
        )

        completed += (
            len(actual_ids)
        )

        print(
            f"새 임베딩 "
            f"{completed:,} / "
            f"{len(missing_article_ids):,}"
        )


# ============================================================
# 최종 검증
# ============================================================

final_count = (
    new_collection.count()
)

print()
print(
    "=" * 70
)

print(
    "최적화 Vector DB 생성 결과"
)

print(
    "=" * 70
)

print(
    "SQLite 전체 조문:",
    f"{sqlite_count:,}",
)

print(
    "기존 벡터 재사용:",
    f"{len(mapped_article_ids):,}",
)

print(
    "새 임베딩:",
    f"{len(missing_article_ids):,}",
)

print(
    "최종 Vector 수:",
    f"{final_count:,}",
)


if final_count == sqlite_count:
    print()
    print(
        "✅ 모든 SQLite 조문에 Vector가 생성되었습니다."
    )

else:
    print()
    print(
        "⚠️ SQLite 조문 수와 Vector 수가 다릅니다."
    )

    print(
        "차이:",
        f"{sqlite_count - final_count:,}",
    )


db.close()
