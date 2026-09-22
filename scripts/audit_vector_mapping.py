import re
import sqlite3
from collections import defaultdict

import chromadb


CHROMA_PATH = "law_db_full"
COLLECTION_NAME = "langchain"
SQLITE_PATH = "law_search.db"

BATCH_SIZE = 5000


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
    기존 Chroma 문서에서
    조문 제목과 본문을 분리한다.
    """

    document = (
        document or ""
    ).strip()

    lines = document.splitlines()

    article_title = ""

    article_line_index = None

    for index, line in enumerate(
        lines[:3]
    ):
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
        body = "\n".join(
            lines
        ).strip()

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


# =========================================================
# SQLite 인덱스
# =========================================================

db = sqlite3.connect(
    SQLITE_PATH
)

db.row_factory = (
    sqlite3.Row
)

rows = db.execute(
    """
    SELECT
        id,
        law_name,
        source_article_no,
        article_title
    FROM articles
    """
).fetchall()

article_index = (
    defaultdict(list)
)

for row in rows:
    key = (
        row["law_name"],
        normalize_article_no(
            row[
                "source_article_no"
            ]
        ),
    )

    article_index[
        key
    ].append(
        (
            row["id"],
            compact(
                row[
                    "article_title"
                ]
            ),
        )
    )

sqlite_count = len(rows)

print(
    "SQLite 조문 수:",
    f"{sqlite_count:,}",
)


# =========================================================
# 기존 Chroma
# =========================================================

client = (
    chromadb.PersistentClient(
        path=CHROMA_PATH
    )
)

collection = (
    client.get_collection(
        COLLECTION_NAME
    )
)

total = collection.count()

print(
    "기존 Chroma 벡터 수:",
    f"{total:,}",
)


# =========================================================
# 통계
# =========================================================

mapped_counts = defaultdict(
    int
)

placeholder_count = 0
no_key_count = 0
mismatch_count = 0
ambiguous_count = 0

examples = {
    "no_key": [],
    "mismatch": [],
    "ambiguous": [],
}


def compare_body(
    candidate_ids,
    old_body,
):
    """
    제목만으로 구별되지 않는 경우
    SQLite 본문과 직접 비교한다.
    """

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
                row["id"]
            )

    return matched


# =========================================================
# 매핑 검사
# =========================================================

processed = 0

for offset in range(
    0,
    total,
    BATCH_SIZE,
):
    result = collection.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=[
            "documents",
            "metadatas",
        ],
    )

    for document, metadata in zip(
        result["documents"],
        result["metadatas"],
    ):
        processed += 1

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

        # ---------------------------------------------
        # 빈 placeholder
        #
        # 예:
        # 제4조 ()
        # ---------------------------------------------

        if (
            not normalized_title
            and not compact(
                old_body
            )
        ):
            placeholder_count += 1
            continue

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

        # ---------------------------------------------
        # 동일 법령 + 원본 조문번호가 없음
        # ---------------------------------------------

        if not candidates:
            no_key_count += 1

            if (
                len(
                    examples["no_key"]
                )
                < 10
            ):
                examples[
                    "no_key"
                ].append(
                    (
                        law_name,
                        source_article_no,
                        old_title,
                    )
                )

            continue

        # ---------------------------------------------
        # 제목 일치 후보
        # ---------------------------------------------

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

        # 제목으로 하나만 특정 가능
        if len(title_matches) == 1:
            mapped_counts[
                title_matches[0]
            ] += 1

            continue

        # ---------------------------------------------
        # 제목으로 못 찾았거나
        # 같은 제목 후보가 여러 개인 경우
        # 본문까지 비교
        # ---------------------------------------------

        if title_matches:
            body_candidate_ids = (
                title_matches
            )
        else:
            body_candidate_ids = [
                article_id
                for (
                    article_id,
                    _,
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
            mapped_counts[
                body_matches[0]
            ] += 1

            continue

        # ---------------------------------------------
        # 후보가 하나뿐인데 제목/본문이 다르면
        # Chroma와 JSON 버전 차이 가능성
        # ---------------------------------------------

        if (
            len(candidates) == 1
            and not body_matches
        ):
            mismatch_count += 1

            if (
                len(
                    examples["mismatch"]
                )
                < 10
            ):
                examples[
                    "mismatch"
                ].append(
                    (
                        law_name,
                        source_article_no,
                        old_title,
                        candidates[0][1],
                    )
                )

            continue

        # ---------------------------------------------
        # 여러 후보를 끝까지 구별 못함
        # ---------------------------------------------

        ambiguous_count += 1

        if (
            len(
                examples["ambiguous"]
            )
            < 10
        ):
            examples[
                "ambiguous"
            ].append(
                (
                    law_name,
                    source_article_no,
                    old_title,
                    len(candidates),
                )
            )

    print(
        f"{processed:,} / "
        f"{total:,} 처리"
    )


# =========================================================
# 결과
# =========================================================

unique_mapped = len(
    mapped_counts
)

mapped_occurrences = sum(
    mapped_counts.values()
)

duplicate_vector_records = sum(
    count - 1
    for count
    in mapped_counts.values()
    if count > 1
)

missing_sqlite = (
    sqlite_count
    - unique_mapped
)

print()
print(
    "=" * 65
)

print(
    "벡터 ↔ SQLite 매핑 검사 결과"
)

print(
    "=" * 65
)

print(
    "기존 Chroma 벡터:",
    f"{total:,}",
)

print(
    "SQLite 조문:",
    f"{sqlite_count:,}",
)

print(
    "매핑된 벡터:",
    f"{mapped_occurrences:,}",
)

print(
    "매핑된 고유 SQLite 조문:",
    f"{unique_mapped:,}",
)

print(
    "동일 조문에 중복 매핑된 벡터:",
    f"{duplicate_vector_records:,}",
)

print(
    "빈 placeholder:",
    f"{placeholder_count:,}",
)

print(
    "법령/조문번호 매칭 없음:",
    f"{no_key_count:,}",
)

print(
    "내용 불일치:",
    f"{mismatch_count:,}",
)

print(
    "모호해서 매핑 실패:",
    f"{ambiguous_count:,}",
)

print(
    "기존 벡터를 찾지 못한 SQLite 조문:",
    f"{missing_sqlite:,}",
)


for name, values in examples.items():

    if not values:
        continue

    print()
    print(
        f"=== {name} 예시 ==="
    )

    for value in values:
        print(
            value
        )


db.close()
