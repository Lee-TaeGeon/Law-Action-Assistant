import argparse
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = (
    BASE_DIR
    / "korean_law_full_dataset.json"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "law_search.db"
)


# =========================================================
# 텍스트 유틸
# =========================================================

def clean_text(value) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    return text.strip()


def normalize_article_number(
    value,
) -> str:
    """
    조문 번호를 내부 형식으로 정리한다.

    예:
    43       -> 43
    43의2    -> 43의2
    제43조   -> 43
    제43조의2 -> 43의2
    """

    text = clean_text(value)

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    match = re.fullmatch(
        r"제?(\d+)조?(?:의(\d+))?",
        text,
    )

    if match:
        base = match.group(1)
        sub = match.group(2)

        if sub:
            return f"{base}의{sub}"

        return base

    # 현재 데이터에서 예상하지 못한 형식은
    # 임의 변환하지 않고 원본을 보존한다.
    return text


def format_article_number(
    article_no: str,
) -> str:
    article_no = clean_text(
        article_no
    )

    if not article_no:
        return ""

    if "의" in article_no:
        base, sub = article_no.split(
            "의",
            1,
        )

        return f"제{base}조의{sub}"

    return f"제{article_no}조"


# =========================================================
# 조문 본문 생성
# =========================================================

def append_sub_item(
    lines: list[str],
    sub_item: dict,
):
    content = (
        sub_item.get("sub_content")
        or sub_item.get("item_content")
        or sub_item.get("content")
        or ""
    )

    content = clean_text(content)

    if content:
        lines.append(
            f"        {content}"
        )


def append_item(
    lines: list[str],
    item: dict,
):
    content = (
        item.get("item_content")
        or item.get("content")
        or ""
    )

    content = clean_text(content)

    if content:
        lines.append(
            f"    {content}"
        )

    for sub_item in item.get(
        "sub_items",
        [],
    ):
        if isinstance(
            sub_item,
            dict,
        ):
            append_sub_item(
                lines,
                sub_item,
            )


def build_article_body(
    article: dict,
) -> str:
    """
    조문의 paragraphs/items/sub_items를
    하나의 검색용 본문으로 만든다.
    """

    lines = []

    for paragraph in article.get(
        "paragraphs",
        [],
    ):
        if not isinstance(
            paragraph,
            dict,
        ):
            continue

        para_content = clean_text(
            paragraph.get(
                "para_content"
            )
        )

        if para_content:
            lines.append(
                para_content
            )

        for item in paragraph.get(
            "items",
            [],
        ):
            if isinstance(
                item,
                dict,
            ):
                append_item(
                    lines,
                    item,
                )

    return "\n".join(
        lines
    ).strip()


def build_full_content(
    law_name: str,
    article_no: str,
    article_title: str,
    body: str,
) -> str:
    """
    실제 RAG/출처 카드에서 사용할 수 있는
    조문 문자열을 만든다.
    """

    title_line = (
        format_article_number(
            article_no
        )
    )

    if article_title:
        title_line += (
            f" ({article_title})"
        )

    lines = [
        f"<{law_name}>",
        title_line,
    ]

    if body:
        lines.append(
            body
        )

    return "\n".join(
        line
        for line in lines
        if line
    ).strip()


# =========================================================
# 데이터셋 로드
# =========================================================

def load_dataset(
    source_path: Path,
) -> list:
    print(
        f"JSON 로드 중: {source_path}"
    )

    with source_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(
            file
        )

    # 현재 사용 중인 파일은 list 구조지만
    # 혹시 {"data": [...]} 형태여도 대응한다.
    if isinstance(
        dataset,
        dict,
    ):
        dataset = dataset.get(
            "data",
            []
        )

    if not isinstance(
        dataset,
        list,
    ):
        raise ValueError(
            "법령 JSON 최상위 구조가 "
            "list 또는 {'data': [...]} 형식이 아닙니다."
        )

    print(
        f"법령 수: {len(dataset):,}"
    )

    return dataset


# =========================================================
# 조문 정규화
# =========================================================

def prepare_law_articles(
    law: dict,
):
    """
    같은 법령 내부의 조문을 정규화한다.

    중요:
    원본 데이터의 가지번호가

        43
        43
        43

    처럼 저장된 경우:

        43
        43의2
        43의3

    로 보정한다.

    단, 이것은 현재 데이터 오류를 보완하기 위한
    휴리스틱이므로 source_article_no도 함께 보존한다.
    """

    raw_articles = law.get(
        "data",
        [],
    )

    if not isinstance(
        raw_articles,
        list,
    ):
        return

    # 이미 정상적으로 가지번호가 들어 있는 경우
    # 휴리스틱 번호와 충돌하지 않도록 먼저 예약한다.
    reserved_numbers = set()

    for article in raw_articles:
        if not isinstance(
            article,
            dict,
        ):
            continue

        number = normalize_article_number(
            article.get(
                "article_no"
            )
        )

        if (
            number
            and "의" in number
        ):
            reserved_numbers.add(
                number
            )

    # 완전히 같은 원본 레코드 제거
    seen_raw_records = set()

    # 실제 사용된 정규화 조문번호
    used_numbers = set()

    # base 조문이 몇 번 등장했는지 기록
    base_occurrences = {}

    for source_order, article in enumerate(
        raw_articles,
        start=1,
    ):
        if not isinstance(
            article,
            dict,
        ):
            continue

        source_article_no = (
            normalize_article_number(
                article.get(
                    "article_no"
                )
            )
        )

        article_title = clean_text(
            article.get(
                "article_title"
            )
        )

        body = build_article_body(
            article
        )

        # 완전히 비어 있는 placeholder 제거
        if (
            not article_title
            and not body
        ):
            yield {
                "skip_reason":
                    "empty_placeholder"
            }

            continue

        # 조문번호가 아예 없는 경우도
        # 검색 DB에는 넣지 않는다.
        if not source_article_no:
            yield {
                "skip_reason":
                    "missing_article_no"
            }

            continue

        body_hash = hashlib.sha256(
            body.encode(
                "utf-8"
            )
        ).hexdigest()

        raw_signature = (
            source_article_no,
            article_title,
            body_hash,
        )

        # 같은 법령 안에서
        # 번호 + 제목 + 본문이 완전히 동일한 경우 제거
        if raw_signature in seen_raw_records:
            yield {
                "skip_reason":
                    "exact_duplicate"
            }

            continue

        seen_raw_records.add(
            raw_signature
        )

        normalization_method = (
            "original"
        )

        # 이미 가지번호가 있다면 그대로 보존
        if "의" in source_article_no:
            normalized_article_no = (
                source_article_no
            )

        else:
            base = source_article_no

            occurrence = (
                base_occurrences.get(
                    base,
                    0,
                )
                + 1
            )

            base_occurrences[
                base
            ] = occurrence

            if occurrence == 1:
                normalized_article_no = (
                    base
                )

            else:
                branch_no = 2

                while True:
                    candidate = (
                        f"{base}의{branch_no}"
                    )

                    if (
                        candidate
                        not in reserved_numbers
                        and candidate
                        not in used_numbers
                    ):
                        normalized_article_no = (
                            candidate
                        )

                        break

                    branch_no += 1

                normalization_method = (
                    "repeat_branch_heuristic"
                )

        # 혹시 원본 자체가 이상해서
        # 정규화 후에도 번호가 겹친다면
        # 다음 가지번호로 보정한다.
        if (
            normalized_article_no
            in used_numbers
        ):
            base = (
                normalized_article_no
                .split("의", 1)[0]
            )

            branch_no = 2

            while (
                f"{base}의{branch_no}"
                in used_numbers
                or f"{base}의{branch_no}"
                in reserved_numbers
            ):
                branch_no += 1

            normalized_article_no = (
                f"{base}의{branch_no}"
            )

            normalization_method = (
                "collision_heuristic"
            )

        used_numbers.add(
            normalized_article_no
        )

        yield {
            "skip_reason": None,
            "source_order":
                source_order,
            "source_article_no":
                source_article_no,
            "article_no":
                normalized_article_no,
            "article_title":
                article_title,
            "body":
                body,
            "body_hash":
                body_hash,
            "normalization_method":
                normalization_method,
        }


# =========================================================
# SQLite
# =========================================================

def create_schema(
    connection: sqlite3.Connection,
):
    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE laws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            law_name TEXT NOT NULL UNIQUE,
            source_law_id TEXT
        );

        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            law_id INTEGER NOT NULL,
            law_name TEXT NOT NULL,

            source_order INTEGER NOT NULL,

            source_article_no TEXT NOT NULL,
            article_no TEXT NOT NULL,
            article_display TEXT NOT NULL,

            article_title TEXT NOT NULL DEFAULT '',

            body TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,

            content_hash TEXT NOT NULL,

            normalization_method TEXT NOT NULL,

            FOREIGN KEY (law_id)
                REFERENCES laws(id)
        );

        CREATE INDEX idx_articles_law_name
            ON articles(law_name);

        CREATE INDEX idx_articles_law_article
            ON articles(
                law_name,
                article_no
            );

        CREATE INDEX idx_articles_source_article
            ON articles(
                law_name,
                source_article_no
            );

        CREATE INDEX idx_articles_title
            ON articles(
                article_title
            );

        CREATE INDEX idx_articles_hash
            ON articles(
                content_hash
            );
        """
    )

    # 본문 자체를 FTS 테이블에 또 복사하지 않고
    # articles 테이블을 원문 저장소로 사용하는
    # External Content FTS5 구조.
    cursor.execute(
        """
        CREATE VIRTUAL TABLE articles_fts
        USING fts5(
            law_name,
            article_no,
            article_title,
            content,

            content='articles',
            content_rowid='id',

            tokenize='unicode61'
        )
        """
    )

    connection.commit()


# =========================================================
# DB 구축
# =========================================================

def build_database(
    source_path: Path,
    output_path: Path,
    force: bool,
):
    if not source_path.exists():
        raise FileNotFoundError(
            f"원본 JSON이 없습니다: "
            f"{source_path}"
        )

    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"{output_path} 파일이 이미 있습니다.\n"
                "다시 만들려면 --force 옵션을 사용하세요."
            )

        output_path.unlink()

    temp_path = Path(
        str(output_path)
        + ".tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    dataset = load_dataset(
        source_path
    )

    stats = {
        "laws": 0,
        "raw_articles": 0,
        "inserted": 0,
        "empty_placeholder": 0,
        "missing_article_no": 0,
        "exact_duplicate": 0,
        "repeat_branch_heuristic": 0,
        "collision_heuristic": 0,
    }

    connection = sqlite3.connect(
        temp_path
    )

    try:
        connection.execute(
            "PRAGMA journal_mode=OFF"
        )

        connection.execute(
            "PRAGMA synchronous=OFF"
        )

        connection.execute(
            "PRAGMA temp_store=MEMORY"
        )

        connection.execute(
            "PRAGMA foreign_keys=ON"
        )

        create_schema(
            connection
        )

        cursor = connection.cursor()

        for law_index, law in enumerate(
            dataset,
            start=1,
        ):
            if not isinstance(
                law,
                dict,
            ):
                continue

            law_name = clean_text(
                law.get(
                    "law_name"
                )
            )

            if not law_name:
                continue

            source_law_id = clean_text(
                law.get(
                    "law_id"
                )
            )

            cursor.execute(
                """
                INSERT INTO laws (
                    law_name,
                    source_law_id
                )
                VALUES (?, ?)
                """,
                (
                    law_name,
                    source_law_id,
                ),
            )

            law_id = (
                cursor.lastrowid
            )

            stats["laws"] += 1

            raw_data = law.get(
                "data",
                [],
            )

            if isinstance(
                raw_data,
                list,
            ):
                stats[
                    "raw_articles"
                ] += len(
                    raw_data
                )

            for prepared in (
                prepare_law_articles(
                    law
                )
            ):
                skip_reason = (
                    prepared.get(
                        "skip_reason"
                    )
                )

                if skip_reason:
                    stats[
                        skip_reason
                    ] += 1

                    continue

                article_no = (
                    prepared[
                        "article_no"
                    ]
                )

                article_title = (
                    prepared[
                        "article_title"
                    ]
                )

                body = prepared[
                    "body"
                ]

                full_content = (
                    build_full_content(
                        law_name,
                        article_no,
                        article_title,
                        body,
                    )
                )

                cursor.execute(
                    """
                    INSERT INTO articles (
                        law_id,
                        law_name,

                        source_order,

                        source_article_no,
                        article_no,
                        article_display,

                        article_title,

                        body,
                        content,

                        content_hash,

                        normalization_method
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        law_id,
                        law_name,

                        prepared[
                            "source_order"
                        ],

                        prepared[
                            "source_article_no"
                        ],
                        article_no,
                        format_article_number(
                            article_no
                        ),

                        article_title,

                        body,
                        full_content,

                        hashlib.sha256(
                            full_content.encode(
                                "utf-8"
                            )
                        ).hexdigest(),

                        prepared[
                            "normalization_method"
                        ],
                    ),
                )

                stats[
                    "inserted"
                ] += 1

                method = prepared[
                    "normalization_method"
                ]

                if method in stats:
                    stats[
                        method
                    ] += 1

            if law_index % 100 == 0:
                connection.commit()

                print(
                    f"{law_index:,} / "
                    f"{len(dataset):,} 법령 처리 "
                    f"| 조문 {stats['inserted']:,}"
                )

        connection.commit()

        print()
        print(
            "FTS5 인덱스 생성 중..."
        )

        # External content 테이블의 내용을 기반으로
        # FTS 인덱스 생성
        connection.execute(
            """
            INSERT INTO articles_fts(
                articles_fts
            )
            VALUES ('rebuild')
            """
        )

        connection.commit()

        print(
            "SQLite ANALYZE 실행 중..."
        )

        connection.execute(
            "ANALYZE"
        )

        connection.commit()

        # 운영용 설정으로 복원
        connection.execute(
            "PRAGMA journal_mode=DELETE"
        )

        connection.execute(
            "PRAGMA synchronous=NORMAL"
        )

        # DB 무결성 확인
        result = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()

        if (
            not result
            or result[0] != "ok"
        ):
            raise RuntimeError(
                "SQLite integrity_check 실패: "
                f"{result}"
            )

    except Exception:
        connection.close()

        if temp_path.exists():
            temp_path.unlink()

        raise

    finally:
        try:
            connection.close()
        except Exception:
            pass

    os.replace(
        temp_path,
        output_path,
    )

    size_mb = (
        output_path.stat().st_size
        / 1024
        / 1024
    )

    print()
    print(
        "=" * 60
    )

    print(
        "법령 검색 DB 생성 완료"
    )

    print(
        "=" * 60
    )

    print(
        f"법령 수: "
        f"{stats['laws']:,}"
    )

    print(
        f"원본 조문 수: "
        f"{stats['raw_articles']:,}"
    )

    print(
        f"저장 조문 수: "
        f"{stats['inserted']:,}"
    )

    print(
        f"빈 placeholder 제거: "
        f"{stats['empty_placeholder']:,}"
    )

    print(
        f"조문번호 없음 제거: "
        f"{stats['missing_article_no']:,}"
    )

    print(
        f"완전 중복 제거: "
        f"{stats['exact_duplicate']:,}"
    )

    print(
        f"가지번호 휴리스틱 적용: "
        f"{stats['repeat_branch_heuristic']:,}"
    )

    print(
        f"번호 충돌 보정: "
        f"{stats['collision_heuristic']:,}"
    )

    print(
        f"DB 크기: "
        f"{size_mb:,.1f} MB"
    )

    print(
        f"DB 위치: "
        f"{output_path}"
    )


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "대한민국 법령 JSON을 "
            "검색용 SQLite DB로 변환합니다."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "원본 korean_law_full_dataset.json 경로"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "생성할 SQLite DB 경로"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "기존 출력 DB가 있으면 덮어씁니다."
        ),
    )

    args = parser.parse_args()

    build_database(
        source_path=args.source.resolve(),
        output_path=args.output.resolve(),
        force=args.force,
    )


if __name__ == "__main__":
    main()