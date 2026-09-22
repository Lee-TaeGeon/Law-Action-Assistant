import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =========================================================
# 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = (
    BASE_DIR
    / "law_db_full"
)

SEARCH_DB_PATH = (
    BASE_DIR
    / "law_search.db"
)


# =========================================================
# SQLite
# =========================================================

def get_search_connection():
    """
    법령 검색용 SQLite DB를 읽기 전용으로 연다.

    요청마다 연결하고 닫기 때문에
    FastAPI 멀티스레드 환경에서도 안전하게 사용한다.
    """

    if not SEARCH_DB_PATH.exists():
        raise FileNotFoundError(
            f"법령 검색 DB가 없습니다: "
            f"{SEARCH_DB_PATH}"
        )

    uri = (
        f"file:{SEARCH_DB_PATH}"
        "?mode=ro"
    )

    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=30,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# =========================================================
# 법령명
# =========================================================

@lru_cache(maxsize=1)
def get_law_names():
    """
    Exact Search에서 사용할 법령명을
    SQLite에서 최초 1회 읽는다.

    긴 법령명을 먼저 검사한다.
    """

    db = get_search_connection()

    try:
        rows = db.execute(
            """
            SELECT law_name
            FROM laws
            ORDER BY
                LENGTH(law_name) DESC
            """
        ).fetchall()

        return [
            row["law_name"]
            for row in rows
        ]

    finally:
        db.close()


# =========================================================
# 문자열 정규화
# =========================================================

def normalize_title(
    title: str | None,
):
    """
    제목 비교용 정규화.

    사기죄 -> 사기
    """

    if not title:
        return ""

    title = re.sub(
        r"\s+",
        "",
        str(title),
    )

    if title.endswith("죄"):
        title = title[:-1]

    return title


def compact_text(
    text: str | None,
):
    if not text:
        return ""

    return re.sub(
        r"\s+",
        "",
        str(text),
    )


def normalize_article_number(
    article_no: str | None,
):
    """
    제347조       -> 347
    제347조의2    -> 347의2
    347           -> 347
    347의2        -> 347의2
    """

    if article_no is None:
        return ""

    text = re.sub(
        r"\s+",
        "",
        str(article_no),
    )

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


def format_article_number(
    article_no: str | None,
):
    """
    347 -> 제347조
    347의2 -> 제347조의2
    """

    article_no = (
        normalize_article_number(
            article_no
        )
    )

    if not article_no:
        return None

    if "의" in article_no:
        base, sub = article_no.split(
            "의",
            1,
        )

        return (
            f"제{base}조의{sub}"
        )

    return f"제{article_no}조"


# =========================================================
# 사용자 질문에서 명시된 법령 + 조문 추출
# =========================================================

def extract_explicit_law_refs(
    query: str,
):
    """
    형법 제347조(사기죄)

    ->

    {
        law_name: 형법,
        article_no: 347,
        title_hint: 사기죄
    }
    """

    refs = []

    for law_name in get_law_names():

        if law_name not in query:
            continue

        pattern = (
            rf"{re.escape(law_name)}\s*"
            rf"제\s*(\d+)\s*조"
            rf"(?:의\s*(\d+))?"
            rf"(?:\s*\(([^)]+)\))?"
        )

        for match in re.finditer(
            pattern,
            query,
        ):
            article_no = (
                match.group(1)
            )

            if match.group(2):
                article_no += (
                    f"의{match.group(2)}"
                )

            refs.append(
                {
                    "law_name":
                        law_name,

                    "article_no":
                        article_no,

                    "title_hint":
                        match.group(3),
                }
            )

    return refs


# =========================================================
# SQLite Row -> RAG Source
# =========================================================

def row_to_source(
    row,
    source_type: str,
):
    return {
        "law_name":
            row["law_name"],

        "article":
            row["article_display"],

        "article_title":
            row["article_title"],

        "content":
            row["content"],

        "source_type":
            source_type,
    }


# =========================================================
# Exact Search
# =========================================================

def exact_search_laws(
    query: str,
):
    """
    사용자가 직접 명시한 법령/조문을 검색한다.

    예:
    형법 제347조
    근로기준법 제43조의2
    """

    refs = (
        extract_explicit_law_refs(
            query
        )
    )

    if not refs:
        return []

    db = get_search_connection()

    results = []

    try:
        for ref in refs:

            rows = db.execute(
                """
                SELECT
                    law_name,
                    article_no,
                    article_display,
                    article_title,
                    content

                FROM articles

                WHERE law_name = ?
                  AND article_no = ?

                ORDER BY source_order

                LIMIT 10
                """,
                (
                    ref["law_name"],
                    ref["article_no"],
                ),
            ).fetchall()

            if not rows:
                continue

            title_hint = (
                normalize_title(
                    ref.get(
                        "title_hint"
                    )
                )
            )

            selected_rows = rows

            if title_hint:

                title_matches = [
                    row
                    for row in rows
                    if normalize_title(
                        row[
                            "article_title"
                        ]
                    )
                    == title_hint
                ]

                if title_matches:
                    selected_rows = (
                        title_matches
                    )

            for row in selected_rows:
                results.append(
                    row_to_source(
                        row,
                        "exact",
                    )
                )

    finally:
        db.close()

    return results


# =========================================================
# Keyword Search
# =========================================================

def split_search_phrases(
    query: str,
):
    """
    Query Rewriter 출력:

    임금체불, 임금 지급,
    미지급 임금, 체불임금

    형태를 검색 표현 단위로 분리한다.
    """

    raw_phrases = re.split(
        r"[,;\n]+",
        query,
    )

    phrases = []

    seen = set()

    for phrase in raw_phrases:

        phrase = phrase.strip()

        phrase = re.sub(
            r"^[\-\*\d\.\)\s]+",
            "",
            phrase,
        )

        if not phrase:
            continue

        key = compact_text(
            phrase
        )

        if (
            not key
            or key in seen
        ):
            continue

        seen.add(key)

        phrases.append(
            phrase
        )

        if len(phrases) >= 10:
            break

    if not phrases:
        query = query.strip()

        if query:
            phrases.append(
                query
            )

    return phrases


def extract_search_terms(
    phrase: str,
):
    """
    FTS에 안전하게 넣을 수 있는 검색어만 추출한다.
    """

    terms = re.findall(
        r"[0-9A-Za-z가-힣]+",
        phrase,
    )

    cleaned = []

    seen = set()

    for term in terms:

        term = normalize_title(
            term
        )

        if not term:
            continue

        # 한 글자 검색어는 노이즈가 너무 많으므로 제외.
        if (
            len(term) < 2
            and not term.isdigit()
        ):
            continue

        if term in seen:
            continue

        seen.add(term)

        cleaned.append(
            term
        )

        if len(cleaned) >= 6:
            break

    return cleaned


def build_fts_query(
    terms,
    operator="AND",
):
    if not terms:
        return ""

    return (
        f" {operator} ".join(
            f"{term}*"
            for term in terms
        )
    )


def calculate_title_priority(
    article_title: str,
    phrase: str,
):
    """
    검색 결과 우선순위.

    0 = 제목 완전 일치
    1 = 제목에 검색 표현 포함
    2 = 검색 단어가 제목에 포함
    3 = 본문 등에만 등장
    """

    title = normalize_title(
        article_title
    )

    phrase_normalized = (
        normalize_title(
            phrase
        )
    )

    if (
        title
        and title
        == phrase_normalized
    ):
        return 0

    if (
        title
        and phrase_normalized
        and phrase_normalized
        in title
    ):
        return 1

    terms = extract_search_terms(
        phrase
    )

    for term in terms:
        if term in title:
            return 2

    return 3


def keyword_search_laws(
    query: str,
    k: int = 12,
):
    """
    SQLite FTS5 + 조문 제목 직접 검색.

    한국어 띄어쓰기 차이도 보완한다.

    예:
    임금지급
        ->
    임금 지급

    재산분할
        ->
    재산분할청구권
    """

    phrases = split_search_phrases(
        query
    )

    if not phrases:
        return []

    db = get_search_connection()

    candidates = {}

    def add_candidate(
        row,
        phrase,
        score=0.0,
    ):
        article_id = row["id"]

        priority = (
            calculate_title_priority(
                row["article_title"],
                phrase,
            )
        )

        if article_id not in candidates:

            candidates[
                article_id
            ] = {
                "law_name":
                    row["law_name"],

                "article_no":
                    row["article_no"],

                "article_title":
                    row["article_title"],

                "content":
                    row["content"],

                "match_count":
                    0,

                "best_priority":
                    99,

                "best_score":
                    float("inf"),
            }

        candidate = (
            candidates[
                article_id
            ]
        )

        candidate[
            "match_count"
        ] += 1

        candidate[
            "best_priority"
        ] = min(
            candidate[
                "best_priority"
            ],
            priority,
        )

        candidate[
            "best_score"
        ] = min(
            candidate[
                "best_score"
            ],
            score,
        )

    try:
        for phrase in phrases:

            # =================================================
            # 1. 조문 제목 직접 검색
            #
            # 공백 제거 후 비교한다.
            #
            # 임금지급
            # 임금 지급
            #
            # 둘을 같은 표현으로 처리한다.
            # =================================================

            normalized_phrase = (
                compact_text(
                    phrase
                )
            )

            if normalized_phrase:

                title_rows = db.execute(
                    """
                    SELECT
                        id,
                        law_name,
                        article_no,
                        article_title,
                        content

                    FROM articles

                    WHERE
                        REPLACE(
                            article_title,
                            ' ',
                            ''
                        ) LIKE ?

                    ORDER BY
                        CASE
                            WHEN REPLACE(
                                article_title,
                                ' ',
                                ''
                            ) = ?
                            THEN 0

                            ELSE 1
                        END,

                        LENGTH(
                            article_title
                        ) ASC

                    LIMIT 30
                    """,
                    (
                        f"%{normalized_phrase}%",
                        normalized_phrase,
                    ),
                ).fetchall()

                for row in title_rows:

                    add_candidate(
                        row,
                        phrase,
                        score=-100.0,
                    )

            # =================================================
            # 2. FTS5 검색
            # =================================================

            terms = extract_search_terms(
                phrase
            )

            if not terms:
                continue

            fts_query = (
                build_fts_query(
                    terms,
                    operator="AND",
                )
            )

            try:
                rows = db.execute(
                    """
                    SELECT
                        rowid AS id,
                        law_name,
                        article_no,
                        article_title,
                        content,

                        bm25(
                            articles_fts,
                            3.0,
                            1.0,
                            8.0,
                            1.0
                        ) AS score

                    FROM articles_fts

                    WHERE
                        articles_fts
                        MATCH ?

                    ORDER BY
                        score ASC

                    LIMIT 30
                    """,
                    (
                        fts_query,
                    ),
                ).fetchall()

            except sqlite3.OperationalError:
                rows = []

            # =================================================
            # AND 결과가 없으면 OR 검색
            # =================================================

            if (
                not rows
                and len(terms) > 1
            ):
                fallback_query = (
                    build_fts_query(
                        terms,
                        operator="OR",
                    )
                )

                try:
                    rows = db.execute(
                        """
                        SELECT
                            rowid AS id,
                            law_name,
                            article_no,
                            article_title,
                            content,

                            bm25(
                                articles_fts,
                                3.0,
                                1.0,
                                8.0,
                                1.0
                            ) AS score

                        FROM articles_fts

                        WHERE
                            articles_fts
                            MATCH ?

                        ORDER BY
                            score ASC

                        LIMIT 20
                        """,
                        (
                            fallback_query,
                        ),
                    ).fetchall()

                except sqlite3.OperationalError:
                    rows = []

            for row in rows:

                add_candidate(
                    row,
                    phrase,
                    score=row["score"],
                )

    finally:
        db.close()

    # =====================================================
    # 최종 Ranking
    #
    # 1. 조문 제목 직접 일치
    # 2. 여러 검색 표현에서 반복 등장
    # 3. BM25
    # =====================================================

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            item[
                "best_priority"
            ],

            -item[
                "match_count"
            ],

            item[
                "best_score"
            ],
        ),
    )

    results = []

    for item in ranked[:k]:

        results.append(
            {
                "law_name":
                    item["law_name"],

                "article":
                    format_article_number(
                        item[
                            "article_no"
                        ]
                    ),

                "article_title":
                    item[
                        "article_title"
                    ],

                "content":
                    item[
                        "content"
                    ],

                "source_type":
                    "keyword",
            }
        )

    return results


# =========================================================
# ChromaDB
# =========================================================

@lru_cache(maxsize=1)
def get_vector_db():
    """
    기존 Chroma 벡터 DB는 그대로 사용한다.
    """

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=(
                "jhgan/"
                "ko-sroberta-multitask"
            ),
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings":
                    True,
            },
        )
    )

    return Chroma(
        persist_directory=str(
            VECTOR_DB_PATH
        ),
        embedding_function=(
            embeddings
        ),
    )


# =========================================================
# Vector 문서 정보 추출
# =========================================================

def extract_article_number_from_content(
    content: str,
):
    match = re.search(
        r"제\s*(\d+)\s*조"
        r"(?:의\s*(\d+))?",
        content or "",
    )

    if not match:
        return ""

    article_no = match.group(1)

    if match.group(2):
        article_no += (
            f"의{match.group(2)}"
        )

    return article_no


def extract_article_title_from_content(
    content: str,
):
    match = re.search(
        r"제\s*\d+\s*조"
        r"(?:의\s*\d+)?"
        r"\s*\(([^)]*)\)",
        content or "",
    )

    if not match:
        return ""

    return (
        match.group(1)
        .strip()
    )


# =========================================================
# Vector 결과 -> SQLite의 정규화된 조문으로 연결
# =========================================================

def resolve_vector_source(
    db,
    doc,
):
    law_name = (
        doc.metadata.get(
            "law_name"
        )
        or doc.metadata.get(
            "title"
        )
        or "관련 법령"
    )

    law_name = str(
        law_name
    ).strip()

    raw_article_no = (
        doc.metadata.get(
            "article_no"
        )
    )

    if raw_article_no is None:
        raw_article_no = (
            extract_article_number_from_content(
                doc.page_content
            )
        )

    raw_article_no = (
        normalize_article_number(
            raw_article_no
        )
    )

    vector_title = (
        extract_article_title_from_content(
            doc.page_content
        )
    )

    if (
        not law_name
        or not raw_article_no
    ):
        return {
            "law_name":
                law_name,

            "article":
                format_article_number(
                    raw_article_no
                ),

            "article_title":
                vector_title,

            "content":
                doc.page_content,

            "source_type":
                "vector",
        }

    rows = db.execute(
        """
        SELECT
            law_name,
            source_article_no,
            article_no,
            article_display,
            article_title,
            content,
            source_order

        FROM articles

        WHERE law_name = ?
          AND (
                source_article_no = ?
                OR article_no = ?
              )

        ORDER BY source_order
        """,
        (
            law_name,
            raw_article_no,
            raw_article_no,
        ),
    ).fetchall()

    if not rows:
        return {
            "law_name":
                law_name,

            "article":
                format_article_number(
                    raw_article_no
                ),

            "article_title":
                vector_title,

            "content":
                doc.page_content,

            "source_type":
                "vector",
        }

    # 가지번호가 원본에서 손상된 경우
    # Chroma 문서의 조문 제목으로 정확한 조문을 찾는다.
    normalized_vector_title = (
        normalize_title(
            vector_title
        )
    )

    if normalized_vector_title:

        for row in rows:

            if (
                normalize_title(
                    row[
                        "article_title"
                    ]
                )
                == normalized_vector_title
            ):
                return row_to_source(
                    row,
                    "vector",
                )

    # 제목이 정확히 추출되지 않은 경우
    # Chroma 본문에 조문 제목이 포함되는지 검사한다.
    compact_document = (
        compact_text(
            doc.page_content
        )
    )

    for row in rows:

        title = compact_text(
            row[
                "article_title"
            ]
        )

        if (
            title
            and title
            in compact_document
        ):
            return row_to_source(
                row,
                "vector",
            )

    # 이미 정상적인 가지번호가 있는 경우 우선.
    for row in rows:

        if (
            row["article_no"]
            == raw_article_no
        ):
            return row_to_source(
                row,
                "vector",
            )

    # 마지막 fallback.
    return row_to_source(
        rows[0],
        "vector",
    )


# =========================================================
# Vector / MMR Search
# =========================================================

def search_laws(
    question: str,
    category: str,
    k: int = 4,
):
    """
    기존 ChromaDB MMR 검색.

    단, 검색 결과의 실제 조문 내용과
    정규화된 가지번호는 SQLite에서 다시 가져온다.
    """

    vector_db = (
        get_vector_db()
    )

    query = f"""
법률 분야: {category}

사용자 질문:
{question}

이 질문을 해결하는 데 직접적으로 관련된
대한민국 법령과 조문
"""

    retriever = (
        vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,

                "fetch_k": max(
                    20,
                    k * 3,
                ),

                "lambda_mult":
                    0.75,
            },
        )
    )

    docs = retriever.invoke(
        query
    )

    db = get_search_connection()

    results = []

    seen = set()

    try:
        for doc in docs:

            source = (
                resolve_vector_source(
                    db,
                    doc,
                )
            )

            key = (
                source.get(
                    "law_name",
                    "",
                ),
                source.get(
                    "article",
                    "",
                ),
            )

            if key in seen:
                continue

            seen.add(key)

            results.append(
                source
            )

            if len(results) >= k:
                break

    finally:
        db.close()

    return results