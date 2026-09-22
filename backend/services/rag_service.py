import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import (
    maximal_marginal_relevance,
)


# =========================================================
# 경로
# =========================================================

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.getenv(
        "LAW_DATA_DIR",
        str(BASE_DIR),
    )
)

VECTOR_DB_PATH = (
    DATA_DIR
    / "law_db_optimized"
)

VECTOR_COLLECTION_NAME = (
    "law_articles"
)

SEARCH_DB_PATH = (
    DATA_DIR
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
# ChromaDB / Embedding
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Vector 검색에 사용할 임베딩 모델.

    기존 법령 Vector DB를 만들 때와 동일하게
    jhgan/ko-sroberta-multitask 모델을 사용하고
    embedding을 L2 norm 1로 정규화한다.
    """

    return HuggingFaceEmbeddings(
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


@lru_cache(maxsize=1)
def get_vector_db():
    """
    최적화된 Chroma Vector DB의
    law_articles collection을 반환한다.

    새 Vector DB에는
    법령 본문이나 metadata를 넣지 않고
    SQLite articles.id를 Chroma id로 사용한다.
    """

    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            "Vector DB가 없습니다: "
            f"{VECTOR_DB_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(
            VECTOR_DB_PATH
        )
    )

    try:
        collection = (
            client.get_collection(
                VECTOR_COLLECTION_NAME
            )
        )

    except Exception as exc:
        raise RuntimeError(
            "Vector collection을 "
            "불러올 수 없습니다: "
            f"{VECTOR_COLLECTION_NAME}"
        ) from exc

    return collection


# =========================================================
# MMR
# =========================================================

def cosine_similarity_matrix(
    a,
    b,
):
    """
    a: (N, D)
    b: (M, D)

    cosine similarity matrix:
    (N, M)
    """

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    if a.ndim == 1:
        a = a.reshape(
            1,
            -1,
        )

    if b.ndim == 1:
        b = b.reshape(
            1,
            -1,
        )

    a_norm = np.linalg.norm(
        a,
        axis=1,
        keepdims=True,
    )

    b_norm = np.linalg.norm(
        b,
        axis=1,
        keepdims=True,
    )

    # 0으로 나누는 것을 방지한다.
    a_norm = np.maximum(
        a_norm,
        1e-12,
    )

    b_norm = np.maximum(
        b_norm,
        1e-12,
    )

    return (
        (a / a_norm)
        @ (b / b_norm).T
    )


def maximal_marginal_relevance(
    query_embedding,
    candidate_embeddings,
    k: int,
    lambda_mult: float = 0.75,
):
    """
    기존 LangChain Chroma Retriever에서 사용하던
    MMR(Maximal Marginal Relevance)을
    직접 구현한다.

    관련성뿐 아니라 이미 선택된 문서와의
    중복성도 함께 고려한다.
    """

    candidate_embeddings = np.asarray(
        candidate_embeddings,
        dtype=np.float32,
    )

    if (
        candidate_embeddings.ndim != 2
        or len(candidate_embeddings) == 0
        or k <= 0
    ):
        return []

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    k = min(
        k,
        len(candidate_embeddings),
    )

    # Query ↔ 각 후보의 similarity
    similarity_to_query = (
        cosine_similarity_matrix(
            candidate_embeddings,
            query_embedding,
        )[:, 0]
    )

    # 첫 번째는 Query와 가장 유사한 후보
    first_index = int(
        np.argmax(
            similarity_to_query
        )
    )

    selected = [
        first_index
    ]

    selected_set = {
        first_index
    }

    while len(selected) < k:

        selected_vectors = (
            candidate_embeddings[
                selected
            ]
        )

        # 각 후보와 이미 선택된 후보 간 유사도
        similarity_to_selected = (
            cosine_similarity_matrix(
                candidate_embeddings,
                selected_vectors,
            )
        )

        best_index = None
        best_score = -float(
            "inf"
        )

        for index in range(
            len(candidate_embeddings)
        ):
            if index in selected_set:
                continue

            redundancy = float(
                np.max(
                    similarity_to_selected[
                        index
                    ]
                )
            )

            relevance = float(
                similarity_to_query[
                    index
                ]
            )

            score = (
                lambda_mult
                * relevance
                - (
                    1.0
                    - lambda_mult
                )
                * redundancy
            )

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            break

        selected.append(
            best_index
        )

        selected_set.add(
            best_index
        )

    return selected


# =========================================================
# Vector 문서 정보 추출
# =========================================================
#
# 아래 함수들은 기존 law_db_full용으로 사용했던
# 보정 함수다.
#
# 새 law_db_optimized에서는 Chroma id 자체가
# SQLite articles.id이므로 더 이상 사용하지 않는다.
#
# 현재 전환 테스트 단계에서는 기존 함수가 파일에
# 남아 있어도 문제없다.
# 테스트가 끝난 뒤 삭제해도 된다.
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
#
# 기존 DB 호환용 함수.
#
# 새 Vector 검색에서는 사용하지 않는다.
# 기존 코드를 당장 삭제하지 않기 위해 남겨둔다.
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

    for row in rows:

        if (
            row["article_no"]
            == raw_article_no
        ):
            return row_to_source(
                row,
                "vector",
            )

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
    최적화된 Chroma Vector DB를 이용해
    의미 기반 법령 검색을 수행한다.

    구조:

    질문
      ↓
    ko-sroberta embedding
      ↓
    Chroma L2 후보 검색
      ↓
    MMR 재정렬
      ↓
    article_id
      ↓
    law_search.db 조회

    Vector DB에는 본문이나 metadata를 저장하지 않고
    SQLite articles.id만 Chroma id로 사용한다.
    """

    if k <= 0:
        return []

    vector_db = (
        get_vector_db()
    )

    embedding_model = (
        get_embedding_model()
    )

    query = question.strip()

    # 기존 LangChain Retriever 설정과 동일
    fetch_k = max(
        20,
        k * 3,
    )

    lambda_mult = 0.75

    # 기존 Vector DB 생성 방식과 동일하게
    # normalize_embeddings=True 사용
    query_embedding = (
        embedding_model.embed_query(
            query
        )
    )

    # -----------------------------------------------------
    # 1. Chroma에서 후보 검색
    # -----------------------------------------------------

    raw_results = (
        vector_db.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=fetch_k,
            include=[
                "embeddings",
                "distances",
            ],
        )
    )

    result_ids = raw_results.get(
        "ids"
    )

    if (
        result_ids is None
        or len(result_ids) == 0
        or len(result_ids[0]) == 0
    ):
        return []

    candidate_ids = (
        result_ids[0]
    )

    result_embeddings = (
        raw_results.get(
            "embeddings"
        )
    )

    if result_embeddings is None:
        return []

    candidate_vectors = np.asarray(
        result_embeddings[0],
        dtype=np.float32,
    )

    if (
        candidate_vectors.ndim != 2
        or len(candidate_vectors) == 0
    ):
        return []

    # -----------------------------------------------------
    # 2. MMR
    # -----------------------------------------------------

    selected_indexes = (
        maximal_marginal_relevance(
            query_embedding=(
                query_embedding
            ),
            candidate_embeddings=(
                candidate_vectors
            ),
            k=k,
            lambda_mult=(
                lambda_mult
            ),
        )
    )

    if not selected_indexes:
        return []

    # -----------------------------------------------------
    # 3. article_id → SQLite
    # -----------------------------------------------------

    db = (
        get_search_connection()
    )

    results = []

    seen_article_ids = set()

    try:

        for index in selected_indexes:

            if (
                index < 0
                or index
                >= len(candidate_ids)
            ):
                continue

            raw_article_id = (
                candidate_ids[
                    index
                ]
            )

            try:
                article_id = int(
                    raw_article_id
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                article_id
                in seen_article_ids
            ):
                continue

            seen_article_ids.add(
                article_id
            )

            row = db.execute(
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

                WHERE id = ?
                """,
                (
                    article_id,
                ),
            ).fetchone()

            if row is None:
                continue

            results.append(
                row_to_source(
                    row,
                    "vector",
                )
            )

            if len(results) >= k:
                break

    finally:
        db.close()

    return results