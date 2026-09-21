import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =========================================================
# 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DB_PATH = BASE_DIR / "law_db_full"

DATASET_PATH = BASE_DIR / "korean_law_full_dataset.json"


# =========================================================
# 법령 JSON 로드
# =========================================================

@lru_cache(maxsize=1)
def get_law_dataset():
    """
    전체 법령 JSON 데이터를 최초 1회만 메모리에 로드한다.
    """
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_law_index():
    """
    법령명을 key로 빠르게 조회할 수 있는 인덱스.
    """
    dataset = get_law_dataset()

    return {
        law.get("law_name"): law
        for law in dataset
        if law.get("law_name")
    }


@lru_cache(maxsize=1)
def get_law_names():
    """
    긴 법령명을 먼저 검사할 수 있도록 길이순으로 정렬한다.
    """
    return sorted(
        get_law_index().keys(),
        key=len,
        reverse=True,
    )


# =========================================================
# 문자열 정규화
# =========================================================

def normalize_title(title: str | None):
    """
    예:
    사기죄 -> 사기
    """

    if not title:
        return ""

    title = re.sub(
        r"\s+",
        "",
        title,
    )

    if title.endswith("죄"):
        title = title[:-1]

    return title


def format_article_number(article_no: str):
    """
    347     -> 제347조
    347의2  -> 제347조의2
    """

    article_no = str(article_no).strip()

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
# 유실된 가지번호 복원
# =========================================================

def is_empty_article(article: dict):
    """
    데이터 파싱 과정에서 생긴 빈 placeholder인지 확인한다.
    """

    title = str(
        article.get(
            "article_title",
            "",
        )
    ).strip()

    paragraphs = article.get(
        "paragraphs",
        [],
    )

    return (
        not title
        and not paragraphs
    )


def get_normalized_articles(law: dict):
    """
    데이터셋에서 유실된 가지번호를 복원한다.

    예:

    43 / 임금 지급
    43 / 체불사업주 명단 공개
    43 / 임금등 체불자료의 제공

    ->

    43
    43의2
    43의3
    """

    results = []

    previous_base = None
    sequence = 0

    for article in law.get(
        "data",
        [],
    ):
        base_no = str(
            article.get(
                "article_no",
                "",
            )
        ).strip()

        if not base_no:
            continue

        # 빈 placeholder는 제외
        if is_empty_article(article):
            continue

        # 이미 가지번호가 정상적으로 들어있는 경우
        if "의" in base_no:
            normalized_no = base_no

            previous_base = None
            sequence = 0

        else:
            if base_no != previous_base:
                previous_base = base_no
                sequence = 1
            else:
                sequence += 1

            if sequence == 1:
                normalized_no = base_no
            else:
                normalized_no = (
                    f"{base_no}의{sequence}"
                )

        normalized_article = dict(
            article
        )

        normalized_article[
            "normalized_article_no"
        ] = normalized_no

        results.append(
            normalized_article
        )

    return results


def get_article_no(article: dict):
    """
    정규화된 조문번호를 우선 사용한다.
    """

    return str(
        article.get(
            "normalized_article_no",
            article.get(
                "article_no",
                "",
            ),
        )
    ).strip()


# =========================================================
# JSON 조문 -> 문자열 변환
# =========================================================

def format_exact_article(
    law_name: str,
    article: dict,
):
    """
    JSON 조문 데이터를 LLM이 사용할 문자열로 변환한다.
    """

    article_no = get_article_no(
        article
    )

    article_title = article.get(
        "article_title",
        "",
    )

    lines = [
        f"<{law_name}>",
        (
            f"{format_article_number(article_no)} "
            f"({article_title})"
        ),
    ]

    for paragraph in article.get(
        "paragraphs",
        [],
    ):
        para_content = paragraph.get(
            "para_content"
        )

        if para_content:
            lines.append(
                para_content
            )

        for item in paragraph.get(
            "items",
            [],
        ):
            item_content = item.get(
                "item_content"
            )

            if item_content:
                lines.append(
                    f"    {item_content}"
                )

            for sub_item in item.get(
                "sub_items",
                [],
            ):
                sub_content = sub_item.get(
                    "sub_content"
                )

                if sub_content:
                    lines.append(
                        f"        {sub_content}"
                    )

    return "\n".join(
        lines
    )


# =========================================================
# 사용자 질문에서 법령 + 조문 추출
# =========================================================

def extract_explicit_law_refs(
    query: str,
):
    """
    예:

    형법 제347조(사기죄)

    ->

    {
        "law_name": "형법",
        "article_no": "347",
        "title_hint": "사기죄"
    }
    """

    refs = []

    law_names = get_law_names()

    for law_name in law_names:

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
            article_no = match.group(1)

            if match.group(2):
                article_no += (
                    f"의{match.group(2)}"
                )

            refs.append(
                {
                    "law_name": law_name,
                    "article_no": article_no,
                    "title_hint": match.group(3),
                }
            )

    return refs


# =========================================================
# Exact Search
# =========================================================

def exact_search_laws(
    query: str,
):
    """
    사용자가 직접 입력한 법령/조문만 정확 조회한다.

    Query Rewriter가 생성한 추정 조문은 사용하지 않는다.
    """

    refs = extract_explicit_law_refs(
        query
    )

    law_index = get_law_index()

    results = []

    normalized_query = normalize_title(
        query
    )

    for ref in refs:

        law_name = ref[
            "law_name"
        ]

        target_no = ref[
            "article_no"
        ]

        title_hint = normalize_title(
            ref.get(
                "title_hint"
            )
        )

        law = law_index.get(
            law_name
        )

        if not law:
            continue

        article_candidates = []

        for article in get_normalized_articles(
            law
        ):
            article_no = get_article_no(
                article
            )

            if article_no == target_no:
                article_candidates.append(
                    article
                )

        if not article_candidates:
            continue

        # 제목 힌트가 있을 경우
        if title_hint:

            title_matches = [
                article
                for article
                in article_candidates
                if normalize_title(
                    article.get(
                        "article_title"
                    )
                )
                == title_hint
            ]

            if title_matches:
                article_candidates = (
                    title_matches
                )

        # 같은 번호 후보가 여러 개일 경우
        elif len(
            article_candidates
        ) > 1:

            query_matches = []

            for article in article_candidates:

                article_title = (
                    normalize_title(
                        article.get(
                            "article_title"
                        )
                    )
                )

                if (
                    article_title
                    and article_title
                    in normalized_query
                ):
                    query_matches.append(
                        article
                    )

            if query_matches:

                query_matches.sort(
                    key=lambda article: len(
                        normalize_title(
                            article.get(
                                "article_title"
                            )
                        )
                    ),
                    reverse=True,
                )

                article_candidates = [
                    query_matches[0]
                ]

            else:
                article_candidates = [
                    article_candidates[0]
                ]

        for article in article_candidates:

            article_no = get_article_no(
                article
            )

            article_title = article.get(
                "article_title",
                "",
            )

            results.append(
                {
                    "law_name": law_name,
                    "article": (
                        format_article_number(
                            article_no
                        )
                    ),
                    "article_title": (
                        article_title
                    ),
                    "content": (
                        format_exact_article(
                            law_name,
                            article,
                        )
                    ),
                    "source_type": "exact",
                }
            )

    return results


# =========================================================
# Keyword Search
# =========================================================

KEYWORD_STOPWORDS = {
    "관련",
    "법률",
    "법령",
    "규정",
    "절차",
    "방법",
    "경우",
    "확인",
    "신청",
    "조사",
    "기준",
    "사용자",
    "질문",
    "대한민국",
}


@lru_cache(maxsize=128)
def keyword_search_laws(
    query: str,
    k: int = 12,
):
    """
    법률 표현을 이용한 문자열 기반 검색.

    예:
    - 임금 지급
    - 미지급 임금
    - 재산분할
    - 보증금 반환
    """

    dataset = get_law_dataset()

    phrases = [
        phrase.strip()
        for phrase in re.split(
            r"[,\n]",
            query,
        )
        if len(
            phrase.strip()
        ) >= 2
    ]

    tokens = {
        token
        for token in re.findall(
            r"[가-힣A-Za-z0-9]+",
            query,
        )
        if (
            len(token) >= 2
            and token
            not in KEYWORD_STOPWORDS
        )
    }

    # 공백을 제거한 전체 Query
    normalized_query = re.sub(
        r"\s+",
        "",
        query,
    )

    scored_results = []

    for law in dataset:

        law_name = law.get(
            "law_name",
            "",
        )

        normalized_law_name = re.sub(
            r"\s+",
            "",
            law_name,
        )

        for article in get_normalized_articles(
            law
        ):
            article_no = get_article_no(
                article
            )

            if not article_no:
                continue

            article_title = (
                article.get(
                    "article_title",
                    "",
                )
                or ""
            )

            normalized_article_title = re.sub(
                r"\s+",
                "",
                article_title,
            )

            content = format_exact_article(
                law_name,
                article,
            )

            searchable_text = (
                f"{law_name} "
                f"{article_title} "
                f"{content}"
            )

            normalized_text = re.sub(
                r"\s+",
                "",
                searchable_text,
            )

            score = 0

            substantive_match = False

            # -------------------------------------------------
            # 조문 제목 자체가 검색 질의 안에 포함되는 경우
            #
            # 예:
            # 조문명: 임금 지급
            # Query: 임금지급청구, 임금지급지연
            #
            # -> 강한 가산점
            # -------------------------------------------------

            if (
                normalized_article_title
                and len(
                    normalized_article_title
                ) >= 2
                and normalized_article_title
                in normalized_query
            ):
                score += 30

                substantive_match = True

            # -------------------------------------------------
            # 핵심 구문 일치
            # -------------------------------------------------

            for phrase in phrases:

                normalized_phrase = re.sub(
                    r"\s+",
                    "",
                    phrase,
                )

                if not normalized_phrase:
                    continue

                # 법령명 자체만으로 높은 점수를 주지 않음
                if (
                    normalized_phrase
                    == normalized_law_name
                ):
                    continue

                if (
                    normalized_phrase
                    in normalized_text
                ):
                    score += 10

                    substantive_match = True

                    # 조문 제목에 직접 포함되면 추가 점수
                    if (
                        normalized_phrase
                        in normalized_article_title
                    ):
                        score += 8

            # -------------------------------------------------
            # 개별 키워드 일치
            # -------------------------------------------------

            for token in tokens:

                # 법령명 자체는 건너뜀
                if token == law_name:
                    continue

                if token in article_title:
                    score += 7

                    substantive_match = True

                elif token in searchable_text:
                    score += 1

                    substantive_match = True

            # 실제 쟁점과 관련된 내용이 있는 경우에만
            # 법령명 일치 보너스
            if (
                substantive_match
                and law_name in query
            ):
                score += 2

            if score <= 0:
                continue

            scored_results.append(
                (
                    score,
                    {
                        "law_name": law_name,
                        "article": (
                            format_article_number(
                                article_no
                            )
                        ),
                        "article_title": (
                            article_title
                        ),
                        "content": content,
                        "source_type": "keyword",
                    },
                )
            )

    # 점수 높은 순
    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results = []

    seen = set()

    for score, source in scored_results:

        key = (
            source["law_name"],
            source["article"],
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        results.append(
            source
        )

        if len(results) >= k:
            break

    return results


# =========================================================
# ChromaDB
# =========================================================

@lru_cache(maxsize=1)
def get_vector_db():
    """
    임베딩 모델과 ChromaDB를 최초 1회만 로드한다.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "jhgan/"
            "ko-sroberta-multitask"
        ),
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return Chroma(
        persist_directory=str(
            DB_PATH
        ),
        embedding_function=embeddings,
    )


# =========================================================
# Vector Search 결과의 조문번호 추출
# =========================================================

def extract_article(
    content: str,
):
    """
    제347조
    제347조의2

    형태를 추출한다.
    """

    match = re.search(
        r"제\s*\d+\s*조"
        r"(?:의\s*\d+)?",
        content,
    )

    if match:

        return re.sub(
            r"\s+",
            "",
            match.group(),
        )

    return None


# =========================================================
# Vector 결과를 JSON 원본과 다시 연결
# =========================================================

@lru_cache(maxsize=512)
def resolve_article_from_dataset(
    law_name: str,
    article_number: str,
):
    """
    Vector Search 결과의

    법령명 + 조문번호

    를 이용해 JSON 원본에서
    정확한 조문 제목과 본문을 가져온다.

    예:

    근로기준법 + 제43조

    ->

    임금 지급
    """

    if (
        not law_name
        or not article_number
    ):
        return None

    match = re.match(
        r"제(\d+)조(?:의(\d+))?",
        article_number,
    )

    if not match:
        return None

    target_no = match.group(1)

    if match.group(2):
        target_no += (
            f"의{match.group(2)}"
        )

    law = get_law_index().get(
        law_name
    )

    if not law:
        return None

    for article in get_normalized_articles(
        law
    ):
        article_no = get_article_no(
            article
        )

        if article_no != target_no:
            continue

        return {
            "article_title": (
                article.get(
                    "article_title",
                    "",
                )
            ),
            "content": (
                format_exact_article(
                    law_name,
                    article,
                )
            ),
        }

    return None


# =========================================================
# Vector / MMR Search
# =========================================================

def search_laws(
    question: str,
    category: str,
    k: int = 8,
):
    """
    ChromaDB 기반 의미 검색.

    MMR:
    관련성 + 문서 다양성을 함께 고려한다.
    """

    db = get_vector_db()

    query = f"""
법률 분야: {category}

사용자 질문:
{question}

이 질문의 핵심 법적 쟁점과 직접적으로 관련된
대한민국 법령과 조문
"""

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(
                20,
                k * 3,
            ),
            "lambda_mult": 0.75,
        },
    )

    docs = retriever.invoke(
        query
    )

    results = []

    seen = set()

    for doc in docs:

        law_name = doc.metadata.get(
            "law_name",
            doc.metadata.get(
                "title",
                "관련 법령",
            ),
        )

        article = extract_article(
            doc.page_content
        )

        key = (
            law_name.strip(),
            article
            or doc.page_content[:80],
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        article_title = None

        content = doc.page_content

        # -------------------------------------------------
        # Vector 결과의 법령명 + 조문번호가 있으면
        # JSON 원본에서 조문 제목과 정확한 본문을 가져온다.
        # -------------------------------------------------

        if article:

            resolved = (
                resolve_article_from_dataset(
                    law_name,
                    article,
                )
            )

            if resolved:

                article_title = (
                    resolved[
                        "article_title"
                    ]
                )

                content = (
                    resolved[
                        "content"
                    ]
                )

        results.append(
            {
                "law_name": law_name,
                "article": article,
                "article_title": (
                    article_title
                ),
                "content": content,
                "source_type": "vector",
            }
        )

        if len(results) >= k:
            break

    return results