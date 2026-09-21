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
    법령명을 이용해 빠르게 법령 객체를 찾을 수 있도록
    {법령명: 법령데이터} 형태의 인덱스를 생성한다.
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
    긴 법령명을 먼저 검사할 수 있도록
    법령명을 길이순으로 정렬한다.
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
    '사기죄'와 '사기'처럼 제목 표현이 약간 다른 경우를
    비교하기 쉽게 정규화한다.
    """

    if not title:
        return ""

    title = re.sub(
        r"\s+",
        "",
        title,
    )

    # 예:
    # 사기죄 -> 사기
    if title.endswith("죄"):
        title = title[:-1]

    return title


def format_article_number(article_no: str):
    """
    DB의 조문 번호를 사용자에게 보여줄 형식으로 변환한다.

    347 -> 제347조
    347의2 -> 제347조의2
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
# Exact Search용 조문 문자열 생성
# =========================================================

def format_exact_article(
    law_name: str,
    article: dict,
):
    """
    JSON의 조문 데이터를 LLM이 사용할 수 있는
    문자열 형태로 변환한다.
    """

    article_no = str(
        article.get(
            "article_no",
            "",
        )
    ).strip()

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

    return "\n".join(lines)


# =========================================================
# 검색 질의에서 명시적인 법령 + 조문 추출
# =========================================================

def extract_explicit_law_refs(
    query: str,
):
    """
    예:

    형법 제347조(사기죄)

    ↓

    {
        "law_name": "형법",
        "article_no": "347",
        "title_hint": "사기죄"
    }
    """

    refs = []

    law_names = get_law_names()

    for law_name in law_names:

        # 질의에 법령명 자체가 없으면 건너뜀
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

            # 제347조의2 같은 경우
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
# 정확한 법령 / 조문 검색
# =========================================================

def exact_search_laws(query: str):
    refs = extract_explicit_law_refs(query)

    law_index = get_law_index()

    results = []

    normalized_query = normalize_title(query)

    for ref in refs:
        law_name = ref["law_name"]
        target_no = ref["article_no"]

        title_hint = normalize_title(
            ref.get("title_hint")
        )

        law = law_index.get(law_name)

        if not law:
            continue

        article_candidates = []

        for article in law.get("data", []):
            article_no = str(
                article.get(
                    "article_no",
                    "",
                )
            ).strip()

            if article_no == target_no:
                article_candidates.append(
                    article
                )

        if not article_candidates:
            continue

        # 1. 형법 제347조(사기죄)처럼
        # 제목 힌트가 있는 경우 제목까지 정확히 비교
        if title_hint:
            title_matches = [
                article
                for article in article_candidates
                if normalize_title(
                    article.get("article_title")
                ) == title_hint
            ]

            if title_matches:
                article_candidates = title_matches

        # 2. 제목 힌트가 없어도
        # 사용자 검색 질의에 조문 제목이 포함되어 있으면 우선 선택
        elif len(article_candidates) > 1:
            query_matches = []

            for article in article_candidates:
                article_title = normalize_title(
                    article.get("article_title")
                )

                if (
                    article_title
                    and article_title in normalized_query
                ):
                    query_matches.append(article)

            if query_matches:
                # 더 구체적인 제목 우선
                query_matches.sort(
                    key=lambda article: len(
                        normalize_title(
                            article.get("article_title")
                        )
                    ),
                    reverse=True,
                )

                article_candidates = [
                    query_matches[0]
                ]

            else:
                # 그래도 구별되지 않으면 첫 번째 후보만 사용
                article_candidates = [
                    article_candidates[0]
                ]

        for article in article_candidates:
            article_no = str(
                article.get(
                    "article_no",
                    "",
                )
            ).strip()

            article_title = article.get(
                "article_title",
                "",
            )

            results.append(
                {
                    "law_name": law_name,
                    "article": format_article_number(
                        article_no
                    ),
                    "article_title": article_title,
                    "content": format_exact_article(
                        law_name,
                        article,
                    ),
                    "source_type": "exact",
                }
            )

    return results


# =========================================================
# ChromaDB
# =========================================================

@lru_cache(maxsize=1)
def get_vector_db():
    """
    HuggingFace 임베딩 모델과 ChromaDB를
    서버 실행 중 1번만 로드한다.
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
# Vector Search 결과에서 조문번호 추출
# =========================================================

def extract_article(
    content: str,
):
    """
    문서 내용에서

    제347조
    제347조의2

    같은 표현을 추출한다.
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
# Vector / MMR 검색
# =========================================================

def search_laws(
    question: str,
    category: str,
    k: int = 4,
):
    """
    ChromaDB를 이용한 의미 기반 법령 검색.

    MMR을 사용하여
    관련성과 결과 다양성을 함께 고려한다.
    """

    db = get_vector_db()

    query = f"""
법률 분야: {category}

사용자 질문:
{question}

이 질문을 해결하는 데 직접적으로 관련된
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

        # 같은 법령 + 같은 조문 중복 제거
        key = (
            law_name.strip(),
            article
            or doc.page_content[:80],
        )

        if key in seen:
            continue

        seen.add(key)

        results.append(
            {
                "law_name": law_name,
                "article": article,
                "content": (
                    doc.page_content
                ),
                "source_type": (
                    "vector"
                ),
            }
        )

        if len(results) >= k:
            break

    return results