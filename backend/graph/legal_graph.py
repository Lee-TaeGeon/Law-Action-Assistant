import os
import re
from functools import lru_cache
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.services.rag_service import (
    search_laws,
    exact_search_laws,
    keyword_search_laws,
)


load_dotenv()


# =========================================================
# State
# =========================================================

class AgentState(TypedDict, total=False):
    question: str
    category: str
    search_query: str
    candidate_sources: List[Dict]
    sources: List[Dict]
    answer: str


# =========================================================
# LLM
# =========================================================

@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY가 설정되어 있지 않습니다."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.1,

        # Groq TPM 사용량을 과도하게 잡지 않도록 제한
        max_tokens=1000,
    )


# =========================================================
# 1. 질문 분야 분류
# =========================================================

def classifier_node(
    state: AgentState,
):
    llm = get_llm()

    prompt = f"""
당신은 대한민국 법률 상담 질문 분류기입니다.

다음 질문을 아래 분야 중
가장 적절한 하나로 분류하세요.

- 민사
- 형사
- 가사
- 노동
- 행정
- 기타
- 상대방이 처음부터 속여 돈이나 재산상 이익을 취득했을 가능성이
  핵심 쟁점인 질문은 형사로 분류하세요.

- 단순 계약불이행, 환불, 대금 반환처럼
  범죄 성립 여부가 핵심이 아닌 계약상 분쟁은 민사로 분류하세요.

예:
- 돈을 보냈는데 물건을 보내지 않고 연락을 끊음
  → 형사

- 물건을 받았지만 하자가 있어 환불을 원함
  → 민사

질문:
{state["question"]}

분야 이름만 출력하세요.
"""

    response = (
        llm.invoke(prompt)
        .content
        .strip()
    )

    allowed_categories = {
        "민사",
        "형사",
        "가사",
        "노동",
        "행정",
        "기타",
    }

    category = "기타"

    for item in allowed_categories:
        if item in response:
            category = item
            break

    return {
        "category": category,
    }


# =========================================================
# 2. Query Rewrite
# =========================================================

def query_rewriter_node(
    state: AgentState,
):
    llm = get_llm()

    prompt = f"""
당신은 대한민국 법령 검색을 위한
검색 질의 생성기입니다.

사용자의 질문을 법령 원문 검색에 적합한
짧고 독립적인 핵심 검색 표현으로 변환하세요.

중요 규칙:

1. 사용자의 원래 의미를 유지하세요.

2. 검색 표현은 반드시
   짧은 법률 개념 또는 조문 제목 형태로 작성하세요.

3. 하나의 검색 표현에 너무 많은 단어를
   붙이지 마세요.

예:

나쁜 예:
- 임금 지급 청구
- 임금 체불 구제 절차
- 재산분할 관련 소송 절차

좋은 예:
- 임금
- 임금 지급
- 임금체불
- 체불임금
- 재산분할
- 재산분할청구권

4. 사용자의 문제를 직접 규정하는
   기본적인 권리ㆍ의무 표현을 반드시 포함하세요.

예:

- 월급을 받지 못함
  → 임금, 임금 지급, 임금체불, 체불임금

- 이혼 재산 문제
  → 재산분할, 재산분할청구권, 부부재산

- 중고거래에서 돈을 받고 물건을 보내지 않음
  → 사기, 재산상 이익, 기망

5. 구제절차나 신고 방법보다
   먼저 실체적인 권리ㆍ의무ㆍ책임에 해당하는
   검색어를 출력하세요.

6. 법령명은 검색에 도움이 되는 경우에만
   뒤쪽에 추가하세요.

7. 사용자가 직접 언급하지 않은
   구체적인 조문 번호는 절대 생성하지 마세요.

8. 특정 조문 번호를 추측하지 마세요.

9. 검색 표현은 최대 8개만 출력하세요.

10. 각 검색 표현은 쉼표로 구분하세요.

11. 설명 문장은 출력하지 마세요.

12. 사용자가 "어떻게 해야 하나요", "신고할 수 있나요",
    "고소할 수 있나요"처럼 대응 방법을 묻는 경우에는
    실체적인 권리ㆍ의무ㆍ책임뿐 아니라
    그 대응을 직접 뒷받침할 절차상 법률 개념도
    검색어에 포함하세요.

13. 절차 검색어는 막연한 "절차", "신고"보다
    실제 법령 조문 제목에 사용될 가능성이 높은
    구체적인 법률 용어를 사용하세요.

예:

- 돈을 보냈는데 물건을 보내지 않고 연락을 끊음
  → 사기, 기망, 재산상 이익, 피해자 고소, 고소권자

- 임금을 받지 못함
  → 임금, 임금 지급, 임금체불, 체불 임금 확인

- 이혼 중 재산 문제
  → 재산분할, 재산분할청구권, 재산분할 심판


법률 분야:
{state["category"]}

사용자 질문:
{state["question"]}
"""

    search_query = (
        llm.invoke(prompt)
        .content
        .strip()
    )

    return {
        "search_query": search_query,
    }


# =========================================================
# 3. Hybrid Retrieval
# =========================================================

def researcher_node(
    state: AgentState,
):
    search_query = state["search_query"]

    # -----------------------------------------------------
    # Exact Search
    #
    # 사용자가 직접 입력한 법령/조문만 대상으로 한다.
    # Query Rewriter가 만든 조문 번호를 Exact Search에
    # 사용하면 잘못된 조문을 추측할 위험이 있다.
    # -----------------------------------------------------

    exact_sources = exact_search_laws(
        state["question"]
    )

    # -----------------------------------------------------
    # Keyword Search
    # -----------------------------------------------------

    keyword_sources = keyword_search_laws(
        search_query,
        k=12,
    )

    # -----------------------------------------------------
    # Vector / MMR Search
    # -----------------------------------------------------

    vector_sources = search_laws(
        question=search_query,
        category=state["category"],
        k=8,
    )

    # -----------------------------------------------------
    # Exact + Keyword + Vector 병합 및 중복 제거
    # -----------------------------------------------------

    combined = []
    seen = set()

    for source in (
        exact_sources
        + keyword_sources
        + vector_sources
    ):
        key = (
            source.get("law_name"),
            source.get("article"),
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(source)

    # =====================================================
    # 디버깅 출력
    # =====================================================

    print(
        "\n=== 검색 질의 ==="
    )

    print(
        search_query
    )

    print(
        "\n=== Exact 검색 ==="
    )

    if not exact_sources:
        print(
            "Exact 검색 결과 없음"
        )

    for source in exact_sources:
        print(
            source.get("law_name"),
            source.get("article"),
            source.get("article_title"),
        )

    print(
        "\n=== Keyword 검색 ==="
    )

    if not keyword_sources:
        print(
            "Keyword 검색 결과 없음"
        )

    for source in keyword_sources:
        print(
            source.get("law_name"),
            source.get("article"),
            source.get("article_title"),
        )

    print(
        "\n=== Vector 검색 ==="
    )

    if not vector_sources:
        print(
            "Vector 검색 결과 없음"
        )

    for source in vector_sources:
        print(
            source.get("law_name"),
            source.get("article"),
            source.get("article_title"),
        )

    print(
        "\n=== 병합 후보 ==="
    )

    for index, source in enumerate(
        combined
    ):
        print(
            f"[{index}]",
            source.get("law_name"),
            source.get("article"),
            source.get("article_title"),
            source.get("source_type"),
        )

    return {
        "candidate_sources": combined,
    }


# =========================================================
# Reranker 후보 압축
# =========================================================

def select_reranker_candidates(
    all_candidates: List[Dict],
) -> List[Dict]:
    """
    Reranker 프롬프트가 지나치게 커지는 것을 막으면서
    Exact / Keyword / Vector 검색 결과를 균형 있게 유지한다.
    """

    exact_candidates = [
        source
        for source in all_candidates
        if source.get("source_type") == "exact"
    ]

    keyword_candidates = [
        source
        for source in all_candidates
        if source.get("source_type") == "keyword"
    ]

    vector_candidates = [
        source
        for source in all_candidates
        if source.get("source_type") == "vector"
    ]

    # 검색 방식별 후보 균형 유지
    initial_candidates = (
        exact_candidates[:2]
        + keyword_candidates[:3]
        + vector_candidates[:2]
    )

    unique_candidates = []
    seen = set()

    for source in initial_candidates:
        key = (
            source.get("law_name"),
            source.get("article"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(source)

        if len(unique_candidates) >= 7:
            break

    # source_type이 예상과 다른 후보가 있거나
    # 중복 때문에 7개보다 적어진 경우 상위 검색 결과로 보충
    if len(unique_candidates) < 7:
        for source in all_candidates:
            key = (
                source.get("law_name"),
                source.get("article"),
            )

            if key in seen:
                continue

            seen.add(key)
            unique_candidates.append(source)

            if len(unique_candidates) >= 7:
                break

    return unique_candidates[:7]


# =========================================================
# 4. LLM Reranker
# =========================================================
def filter_scope_mismatch_candidates(
    question: str,
    candidates: List[Dict],
):
    """
    특정 신분/상황에만 적용되는 조문이
    일반 질문에 섞이는 것을 1차적으로 차단한다.
    """

    question_text = re.sub(
        r"\s+",
        "",
        question,
    )

    scope_rules = [
        {
            "source_terms": [
                "미성년자",
                "18세미만",
                "연소자",
            ],
            "question_terms": [
                "미성년자",
                "18세미만",
                "청소년",
                "고등학생",
                "중학생",
            ],
        },
        {
            "source_terms": [
                "선원",
            ],
            "question_terms": [
                "선원",
                "선박",
                "해운",
            ],
        },
        {
            "source_terms": [
                "건설업",
                "건설근로자",
                "수급인",
                "하수급인",
            ],
            "question_terms": [
                "건설",
                "공사",
                "하청",
                "도급",
            ],
        },
        {
            "source_terms": [
                "퇴직한근로자",
            ],
            "question_terms": [
                "퇴직",
                "퇴사",
                "그만뒀",
            ],
        },
    ]

    filtered = []

    for source in candidates:
        content = re.sub(
            r"\s+",
            "",
            source.get(
                "content",
                "",
            ),
        )

        mismatch = False

        for rule in scope_rules:

            source_matches = any(
                term in content
                for term in rule[
                    "source_terms"
                ]
            )

            if not source_matches:
                continue

            question_matches = any(
                term in question_text
                for term in rule[
                    "question_terms"
                ]
            )

            if not question_matches:
                mismatch = True
                break

        if not mismatch:
            filtered.append(
                source
            )

        else:
            print(
                "[적용대상 불일치 제외]",
                source.get(
                    "law_name"
                ),
                source.get(
                    "article"
                ),
                source.get(
                    "article_title"
                ),
            )

    return filtered





def reranker_node(
    state: AgentState,
):
    llm = get_llm()

    all_candidates = state.get(
        "candidate_sources",
        [],
    )

    all_candidates = (
    filter_scope_mismatch_candidates(
        state["question"],
        all_candidates,
    )
)

    candidates = select_reranker_candidates(
        all_candidates
    )

    if not candidates:
        return {
            "sources": [],
        }

    print(
        "\n=== Reranker 입력 후보 ==="
    )

    for index, source in enumerate(
        candidates
    ):
        print(
            f"[{index}]",
            source.get("law_name"),
            source.get("article"),
            source.get("article_title"),
            source.get("source_type"),
        )

    candidate_parts = []

    for index, source in enumerate(
        candidates
    ):
        content = (
            source.get("content")
            or ""
        )

        # Groq TPM 초과를 방지하기 위해
        # Reranker에는 조문 앞부분만 전달한다.
        content = content[:400]

        candidate_parts.append(
            (
                f"[{index}]\n"
                f"검색방식: "
                f"{source.get('source_type', '')}\n"
                f"법령: "
                f"{source.get('law_name', '')} "
                f"{source.get('article') or ''}\n"
                f"조문명: "
                f"{source.get('article_title') or ''}\n"
                f"내용:\n"
                f"{content}"
            )
        )

    candidate_text = "\n\n".join(
        candidate_parts
    )

    prompt = f"""
당신은 대한민국 법률 RAG 검색 결과 평가기입니다.

사용자의 질문 해결에 직접 필요한 법령을 선택하세요.

선택 원칙:

1. 가장 먼저 사용자의 권리ㆍ의무ㆍ책임을
   직접 규정하는 핵심 조문을 찾으세요.

2. 질문의 핵심 행위 자체를 규정하는 조문을
   특별 구제제도나 행정절차 조문보다 우선하세요.

예:

- 임금을 받지 못한 질문이라면
  임금 지급 의무를 직접 규정하는 조문을 우선합니다.

- 그 다음 미지급 임금의 지연이자,
  손해배상 또는 구제절차 등을
  보조 근거로 선택할 수 있습니다.

3. 단순히 같은 단어가 등장한다는 이유로
   법령을 선택하지 마세요.

4. 행정기관 조직, 위원회 구성,
   자료 제공, 명단 공개 등의 규정은
   사용자 질문이 직접 해당 내용을 묻는 경우가 아니라면
   핵심 근거로 선택하지 마세요.

5. 질문과 직접 관련된 일반적인 법적 의무가 있다면
   특수한 상황에서만 적용되는 법령보다 우선하세요.

6. 조문 제목이 질문의 핵심 쟁점과 직접 일치하면
   중요하게 평가하세요.

7. 사용자의 질문을 해결하는 데 필요한
   실질적인 법적 근거를 우선하세요.

8. 최대 4개만 선택하세요.

9. 관련성이 충분한 법령이 1~3개뿐이라면
   억지로 4개를 채우지 마세요.

10. 같은 단어나 유사한 개념이 등장하더라도
    사용자 질문과 법률관계가 다른 조문은 제외하세요.

    예를 들어 이혼 문제와 상속 문제,
    임금체불 문제와 부당해고 문제처럼
    법률 분야가 비슷하더라도 실제 쟁점이 다르면
    핵심 근거로 선택하지 마세요.

11. 실체적인 권리ㆍ의무를 직접 규정하는 조문이 있다면
    일반적인 소송 절차나 관할 조문보다 우선하세요.

12. 조문 제목만 보고 선택하지 마세요.
    반드시 조문 본문에서 적용 대상과 적용 조건을 확인하세요.

13. 특정 대상이나 특정 상황에만 적용되는 조문은
    사용자 질문에 그 조건이 명시되어 있지 않다면
    선택하지 마세요.

    예:
    - 미성년자에게만 적용되는 조문
    - 선원에게만 적용되는 조문
    - 건설업에만 적용되는 조문
    - 특정 산업 종사자에게만 적용되는 조문
    - 임산부에게만 적용되는 조문
    - 퇴직한 근로자에게만 적용되는 조문

14. 예를 들어 근로기준법 제68조는
    미성년자가 독자적으로 임금을 청구할 수 있다는
    특별 규정이므로,
    사용자가 미성년자라는 사실이 없는 일반 임금체불 질문에는
    선택하지 마세요.

15. 일반적인 규정과 특별한 대상에게만 적용되는 규정이
    동시에 후보에 있다면,
    사용자의 사실관계에 특별 조건이 확인되지 않는 한
    일반 규정을 우선하세요.

16. "본장의 죄", "본절의 죄", "이 장의 죄",
    "전조의 죄"처럼 적용 범위가 제한된 조문은
    일반적인 규정으로 해석하지 마세요.

17. 위와 같은 조문은 해당 장ㆍ절ㆍ참조 조문이
    사용자의 사건과 실제로 관련되어 있는 경우에만
    선택하세요.

18. 조문 제목이 "고소", "신청", "청구"처럼
    사용자의 질문과 같더라도,
    조문 본문이 특정 범죄ㆍ특정 장ㆍ특정 대상에만
    적용되는 규정이라면 제외하세요.

19. 일반적인 절차상 권리를 직접 규정하는 조문과
    특정 범죄에만 적용되는 특별 규정이 동시에 있다면,
    사용자의 사건이 특별 규정의 적용 대상이라고
    확인되지 않는 한 일반 규정을 우선하세요.

예:
- 형사소송법의 "범죄로 인한 피해자는 고소할 수 있다"는
  일반적인 고소권 규정은 피해자의 고소 가능성을
  판단하는 근거가 될 수 있습니다.

- "본장의 죄는 고소가 있어야 공소를 제기할 수 있다"는
  조문은 해당 장의 범죄에만 적용되므로,
  다른 범죄의 일반적인 고소 근거로 사용하면 안 됩니다.

사용자 질문:
{state["question"]}

법률 분야:
{state["category"]}

후보 법령:

{candidate_text}

선택할 후보 번호만 쉼표로 구분해서 출력하세요.

예:
0,2,5

다른 문장은 출력하지 마세요.
"""

    response = (
        llm.invoke(prompt)
        .content
        .strip()
    )

    # 아래와 같은 응답 형식을 모두 처리한다.
    #
    # 0,2,5
    # [0, 2, 5]

    index_values = re.findall(
        r"\d+",
        response,
    )

    selected = []
    seen_indices = set()

    for value in index_values:
        index = int(value)

        if index in seen_indices:
            continue

        if (
            0 <= index
            < len(candidates)
        ):
            selected.append(
                candidates[index]
            )

            seen_indices.add(
                index
            )

        if len(selected) >= 4:
            break

    print(
        "\n=== Reranker 원본 응답 ==="
    )

    print(
        response
    )

    print(
        "\n=== 최종 선택 법령 ==="
    )

    if not selected:
        print(
            "최종 선택 법령 없음"
        )

    for source in selected:
        print(
            source.get("law_name"),
            source.get("article"),
            source.get("article_title"),
            source.get("source_type"),
        )

    return {
        "sources": selected,
    }


# =========================================================
# 5. Generator
# =========================================================

def generator_node(
    state: AgentState,
):
    llm = get_llm()

    sources = state.get(
        "sources",
        [],
    )

    # =====================================================
    # 검색된 법령을 Generator용 Context로 변환
    # =====================================================

    if sources:

        context_parts = []

        for index, source in enumerate(
            sources
        ):
            content = source.get(
                "content",
                "",
            )

            # 토큰 사용량 방지
            content = content[:900]

            context_parts.append(
                (
                    f"[출처 {index + 1}]\n"
                    f"법령: "
                    f"{source['law_name']} "
                    f"{source.get('article') or ''}\n"
                    f"조문명: "
                    f"{source.get('article_title') or ''}\n"
                    f"조문 내용:\n"
                    f"{content}"
                )
            )

        context = "\n\n".join(
            context_parts
        )

    else:

        context = (
            "검색된 관련 법령이 없습니다."
        )

    # =====================================================
    # Generator Prompt
    # =====================================================

    prompt = f"""
당신은 대한민국 법률정보를
일반 사용자에게 이해하기 쉽게 설명하는 AI입니다.

가장 중요한 원칙은
'검색된 법령에 근거한 내용만 답변한다'입니다.

일반적인 법률 상식이나
당신이 알고 있는 다른 법률 지식을 이용하여
검색 자료에 없는 내용을 추가하지 마세요.


[핵심 규칙]

1. 아래에 제공된 검색 법령을
   가장 우선적인 근거로 사용하세요.

2. 검색 자료에 없는 법령명이나
   조문 번호를 임의로 만들어내지 마세요.

3. 법률적 결론이나 승소 가능성 등을
   확정적으로 단정하지 마세요.

4. 검색된 조문에서 직접 확인되는
   권리, 의무, 청구 가능성, 요건만 설명하세요.

5. 사용자가 실제로 취할 수 있는 대응 방법도
   현재 검색된 조문에서 근거를 확인할 수 있는
   내용만 제시하세요.

6. 검색된 법령에 직접 나타나지 않는
   기관명, 신고기관, 접수처를
   임의로 추가하지 마세요.

예:
- 고용노동부
- 노동청
- 경찰
- 법원
- 주민센터

검색 자료에서 직접 확인되지 않았다면
구체적인 기관을 단정해서 안내하지 마세요.

7. 검색된 법령에 직접 나타나지 않는
   신청 절차, 신고 절차, 소송 절차,
   제출 방법을 임의로 만들어내지 마세요.

8. 검색된 법령에 없는
   구체적인 제출 서류를 만들어내지 마세요.

예:
- 내용증명
- 계약서
- 녹취록
- 문자메시지
- 신청서

현재 검색 자료에서 확인되지 않는다면
필수 서류처럼 표현하지 마세요.

9. 검색된 법령에서 직접 확인할 수 없는
   기간, 금액, 이율, 시효,
   처리기간을 임의로 추가하지 마세요.

10. 검색된 법령에서 확인되지 않는 내용이
    답변에 필요하다면 다음과 같이 표현하세요.

    "현재 검색된 법령만으로는
    구체적인 절차까지 확인하기 어렵습니다."

11. 실무적으로 가능해 보이는 대응이라도
    검색 자료에서 직접 근거가 없다면
    '대응 방법' 목록에 넣지 마세요.

12. 검색된 법령의 내용을
    다른 의미로 확대해석하지 마세요.

예:

어떤 조문이 국가기관의 대위권을 규정한다면
이를 사용자가 직접 행사하는 대위권으로
바꾸어 설명하면 안 됩니다.

13. 조문의 주체를 반드시 확인하세요.

예:

- 근로자
- 사용자
- 사업주
- 고용노동부장관
- 법원
- 국가

누가 권리를 행사하거나
의무를 부담하는지 바꾸지 마세요.

14. 각 대응 방법에는 가능하면
    그 근거가 되는 법령과 조문을
    자연스럽게 명시하세요.

예:

"근로기준법 제43조에 따르면
사용자는 임금을 지급해야 합니다."

15. 검색된 조문 중
    질문과 직접 관련 없는 조문은
    답변에 억지로 포함하지 마세요.

16. 같은 법령이 검색됐더라도
    조문 내용이 현재 상황과 직접 관계없다면
    사용하지 않아도 됩니다.

17. 검색 근거가 부족하다면
    대응 방법의 개수를 억지로 늘리지 마세요.

18. 표(Table)는 사용하지 마세요.

19. HTML 태그를 사용하지 마세요.

20. 법령 원문을 길게 그대로 복사하지 말고
    사용자가 이해하기 쉬운 말로 요약하세요.

21. 전체 답변은 너무 길지 않게 작성하세요.

22. 검색 자료에 직접 등장하지 않는 구체적인 예시도 제시하지 마세요.
    예: 내용증명, 문자메시지, 녹취, 계약서 등의 수단을 임의로 예시로 들지 마세요.

23. "법령에는 명시되어 있지 않지만", "일반적으로", "예를 들어"라는 표현을 사용하여
    검색 자료에 없는 지식을 우회적으로 추가하지 마세요.

24. 검색된 법령에 직접 근거가 없는
    증거자료, 준비자료, 필요 서류를
    일반적인 법률 지식으로 추정하지 마세요.

    예:
    근로계약서, 급여명세서, 문자메시지,
    녹취록, 계좌내역 등을
    검색 자료에 없는데 임의로 권하지 마세요.

25. 검색된 자료에 등장하지 않는
    기관이나 상담기관을 추천하지 마세요.

    예:
    법률구조기관, 변호사, 경찰,
    주민센터 등의 기관을
    일반적인 조언으로 추가하지 마세요.

26. 답변 마지막에 일반적인
    "전문가와 상담하세요",
    "법률기관에 문의하세요" 등의 문장을
    자동으로 추가하지 마세요.

    추가 확인이 필요한 경우에는
    "현재 검색된 법령만으로는
    구체적인 절차를 확인하기 어렵습니다."
    정도로만 설명하세요.

27. 특정 대상에게만 적용되는 조문을 일반화하지 마세요.

    예:
    "미성년자는 독자적으로 임금을 청구할 수 있다"는 조문을
    "근로자는 임금을 청구할 수 있다"로 바꾸어 설명하면 안 됩니다.

    사용자에게 해당 적용 조건이 확인되지 않는다면
    그 조문을 답변 근거로 사용하지 마세요.

28. 현재 제공된 검색 법령에 기간, 금액, 이율, 시효,
    적용 대상 등이 명시적으로 규정되어 있다면
    그 내용을 답변에 사용할 수 있습니다.

    검색된 법령에서 직접 확인되는 내용까지
    "확인하기 어렵다"고 표현하지 마세요.

    단, 검색 법령에 없는 숫자나 조건을
    추측해서 추가해서는 안 됩니다.

29. 사용자가 질문에서 제공하지 않은 사실관계를
    임의로 추정하거나 만들어내지 마세요.

    예:
    - 체불 기간
    - 근속 기간
    - 나이
    - 계약 형태
    - 퇴직 여부
    - 피해 금액
    - 사업장의 업종

    사용자가 "월급을 받지 못했다"고만 했다면
    "2개월째", "3개월째"처럼 기간을 만들어내지 마세요.

30. 검색된 절차법·시행령·규칙의 조문이
    특정 사항만 규정하고 있다면,
    해당 조문이 전체 절차를 규정하는 것처럼
    확대하여 설명하지 마세요.


31. 사용자가 제공한 사실과 법적으로 확정된 사실을 구분하세요.

    사용자의 사실만으로 범죄 성립요건이 모두 확인되지 않았다면
    "해당한다", "기망하였다", "사기죄가 성립한다"라고
    확정적으로 표현하지 마세요.

    대신:
    "해당할 가능성이 있습니다",
    "추가적인 사실 확인이 필요합니다"
    처럼 표현하세요.

32. 범죄 구성요건을 규정하는 조문만 검색된 경우,
    고소ㆍ고발 방법, 접수기관, 수사 절차 등을
    임의로 추가하지 마세요.

    해당 절차를 설명하려면 그 절차를 뒷받침하는
    별도의 검색 법령이 있어야 합니다.

[답변 형식]

## 상황 분석

검색된 법령으로 직접 확인되는
사용자의 법적 상황을 2~4문장으로 설명하세요.


## 대응 방법

검색된 법령에 직접 근거가 있는 대응만
번호 목록으로 작성하세요.

적절한 근거가 1~2개라면
1~2개만 작성해도 됩니다.

각 항목은 가능하면 다음 형태로 작성하세요.

1. **대응 내용**
   - 근거가 되는 법령과 조문을 설명
   - 해당 조문에서 직접 확인되는 범위까지만 설명


## 주의사항

검색된 법령만으로 확인할 수 없는 내용,
사실관계에 따라 달라질 수 있는 부분,
추가 확인이 필요한 부분을 설명하세요.


[사용자 질문]

{state["question"]}


[법률 분야]

{state["category"]}


[검색된 법령]

{context}
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content,
    }

# =========================================================
# LangGraph 구성
# =========================================================

workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "classifier",
    classifier_node,
)

workflow.add_node(
    "query_rewriter",
    query_rewriter_node,
)

workflow.add_node(
    "researcher",
    researcher_node,
)

workflow.add_node(
    "reranker",
    reranker_node,
)

workflow.add_node(
    "generator",
    generator_node,
)


workflow.set_entry_point(
    "classifier"
)


workflow.add_edge(
    "classifier",
    "query_rewriter",
)

workflow.add_edge(
    "query_rewriter",
    "researcher",
)

workflow.add_edge(
    "researcher",
    "reranker",
)

workflow.add_edge(
    "reranker",
    "generator",
)

workflow.add_edge(
    "generator",
    END,
)


# =========================================================
# Memory
# =========================================================

memory = MemorySaver()

legal_graph = workflow.compile(
    checkpointer=memory
)


# =========================================================
# FastAPI 호출 함수
# =========================================================

def run_legal_chat(
    question: str,
    session_id: str,
):
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    return legal_graph.invoke(
        {
            "question": question,
        },
        config=config,
    )