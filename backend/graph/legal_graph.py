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
핵심 검색어와 법률 개념으로 확장하세요.

규칙:

1. 사용자의 원래 의미를 유지하세요.

2. 사용자가 사용한 표현뿐 아니라
   법령 조문 제목이나 법률 문장에서
   사용될 가능성이 높은 동의 표현도 포함하세요.

예:

- 임금을 못 받음
  → 임금체불, 임금 지급, 미지급 임금, 체불임금

- 전세금을 못 돌려받음
  → 보증금 반환, 임대차보증금, 반환청구

- 이혼 재산 문제
  → 재산분할, 재산분할청구, 혼인 중 재산

3. 핵심 권리ㆍ의무ㆍ청구ㆍ구제 방법을 포함하세요.

4. 관련 법령명은 검색에 도움이 된다면
   검색 키워드로 포함할 수 있습니다.

5. 사용자가 직접 언급하지 않은
   구체적인 조문 번호는 절대 생성하지 마세요.

6. 특정 조문 번호를 추측하지 마세요.

7. 너무 일반적인 표현보다
   실제 법령 본문에 등장할 법률 용어를 우선하세요.

8. 핵심 검색 표현을 쉼표로 구분하세요.

9. 설명 문장은 출력하지 마세요.

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

def reranker_node(
    state: AgentState,
):
    llm = get_llm()

    all_candidates = state.get(
        "candidate_sources",
        [],
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

    if sources:
        context_parts = []

        for index, source in enumerate(
            sources
        ):
            content = (
                source.get("content")
                or ""
            )

            # 최종 답변 생성 시에도 너무 긴 조문 전체를
            # LLM에 전달하지 않는다.
            content = content[:900]

            context_parts.append(
                (
                    f"[출처 {index + 1}]\n"
                    f"법령: "
                    f"{source.get('law_name', '')} "
                    f"{source.get('article') or ''}\n"
                    f"조문명: "
                    f"{source.get('article_title') or ''}\n"
                    f"내용:\n"
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

    prompt = f"""
당신은 대한민국 법률정보를 설명하는 AI입니다.

반드시 제공된 검색 법령 자료를
우선 근거로 답변하세요.

규칙:

1. 검색 자료에 없는 법령명이나
   조문 번호를 임의로 만들어내지 마세요.

2. 검색 자료만으로 적용 여부를 확정하기 어렵다면
   불확실하다고 명확히 설명하세요.

3. 법률적 결론을 단정하지 마세요.

4. 사용자가 실제로 취할 수 있는
   행동을 중심으로 설명하세요.

5. 표(Table)는 사용하지 마세요.

6. HTML 태그를 사용하지 마세요.

7. 검색된 법령 원문을
   답변에 길게 그대로 복사하지 마세요.

8. "관련 법령" 섹션은 따로 만들지 마세요.

   관련 법령은 React 화면에서
   별도 출처 카드로 제공합니다.

9. 전체 답변은 너무 길지 않게 작성하세요.

10. 검색된 법령 자료에서 직접 확인할 수 없는
    구체적인 기간, 금액, 요건, 기관 절차,
    시효, 제도명 등을 임의로 추가하지 마세요.

11. 일반적으로 알고 있는 법률 지식보다
    현재 제공된 검색 자료를 우선하세요.

12. 검색 자료에서 확인되지 않는 내용이 필요한 경우
    "현재 검색된 법령만으로는 확인하기 어렵습니다."
    라고 설명하세요.

13. 각 대응 방법은 현재 제공된 검색 자료 중
    적어도 하나에 근거가 있는 내용만 작성하세요.

14. 검색된 법령 중 질문과 직접 관련 없는 내용은
    억지로 답변에 포함하지 마세요.

15. 검색된 법령이 없다면
    법령명이나 구체적인 법적 절차를 추측하지 말고
    현재 검색 결과만으로 답변하기 어렵다고 설명하세요.

반드시 아래 형식으로 답변하세요.

## 상황 분석

사용자의 상황과 핵심 쟁점을
2~4문장으로 설명합니다.

## 대응 방법

검색된 법령을 근거로 실제로 취할 수 있는 행동을
번호 목록으로 3~6개 제시합니다.

단, 검색 자료가 부족하다면
억지로 3개 이상 만들지 마세요.

## 주의사항

사실관계에 따라 달라질 수 있는 부분과
검색된 법령만으로 확인하기 어려운 내용을 설명합니다.

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