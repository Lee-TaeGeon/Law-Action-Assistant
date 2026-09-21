import os
from functools import lru_cache
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from backend.services.rag_service import (
    search_laws,
    exact_search_laws,
)
from backend.services.rag_service import search_laws


load_dotenv()


class AgentState(TypedDict, total=False):
    question: str
    category: str
    search_query: str
    candidate_sources: List[Dict[str, str]]
    sources: List[Dict[str, str]]
    answer: str


@lru_cache
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


def classifier_node(state: AgentState):
    llm = get_llm()

    prompt = f"""
당신은 대한민국 법률 상담 질문 분류기입니다.

다음 질문을 아래 분야 중 가장 적절한 하나로 분류하세요.

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

    response = llm.invoke(prompt).content.strip()

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


def researcher_node(state: AgentState):
    search_query = state["search_query"]

    # 1. 정확한 법령 / 조문 검색
    exact_sources = exact_search_laws(
        search_query
    )

    # 2. 의미 기반 Vector / MMR 검색
    vector_sources = search_laws(
        question=search_query,
        category=state["category"],
        k=8,
    )

    # 3. Exact + Vector 결과 병합
    combined = []

    seen = set()

    for source in (
        exact_sources
        + vector_sources
    ):
        key = (
            source["law_name"],
            source.get("article"),
        )

        if key in seen:
            continue

        seen.add(key)

        combined.append(
            source
        )

    # -------------------------
    # 디버깅 출력
    # -------------------------

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
            source["law_name"],
            source.get("article"),
        )

    print(
        "\n=== Vector 검색 ==="
    )

    for source in vector_sources:
        print(
            source["law_name"],
            source.get("article"),
        )

    print(
        "\n=== 병합 후보 ==="
    )

    for source in combined:
        print(
            source["law_name"],
            source.get("article"),
            source.get(
                "source_type"
            ),
        )

    return {
        "candidate_sources": combined,
    }

    

def query_rewriter_node(state: AgentState):
    llm = get_llm()

    prompt = f"""
당신은 대한민국 법령 검색을 위한 검색어 생성기입니다.

사용자 질문을 ChromaDB 법령 검색에 적합한 검색어로 변환하세요.

규칙:
- 사용자의 원래 질문 의미를 유지하세요.
- 법률적으로 중요한 행위와 쟁점을 키워드로 포함하세요.
- 적용될 가능성이 있는 법률 분야와 절차 관련 표현을 포함하세요.
- 특정 법령이나 조문을 확신할 수 없다면 임의로 만들어내지 마세요.
- 설명하지 말고 검색 문자열 한 줄만 출력하세요.

분야:
{state["category"]}

사용자 질문:
{state["question"]}
"""

    search_query = llm.invoke(prompt).content.strip()

    return {
        "search_query": search_query
    }

def reranker_node(state: AgentState):
    llm = get_llm()

    candidates = state.get("candidate_sources", [])

    if not candidates:
        return {"sources": []}

    candidate_text = "\n\n".join(
        [
            f"[{index}] {source['law_name']} {source.get('article') or ''}\n"
            f"{source['content']}"
            for index, source in enumerate(candidates)
        ]
    )

    prompt = f"""
당신은 대한민국 법률 RAG 검색 결과 평가기입니다.

사용자의 질문을 해결하는 데 직접적으로 도움이 되는 법령만 선택하세요.

선택 기준:
- 사용자의 상황과 직접 관련이 있는가
- 실제 대응 절차나 법적 근거를 제공하는가
- 단순히 '사기', '피해자', '금융'이라는 단어가 비슷한 것만으로 선택하지 마세요
- 간접적이거나 관련성이 낮은 조문은 제외하세요
- 최대 4개만 선택하세요

사용자 질문:
{state["question"]}

후보 법령:
{candidate_text}

선택할 후보 번호만 쉼표로 출력하세요.
예:
0,2,5
"""

    response = llm.invoke(prompt).content.strip()

    selected = []

    for value in response.split(","):
        try:
            index = int(value.strip())

            if 0 <= index < len(candidates):
                selected.append(candidates[index])

        except ValueError:
            continue

    selected = selected[:4]

    print("\n=== 최종 선택 법령 ===")
    for source in selected:
        print(
            source["law_name"],
            source.get("article"),
        )

    return {
        "sources": selected,
    }


def generator_node(state: AgentState):
    llm = get_llm()

    sources = state.get("sources", [])

    if sources:
        context = "\n\n".join(
            [
                f"[출처 {index + 1}] {source['law_name']}\n"
                f"{source['content']}"
                for index, source in enumerate(sources)
            ]
        )
    else:
        context = "검색된 법령이 없습니다."

    prompt = f"""
당신은 대한민국 법률정보를 설명하는 AI입니다.

반드시 제공된 검색 법령 자료를 우선 근거로 답변하세요.

규칙:
1. 검색 자료에 없는 법령명이나 조문을 임의로 만들어내지 마세요.
2. 적용 여부가 불확실하면 명확히 불확실하다고 설명하세요.
3. 법률적 결론을 단정하지 마세요.
4. 사용자가 바로 이해할 수 있도록 간결하게 답변하세요.
5. 표(Table)는 사용하지 마세요.
6. HTML 태그를 사용하지 마세요.
7. 검색된 법령 원문을 답변에 길게 복사하지 마세요.
8. "관련 법령" 섹션은 만들지 마세요.
   관련 법령 원문은 프론트엔드의 별도 출처 카드에서 제공합니다.
9. 전체 답변은 너무 길지 않게 작성하세요.

반드시 아래 형식으로 답변하세요.

## 상황 분석
사용자의 상황을 2~4문장으로 정리합니다.

## 대응 방법
실제로 취할 수 있는 행동을 번호 목록으로 3~6개 제시합니다.

## 주의사항
사실관계에 따라 달라질 수 있는 부분과 확인할 사항을 간단히 설명합니다.

[질문]
{state["question"]}

[분야]
{state["category"]}

[검색된 법령]
{context}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
    }


workflow = StateGraph(AgentState)

workflow.add_node("classifier", classifier_node)
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("classifier")

workflow.add_edge("classifier", "query_rewriter")
workflow.add_edge("query_rewriter", "researcher")
workflow.add_edge("researcher", "reranker")
workflow.add_edge("reranker", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()

legal_graph = workflow.compile(
    checkpointer=memory
)


def run_legal_chat(question: str, session_id: str):
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