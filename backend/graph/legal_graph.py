import os
from functools import lru_cache
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.services.rag_service import search_laws


load_dotenv()


class AgentState(TypedDict, total=False):
    question: str
    category: str
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
    sources = search_laws(
        question=state["question"],
        category=state["category"],
        k=5,
    )

    return {
        "sources": sources,
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

반드시 아래 검색된 법령 자료를 우선 근거로 사용하세요.

규칙:
1. 검색 자료에 없는 법령이나 조문을 임의로 만들어내지 마세요.
2. 적용 여부가 불확실하면 불확실하다고 설명하세요.
3. 법률적 결론을 단정하지 말고 사실관계에 따라 달라질 수 있음을 알려주세요.
4. 답변은 아래 형식을 따르세요.

## 상황 분석
사용자의 상황을 간단히 정리합니다.

## 관련 법령
실제로 검색된 법령을 설명합니다.

## 대응 방법
사용자가 취할 수 있는 실질적인 절차를 순서대로 설명합니다.

## 주의사항
추가로 확인해야 할 사실이나 한계를 설명합니다.

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
workflow.add_node("researcher", researcher_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("classifier")

workflow.add_edge("classifier", "researcher")
workflow.add_edge("researcher", "generator")
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