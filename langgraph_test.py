import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# LangChain & Graph 라이브러리
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 환경 설정 및 키 로드
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Law-Action-Assistant-Graph"

# 2. 상태 정의 (데이터 바구니)
class AgentState(TypedDict):
    question: str      # 사용자 질문
    category: str      # 민사/형사/기타 분류
    context: List[str] # 검색된 법령 내용 (없을 수 있으므로 아래에서 get 처리)
    answer: str        # 최종 답변

# 3. 모델 및 DB 설정
# 태건님 환경에서 성공했던 모델명 'gemini-2.5-flash' 사용
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# 임베딩 및 DB 로드
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
db = Chroma(persist_directory="./law_db_full", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- 4. 노드(Node) 함수 정의 ---

def classifier_node(state: AgentState):
    """질문을 분석하여 민사, 형사, 기타로 분류"""
    print("--- [노드: 분류기] 질문 분석 중... ---")
    prompt = f"다음 법률 질문을 보고 '민사', '형사', '기타' 중 하나로 분류해줘. 단어만 대답해. 질문: {state['question']}"
    response = llm.invoke(prompt)
    return {"category": response.content.strip()}

def legal_researcher_node(state: AgentState):
    """분류된 카테고리에 맞춰 DB에서 법령 검색"""
    print(f"--- [노드: 리서처] {state['category']} 관련 법령 검색 중... ---")
    query = f"{state['category']} {state['question']}"
    docs = retriever.invoke(query)
    context = [d.page_content for d in docs]
    return {"context": context}

def answer_generator_node(state: AgentState):
    """검색된 법령을 바탕으로 최종 답변 생성"""
    print("--- [노드: 생성기] 답변 작성 중... ---")
    
    # [핵심 수정] 검색을 건너뛰어 context가 없을 경우를 대비해 빈 리스트로 처리
    context_list = state.get("context", [])
    context_text = "\n".join(context_list) if context_list else "참고할 법령 데이터가 없습니다."
    
    system_instruction = f"""
    당신은 대한민국의 법률 전문가입니다. 분류된 카테고리({state['category']})에 맞춰 답변하세요.
    법률 관련 질문인 경우 참고 법령을 바탕으로 상세히 설명하고, 
    일상적인 인사나 법률과 무관한 질문인 경우 그에 맞게 친절하게 응대하세요.
    """
    final_prompt = f"{system_instruction}\n\n[참고 법령]\n{context_text}\n\n[질문]\n{state['question']}"
    response = llm.invoke(final_prompt)
    return {"answer": response.content}

# --- 5. 그래프 구성 (Workflow) ---

workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("classifier", classifier_node)
workflow.add_node("researcher", legal_researcher_node)
workflow.add_node("generator", answer_generator_node)

# 🚦 조건부 로직: 분류 결과에 따라 길을 정함
def route_by_category(state: AgentState):
    # AI가 '민사', '형사'라고 분류했을 때만 검색 수행
    if state["category"] in ["민사", "형사"]:
        return "researcher"
    else:
        return "generator"

# 흐름 연결
workflow.set_entry_point("classifier")

# 조건부 에지 추가
workflow.add_conditional_edges(
    "classifier",
    route_by_category,
    {
        "researcher": "researcher",
        "generator": "generator"
    }
)

workflow.add_edge("researcher", "generator")
workflow.add_edge("generator", END)

# 컴파일
app = workflow.compile()

# # 그래프 이미지 저장
# try:
#     graph_png = app.get_graph().draw_mermaid_png()
#     with open("law_graph_structure.png", "wb") as f:
#         f.write(graph_png)
#     print("\n✅ 그래프 이미지가 'law_graph_structure.png'로 저장되었습니다!")
# except Exception as e:
#     print(f"\n❌ 이미지 생성 실패 (무시 가능): {e}")

# --- 6. 실행 테스트 ---
if __name__ == "__main__":
    # 테스트 1: 일상 인사 (검색 건너뜀)
    #test_inputs = {"question": "안녕? 반가워!"}
    
    # 테스트 2: 법률 질문 (검색 수행) - 테스트하고 싶으면 아래 주석 해제
    test_inputs = {"question": "전세금을 못 받고 있어요."}
    
    print("\n🚀 실전 랭그래프 가동!!")
    result = app.invoke(test_inputs)
    
    print("\n================ FINAL ANSWER ================")
    print(result["answer"])
    print("===============================================")