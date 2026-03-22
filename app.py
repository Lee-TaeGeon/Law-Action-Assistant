import streamlit as st
import os
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph 라이브러리
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 페이지 설정
st.set_page_config(
    page_title="Law-Action-Assistant", 
    page_icon="⚖️", 
    layout="wide"
)

# 2. 환경 변수 로드 및 LangSmith 설정
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Law-Action-Assistant-Multi-Legal" 

# 3. 모델 및 DB 로드 (캐싱)
@st.cache_resource
def init_models():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db = Chroma(persist_directory="./law_db_full", embedding_function=embeddings)
    return llm, db

llm, db = init_models()

# --- 4. LangGraph 상태 및 노드 정의 ---

class AgentState(TypedDict):
    question: str
    category: str
    context: List[str]
    answer: str

def classifier_node(state: AgentState):
    """질문을 분석하여 관련된 모든 법률 분야(복합 분야 가능) 추출"""
    print("--- [노드: 분류기] 질문 성격 분석 중... ---")
    prompt = f"""
    당신은 법률 분류 전문가입니다. 다음 질문을 분석하여 관련된 법률 분야를 모두 나열하세요.
    
    1. 법률 관련 질문인 경우: '민사', '형사', '가사', '행정', '노동', '상법' 등 관련 분야를 모두 쉼표로 구분해 적으세요.
       (예: 돈을 빌려가서 안 갚고 잠적한 경우 -> '민사, 형사')
    2. 단순 인사나 법률과 무관한 경우: '일상'이라고만 대답하세요.
    
    질문: {state['question']}
    결과:"""
    response = llm.invoke(prompt)
    return {"category": response.content.strip()}

def legal_researcher_node(state: AgentState):
    """추출된 모든 카테고리를 키워드로 활용해 22만 건 DB 검색"""
    # 여러 카테고리가 섞여 있어도 검색어에 모두 포함하여 정확도 향상
    query = f"{state['category']} 관련 법령 {state['question']}"
    print(f"--- [노드: 리서처] 분야({state['category']}) 기반 검색 중... ---")
    
    # 복합 문제일 수 있으므로 검색 결과(k)를 5개로 늘림
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = [d.page_content for d in docs]
    return {"context": context}

def answer_generator_node(state: AgentState):
    """최종 통합 답변 생성"""
    print("--- [노드: 생성기] 통합 조언 작성 중... ---")
    category = state.get("category", "법률")
    context_list = state.get("context", [])
    context_text = "\n\n".join(context_list) if context_list else "참고할 직접적인 법령 데이터가 없습니다."
    
    system_instruction = f"""
    당신은 대한민국의 법률 전문가입니다. 분석된 분야({category})를 바탕으로 답변하세요.
    1. 복합적인 문제(예: 민사+형사)인 경우 각 관점에서의 대응 방안을 나누어 설명하세요.
    2. 반드시 '### 📝 상황 분석', '### ⚖️ 법적 근거', '### 🛠️ 대응 방법' 순으로 답변하세요.
    3. 답변 끝에 '⚠️ 본 답변은 참고용이며 법적 효력이 없습니다.'를 포함하세요.
    """
    final_prompt = f"{system_instruction}\n\n[참고 법령]\n{context_text}\n\n[질문]\n{state['question']}"
    response = llm.invoke(final_prompt)
    return {"answer": response.content}

# --- 5. 그래프 구성 (Workflow) ---

workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("researcher", legal_researcher_node)
workflow.add_node("generator", answer_generator_node)

def route_by_category(state: AgentState):
    # '일상'이 포함되지 않았다면 법률 검색으로 진행
    if "일상" not in state["category"]:
        return "researcher"
    return "generator"

workflow.set_entry_point("classifier")
workflow.add_conditional_edges(
    "classifier",
    route_by_category,
    {"researcher": "researcher", "generator": "generator"}
)
workflow.add_edge("researcher", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()

# --- 6. Streamlit UI 구성 ---

with st.sidebar:
    st.title("⚖️ 법률 AI 가이드")
    st.info("대한민국 22만 건 법령 기반 복합 법률 에이전트")
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

st.title("⚖️ 상황 맞춤형 법률 AI 에이전트")
st.caption("민사, 형사, 가사 등 복합적인 법률 문제에 대해 통합 솔루션을 제공합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("법률 고민(예: 보이스피싱, 층간소음, 전세사기 등)을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🚀 지능형 법률 분석 중...", expanded=True) as status:
            st.write("🔍 질문의 법적 성격 분석 중 (민사/형사/복합)...")
            inputs = {"question": prompt, "context": []}
            result = app.invoke(inputs)
            
            category = result.get("category", "기타")
            st.write(f"⚖️ 관련 분야 탐지: **{category}**")
            
            if "일상" not in category:
                st.write(f"📚 {category} 관련 법령 검색 및 분석 중...")
            
            status.update(label="✅ 분석 완료!", state="complete", expanded=False)

        st.markdown(result["answer"])
        
        if result.get("context"):
            with st.expander("🔍 분석에 활용된 법령 원문 보기"):
                for i, content in enumerate(result["context"]):
                    st.info(f"📖 참조 {i+1}\n\n{content}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})