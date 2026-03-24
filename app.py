import streamlit as st
import os
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph 라이브러리
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 페이지 설정 및 디자인 (Custom CSS)
st.set_page_config(
    page_title="Law-Action-Assistant", 
    page_icon="⚖️", 
    layout="wide"
)

# 전문적인 느낌을 위한 스타일링 적용
# 전문적인 느낌을 위한 스타일링 적용 (글자색 보정 포함)
st.markdown("""
    <style>
    /* 메인 앱 배경 */
    .stApp { background-color: #f8f9fa; }
    
    /* 사이드바 배경색 고정 */
    section[data-testid="stSidebar"] { 
        background-color: #1e293b !important; 
    }
    
    /* 1. 사이드바 제목(st.title) 색상 변경 */
    section[data-testid="stSidebar"] h1 { 
        color: #ffffff !important; 
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    /* 2. 사이드바 라디오 버튼(메뉴) 글자색 및 크기 */
    div[data-testid="stRadio"] label p {
        color: #f1f5f9 !important;
        font-size: 17px !important;
        font-weight: 500 !important;
    }

    /* 3. 사이드바 내 모든 일반 텍스트 및 캡션 */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span { 
        color: #cbd5e1 !important; 
    }

    /* 채팅 메시지 및 기타 디자인 */
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stStatus { border: 1px solid #e2e8f0; border-radius: 10px; background-color: white; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #3b82f6; color: white; transition: all 0.3s; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-1px); }
    </style>
    """, unsafe_allow_html=True)

# 2. 환경 변수 로드 및 LangSmith 설정
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Law-Action-Assistant-Final-v1.2" 

# 3. 모델 및 DB 로드 (캐싱)
@st.cache_resource
def init_models():
    # Groq Llama 3.3 모델 (초고속 추론 엔진)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )
    # 한국어 특화 임베딩
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # 법령 Vector DB 로드 (경로 주의)
    db = Chroma(persist_directory="./law_db_full", embedding_function=embeddings)
    return llm, db

llm, db = init_models()

# --- 4. LangGraph 상태 및 노드 정의 ---

class AgentState(TypedDict):
    question: str
    category: str
    context: List[str]
    answer: str
    need_search: bool        
    history_category: str    

def classifier_node(state: AgentState):
    last_topic = state.get("history_category", "없음")
    prompt = f"""당신은 법률 상담 전문가입니다. 질문이 [{last_topic}] 연관 질문인지 판단하세요.
    [이전 주제]: {last_topic} | [현재 질문]: {state['question']}
    형식: 분야: [분야] / 검색필요: [YES/NO]"""
    
    response = llm.invoke(prompt).content
    category, need_search = "법률", True
    for line in response.split('\n'):
        if "분야:" in line: category = line.split("분야:")[1].strip()
        if "검색필요:" in line and "NO" in line.upper(): need_search = False

    relevant_keywords = ["합의", "금", "보험", "처벌", "사고", "과실"]
    if last_topic != "없음" and (last_topic in category or any(kw in state['question'] for kw in relevant_keywords)):
        need_search = False
        
    return {"category": category, "need_search": need_search, "history_category": category if need_search else last_topic}

def legal_researcher_node(state: AgentState):
    query = f"{state['category']} 관련 법령 {state['question']}"
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    return {"context": [d.page_content for d in docs]}

def answer_generator_node(state: AgentState):
    category = state.get("category", "법률")
    context_text = "\n\n".join(state.get("context", [])) if state.get("context") else "기존 맥락을 바탕으로 답변하세요."
    system_instruction = f"당신은 대한민국의 유능한 법률 전문가입니다. 분야({category})에 맞춰 상황 분석, 법적 근거, 대응 방법을 상세히 작성하세요."
    final_prompt = f"{system_instruction}\n\n[참고 데이터]\n{context_text}\n\n[현재 질문]\n{state['question']}"
    response = llm.invoke(final_prompt)
    return {"answer": response.content}

# --- 5. 그래프 구성 ---
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("researcher", legal_researcher_node)
workflow.add_node("generator", answer_generator_node)
workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", lambda x: "generator" if not x.get("need_search", True) else "researcher", {"researcher": "researcher", "generator": "generator"})
workflow.add_edge("researcher", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 6. 사이드바 메뉴바 구성 ---
with st.sidebar:
    st.title("⚖️ Law-Action Assistant")
    st.markdown("---")
    
    # 메뉴 선택 (확장성 확보)
    menu = st.radio(
        "Navigation Menu",
        ["💬 지능형 법률 상담", "📄 판결문 분석 (OCR)", "📊 나의 상담 리포트"],
        index=0
    )
    
    st.markdown("---")
    if st.button("🔄 대화 기록 초기화"):
        st.session_state.messages = []
        st.session_state.history_cat = "없음"
        st.rerun()
    st.markdown("---")
    st.caption("v1.2 | Developed by 이태건")
    st.caption("© 2026 Law-Action-Project")

# --- 7. 메뉴별 메인 화면 실행 ---

if menu == "💬 지능형 법률 상담":
    col1, col2 = st.columns([1, 8])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/3252/3252906.png", width=70)
    with col2:
        st.title("지능형 복합 법률 에이전트")
        st.subheader("대한민국 22만 건 법령 기반 AI 상담")

    if "messages" not in st.session_state: st.session_state.messages = []
    config = {"configurable": {"thread_id": "final_law_session_v1"}}

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("사건 내용이나 궁금한 법률 사항을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("⚖️ 법률 데이터 분석 중...", expanded=True) as status:
                st.write("🔍 질문 의도 및 문맥 파악 중...")
                result = app.invoke({"question": prompt}, config=config) 
                cat = result.get("category", "기타")
                if not result.get("need_search", True): st.write(f"📂 **이전 문맥({cat})**을 기억하여 즉시 답변합니다.")
                else: st.write(f"🌐 **{cat}** 분야 법령 데이터를 검색합니다.")
                st.write("✍️ 법률 대응 계획 생성 중...")
                status.update(label=f"✅ {cat} 분야 분석 완료", state="complete", expanded=False)

            st.markdown(result["answer"])
            if result.get("context"):
                with st.expander("📝 참조 법령 및 근거 데이터"):
                    for i, content in enumerate(result["context"]):
                        st.markdown(f"**[참조 {i+1}]**"); st.caption(content); st.divider()
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

elif menu == "📄 판결문 분석 (OCR)":
    st.title("📄 판결문 분석 (OCR)")
    st.subheader("사용자의 판결문을 분석하여 핵심 내용을 요약합니다.")
    st.warning("⚠️ **해당 서비스는 현재 준비 중입니다.**")
    st.markdown("""
    #### 🚀 업데이트 예정 기능:
    - PDF/이미지 기반 판결문 텍스트 추출 (OCR)
    - 복잡한 판결문 요약 및 핵심 쟁점 자동 도출
    - 현재 상담 중기 에이전트와 연동하여 유사 사례 매칭
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2911/2911230.png", width=150)

elif menu == "📊 나의 상담 리포트":
    st.title("📊 나의 상담 리포트")
    st.subheader("상담 이력을 분석하여 시각화된 리포트를 제공합니다.")
    st.info("📅 **2026년 상반기 업데이트 예정입니다.**")
    st.markdown("""
    #### 📈 제공 예정 데이터:
    - 주요 상담 카테고리 통계 (민사/형사/가사 등)
    - 상담 이력 기반 개인 맞춤형 법률 가이드북 PDF 생성
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3595/3595490.png", width=150)