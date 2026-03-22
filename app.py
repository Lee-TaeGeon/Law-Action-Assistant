import streamlit as st
import os
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph 라이브러리
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq  
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
os.environ["LANGCHAIN_PROJECT"] = "Law-Action-Assistant-Final-Fix" 

# 3. 모델 및 DB 로드 (캐싱)
@st.cache_resource
def init_models():
    # 2. Gemini 대신 Groq 모델로 선언
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", # 혹은 "llama3-8b-8192" (더 빠름)
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )
    # # 2.5-flash 대신 가장 안정적인 1.5-flash 사용 권장 (쿼터 이슈 방지)
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash", 
    #     google_api_key=os.getenv("GEMINI_API_KEY"),
    #     temperature=0.1
    # )
    
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
    need_search: bool        
    history_category: str    

def classifier_node(state: AgentState):
    print("--- [노드: 분류기] 문맥 파악 및 검색 필요성 정밀 판단 중... ---")
    last_topic = state.get("history_category", "없음")
    
    prompt = f"""
    당신은 법률 상담 효율화 전문가입니다.
    현재 질문이 이전 대화의 주제인 [{last_topic}]의 연장선상에 있는지 판단하세요.

    [이전 주제]: {last_topic}
    [현재 질문]: {state['question']}

    판단 기준:
    1. 분야: 질문의 법률 분야(민사, 형사 등)를 적으세요.
    2. 검색필요 (YES/NO): 
       - 만약 질문이 [{last_topic}]과 관련된 '합의금', '처벌', '절차' 등에 대한 것이라면 무조건 'NO'라고 하세요.
       - 주제가 완전히 바뀌거나 첫 질문일 때만 'YES'라고 하세요.
    
    답변 형식:
    분야: [분야명]
    검색필요: [YES/NO]
    """
    
    response = llm.invoke(prompt).content
    print(f"--- [분류기 AI 응답]:\n{response}\n------------------")

    category = "법률"
    need_search = True
    
    for line in response.split('\n'):
        if "분야:" in line:
            category = line.split("분야:")[1].strip()
        if "검색필요:" in line:
            if "NO" in line.upper():
                need_search = False

    # [수정된 핵심 로직] 
    # 이전 주제가 '교통사고'인데 현재가 '민사'나 '형사'로 나와도 
    # 맥락상 같은 사건이면 검색을 강제로 막습니다.
    relevant_keywords = ["합의", "금", "보험", "처벌", "사고", "과실"]
    is_related = any(kw in state['question'] for kw in relevant_keywords)

    if last_topic != "없음" and (last_topic in category or is_related):
        print(f"!!! 맥락 유지 확인 ({last_topic} -> {category}): 검색 강제 차단 !!!")
        need_search = False
        
    return {
        "category": category, 
        "need_search": need_search, 
        "history_category": category if need_search else last_topic # 검색 안 할 땐 주제 유지
    }

def legal_researcher_node(state: AgentState):
    """새로운 법령 검색이 필요한 경우에만 실행"""
    query = f"{state['category']} 관련 법령 {state['question']}"
    print(f"--- [노드: 리서처] 분야({state['category']}) 기반 신규 검색 중... ---")
    
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = [d.page_content for d in docs]
    return {"context": context}

def answer_generator_node(state: AgentState):
    """최종 통합 답변 생성 (가이드라인을 강화하여 답변 분량 확보)"""
    print("--- [노드: 생성기] 답변 작성 중... ---")
    category = state.get("category", "법률")
    context_list = state.get("context", [])
    
    # 문맥이 없어도 이전 대화 내용을 최대한 활용하도록 지시
    if not context_list:
        context_text = "제공된 새로운 법령 데이터는 없으나, 이전 대화에서 언급된 법적 사실과 일반적인 법률 지식을 바탕으로 상세히 답변하세요."
    else:
        context_text = "\n\n".join(context_list)
    
    system_instruction = f"""
    당신은 대한민국의 유능하고 친절한 법률 전문가입니다. 사용자의 질문에 대해 신뢰감을 줄 수 있도록 상세하고 깊이 있게 답변하세요.
    
    [답변 지침]:
    1. 분야({category})에 맞는 전문 용어를 적절히 섞어 신뢰도를 높이세요.
    2. 이전 대화 맥락이 있다면 이를 반드시 언급하며 답변의 연속성을 유지하세요.
    3. 각 섹션(상황 분석, 법적 근거, 대응 방법)은 최소 3~4문장 이상의 구체적인 설명을 포함해야 합니다.
    4. 법적 근거 섹션에서는 관련된 법 조항의 취지나 판례의 경향을 풀어서 설명하세요.
    5. 답변은 반드시 한국어로 작성하며, 전문가다운 격식 있는 문체를 사용하세요.
    6. 답변 끝에 '⚠️ 본 답변은 참고용이며 법적 효력이 없습니다.'를 포함하세요.
    
    [답변 구조]:
    ### 📝 상황 분석
    (사용자의 상황을 법적으로 재해석하고 핵심 쟁점을 짚어주세요)

    ### ⚖️ 법적 근거
    (참고 데이터와 법률 지식을 바탕으로 상세한 근거를 제시하세요)

    ### 🛠️ 대응 방법
    (사용자가 당장 실천할 수 있는 구체적인 단계별 대안을 제시하세요)
    """
    
    final_prompt = f"{system_instruction}\n\n[참고 데이터]\n{context_text}\n\n[현재 질문]\n{state['question']}"
    response = llm.invoke(final_prompt)
    return {"answer": response.content}
# --- 5. 그래프 구성 ---

workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("researcher", legal_researcher_node)
workflow.add_node("generator", answer_generator_node)

def route_by_logic(state: AgentState):
    if not state.get("need_search", True):
        return "generator"
    return "researcher"

workflow.set_entry_point("classifier")
workflow.add_conditional_edges(
    "classifier",
    route_by_logic,
    {"researcher": "researcher", "generator": "generator"}
)
workflow.add_edge("researcher", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 6. Streamlit UI ---

with st.sidebar:
    st.title("⚖️ 법률 AI 가이드")
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.session_state.history_cat = "없음" # 카테고리 초기화
        st.rerun()

st.title("⚖️ 지능형 법률 에이전트")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 스레드 ID 고정 (메모리 유지 핵심)
config = {"configurable": {"thread_id": "final_law_session_v1"}}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 문맥 분석 중...", expanded=True) as status:
            # [수정 핵심] context를 []로 강제 초기화하지 않고 전달
            inputs = {"question": prompt} 
            result = app.invoke(inputs, config=config) 
            
            cat = result.get("category", "기타")
            search_flag = result.get("need_search", True)
            
            if not search_flag:
                st.write(f"💡 주제(**{cat}**)를 기억하고 있습니다. 추가 검색 없이 답변합니다.")
            else:
                st.write(f"🔍 새로운 주제(**{cat}**)에 대한 법령을 검색합니다.")
            
            status.update(label="✅ 분석 완료!", state="complete", expanded=False)

        st.markdown(result["answer"])
        
        if result.get("context"):
            with st.expander("🔍 참조 법령 원문"):
                for i, content in enumerate(result["context"]):
                    st.info(f"📖 참조 {i+1}\n\n{content}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})