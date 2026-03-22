import os
from dotenv import load_dotenv 
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- [세팅] ---
# 1. .env 파일 로드
load_dotenv()

# 2. 키 가져오기 (이제 코드에 직접 안 써도 됩니다!)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-flash-latest')

# DB 및 임베딩 설정 (태건님 기존 설정)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
db = Chroma(persist_directory="./law_db_full", embedding_function=embeddings)

# 대화 내역을 저장할 리스트
chat_history = []

def ask_legal_chat(user_question):
    global chat_history
    try:
        # 1. 현재 질문과 관련된 법령 검색
        docs = db.similarity_search(user_question, k=5)
        context = "\n\n".join([d.page_content for d in docs])
        
        # 2. 이전 대화 요약본 만들기 (맥락 유지용)
        history_text = "\n".join([f"사용자: {h['q']}\nAI: {h['a']}" for h in chat_history[-3:]]) # 최근 3개 대화만 유지
        
        # 3. 프롬프트 구성 (지시사항 강화)
        prompt = f"""
        너는 대한민국의 엄격하고 정확한 '법률 전문 상담 AI'야. 
        사용자의 질문에 답할 때, 반드시 아래 [참고 법령 데이터]에서 근거를 찾아 답변해.

        ### [필수 규칙] ###
        1. 모든 답변에는 반드시 관련 법령 명칭과 구체적인 조항(예: 민법 제686조)을 명시할 것.
        2. 제공된 데이터에 없는 내용은 '법령 데이터에 근거가 부족하다'고 말하고 아는 선에서 답할 것.
        3. 이전 대화 맥락을 고려하되, 새로운 질문마다 다시 한번 관련 법 조항을 언급하며 전문성을 유지할 것.

        [이전 대화 맥락]:
        {history_text}
        
        [참고 법령 데이터]:
        {context}
        
        [현재 사용자의 질문]:
        {user_question}
        
        답변 형식: 
        - 상황 분석
        - 관련 법 조항 및 근거
        - 실천적인 해결 방안
        """
        
        # 4. 답변 생성
        response = model.generate_content(prompt)
        answer = response.text
        
        # 5. 대화 내역에 저장
        chat_history.append({"q": user_question, "a": answer})
        return answer
    
    except Exception as e:
        return f"❌ 에러 발생: {str(e)}"

# --- [무한 채팅 루프] ---
if __name__ == "__main__":
    print("⚖️ 법률 상담 챗봇이 시작되었습니다! (종료하려면 'exit' 입력)")
    print("-" * 50)
    
    while True:
        user_input = input("\n👤 나: ")
        if user_input.lower() in ['exit', 'quit', '종료']:
            print("상담을 종료합니다. 안녕히 가세요!")
            break
            
        print("\n🤖 챗봇 답변 생성 중...")
        result = ask_legal_chat(user_input)
        print(f"\n🤖 AI: {result}")
        print("-" * 50)