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

# 방법 1: 모델명을 'models/gemini-1.5-flash'로 경로까지 명시해줍니다.
# 그래도 안 되면 'gemini-pro'로 테스트해볼 수도 있습니다.
model_name = 'models/gemini-flash-latest'
model = genai.GenerativeModel(model_name)

# 임베딩 및 DB 설정
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
db = Chroma(persist_directory="./law_db_full", embedding_function=embeddings)

# --- [챗봇 함수] ---
def ask_legal_bot(user_question):
    try:
        docs = db.similarity_search(user_question, k=5)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""
        너는 대한민국의 친절한 법률 상담 AI야. 
        아래 제공된 [법령 데이터]를 바탕으로 사용자의 [질문]에 답변해줘.
        
        [법령 데이터]:
        {context}
        
        [질문]:
        {user_question}
        
        [답변 지침]:
        1. 반드시 제공된 법령 내용을 근거로 답변할 것.
        2. 일반인도 이해하기 쉽게 풀어서 설명하고, 해결 방안을 제시할 것.
        3. 구체적인 조항(예: 민법 제000조)을 언급하며 답변할 것.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        # 에러가 나면 어떤 모델들이 사용 가능한지 리스트를 출력하도록 추가했습니다.
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return f"❌ 에러 발생: {str(e)}\n\n💡 사용 가능한 모델 목록:\n" + "\n".join(available_models)

# --- [실행] ---
if __name__ == "__main__":
    question = "식당에서 일을하고 돈을 받았는데 법적인걸 안 지켜서 준것 같아 금액이 좀 적은데 어떻게 하지? "
    print(f"\n💬 질문: {question}\n")
    print("🤖 챗봇 답변 생성 중...")
    print("-" * 50)
    print(ask_legal_bot(question))