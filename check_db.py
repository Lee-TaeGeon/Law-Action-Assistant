import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 어제와 동일한 임베딩 모델 설정
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cpu'} # GPU 사용 시 'cuda'
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 2. 구축된 DB 로드 (경로: ./law_db_full)
db_path = "./law_db_full"
if not os.path.exists(db_path):
    print(f"❌ DB 폴더를 찾을 수 없습니다: {db_path}")
else:
    # LangChain의 Chroma 객체를 통해 로드
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # 3. 테스트 질문 던져보기
    query = "프리랜서가 계약금도 못 받고 일만 시켰을 때 대처법 알려줘"
    
    print(f"\n🔍 질문: {query}")
    print("=" * 60)

    # 4. 유사도 검색 (상위 3개 추출)
    docs = db.similarity_search(query, k=3)

    # 5. 결과 출력
    if not docs:
        print("❓ 검색 결과가 없습니다. DB 구축 상태를 확인해 보세요.")
    for i, doc in enumerate(docs):
        law_name = doc.metadata.get('law_name', '알 수 없음')
        article_no = doc.metadata.get('article_no', '-')
        print(f"[{i+1}] 관련 법령: {law_name} (제{article_no}조)")
        print(f"내용:\n{doc.page_content}")
        print("-" * 60)