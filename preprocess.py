import os
import json
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_split_law_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    print(f"📖 [1단계] 검증된 로직으로 {len(data)}개 법령 분리 중...")
    
    for law in tqdm(data, desc="법전 구조화 중"):
        law_name = law.get('law_name', '알 수 없음')
        
        for article in law.get('data', []):
            article_no = article.get('article_no', '')
            article_title = article.get('article_title', '')
            
            # 태건님이 확인하신 서식 그대로 (불필요한 본문 로직 제거)
            full_text = f"<{law_name}>\n"
            full_text += f"제{article_no}조 ({article_title})\n"
            
            for para in article.get('paragraphs', []):
                p_no = para.get('para_no', '').strip()
                p_content = para.get('para_content', '').strip()
                if p_no or p_content:
                    full_text += f"  [항] {p_no} {p_content}\n"
                
                for item in para.get('items', []):
                    i_no = item.get('item_no', '').strip()
                    i_content = item.get('item_content', '').strip()
                    # 중복 제거 및 [호] 표시
                    clean_i = i_content.replace(i_no, "").strip()
                    full_text += f"    └─[호] {i_no} {clean_i}\n"
                    
                    for sub in item.get('sub_items', []):
                        s_no = sub.get('sub_no', '').strip()
                        s_content = sub.get('sub_content', '').strip()
                        full_text += f"      └─[목] {s_no} {s_content}\n"

            # 생성된 텍스트를 문서 객체로 변환
            documents.append(Document(
                page_content=full_text.strip(),
                metadata={"law_name": law_name, "article_no": article_no}
            ))
            
    return documents

def create_vector_db_full(documents):
    if not documents:
        print("❌ 임베딩할 문서가 없습니다.")
        return
        
    print(f"🚀 [2단계] 전체 {len(documents)}개 조문 임베딩 시작!")
    
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} # GPU가 있다면 'cuda' 권장
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # 배치 처리 (1,000개씩 끊어서 저장)
    batch_size = 1000
    db = None
    
    for i in tqdm(range(0, len(documents), batch_size), desc="DB 구축 중"):
        batch = documents[i : i + batch_size]
        if db is None:
            db = Chroma.from_documents(
                documents=batch, 
                embedding=embeddings, 
                persist_directory="./law_db_full"
            )
        else:
            db.add_documents(batch)
    
    print("\n✅ 모든 법령 구축 완료! './law_db_full' 폴더를 확인하세요.")

if __name__ == "__main__":
    target_file = "korean_law_full_dataset.json" 
    
    # 1. 데이터 로드 (확인된 고정밀 로직 반영)
    docs = load_and_split_law_data(target_file)
    
    if docs:
        # 2. 임베딩 및 DB 저장
        create_vector_db_full(docs)