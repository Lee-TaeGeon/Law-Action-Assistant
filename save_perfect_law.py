import json
from tqdm import tqdm

def save_perfect_law_texts(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📂 [고정밀 텍스트 추출] {len(data)}개 법령 분석 중...")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for law in tqdm(data, desc="법전 구조화 중"):
            law_name = law.get('law_name', '알 수 없음')
            
            for article in law.get('data', []):
                article_no = article.get('article_no', '')
                article_title = article.get('article_title', '')
                
                # [조] 단위 시작
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
                        
                        # [목]이나 그 하위 단계까지 싹 긁어오기
                        for sub in item.get('sub_items', []):
                            s_no = sub.get('sub_no', '').strip()
                            s_content = sub.get('sub_content', '').strip()
                            full_text += f"      └─[목] {s_no} {s_content}\n"

                
                # 파일에 저장
                f_out.write(full_text.strip() + "\n")
                f_out.write("=" * 60 + "\n") # 조문 간 확실한 구분선

    print(f"\n✅ 구조화 완료! '{output_path}' 파일을 확인해 보세요.")

# 실행
save_perfect_law_texts("korean_law_full_dataset.json", "perfect_law_book.txt")