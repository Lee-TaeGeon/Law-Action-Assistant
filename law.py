import requests
import xml.etree.ElementTree as ET
import json
import time
import os

# 태건님의 API 키
API_KEY = "ltg3571128"

def get_all_law_list():
    """1단계: 대한민국 모든 법령의 '이름'과 'ID' 목록을 가져옵니다."""
    all_laws = []
    page = 1
    rows_per_page = 1000
    
    print("📢 [1단계] 전체 법령 목록 수집 시작...")
    while True:
        url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={API_KEY}&target=law&type=XML&numOfRows={rows_per_page}&page={page}"
        res = requests.get(url)
        root = ET.fromstring(res.content)
        
        total_count = int(root.findtext("totalCnt", "0"))
        law_elements = root.findall(".//law")
        
        if not law_elements: break
            
        for law in law_elements:
            all_laws.append({
                "law_id": law.findtext("법령ID"),
                "law_name": law.findtext("법령명한글")
            })
        
        print(f"✅ 목록 수집 중... ({len(all_laws)} / {total_count})")
        if len(all_laws) >= total_count: break
        page += 1
        time.sleep(0.5)
        
    return all_laws

def get_law_detail_structured(law_id, law_name):
    """2단계: 특정 법령의 조, 항, 호, 목 구조를 정밀 추출합니다."""
    url = f"https://www.law.go.kr/DRF/lawService.do?OC={API_KEY}&target=law&ID={law_id}&type=XML"
    try:
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        articles = root.findall(".//조문단위")
        
        law_data = []
        for art in articles:
            article_info = {
                "article_no": art.findtext("조문번호"),
                "article_title": art.findtext("조문제목", ""),
                "paragraphs": []
            }
            # 항(Paragraph) 추출
            for para in art.findall("항"):
                para_info = {
                    "para_no": para.findtext("항번호", ""),
                    "para_content": para.findtext("항내용", "").strip(),
                    "items": []
                }
                # 호(Item) 추출
                for item in para.findall("호"):
                    item_info = {
                        "item_no": item.findtext("호번호", ""),
                        "item_content": item.findtext("호내용", "").strip(),
                        "sub_items": []
                    }
                    # 목(Sub-item) 추출
                    for sub in item.findall("목"):
                        item_info["sub_items"].append({
                            "sub_no": sub.findtext("목번호"),
                            "sub_content": sub.findtext("목내용", "").strip()
                        })
                    para_info["items"].append(item_info)
                article_info["paragraphs"].append(para_info)
            law_data.append(article_info)
        return {"law_name": law_name, "law_id": law_id, "data": law_data}
    except:
        return None

def start_project():
    # 1. 목록 먼저 가져오기
    law_list = get_all_law_list()
    
    # 2. 상세 내용 수집 시작
    print(f"\n🚀 [2단계] 총 {len(law_list)}개 법령 상세 수집 시작 (조,항,호 포함)")
    
    final_batch = []
    for i, law in enumerate(law_list):
        print(f"[{i+1}/{len(law_list)}] '{law['law_name']}' 수집 중...")
        
        detail = get_law_detail_structured(law['law_id'], law['law_name'])
        if detail:
            final_batch.append(detail)
        
        time.sleep(0.3) # 서버 보호용 휴식

        # 100개마다 파일로 분할 저장 (데이터 손실 방지)
        if (i + 1) % 100 == 0:
            batch_num = (i + 1) // 100
            with open(f"law_data_batch_{batch_num}.json", "w", encoding="utf-8") as f:
                json.dump(final_batch, f, ensure_ascii=False, indent=4)
            print(f"💾 {batch_num}번 배치 저장 완료!")
            final_batch = [] # 메모리 초기화

    # 나머지 데이터 저장
    if final_batch:
        with open("law_data_final.json", "w", encoding="utf-8") as f:
            json.dump(final_batch, f, ensure_ascii=False, indent=4)

    print("\n🎊 모든 법령의 조, 항, 호 수집이 끝났습니다!")

if __name__ == "__main__":
    start_project()