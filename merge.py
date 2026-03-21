import json
import os

def merge_legal_data():
    combined_data = []
    total_files = 0

    # 1. law_data_batch_1 ~ 55까지 합치기
    for i in range(1, 56):
        filename = f'law_data_batch_{i}.json'
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    combined_data.extend(data)
                    total_files += 1
                except Exception as e:
                    print(f"❌ {filename} 읽기 오류: {e}")
        else:
            print(f"⚠️ {filename} 파일을 찾을 수 없습니다. (건너뜀)")

    # 2. 마지막 law_data_final.json 합치기
    final_file = 'law_data_final.json'
    if os.path.exists(final_file):
        with open(final_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                combined_data.extend(data)
                total_files += 1
                print(f"✅ {final_file} 통합 완료!")
            except Exception as e:
                print(f"❌ {final_file} 읽기 오류: {e}")

    # 3. 최종 결과 저장
    output_name = 'korean_law_full_dataset.json'
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print("\n" + "="*45)
    print(f"🎊 데이터 통합 작업이 완료되었습니다!")
    print(f"📁 합쳐진 파일 수: {total_files}개")
    print(f"📊 최종 수집 법령 개수: {len(combined_data)}개")
    print(f"💾 저장된 파일명: {output_name}")
    print("="*45)

if __name__ == "__main__":
    merge_legal_data()