"""법령 수집 JSON을 변경하지 않고 구조상 이상 징후를 점검합니다.

사용법:
    python scripts/audit_law_json.py law_data_batch_1.json
    python scripts/audit_law_json.py korean_law_full_dataset.json
    python scripts/audit_law_json.py .  # law_data_batch_*.json + law_data_final.json

실행 위치에 law_audit_report.json만 생성합니다.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def article_body_present(article):
    for paragraph in article.get("paragraphs", []):
        if not isinstance(paragraph, dict):
            continue
        if str(paragraph.get("para_content") or "").strip():
            return True
        for item in paragraph.get("items", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("item_content") or item.get("content") or "").strip():
                return True
            for sub in item.get("sub_items", []):
                if isinstance(sub, dict) and str(sub.get("sub_content") or sub.get("content") or "").strip():
                    return True
    return False


def analyze_one_file(path, stats, examples, law_names, max_examples=25):
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if isinstance(dataset, dict):
        dataset = dataset.get("data", [])
    if not isinstance(dataset, list):
        raise ValueError(f"{path}: 최상위 데이터가 법령 목록(list)이 아닙니다.")

    stats["file_count"] += 1
    for law in dataset:
        if not isinstance(law, dict):
            stats["invalid_law_records"] += 1
            continue
        stats["law_count"] += 1
        law_name = str(law.get("law_name") or "").strip()
        if not law_name:
            stats["missing_law_name"] += 1
        law_names[law_name] += 1

        # 최초 수집 데이터에 날짜 정보가 보존되어 있는지 확인
        if not any(key in law for key in ("시행일자", "시행일", "effective_date", "enforcement_date")):
            stats["law_without_effective_date_field"] += 1
        if not any(key in law for key in ("수집일", "collected_at", "fetched_at")):
            stats["law_without_collection_date_field"] += 1

        articles = law.get("data", [])
        if not isinstance(articles, list):
            stats["invalid_article_lists"] += 1
            continue
        by_number = defaultdict(list)
        for article in articles:
            if not isinstance(article, dict):
                stats["invalid_article_records"] += 1
                continue
            stats["article_count"] += 1
            number = str(article.get("article_no") or "").strip()
            title = str(article.get("article_title") or "").strip()
            has_body = article_body_present(article)
            if not number:
                stats["missing_article_number"] += 1
            if not title and not has_body:
                stats["empty_title_and_body"] += 1
            elif title and not has_body:
                stats["title_only_without_body"] += 1
            if "article_branch_no" not in article and "조문가지번호" not in article:
                stats["article_without_branch_number_field"] += 1
            if number:
                by_number[number].append({"title": title, "has_body": has_body})

        duplicated_in_law = False
        for number, entries in by_number.items():
            if len(entries) <= 1:
                continue
            duplicated_in_law = True
            stats["repeated_number_groups"] += 1
            stats["repeated_number_records"] += len(entries)
            if len(examples) < max_examples:
                examples.append({
                    "law_name": law_name,
                    "article_no": number,
                    "records": entries[:8],
                })
        if duplicated_in_law:
            stats["laws_with_repeated_number"] += 1


def main():
    parser = argparse.ArgumentParser(description="법령 JSON 읽기 전용 품질 검사")
    parser.add_argument("source", type=Path, help="수집 JSON 파일 또는 배치 JSON 폴더")
    args = parser.parse_args()
    source = args.source.resolve()
    if source.is_dir():
        paths = sorted(source.glob("law_data_batch_*.json"))
        last_file = source / "law_data_final.json"
        if last_file.is_file():
            paths.append(last_file)
    else:
        paths = [source]
    if not paths:
        parser.error("검사할 배치 JSON 파일을 찾지 못했습니다.")

    stats = Counter()
    examples = []
    law_names = Counter()
    for path in paths:
        print(f"검사 중: {path.name}")
        analyze_one_file(path, stats, examples, law_names)

    stats["repeated_law_names"] = sum(count - 1 for count in law_names.values() if count > 1)
    report = {
        "source_files": [str(path) for path in paths],
        "statistics": dict(stats),
        "repeated_article_number_examples": examples,
        "notice": "중복 번호와 본문 없음은 데이터 품질 검토 대상이며, 실제 법령 오류라고 단정할 수 없습니다. 공식 원문과 대조해야 합니다.",
    }
    report_path = Path.cwd() / "law_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 검사 결과 ===")
    for key, value in stats.items():
        print(f"{key}: {value:,}")
    print(f"\n보고서: {report_path}")
    print("원본 JSON과 기존 SQLite/ChromaDB는 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
