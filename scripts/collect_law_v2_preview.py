"""국가법령정보센터 법령 1건 수집 형식 검증용 (기존 데이터 무변경).

사용 예:
    python scripts/collect_law_v2_preview.py
    python scripts/collect_law_v2_preview.py --law-id 014808

기본 출력 폴더는 저장소 바깥의 Law-Action-Assistant-Preview.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "Law-Action-Assistant-Preview"
API_URL = "https://www.law.go.kr/DRF/lawService.do"


def xml_text(node: ET.Element, *paths: str) -> str:
    """검색 가능한 경로에서 첫 번째 값 반환. XML 필드가 없으면 빈 문자열."""
    for path in paths:
        value = node.findtext(path)
        if value is not None and value.strip():
            return value.strip()
    return ""


def parse_law_xml(xml_bytes: bytes, requested_law_id: str) -> dict:
    root = ET.fromstring(xml_bytes)
    articles_xml = root.findall(".//조문단위")
    if not articles_xml:
        # API 에러/XML 형식 변경을 정상 수집으로 오인하지 않도록 막음.
        raise ValueError("조문단위가 없습니다. 인증 권한 또는 API 응답 형식을 확인하세요.")

    articles = []
    for article in articles_xml:
        paragraphs = []
        for para in article.findall("항"):
            items = []
            for item in para.findall("호"):
                sub_items = [
                    {
                        "sub_no": xml_text(sub, "목번호"),
                        "sub_content": xml_text(sub, "목내용"),
                    }
                    for sub in item.findall("목")
                ]
                items.append(
                    {
                        "item_no": xml_text(item, "호번호"),
                        "item_content": xml_text(item, "호내용"),
                        "sub_items": sub_items,
                    }
                )
            paragraphs.append(
                {
                    "para_no": xml_text(para, "항번호"),
                    "para_content": xml_text(para, "항내용"),
                    "items": items,
                }
            )

        number = xml_text(article, "조문번호")
        branch = xml_text(article, "조문가지번호")
        display = f"제{number}조"
        if branch and branch.isdigit() and int(branch) > 0:
            display += f"의{int(branch)}"
        elif branch and not branch.isdigit():
            display += f"(가지번호: {branch})"

        articles.append(
            {
                "article_no": number,
                "article_branch_no": branch,  # 공식 API 원본 필드(0이나 공백도 보존)
                "article_display": display,
                "article_title": xml_text(article, "조문제목"),
                "article_body": xml_text(article, "조문내용"),
                "article_effective_date": xml_text(article, "조문시행일자"),
                "article_status": xml_text(article, "조문여부"),
                "article_revision_type": xml_text(article, "조문제개정유형"),
                "article_changed": xml_text(article, "조문변경여부"),
                "paragraphs": paragraphs,
            }
        )

    result = {
        "source": "국가법령정보센터 공식 API",
        "api_target": "law",  # 공포일 기준 본문; 시행일 기준 API(eflaw)와 구분 필요
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "law_id": xml_text(root, ".//기본정보/법령ID", ".//법령ID") or requested_law_id,
        "law_name": xml_text(root, ".//기본정보/법령명_한글", ".//법령명_한글", ".//법령명한글"),
        "promulgation_date": xml_text(root, ".//기본정보/공포일자", ".//공포일자"),
        "promulgation_number": xml_text(root, ".//기본정보/공포번호", ".//공포번호"),
        "effective_date": xml_text(root, ".//기본정보/시행일자", ".//시행일자"),
        "revision_type": xml_text(root, ".//기본정보/제개정구분", ".//제개정구분"),
        "data": articles,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="법령 1건의 공식 API 원문/구조화 결과 검증")
    parser.add_argument("--law-id", default="010719", help="조회할 법령 ID (기본값: 010719)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="검증 파일 저장 폴더")
    args = parser.parse_args()

    if not args.law_id.isdigit():
        parser.error("--law-id는 숫자로 입력하세요.")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("LAW_API_KEY")
    if not api_key:
        raise SystemExit("LAW_API_KEY를 찾을 수 없습니다. 프로젝트 .env 파일을 확인하세요.")

    print(f"법령 1건 조회: ID={args.law_id} (기존 법령 DB 수정 없음)")
    try:
        response = requests.get(
            API_URL,
            params={"OC": api_key, "target": "law", "ID": args.law_id, "type": "XML"},
            timeout=(5, 25),
        )
        if response.status_code != 200:
            raise RuntimeError(f"API HTTP 상태 코드: {response.status_code}")
        data = parse_law_xml(response.content, args.law_id)
    except requests.RequestException as exc:
        # 요청 URL에 API 인증값이 포함될 수 있으므로 예외 문자열은 출력하지 않음.
        raise SystemExit(f"API 요청 실패: {type(exc).__name__} (인증키/URL은 출력하지 않음)") from None
    except (ValueError, ET.ParseError, RuntimeError) as exc:
        raise SystemExit(f"응답 검증 실패: {exc}") from None

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"law_{args.law_id}_original.xml"
    json_path = output / f"law_{args.law_id}_preview.json"
    if raw_path.exists() or json_path.exists():
        raise SystemExit("출력 파일이 이미 존재합니다. 덮어쓰지 않으니 기존 파일을 확인하세요.")
    raw_path.write_bytes(response.content)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter((article["article_no"], article["article_branch_no"]) for article in data["data"])
    duplicate_keys = sum(1 for count in counts.values() if count > 1)
    branched = sum(1 for item in data["data"] if item["article_branch_no"].isdigit() and int(item["article_branch_no"]) > 0)
    no_para_with_body = sum(1 for item in data["data"] if not item["paragraphs"] and item["article_body"])

    print(f"법령명: {data['law_name'] or '(응답에서 확인되지 않음)'}")
    print(f"공포일: {data['promulgation_date'] or '(없음)'} / 시행일: {data['effective_date'] or '(없음)'}")
    print(f"조문 수: {len(data['data'])}")
    print(f"공식 가지번호가 있는 조문 수: {branched}")
    print(f"조문번호+가지번호까지 동일한 중복 키 그룹 수: {duplicate_keys}")
    print(f"항이 없지만 조문내용은 존재하는 조문 수: {no_para_with_body}")
    for article in data["data"][:8]:
        print(f"  - {article['article_display']} {article['article_title']} (본문 {len(article['article_body'])}자)")
    print(f"원본 XML 저장: {raw_path}")
    print(f"검증 JSON 저장: {json_path}")
    print("주의: 수집 성공만으로 데이터의 법적 적용 시점이 검증된 것은 아닙니다.")


if __name__ == "__main__":
    main()
