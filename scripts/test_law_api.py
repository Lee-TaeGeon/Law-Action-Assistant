"""국가법령정보센터 XML 응답 구조를 확인하는 단일 법령 읽기 전용 테스트.

실행 (프로젝트 루트): python scripts/test_law_api.py
다른 법령: python scripts/test_law_api.py --id 014656
원본 JSON/SQLite/ChromaDB 또는 법령 수집 코드에는 변경을 가하지 않습니다.
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def get_text(element, tag):
    if element is None:
        return ""
    return (element.findtext(tag) or "").strip()


def brief(value, limit=110):
    value = " ".join((value or "").split())
    return value[:limit] + ("..." if len(value) > limit else "")


def main():
    parser = argparse.ArgumentParser(description="법령 API XML 구조 검사 (읽기 전용)")
    parser.add_argument("--id", default="010719", help="조회할 법령 ID (기본 010719)")
    args = parser.parse_args()

    api_key = os.getenv("LAW_API_KEY")
    if not api_key:
        raise SystemExit("오류: 프로젝트 최상위 .env 파일에 LAW_API_KEY가 없습니다.")

    print(f"공식 법령 본문 API 조회 중: 법령 ID {args.id}")
    try:
        response = requests.get(
            "https://www.law.go.kr/DRF/lawService.do",
            params={"OC": api_key, "target": "law", "ID": args.id, "type": "XML"},
            timeout=30,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except requests.RequestException as exc:
        raise SystemExit(f"API 요청 실패: {type(exc).__name__} (키나 전체 URL은 공유하지 마세요)") from None
    except ET.ParseError:
        raise SystemExit("XML 파싱 실패: API 승인/응답 내용을 확인해야 합니다. 인증키는 공유하지 마세요.") from None

    info = root.find(".//기본정보")
    articles = root.findall(".//조문단위")
    print(f"XML 최상위 태그: {root.tag}")
    print(f"기본정보 발견: {'예' if info is not None else '아니오'}")
    for label, tag in [
        ("법령명", "법령명_한글"), ("법령 ID", "법령ID"),
        ("공포일자", "공포일자"), ("시행일자", "시행일자"),
        ("제개정구분", "제개정구분"),
    ]:
        print(f"{label}: {get_text(info, tag) or '(응답에서 찾지 못함)'}")
    print(f"조문단위 개수: {len(articles)}")

    if not articles:
        print("조문단위를 찾지 못했습니다. 상위 태그 목록:", [c.tag for c in list(root)[:15]])
        raise SystemExit("실제 XML 구조를 먼저 확인해야 합니다. 기존 DB는 변경되지 않았습니다.")

    branch_articles = [a for a in articles if get_text(a, "조문가지번호") not in ("", "0", "00")]
    numbered_duplicates = {}
    for a in articles:
        no = get_text(a, "조문번호")
        numbered_duplicates[no] = numbered_duplicates.get(no, 0) + 1
    print(f"가지번호가 있는 조문: {len(branch_articles)}개")
    print(f"기본 조문번호가 중복된 그룹: {sum(c > 1 for c in numbered_duplicates.values())}개")
    print(f"조문내용 필드가 채워진 조문: {sum(bool(get_text(a, '조문내용')) for a in articles)}개")
    print(f"항 없이 조문내용이 있는 조문: {sum(not a.findall('항') and bool(get_text(a, '조문내용')) for a in articles)}개")

    samples = [("첫 번째 조문", articles[0])]
    if branch_articles and branch_articles[0] is not articles[0]:
        samples.append(("가지번호 조문 예시", branch_articles[0]))
    content_only = next((a for a in articles if not a.findall("항") and get_text(a, "조문내용")), None)
    if content_only is not None and all(content_only is not a for _, a in samples):
        samples.append(("항 없는 조문 예시", content_only))

    for name, art in samples:
        print(f"\n[{name}]")
        for tag in ("조문번호", "조문가지번호", "조문제목", "조문시행일자", "조문여부"):
            print(f"  {tag}: {get_text(art, tag) or '(비어 있음)'}")
        print(f"  조문내용: {brief(get_text(art, '조문내용')) or '(비어 있음)'}")
        print(f"  항 개수: {len(art.findall('항'))}")

    print("\n검사 완료: 기존 파일·DB는 변경하지 않았습니다.")
    print("위 결과만 공유하세요. .env, 인증키, 요청 URL 전체는 공유하지 마세요.")


if __name__ == "__main__":
    main()
