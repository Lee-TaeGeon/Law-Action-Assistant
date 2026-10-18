"""법령 원본 XML의 조문번호 중복 원인을 읽기 전용으로 조사합니다.

실행: python scripts/inspect_law_duplicates.py
대상: 국가법령정보센터 법령 ID 014656 (기본값)
기존 law.py, JSON, SQLite, ChromaDB는 수정하지 않습니다.
인증키는 화면과 저장 파일명에 출력하지 않습니다.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv


def text_of(element: ET.Element, tag: str) -> str:
    return (element.findtext(tag) or "").strip()


def preview(text: str, limit: int = 100) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def inspect(raw: bytes) -> tuple[int, int]:
    root = ET.fromstring(raw)
    nodes = root.findall(".//조문단위")
    if root.tag != "법령" or not nodes:
        raise ValueError("법령 XML 형식이 아닙니다. 요청 결과를 확인하세요.")

    groups: dict[tuple[str, str], list[tuple[int, ET.Element]]] = defaultdict(list)
    for position, node in enumerate(nodes, start=1):
        no = text_of(node, "조문번호")
        branch = text_of(node, "조문가지번호")
        if no:
            try:
                key = (str(int(no)), str(int(branch or 0)))
            except ValueError:
                key = (no, branch)
            groups[key].append((position, node))

    duplicates = [(key, entries) for key, entries in groups.items() if len(entries) > 1]
    print(f"원본 조문단위: {len(nodes)}개 / 공식번호 중복 그룹: {len(duplicates)}개")
    if not duplicates:
        print("중복 조문번호가 발견되지 않았습니다.")
        return len(nodes), 0

    for (no, branch), entries in duplicates:
        display = f"제{no}조" + (f"의{branch}" if branch not in ("", "0") else "")
        print(f"\n[{display}] 같은 번호의 XML 조문단위 {len(entries)}개")
        for position, node in entries:
            print(f"  - 위치: {position}")
            # 값을 유지하여 실제 XML의 조문 구분 필드를 확인합니다.
            for tag in (
                "조문여부", "조문번호", "조문가지번호", "조문일련번호",
                "조문제목", "조문내용", "조문시행일자", "조문변경여부", "조문제개정유형",
            ):
                value = text_of(node, tag)
                print(f"    {tag}: {preview(value) if value else '(없음)'}")
            print(f"    항 개수: {len(node.findall('항'))}개")
            other_tags = sorted({child.tag for child in node if child.tag not in {
                "조문여부", "조문번호", "조문가지번호", "조문일련번호",
                "조문제목", "조문내용", "조문시행일자", "조문변경여부", "조문제개정유형", "항",
            }})
            if other_tags:
                print(f"    그 외 직접 자식 태그: {', '.join(other_tags)}")
    return len(nodes), len(duplicates)


def main() -> None:
    parser = argparse.ArgumentParser(description="법령 원본 XML의 조문번호 중복 원인 조사")
    parser.add_argument("--law-id", default="014656", help="법령 ID (기본 014656)")
    args = parser.parse_args()
    if not args.law_id.isdigit():
        parser.error("법령 ID는 숫자만 입력하세요.")

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    api_key = os.getenv("LAW_API_KEY")
    if not api_key:
        raise SystemExit("LAW_API_KEY가 없습니다. 프로젝트 .env를 확인하세요.")

    output_dir = project_root.parent / "Law-Action-Assistant-LawDataV2" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"law_{args.law_id}_diagnostic.xml"
    if xml_path.exists():
        print(f"기존 원본 XML 재사용: {xml_path}")
        raw = xml_path.read_bytes()
    else:
        print(f"국가법령정보센터 API에서 법령 ID {args.law_id} 조회 중...")
        try:
            response = requests.get(
                "https://www.law.go.kr/DRF/lawService.do",
                params={"OC": api_key, "target": "law", "ID": args.law_id, "type": "XML"},
                timeout=(8, 40),
            )
            response.raise_for_status()
            raw = response.content
            # XML 형식인 경우에만 파일로 저장
            root = ET.fromstring(raw)
            if root.tag != "법령" or not root.findall(".//조문단위"):
                raise ValueError("정상적인 법령 XML 응답이 아닙니다.")
            with xml_path.open("xb") as f:
                f.write(raw)
            print(f"원본 XML 저장: {xml_path}")
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            # 예외 내의 URL/인증키가 화면에 노출될 수 있으므로 종류만 알림.
            raise SystemExit(f"조회 실패: {type(exc).__name__}. API 설정 및 연결 상태를 확인하세요.") from None

    try:
        inspect(raw)
    except (ET.ParseError, ValueError) as exc:
        raise SystemExit(f"XML 검사 실패: {exc}") from None

    print("\n검사 종료. 기존 JSON, law_search.db 및 ChromaDB는 변경하지 않았습니다.")
    print("터미널에 출력된 검사 결과만 공유하세요. .env나 API 인증키는 공유하지 마세요.")


if __name__ == "__main__":
    main()
