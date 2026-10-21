"""law_v4 실패 법령의 공식 XML 구조를 읽기 전용으로 진단한다.

프로젝트의 scripts/inspect_failed_laws_v4.py에 저장한 뒤 실행:
  python scripts/inspect_failed_laws_v4.py --log "C:\\dev\\Law-Action-Assistant-LawDataV2\\full_collection\\logs\\failures.jsonl"

- 기존 수집 데이터·DB·백업을 수정하지 않음
- 실패 로그의 고유 법령 ID만 공식 API에서 조회
- 인증정보/원문 전체를 출력하거나 저장하지 않음
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from law_v3 import DETAIL_URL, create_session, fetch_xml, xml_text


def load_failed_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"실패 로그 파일이 없습니다: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"실패 로그 {line_number}번째 줄 JSON 형식 오류") from exc
        law_id = record.get("law_id", "")
        if not isinstance(law_id, str) or not law_id.isdigit():
            raise ValueError(f"실패 로그 {line_number}번째 줄 법령 ID 오류")
        if law_id not in seen:
            seen.add(law_id)
            ids.append(law_id)
    return ids


def summarize(raw: bytes, requested_id: str) -> str:
    root = ET.fromstring(raw)
    if root.tag != "법령":
        return f"{requested_id}: 법령 XML 아님 (루트: {root.tag})"
    law_name = xml_text(root, ".//기본정보/법령명_한글", ".//법령명_한글") or "(법령명 없음)"
    nodes = root.findall(".//조문단위")
    types = Counter(xml_text(n, "조문여부") or "(없음)" for n in nodes)
    numbered = sum(xml_text(n, "조문여부") == "조문" for n in nodes)
    unnumbered = [n for n in nodes if xml_text(n, "조문여부") != "조문"]
    sample = ""
    for node in unnumbered:
        body = " ".join(xml_text(node, "조문내용").split())
        if body:
            sample = body[:105] + ("..." if len(body) > 105 else "")
            break
    result = (
        f"{requested_id}: {law_name}\n"
        f"    조문단위 {len(nodes)}개 / 번호 있는 조문 {numbered}개 / 유형 {dict(types)}"
    )
    if sample:
        result += f"\n    비조문 내용 예시: {sample}"
    if len(nodes) == 1 and numbered == 0 and sample:
        result += "\n    [검토 후보] 단일 전문 노드의 무조문형 법령"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="law_v4 실패 법령의 실제 XML 구조 조사(읽기 전용)")
    parser.add_argument("--log", required=True, type=Path, help="기존 failures.jsonl 경로")
    parser.add_argument("--max", type=int, default=30, help="최대 진단 건수(기본 30)")
    args = parser.parse_args()
    if args.max < 1:
        parser.error("--max는 1 이상이어야 합니다.")

    law_ids = load_failed_ids(args.log)
    if not law_ids:
        print("실패 로그에 법령 ID가 없습니다.")
        return 0
    law_ids = law_ids[: args.max]
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("LAW_API_KEY")
    if not api_key:
        raise SystemExit("LAW_API_KEY가 없습니다. 프로젝트 .env를 확인하세요.")

    print(f"실패 법령 구조 조사: {len(law_ids)}건 (읽기 전용)")
    print("기존 법령 데이터와 DB는 변경하지 않습니다.\n")
    success = 0
    errors = 0
    with create_session() as session:
        for index, law_id in enumerate(law_ids, 1):
            try:
                raw = fetch_xml(session, DETAIL_URL, api_key, target="law", ID=law_id)
                print(f"[{index}/{len(law_ids)}] {summarize(raw, law_id)}")
                success += 1
            except (RuntimeError, ValueError, ET.ParseError) as exc:
                # 오류에 인증 키 등이 포함되지 않도록 오류 유형만 출력
                print(f"[{index}/{len(law_ids)}] {law_id}: 확인 실패 ({type(exc).__name__})")
                errors += 1
    print(f"\n진단 완료: 조회 {success}건 / 조회 실패 {errors}건")
    print("수집기 코드를 바꾸거나 실패 법령을 새로 저장하지 않았습니다.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
