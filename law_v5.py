"""국가법령정보센터 법령 수집기 v5: 번호 없는 단문 법령 안전 수집.

v3 파싱 결과 및 v4 목록 스냅샷·원자적 저장·이어받기 기능을 재사용한다.
기존 law_v3.py, law_v4.py 파일을 변경하지 않고 프로젝트 루트에 이 파일을 추가한다.

단계별 실행:
    python law_v5.py --law-id 014378    # 번호 없는 법령 한 건만 테스트
    python law_v5.py --retry-failures   # v4 실패 로그에 있는 법령만 재시도
    python law_v5.py --all              # 앞 단계 검증 후 전체 목록 이어받기

저장 형식:
- 번호가 있는 법령: v4와 동일한 data 배열 (기존 파일 호환)
- 번호가 없는 단문 법령: data=[], document_structure='unnumbered_body',
  unnumbered_provisions 및 non_article_nodes에 공식 원문/유형을 보존
- 없는 제1조 등 임의 조문번호는 생성하지 않는다.

주의: 기존 build_law_search_db_v2.py는 data=[]를 거부한다. 전체 SQLite를
만들기 전에 비조문 단문 법령을 따로 저장·검색하도록 DB 빌더를 개정해야 한다.
API target=law 스냅샷은 역사상 모든 시행 버전의 저장을 뜻하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from dotenv import load_dotenv

import law_v4 as v4


def is_unnumbered_document(record: dict) -> bool:
    """실제 조문이 없고 비조문 원문만 있는 단문 법령인지 보수적으로 판별."""
    if not isinstance(record.get("data"), list) or record["data"]:
        return False
    nodes = record.get("non_article_nodes")
    if not isinstance(nodes, list) or len(nodes) != 1:
        return False
    node = nodes[0]
    if not isinstance(node, dict) or node.get("node_type") != "전문":
        return False
    # 검증된 예외 유형: 공식 '조문번호' 없이 본문 전체를 담은 전문 1개.
    if str(node.get("node_number") or "").strip() not in ("", "0"):
        return False
    if str(node.get("node_branch_number") or "").strip() not in ("", "0"):
        return False
    text = str(node.get("text") or "").strip()
    if len(text) < 8:
        return False
    # 장/절의 제목만 덩그러니 있는 노드를 실질 법령으로 오인하지 않는다.
    if text.replace(" ", "").startswith(("제1장", "제1절", "제1편", "제1관")) and len(text) < 30:
        return False
    return bool(str(record.get("law_name") or "").strip())


def parse_validated_v5(raw: bytes, expected_id: str) -> dict:
    record = v4.parse_law_xml(raw, expected_id)
    returned_id = str(record.get("law_id") or "")
    if not returned_id.isdigit() or int(returned_id) != int(expected_id):
        raise ValueError(f"법령 ID 불일치: 요청 {expected_id}, 응답 {returned_id}")

    if record["data"]:
        record["document_structure"] = "numbered_articles"
        return v4.annotate_source(record, raw)

    if not is_unnumbered_document(record):
        raise ValueError("실제 조문 0개: 단문 법령으로 검증할 수 없는 응답 (원본 조사 필요)")

    source = record["non_article_nodes"][0]
    record["document_structure"] = "unnumbered_body"
    # 데이터 소비자가 '장·절 제목'과 '법령 실질 본문'을 구분할 수 있게 별도 필드 제공.
    record["unnumbered_provisions"] = [{
        "source_order": source["source_order"],
        "node_type": source["node_type"],
        "node_number": source["node_number"],
        "node_branch_number": source["node_branch_number"],
        "text": source["text"],
        "effective_date": source["effective_date"],
    }]
    return v4.annotate_source(record, raw)


def verify_pair_v5(xml_path: Path, json_path: Path, expected_id: str) -> dict:
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("기존 JSON 형식 오류")
    if record.get("data"):
        # 번호 있는 법령은 v4 검증 규칙(본문/ID/원문 해시)을 그대로 사용.
        return v4.verify_pair(xml_path, json_path, expected_id)

    raw = xml_path.read_bytes()
    if str(record.get("law_id", "")).isdigit() is False or int(record["law_id"]) != int(expected_id):
        raise ValueError("기존 단문 법령 ID 불일치")
    if record.get("source_xml_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("기존 단문 법령 XML·JSON 해시 불일치")
    if ET.fromstring(raw).tag != "법령":
        raise ValueError("기존 XML 루트 오류")
    if record.get("document_structure") != "unnumbered_body" or not is_unnumbered_document(record):
        raise ValueError("기존 JSON의 비조문 단문 법령 표기가 유효하지 않음")
    provisions = record.get("unnumbered_provisions")
    if not isinstance(provisions, list) or len(provisions) != 1 or provisions[0] != record["non_article_nodes"][0]:
        raise ValueError("단문 법령 원문 필드 불일치")
    # 원본 XML을 파싱하여 진짜로 번호 있는 조문이 없는지 다시 점검.
    original = v4.parse_law_xml(raw, expected_id)
    if original["data"] or not is_unnumbered_document(original):
        raise ValueError("기존 단문 법령 분류와 원본 XML이 불일치")
    if original["non_article_nodes"][0] != provisions[0]:
        raise ValueError("기존 단문 법령 본문이 원본 XML과 불일치")
    return record


def collect_one_v5(session, api_key: str, output: Path, law_id: str) -> tuple[str, str]:
    xml_path = output / "raw_xml" / f"law_{law_id}.xml"
    json_path = output / "structured_json" / f"law_{law_id}.json"
    if xml_path.exists() and json_path.exists():
        record = verify_pair_v5(xml_path, json_path, law_id)
        return "skipped", f"검증된 기존 파일: {record['law_name']} ({record.get('document_structure', 'numbered_articles')})"
    if json_path.exists():
        raise ValueError("XML 없이 JSON만 있습니다. 기존 파일 유지, 수동 확인 필요")

    if xml_path.exists():
        raw = xml_path.read_bytes()
        status = "recovered"
    else:
        raw = v4.fetch_xml(session, v4.DETAIL_URL, api_key, target=v4.DETAIL_TARGET, ID=law_id)
        status = "collected"

    record = parse_validated_v5(raw, law_id)
    payload = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if status == "collected":
        v4.write_once_atomic(xml_path, raw)
    v4.write_once_atomic(json_path, payload)
    kind = record["document_structure"]
    count = len(record["data"])
    return status, f"{record['law_name']} ({kind}, 번호 있는 조문 {count}개)"


def failure_ids(output: Path, catalog_ids: list[str]) -> list[str]:
    logfile = output / "logs" / "failures.jsonl"
    if not logfile.exists():
        raise FileNotFoundError(f"실패 로그가 없습니다: {logfile}")
    allowed = set(catalog_ids)
    result: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(logfile.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        law_id = str(event.get("law_id", ""))
        if law_id not in allowed:
            raise ValueError(f"실패 로그 {line_number}행: 목록 스냅샷에 없는 법령 ID")
        if law_id not in seen:
            seen.add(law_id)
            result.append(law_id)
    if not result:
        raise ValueError("재시도할 실패 법령이 없습니다.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="법령 수집 v5: 조문번호 없는 단문 법령 지원 및 안전한 재개")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-only", action="store_true", help="목록 스냅샷만 확인")
    mode.add_argument("--limit", type=int, help="목록의 앞 N개 수집")
    mode.add_argument("--all", action="store_true", help="전체 목록 수집")
    mode.add_argument("--law-id", action="append", help="특정 법령 ID (반복 가능)")
    mode.add_argument("--retry-failures", action="store_true", help="v4 실패 로그에 기록된 법령만 재시도")
    parser.add_argument("--output", type=Path, default=v4.DEFAULT_OUTPUT)
    parser.add_argument("--pause", type=float, default=0.7)
    parser.add_argument("--max-failures", type=int, default=10)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")
    if args.pause < 0.2 or args.max_failures < 1:
        parser.error("--pause는 0.2 이상, --max-failures는 1 이상이어야 합니다.")
    output = args.output.resolve()
    try:
        v4.validate_external_output(output)
    except ValueError as exc:
        parser.error(str(exc))
    load_dotenv(v4.PROJECT_ROOT / ".env")
    api_key = os.getenv("LAW_API_KEY", "").strip()
    if not api_key:
        parser.error("LAW_API_KEY가 없습니다. 프로젝트 .env를 확인하세요.")

    with v4.create_session() as session:
        catalog = v4.load_or_create_catalog(session, api_key, output, args.pause)
        all_ids = catalog["law_ids"]
        if args.list_only:
            print(f"목록: {len(all_ids):,}개, 본문 다운로드 없음")
            return 0
        if args.limit is not None:
            ids = all_ids[:args.limit]
        elif args.law_id:
            if any(not law_id.isdigit() for law_id in args.law_id):
                parser.error("법령 ID는 숫자여야 합니다.")
            invalid = [i for i in args.law_id if i not in set(all_ids)]
            if invalid:
                parser.error(f"목록에 없는 ID: {', '.join(invalid)}")
            ids = list(dict.fromkeys(args.law_id))
        elif args.retry_failures:
            ids = failure_ids(output, all_ids)
        else:
            ids = all_ids

        counts = {"collected": 0, "skipped": 0, "recovered": 0, "failed": 0}
        print(f"법령 수집 v5 | 대상 {len(ids):,}개 | 저장: {output}")
        print("기존 운영 DB/백업은 수정하지 않습니다. 원본 조문번호를 임의로 생성하지 않습니다.")
        for index, law_id in enumerate(ids, 1):
            status = None
            try:
                status, description = collect_one_v5(session, api_key, output, law_id)
                counts[status] += 1
                print(f"[{index}/{len(ids)}] {law_id}: [{status}] {description}")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, ET.ParseError, json.JSONDecodeError) as exc:
                counts["failed"] += 1
                v4.append_failure(output, law_id, exc, api_key)
                safe = str(exc).replace(api_key, "[REDACTED]")
                print(f"[{index}/{len(ids)}] {law_id}: [실패] {type(exc).__name__} - {safe[:160]}")
                if counts["failed"] >= args.max_failures:
                    print(f"실패 {args.max_failures}건 도달: 중단 (완료 파일은 보존)")
                    break
            if status is None or status == "collected":
                time.sleep(args.pause)
        print("\n수집 결과: " + ", ".join(f"{k} {v:,}개" for k, v in counts.items()))
        print("주의: 번호 없는 법령은 현재 SQLite 빌더 v2에서 바로 적재할 수 없습니다.")
        return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
