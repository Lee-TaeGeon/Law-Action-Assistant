"""국가법령정보센터 현행법령(공포일 기준) 일괄 수집기 v4.

기존 law_v3.py의 XML 파서를 재사용한다. 운영 DB/기존 v3 수집 파일은 수정하지 않는다.

실행 (프로젝트 루트에 law_v3.py, law_v4.py 둘 다 둘 것):
    python law_v4.py --list-only
    python law_v4.py --limit 20
    python law_v4.py --limit 100
    # 소규모 검증 완료 후에만:
    python law_v4.py --all

출력 기본값: 프로젝트 외부 ../Law-Action-Assistant-LawDataV2/full_collection/
- catalog/law_ids.json: API 목록 스냅샷. 이후 재실행 시 재사용(스냅샷을 갱신하려면 새 출력 폴더 사용)
- raw_xml/law_<ID>.xml, structured_json/law_<ID>.json: 기존 JSON 스키마 유지
- logs/failures.jsonl: 수집 실패 기록

주의: target=law는 현행법령(공포일 기준) 조회이며 사건일 기준 법령 효력 검증,
      연혁 수집, 행정규칙/자치법규/판례 수집, 자동 개정 반영을 뜻하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from dotenv import load_dotenv

from law_v3 import (
    DETAIL_URL,
    PROJECT_ROOT,
    SEARCH_URL,
    create_session,
    fetch_xml,
    parse_law_xml,
    xml_text,
)

DEFAULT_OUTPUT = PROJECT_ROOT.parent / "Law-Action-Assistant-LawDataV2" / "full_collection"
CATALOG_NAME = "law_ids.json"
CATALOG_TARGET = "law"
DETAIL_TARGET = "law"
CATALOG_PAGE_SIZE = 100
MAX_CATALOG_PAGES = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_external_output(output: Path) -> None:
    output = output.resolve()
    if output == PROJECT_ROOT or PROJECT_ROOT in output.parents:
        raise ValueError("수집 경로는 프로젝트 바깥이어야 합니다.")
    # 기존 백업을 수집 대상으로 오인해 덮어쓰는 실수를 예방한다.
    if any("backup" in part.casefold() for part in output.parts):
        raise ValueError("백업 폴더는 수집 출력 경로로 사용할 수 없습니다.")
    # law_v3.py의 기존 출력 경로(raw_xml/structured_json)를 혼용하지 않는다.
    if not (output / "catalog" / CATALOG_NAME).exists():
        if (output / "raw_xml").exists() or (output / "structured_json").exists():
            raise ValueError("기존 수집 파일이 있는 출력 폴더입니다. 새 하위 폴더를 사용하세요.")


def write_once_atomic(path: Path, payload: bytes) -> None:
    """같은 디렉터리에 임시 파일을 쓴 다음 하드링크로 등록: 기존 파일 덮어쓰기 금지."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # 대상이 이미 있으면 FileExistsError. os.replace와 달리 덮어쓰지 않는다.
        os.link(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def read_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("api_target") != CATALOG_TARGET:
        raise ValueError("목록 스냅샷의 API 대상이 다릅니다.")
    ids = catalog.get("law_ids")
    total = catalog.get("total_count")
    if (
        not isinstance(ids, list)
        or not isinstance(total, int)
        or total < 1
        or len(ids) != total
        or len(ids) != len(set(ids))
        or any(not isinstance(i, str) or not i.isdigit() for i in ids)
    ):
        raise ValueError("목록 스냅샷에 중복·누락·비정상 ID가 있습니다.")
    return catalog


def fetch_full_catalog(session, api_key: str, pause: float) -> dict:
    """전체 ID 목록을 끝까지 조회하고 API totalCnt와 정확히 일치해야만 승인."""
    ids: list[str] = []
    seen: set[str] = set()
    announced_total: int | None = None

    for page in range(1, MAX_CATALOG_PAGES + 1):
        raw = fetch_xml(
            session, SEARCH_URL, api_key,
            target=CATALOG_TARGET, display=CATALOG_PAGE_SIZE, page=page,
        )
        root = ET.fromstring(raw)
        total_value = xml_text(root, "totalCnt")
        if not total_value.isdigit():
            raise ValueError(f"목록 {page}쪽: totalCnt를 확인할 수 없습니다.")
        total = int(total_value)
        if total < 1:
            raise ValueError("전체 법령 개수가 0입니다. API 설정을 확인하세요.")
        if announced_total is None:
            announced_total = total
        elif total != announced_total:
            raise ValueError("수집 중 전체 법령 건수가 변했습니다. 새 출력 폴더에서 재시도하세요.")

        nodes = root.findall(".//law")
        if not nodes:
            raise ValueError(f"목록 {page}쪽: 법령 목록이 비어 있습니다.")
        for node in nodes:
            law_id = xml_text(node, "법령ID")
            if not law_id.isdigit():
                raise ValueError(f"목록 {page}쪽: 유효하지 않은 법령ID가 있습니다.")
            if law_id in seen:
                raise ValueError(f"목록 {page}쪽: 중복 법령ID {law_id} 발견. 페이지 변동 확인 필요.")
            seen.add(law_id)
            ids.append(law_id)

        print(f"목록 확인: {len(ids):,} / {total:,} (페이지 {page})")
        if len(ids) > total:
            raise ValueError("API 목록 건수가 totalCnt를 초과합니다.")
        if len(ids) == total:
            break
        time.sleep(pause)
    else:
        raise ValueError("법령 목록 조회 페이지 한도를 초과했습니다.")

    if len(ids) != announced_total:
        raise ValueError("법령 목록 전체 건수 불일치")
    return {
        "source": "국가법령정보센터 공식 API",
        "api_target": CATALOG_TARGET,
        "description": "현행법령 공포일 기준 목록 스냅샷; 사건일 기준 효력 검증 아님",
        "created_at_utc": utc_now(),
        "total_count": len(ids),
        "law_ids": ids,
    }


def load_or_create_catalog(session, api_key: str, output: Path, pause: float) -> dict:
    catalog_path = output / "catalog" / CATALOG_NAME
    if catalog_path.exists():
        catalog = read_catalog(catalog_path)
        print(f"기존 목록 스냅샷 사용: {catalog['total_count']:,}개 ({catalog['created_at_utc']})")
        return catalog
    print("국가법령정보센터에서 전체 법령 목록을 조회합니다 (본문 다운로드는 아직 하지 않음).")
    catalog = fetch_full_catalog(session, api_key, pause)
    write_once_atomic(catalog_path, (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return catalog


def annotate_source(data: dict, raw: bytes) -> dict:
    # 기존 v3의 최상위 JSON 필드/data 배열 구조를 그대로 유지한다.
    data["source_xml_sha256"] = hashlib.sha256(raw).hexdigest()
    data["collection_scope"] = "target=law (현행법령 공포일 기준)"
    return data


def verify_pair(xml_path: Path, json_path: Path, expected_id: str) -> dict:
    raw = xml_path.read_bytes()
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("기존 JSON이 객체가 아닙니다.")
    saved_id = str(record.get("law_id", ""))
    if not saved_id.isdigit() or int(saved_id) != int(expected_id):
        raise ValueError("기존 JSON 법령 ID 불일치")
    if not record.get("law_name") or not isinstance(record.get("data"), list) or not record["data"]:
        raise ValueError("기존 JSON 본문 또는 법령명 누락")
    if record.get("source_xml_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("XML·JSON SHA-256 불일치 (기존 파일 유지)")
    # 해시 일치만으로 문법 검증을 대체하지 않는다.
    if ET.fromstring(raw).tag != "법령":
        raise ValueError("기존 XML 루트가 법령이 아닙니다.")
    return record


def parse_validated(raw: bytes, law_id: str) -> dict:
    data = parse_law_xml(raw, law_id)
    parsed_id = str(data.get("law_id", ""))
    if not parsed_id.isdigit() or int(parsed_id) != int(law_id):
        raise ValueError(f"법령 ID 불일치: 요청 {law_id}, 응답 {parsed_id}")
    if not data["data"]:
        raise ValueError("실제 조문이 0개인 응답입니다. 검토 전까지 저장하지 않습니다.")
    return annotate_source(data, raw)


def collect_one(session, api_key: str, output: Path, law_id: str) -> tuple[str, str]:
    xml_path = output / "raw_xml" / f"law_{law_id}.xml"
    json_path = output / "structured_json" / f"law_{law_id}.json"
    if xml_path.exists() and json_path.exists():
        record = verify_pair(xml_path, json_path, law_id)
        return "skipped", f"기존 데이터 확인: {record['law_name']} ({len(record['data'])}개 조문)"
    if json_path.exists() and not xml_path.exists():
        raise ValueError("JSON만 존재합니다. 기존 파일을 수정하지 않고 중단합니다 (수동 확인 필요).")
    if xml_path.exists():
        # 크래시가 XML 저장 직후 발생했다면 원본 XML을 이용해 JSON만 복구한다.
        raw = xml_path.read_bytes()
        record = parse_validated(raw, law_id)
        json_bytes = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        write_once_atomic(json_path, json_bytes)
        return "recovered", f"XML에서 JSON 복구: {record['law_name']} ({len(record['data'])}개 조문)"

    raw = fetch_xml(session, DETAIL_URL, api_key, target=DETAIL_TARGET, ID=law_id)
    record = parse_validated(raw, law_id)
    json_bytes = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # 두 파일 사이에 중단되는 경우 다음 실행에서 XML만 존재하면 복구할 수 있다.
    write_once_atomic(xml_path, raw)
    write_once_atomic(json_path, json_bytes)
    return "collected", f"{record['law_name']} ({len(record['data'])}개 조문)"


def append_failure(output: Path, law_id: str, exc: Exception, api_key: str) -> None:
    path = output / "logs" / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
    event = {
        "timestamp_utc": utc_now(),
        "law_id": law_id,
        "error_type": type(exc).__name__,
        "message": safe_message[:500],
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="law_v4: 현행법령 목록 저장, 이어받기, 누락·무결성 검증")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-only", action="store_true", help="전체 법령 ID 목록 스냅샷만 저장")
    mode.add_argument("--limit", type=int, help="스냅샷 중 앞 N개 법령만 수집")
    mode.add_argument("--all", action="store_true", help="스냅샷에 수록된 모든 법령 수집")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="프로젝트 외부 신규 데이터 경로")
    parser.add_argument("--pause", type=float, default=0.7, help="API 요청 사이 대기 초 (기본 0.7)")
    parser.add_argument("--max-failures", type=int, default=10, help="실패 누적 시 중단할 건수 (기본 10)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")
    if args.pause < 0.2:
        parser.error("--pause는 0.2초 이상이어야 합니다.")
    if args.max_failures < 1:
        parser.error("--max-failures는 1 이상이어야 합니다.")

    output = args.output.resolve()
    try:
        validate_external_output(output)
    except ValueError as exc:
        parser.error(str(exc))

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("LAW_API_KEY", "").strip()
    if not api_key:
        parser.error("LAW_API_KEY가 없습니다. 프로젝트 .env를 확인하세요.")

    with create_session() as session:
        catalog = load_or_create_catalog(session, api_key, output, args.pause)
        ids = catalog["law_ids"]
        print(f"목록 스냅샷: 총 {len(ids):,}개 | 저장 폴더: {output}")
        print("주의: 해당 스냅샷은 특정 시각 목록이며 조문 버전의 유효기간을 판정하지 않습니다.")
        if args.list_only:
            print("목록 조회 완료. 법령 본문은 다운로드하지 않았습니다.")
            return 0

        if args.limit is not None:
            ids = ids[:args.limit]
        counts = {"collected": 0, "skipped": 0, "recovered": 0, "failed": 0}
        print(f"수집 대상: {len(ids):,}개. 기존 운영 DB/백업/이전 v3 데이터 변경 없음.")
        for idx, law_id in enumerate(ids, start=1):
            status = None
            try:
                status, message = collect_one(session, api_key, output, law_id)
                counts[status] += 1
                print(f"[{idx}/{len(ids)}] {law_id}: [{status}] {message}")
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, ET.ParseError, json.JSONDecodeError) as exc:
                counts["failed"] += 1
                append_failure(output, law_id, exc, api_key)
                print(f"[{idx}/{len(ids)}] {law_id}: [실패] {type(exc).__name__} - {str(exc).replace(api_key, '[REDACTED]')[:150]}")
                if counts["failed"] >= args.max_failures:
                    print(f"실패 {args.max_failures}건에 도달하여 안전하게 중단합니다. 재실행 시 완료된 파일을 확인하고 건너뜁니다.")
                    break
            # 이미 완료한 파일을 건너뛴 경우 추가 요청이 없으므로 대기 생략.
            if status is None or status == "collected":
                time.sleep(args.pause)

        print("\n수집 결과: " + ", ".join(f"{k} {v:,}개" for k, v in counts.items()))
        if counts["failed"]:
            print(f"실패 목록: {output / 'logs' / 'failures.jsonl'}")
        print("기존 DB와 백업은 변경하지 않았습니다. API 범위 밖의 자료는 별도 수집이 필요합니다.")
        return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
