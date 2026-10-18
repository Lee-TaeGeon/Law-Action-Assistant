"""국가법령정보센터 법령 수집기 v3: 기존 DB/JSON을 변경하지 않는 신규 수집용.

실행 (프로젝트 최상위 폴더에 law_v3.py를 둘 것):
    python law_v3.py --law-id 010719 --law-id 010831 --law-id 014656
    python law_v3.py --limit 3
    # 충분한 테스트가 끝난 후에만:
    python law_v3.py --all

출력: 프로젝트 외부 ../Law-Action-Assistant-LawDataV2/
주의: 이 도구는 법령 '수집'만 합니다. 법령 자동 갱신이나 시행일별 효력 판정은 하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "Law-Action-Assistant-LawDataV2"
SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
DETAIL_URL = "https://www.law.go.kr/DRF/lawService.do"


def xml_text(node: ET.Element, *paths: str) -> str:
    for path in paths:
        value = node.findtext(path)
        if value is not None and value.strip():
            return value.strip()
    return ""


def parse_law_xml(raw: bytes, requested_law_id: str) -> dict:
    root = ET.fromstring(raw)
    nodes = root.findall(".//조문단위")
    if root.tag != "법령" or not nodes:
        raise ValueError("정상적인 법령 XML이 아닙니다. API 응답/인증을 확인하세요.")

    name = xml_text(root, ".//기본정보/법령명_한글", ".//법령명_한글", ".//법령명한글")
    if not name:
        raise ValueError("법령명이 없어 수집을 중단합니다.")

    articles: list[dict] = []
    # 국가법령정보센터 XML의 '조문단위'에는 실제 '조문' 외에 장/절 제목인
    # '전문' 노드도 들어 있다. 전문 노드의 조문번호는 뒤따르는 실제 조문과
    # 같을 수 있으므로, 전문을 법률 조문으로 저장하거나 중복 검사하지 않는다.
    non_article_nodes: list[dict] = []
    for order, node in enumerate(nodes, start=1):
        status = xml_text(node, "조문여부")
        if status != "조문":
            # 삭제하지 않고 원본에서 나온 순서와 본문을 보관한다.
            # 예상 밖의 종류도 별도 기록하되, 실제 조문과 섞지는 않는다.
            non_article_nodes.append({
                "source_order": order,
                "node_type": status or "(구분 없음)",
                "node_number": xml_text(node, "조문번호"),
                "node_branch_number": xml_text(node, "조문가지번호"),
                "text": xml_text(node, "조문내용"),
                "effective_date": xml_text(node, "조문시행일자"),
            })
            continue

        number = xml_text(node, "조문번호")
        branch = xml_text(node, "조문가지번호")
        if not number or not number.isdigit():
            raise ValueError("조문번호 누락 또는 비정상 값이 있습니다.")
        if branch and not branch.isdigit():
            raise ValueError(f"제{number}조의 가지번호가 숫자가 아닙니다.")
        display = f"제{int(number)}조"
        if branch and int(branch) != 0:
            display += f"의{int(branch)}"

        paragraphs = []
        for para in node.findall("항"):
            items = []
            for item in para.findall("호"):
                items.append({
                    "item_no": xml_text(item, "호번호"),
                    "item_content": xml_text(item, "호내용"),
                    "sub_items": [
                        {"sub_no": xml_text(sub, "목번호"), "sub_content": xml_text(sub, "목내용")}
                        for sub in item.findall("목")
                    ],
                })
            paragraphs.append({
                "para_no": xml_text(para, "항번호"),
                "para_content": xml_text(para, "항내용"),
                "items": items,
            })

        articles.append({
            "source_order": order,
            "article_no": number,
            "article_branch_no": branch,
            "article_display": display,
            "article_title": xml_text(node, "조문제목"),
            "article_body": xml_text(node, "조문내용"),
            "article_effective_date": xml_text(node, "조문시행일자"),
            "article_status": xml_text(node, "조문여부"),
            "article_revision_type": xml_text(node, "조문제개정유형"),
            "article_changed": xml_text(node, "조문변경여부"),
            "paragraphs": paragraphs,
        })

    # 같은 공식 번호가 중복되는 경우 임의로 '의2'를 만들어서 저장하지 않는다.
    keys = [(a["article_no"], int(a["article_branch_no"] or 0)) for a in articles]
    if len(set(keys)) != len(keys):
        dup = [f"제{num}조" + (f"의{br}" if br else "") for (num, br), n in Counter(keys).items() if n > 1]
        raise ValueError(f"공식 조문 번호 중복: {', '.join(dup[:5])}. 원본 XML 확인 필요")

    return {
        "source": "국가법령정보센터 공식 API",
        "api_target": "law",  # 시행 시점별 검증은 별도 구현
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "law_id": xml_text(root, ".//기본정보/법령ID", ".//법령ID") or requested_law_id,
        "law_name": name,
        "promulgation_date": xml_text(root, ".//기본정보/공포일자", ".//공포일자"),
        "promulgation_number": xml_text(root, ".//기본정보/공포번호", ".//공포번호"),
        "effective_date": xml_text(root, ".//기본정보/시행일자", ".//시행일자"),
        "revision_type": xml_text(root, ".//기본정보/제개정구분", ".//제개정구분"),
        "total_xml_nodes": len(nodes),
        "non_article_nodes": non_article_nodes,
        "data": articles,
    }


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_xml(session: requests.Session, url: str, api_key: str, **params: str | int) -> bytes:
    try:
        response = session.get(url, params={"OC": api_key, "type": "XML", **params}, timeout=(8, 40))
        if response.status_code != 200:
            raise RuntimeError(f"API HTTP {response.status_code}")
        return response.content
    except requests.RequestException as exc:
        # 예외 문자열에 인증키가 포함된 요청 URL이 들어갈 수 있으므로 타입만 표기.
        raise RuntimeError(f"API 네트워크 오류: {type(exc).__name__}") from None


def fetch_law_ids(session: requests.Session, api_key: str, limit: int | None) -> list[str]:
    law_ids: list[str] = []
    seen: set[str] = set()
    page = 1
    while True:
        raw = fetch_xml(session, SEARCH_URL, api_key, target="law", display=100, page=page)
        root = ET.fromstring(raw)
        total_text = xml_text(root, "totalCnt")
        if not total_text.isdigit():
            raise RuntimeError("법령 목록 API에서 전체 건수(totalCnt)를 확인할 수 없습니다.")
        total = int(total_text)
        nodes = root.findall(".//law")
        if not nodes:
            if len(law_ids) < total and (limit is None or len(law_ids) < limit):
                raise RuntimeError(f"페이지 {page}에서 목록이 비어 있습니다. 수집을 중단합니다.")
            break
        for node in nodes:
            law_id = xml_text(node, "법령ID")
            if law_id and law_id not in seen:
                seen.add(law_id)
                law_ids.append(law_id)
                if limit is not None and len(law_ids) >= limit:
                    return law_ids
        print(f"법령 목록 조회: {len(law_ids)} / {total}")
        if len(law_ids) >= total:
            break
        page += 1
        if page > 300:
            raise RuntimeError("목록 페이지 상한(300)을 초과했습니다. API 설정을 확인하세요.")
        time.sleep(0.5)
    return law_ids


def save_new_file(path: Path, content: bytes) -> None:
    # 'x'는 파일이 있으면 실패하며, 기존 데이터 덮어쓰기를 방지한다.
    with path.open("xb") as output:
        output.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="법령 수집 v3: 장·절 제목과 조문 분리, 기존 DB 보존")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--law-id", action="append", help="법령 ID. 여러 번 지정 가능")
    mode.add_argument("--limit", type=int, help="법령 목록의 앞 N개만 시험 수집")
    mode.add_argument("--all", action="store_true", help="전체 법령 수집 (시간이 오래 걸림)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="신규 데이터 저장 경로")
    args = parser.parse_args()
    print("법령 수집기 v3 실행 중 — 장·절 제목(전문)과 실제 조문을 분리합니다.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("LAW_API_KEY")
    if not api_key:
        raise SystemExit("LAW_API_KEY가 없습니다. 프로젝트 .env 설정을 확인하세요.")

    output = args.output.resolve()
    # 프로젝트 내부 및 그 하위 폴더에는 원본 대용량 데이터를 저장하지 않도록 보호.
    if output == PROJECT_ROOT or PROJECT_ROOT in output.parents:
        raise SystemExit("--output은 프로젝트 바깥의 폴더로 지정해야 합니다.")
    raw_dir = output / "raw_xml"
    json_dir = output / "structured_json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    with create_session() as session:
        if args.law_id:
            ids = args.law_id
        elif args.limit is not None or args.all:
            ids = fetch_law_ids(session, api_key, args.limit)
        else:
            ids = ["010719"]  # 아무 옵션 없이 실행해도 전체 수집이 시작되지 않음
        if not all(i.isdigit() for i in ids):
            parser.error("법령 ID는 숫자로 지정해야 합니다.")
        ids = list(dict.fromkeys(ids))

        success = skipped = failed = 0
        print(f"총 {len(ids)}건 수집 예정. 저장: {output}")
        for index, law_id in enumerate(ids, 1):
            xml_path = raw_dir / f"law_{law_id}.xml"
            json_path = json_dir / f"law_{law_id}.json"
            if xml_path.exists() and json_path.exists():
                print(f"[{index}/{len(ids)}] {law_id}: 기존 파일 확인, 건너뜀")
                skipped += 1
                continue
            if xml_path.exists() or json_path.exists():
                print(f"[{index}/{len(ids)}] {law_id}: 저장 파일 일부만 존재, 확인 필요 (수정하지 않음)")
                failed += 1
                continue
            try:
                raw = fetch_xml(session, DETAIL_URL, api_key, target="law", ID=law_id)
                data = parse_law_xml(raw, law_id)
                json_bytes = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                save_new_file(xml_path, raw)
                save_new_file(json_path, json_bytes)
                kinds = dict(Counter(n["node_type"] for n in data["non_article_nodes"]))
                print(
                    f"[{index}/{len(ids)}] {law_id}: {data['law_name']} "
                    f"/ 실제 조문 {len(data['data'])}개 "
                    f"/ 별도 보존한 비조문 노드 {len(data['non_article_nodes'])}개 ({kinds})"
                )
                success += 1
            except (RuntimeError, ValueError, ET.ParseError, OSError) as exc:
                print(f"[{index}/{len(ids)}] {law_id}: 실패 - {exc}")
                failed += 1
            time.sleep(0.5)

    print(f"\n완료: 성공 {success}개 / 기존 파일 건너뜀 {skipped}개 / 실패 {failed}개")
    print("기존 법령 JSON, law_search.db, ChromaDB를 변경하지 않았습니다.")
    print("주의: 이 도구는 자동 업데이트나 시행일별 법령 적용 판정을 수행하지 않습니다.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
