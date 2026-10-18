"""법령 검색 서비스 v2 (검증용, 읽기 전용).

기존 backend/services/rag_service.py의 공개 함수 인터페이스를 유지한다.
운영 파일을 교체하기 전에 별도 파일로 테스트할 것.

환경 변수 (모듈 첫 import 전에 지정):
  LAW_SEARCH_DB_PATH: SQLite 파일 절대 경로
  LAW_VECTOR_DB_PATH: ChromaDB 디렉터리 절대 경로
  LAW_DATA_DIR: 둘 다 없을 때 사용하는 프로젝트 데이터 디렉터리
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from functools import lru_cache
from contextlib import closing
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("LAW_DATA_DIR", str(BASE_DIR))).resolve()
SEARCH_DB_PATH = Path(os.getenv("LAW_SEARCH_DB_PATH", str(DATA_DIR / "law_search.db"))).resolve()
VECTOR_DB_PATH = Path(os.getenv("LAW_VECTOR_DB_PATH", str(DATA_DIR / "law_db_optimized"))).resolve()
VECTOR_COLLECTION_NAME = "law_articles"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def compact_text(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def normalize_title(value: str | None) -> str:
    value = compact_text(value)
    return value[:-1] if value.endswith("죄") else value


def normalize_article_number(value: str | None) -> str:
    text = compact_text(value)
    match = re.fullmatch(r"제?(\d+)조?(?:의(\d+))?", text)
    if not match:
        return text
    return match.group(1) + ("의" + match.group(2) if match.group(2) else "")


def format_article_number(value: str | None) -> str | None:
    number = normalize_article_number(value)
    if not number:
        return None
    return "제" + number.replace("의", "조의", 1) + ("" if "의" in number else "조")


def get_search_connection() -> sqlite3.Connection:
    if not SEARCH_DB_PATH.is_file():
        raise FileNotFoundError(f"법령 SQLite DB가 없습니다: {SEARCH_DB_PATH}")
    db = sqlite3.connect(SEARCH_DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


@lru_cache(maxsize=1)
def get_law_names() -> tuple[str, ...]:
    with closing(get_search_connection()) as db:
        return tuple(row[0] for row in db.execute(
            "SELECT law_name FROM laws ORDER BY LENGTH(law_name) DESC"
        ))


def extract_explicit_law_refs(query: str) -> list[dict]:
    refs = []
    seen = set()
    for law_name in get_law_names():
        if law_name not in query:
            continue
        pattern = rf"{re.escape(law_name)}\s*제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\s*\(([^)]+)\))?"
        for match in re.finditer(pattern, query):
            number = match.group(1) + (f"의{match.group(2)}" if match.group(2) else "")
            key = (law_name, number)
            if key not in seen:
                seen.add(key)
                refs.append({"law_name": law_name, "article_no": number,
                             "title_hint": match.group(3)})
    return refs


def _row_to_source(row: sqlite3.Row, source_type: str) -> dict:
    fields = row.keys()
    result = {
        "law_name": row["law_name"],
        "article": row["article_display"],
        "article_title": row["article_title"],
        "content": row["content"],
        "source_type": source_type,
    }
    if "id" in fields:
        result["article_id"] = int(row["id"])
    if "article_effective_date" in fields and row["article_effective_date"]:
        result["article_effective_date"] = row["article_effective_date"]
    return result


def exact_search_laws(query: str) -> list[dict]:
    refs = extract_explicit_law_refs(query)
    if not refs:
        return []
    result = []
    with closing(get_search_connection()) as db:
        for ref in refs:
            rows = db.execute(
                "SELECT * FROM articles WHERE law_name=? AND article_no=? "
                "ORDER BY source_order LIMIT 10",
                (ref["law_name"], ref["article_no"]),
            ).fetchall()
            title_hint = normalize_title(ref.get("title_hint"))
            matched = [row for row in rows if normalize_title(row["article_title"]) == title_hint]
            for row in (matched if title_hint and matched else rows):
                result.append(_row_to_source(row, "exact"))
    return result


def split_search_phrases(query: str) -> list[str]:
    parts = [re.sub(r"^[\s\d.\-*]+", "", part).strip()
             for part in re.split(r"[,;\n]+", query)]
    result = []
    seen = set()
    for part in parts:
        norm = compact_text(part)
        if norm and norm not in seen:
            result.append(part)
            seen.add(norm)
        if len(result) >= 12:
            break
    return result


def expand_legal_phrases(query: str) -> list[str]:
    """최소한의 검색 표현 확장. 법률적 결론/적용 법률을 확정하지 않는다."""
    phrases = split_search_phrases(query)
    text = compact_text(query)
    hints = []
    if re.search(r"계약|채무|약정", text) and re.search(r"위반|불이행|이행하지|지키지|파기", text):
        hints.extend(["채무불이행", "손해배상"])
    if re.search(r"퇴사|퇴직|퇴사후|퇴직후|그만두|퇴사한", text) and re.search(r"임금|월급|급여|돈|금품", text):
        hints.extend(["금품 청산", "임금 지급", "미지급 임금"])
    if re.search(r"속여|속이|기망|편취|사기", text) and re.search(r"돈|재물|금전|재산|사기", text):
        hints.extend(["사기", "기망"])
    if re.search(r"체불|미지급", text) and re.search(r"임금|급여|월급", text):
        hints.extend(["금품 청산", "임금 지급"])
    if re.search(r"계약|채무", text) and re.search(r"배상|손해", text):
        hints.append("채무불이행")
    seen = {compact_text(phrase) for phrase in phrases}
    for hint in hints:
        if compact_text(hint) not in seen:
            phrases.append(hint)
            seen.add(compact_text(hint))
    return phrases[:16]


def extract_search_terms(phrase: str) -> list[str]:
    seen = set()
    results = []
    for term in re.findall(r"[0-9A-Za-z가-힣]+", phrase):
        term = normalize_title(term)
        if not term or (len(term) < 2 and not term.isdigit()) or term in seen:
            continue
        seen.add(term)
        results.append(term)
    return results[:8]


def _fts_expr(terms: list[str], joiner: str) -> str:
    # FTS5 사용자 검색어는 SQL 바인딩 + FTS 구문 내 인용부호로 보호
    return f" {joiner} ".join('"' + term.replace('"', '""') + '"*' for term in terms)


def _title_priority(title: str, phrase: str) -> int:
    norm_title, norm_phrase = normalize_title(title), normalize_title(phrase)
    if not norm_title:
        return 4
    if norm_title == norm_phrase:
        return 0
    if norm_phrase and norm_phrase in norm_title:
        return 1
    if any(term in norm_title for term in extract_search_terms(phrase)):
        return 2
    return 3


def keyword_search_laws(query: str, k: int = 12) -> list[dict]:
    if k <= 0 or not query.strip():
        return []
    phrases = expand_legal_phrases(query)
    # score: 서로 다른 표현·검색 채널에서 등장한 순위 기반 RRF 점수
    candidates: dict[int, dict] = {}

    def add(row, phrase: str, rank: int, channel: str, phrase_idx: int):
        item_id = int(row["id"])
        priority = _title_priority(row["article_title"], phrase)
        # 정확 제목 매칭 우선. 다만 특정 조문을 무조건 1위로 고정하지 않음.
        weight = 3.0 if channel == "title" else 1.0
        if channel == "title":
            weight *= (1.75 if priority == 0 else 1.25 if priority == 1 else 1.0)
        # 원문에 없는 확장어는 원래 표현보다 약간 낮게 반영
        if phrase_idx >= len(split_search_phrases(query)):
            weight *= 0.85
        score = weight / (15 + rank)
        cand = candidates.setdefault(item_id, {
            "row": row, "score": 0.0, "matches": set(),
        })
        # 동일 검색 표현/채널에서 동일 조문이 두 번 가산되지 않도록 보호
        key = (phrase_idx, channel)
        if key not in cand["matches"]:
            cand["matches"].add(key)
            cand["score"] += score

    with closing(get_search_connection()) as db:
        for phrase_idx, phrase in enumerate(phrases):
            normalized = compact_text(phrase)
            if normalized and len(normalized) <= 100:
                rows = db.execute(
                    "SELECT * FROM articles WHERE REPLACE(article_title,' ','') LIKE ? ESCAPE '\\' "
                    "ORDER BY CASE WHEN REPLACE(article_title,' ','') = ? THEN 0 ELSE 1 END, "
                    "LENGTH(article_title) ASC, id ASC LIMIT 50",
                    ("%" + normalized.replace("%", "\\%").replace("_", "\\_") + "%", normalized),
                ).fetchall()
                for rank, row in enumerate(rows, 1):
                    add(row, phrase, rank, "title", phrase_idx)

            terms = extract_search_terms(phrase)
            if not terms:
                continue
            rows = db.execute(
                "SELECT a.*, bm25(articles_fts, 2.0, 1.0, 8.0, 1.0) AS fts_score "
                "FROM articles_fts JOIN articles a ON a.id=articles_fts.rowid "
                "WHERE articles_fts MATCH ? ORDER BY fts_score ASC LIMIT 50",
                (_fts_expr(terms, "AND"),),
            ).fetchall()
            if not rows and len(terms) > 1:
                rows = db.execute(
                    "SELECT a.*, bm25(articles_fts, 2.0, 1.0, 8.0, 1.0) AS fts_score "
                    "FROM articles_fts JOIN articles a ON a.id=articles_fts.rowid "
                    "WHERE articles_fts MATCH ? ORDER BY fts_score ASC LIMIT 30",
                    (_fts_expr(terms, "OR"),),
                ).fetchall()
            for rank, row in enumerate(rows, 1):
                add(row, phrase, rank, "fts", phrase_idx)

    ranked = sorted(candidates.values(), key=lambda x: (-x["score"], int(x["row"]["id"])))
    return [_row_to_source(item["row"], "keyword") for item in ranked[:k]]


@lru_cache(maxsize=1)
def get_embedding_model():
    # 기존 ko-sroberta-multitask 768차원 정규화 임베딩과 동일한 모델 사용
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def get_vector_db():
    import chromadb
    if not VECTOR_DB_PATH.is_dir():
        raise FileNotFoundError(f"ChromaDB 폴더가 없습니다: {VECTOR_DB_PATH}")
    manifest_path = VECTOR_DB_PATH / "_BUILD_COMPLETE.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "complete" or manifest.get("partial_test")
                or Path(str(manifest.get("source_sqlite", ""))).resolve() != SEARCH_DB_PATH):
            raise RuntimeError("ChromaDB 완료 기록과 SQLite 경로가 일치하지 않거나 일부 데이터입니다.")
    return chromadb.PersistentClient(path=str(VECTOR_DB_PATH)).get_collection(VECTOR_COLLECTION_NAME)


def _mmr_indices(query_vector: np.ndarray, candidate_vectors: np.ndarray,
                 k: int, lambda_mult: float = 0.85) -> list[int]:
    if not len(candidate_vectors) or k <= 0:
        return []
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    v = np.asarray(candidate_vectors, dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    sim = v @ q
    selected = [int(np.argmax(sim))]
    while len(selected) < min(k, len(v)):
        remaining = [i for i in range(len(v)) if i not in selected]
        scores = lambda_mult * sim[remaining] - (1-lambda_mult) * np.max(v[remaining] @ v[selected].T, axis=1)
        selected.append(remaining[int(np.argmax(scores))])
    return selected


def search_laws(question: str, category: str = "", k: int = 4) -> list[dict]:
    """기존 그래프와 호환되는 ChromaDB 의미 검색. category로 결과를 배제하지 않음."""
    if k <= 0 or not question.strip():
        return []
    col = get_vector_db()
    count = col.count()
    if count == 0:
        return []
    model = get_embedding_model()
    vector = model.encode([question.strip()], convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)[0]
    fetch_k = min(count, max(40, k * 8))
    raw = col.query(query_embeddings=[vector.tolist()], n_results=fetch_k,
                    include=["embeddings", "distances"])
    ids = (raw.get("ids") or [[]])[0]
    embeddings = raw.get("embeddings")
    if len(ids) == 0 or embeddings is None:
        return []
    indexes = _mmr_indices(vector, np.asarray(embeddings[0], dtype=np.float32), k)
    with closing(get_search_connection()) as db:
        results = []
        for index in indexes:
            item_id = int(ids[index])
            row = db.execute("SELECT * FROM articles WHERE id=?", (item_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"SQLite와 ChromaDB 조문 ID 불일치: {item_id}")
            results.append(_row_to_source(row, "vector"))
    return results


def _soft_domain_bonus(law_name: str, category: str, text: str) -> float:
    """단일 법률 분야로 강제하지 않는 작은 순위 가산점."""
    bonus = 0.0
    mapping = {
        "민사": ("민법",), "형사": ("형법",),
        "노동": ("근로기준법",),
    }
    if category in mapping and law_name in mapping[category]:
        bonus += 0.008
    if ("퇴사" in text or "퇴직" in text or "임금" in text) and law_name == "근로기준법":
        bonus += 0.006
    if ("속여" in text or "기망" in text) and law_name == "형법":
        bonus += 0.006
    if ("계약" in text or "채무" in text) and law_name == "민법":
        bonus += 0.006
    return bonus


def hybrid_search_laws(question: str, search_query: str | None = None,
                       category: str = "", k: int = 8) -> list[dict]:
    """Exact + 키워드 + 의미 검색을 RRF 방식으로 통합.

    새 API. 기존 legal_graph.py는 아직 이 함수를 호출하지 않으므로
    서비스에 적용하려면 그 그래프 호출부를 이후에 수정해야 한다.
    """
    if k <= 0 or not question.strip():
        return []
    keyword_query = (search_query or "").strip()
    # Query Rewriter가 생성한 개념뿐 아니라 원 질문의 표현도 검색
    keyword_query = question if not keyword_query else keyword_query + ", " + question
    channels = [
        ("exact", exact_search_laws(question), 10.0),
        ("keyword", keyword_search_laws(keyword_query, k=max(30, k * 4)), 3.0),
        ("vector", search_laws(question, category, k=max(20, k * 3)), 1.0),
    ]
    candidates: dict[tuple, dict] = {}
    for channel, items, weight in channels:
        for rank, item in enumerate(items, 1):
            identity = (item["law_name"], item["article"])
            cand = candidates.setdefault(identity, {"item": dict(item), "score": 0.0, "types": []})
            cand["score"] += weight / (30 + rank)
            cand["types"].append(channel)
            if channel == "exact" or (channel == "keyword" and cand["item"]["source_type"] == "vector"):
                cand["item"] = dict(item)
    ordered = sorted(candidates.values(), key=lambda c: -(
        c["score"] + _soft_domain_bonus(c["item"]["law_name"], category, question)
    ))
    result = []
    for candidate in ordered[:k]:
        item = candidate["item"]
        item["retrieval_methods"] = sorted(set(candidate["types"]))
        result.append(item)
    return result
