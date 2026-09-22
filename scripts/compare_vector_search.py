import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "jhgan/ko-sroberta-multitask"

OLD_DB_PATH = "law_db_full"
NEW_DB_PATH = "law_db_optimized"
SQLITE_PATH = "law_search.db"

OLD_COLLECTION = "langchain"
NEW_COLLECTION = "law_articles"

TEST_QUERIES = [
    "회사에서 임금을 두 달째 받지 못했습니다",
    "이혼할 때 재산분할을 어떻게 청구할 수 있나요",
    "중고거래에서 돈을 보냈는데 물건을 보내지 않고 연락을 끊었습니다",
    "계약금을 지급했는데 상대방이 계약을 이행하지 않습니다",
]


print("임베딩 모델 로딩...")

model = SentenceTransformer(
    MODEL_NAME
)


old_client = chromadb.PersistentClient(
    path=OLD_DB_PATH
)

old_collection = old_client.get_collection(
    OLD_COLLECTION
)


new_client = chromadb.PersistentClient(
    path=NEW_DB_PATH
)

new_collection = new_client.get_collection(
    NEW_COLLECTION
)


db = sqlite3.connect(
    SQLITE_PATH
)

db.row_factory = sqlite3.Row


def get_article(article_id):
    row = db.execute(
        """
        SELECT
            id,
            law_name,
            article_display,
            article_title
        FROM articles
        WHERE id = ?
        """,
        (article_id,),
    ).fetchone()

    return row


for query in TEST_QUERIES:

    print()
    print("=" * 80)
    print("질문:")
    print(query)
    print("=" * 80)

    vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # -------------------------------------------------
    # 기존 DB
    # -------------------------------------------------

    old_result = old_collection.query(
        query_embeddings=[vector],
        n_results=5,
        include=[
            "metadatas",
            "distances",
        ],
    )

    print()
    print("=== 기존 Vector DB ===")

    for index, (
        metadata,
        distance,
    ) in enumerate(
        zip(
            old_result["metadatas"][0],
            old_result["distances"][0],
        ),
        start=1,
    ):
        print(
            f"{index}. "
            f"{metadata.get('law_name')} "
            f"제{metadata.get('article_no')}조 "
            f"| distance={distance:.4f}"
        )

    # -------------------------------------------------
    # 새 DB
    # -------------------------------------------------

    new_result = new_collection.query(
        query_embeddings=[vector],
        n_results=5,
        include=[
            "distances",
        ],
    )

    print()
    print("=== 최적화 Vector DB ===")

    for index, (
        article_id,
        distance,
    ) in enumerate(
        zip(
            new_result["ids"][0],
            new_result["distances"][0],
        ),
        start=1,
    ):
        article = get_article(
            int(article_id)
        )

        if article is None:
            print(
                f"{index}. article_id={article_id} "
                f"SQLite 조회 실패"
            )
            continue

        title = (
            article["article_title"]
            or ""
        )

        print(
            f"{index}. "
            f"{article['law_name']} "
            f"{article['article_display']} "
            f"{title} "
            f"| distance={distance:.4f}"
        )


db.close()
