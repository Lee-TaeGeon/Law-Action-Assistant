from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "law_db_full"


@lru_cache
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )


def search_laws(question: str, category: str, k: int = 5):
    db = get_vector_db()

    query = f"{category} 관련 법령 {question}"

    docs = db.as_retriever(
        search_kwargs={"k": k}
    ).invoke(query)

    results = []

    for doc in docs:
        law_name = doc.metadata.get(
            "law_name",
            doc.metadata.get("title", "관련 법령"),
        )

        results.append(
            {
                "law_name": law_name,
                "content": doc.page_content,
            }
        )

    return results