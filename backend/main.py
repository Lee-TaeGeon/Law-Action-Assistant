import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    LawSource,
)
from backend.graph.legal_graph import (
    run_legal_chat,
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Law Action Assistant API",
    version="2.0.0",
)


# =========================================================
# CORS
# =========================================================

# 로컬 개발 환경
cors_origins = [
    "http://localhost:5173",
]


# Railway 환경변수 예:
#
# CORS_ORIGINS=https://law-action-assistant.vercel.app
#
# 여러 주소:
#
# CORS_ORIGINS=https://aaa.vercel.app,https://bbb.vercel.app

cors_env = os.getenv(
    "CORS_ORIGINS",
    "",
)

if cors_env:

    for origin in cors_env.split(","):

        origin = origin.strip()

        if (
            origin
            and origin not in cors_origins
        ):
            cors_origins.append(
                origin
            )


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():
    return {
        "service":
            "Law Action Assistant API",
        "status":
            "running",
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():
    return {
        "status":
            "ok",
    }


# =========================================================
# Chat API
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    try:

        result = run_legal_chat(
            question=(
                request.question
            ),
            session_id=(
                request.session_id
            ),
        )

        sources = [
            LawSource(
                law_name=source[
                    "law_name"
                ],
                article=source.get(
                    "article"
                ),
                content=source[
                    "content"
                ],
            )
            for source
            in result.get(
                "sources",
                [],
            )
        ]

        return ChatResponse(
            category=result.get(
                "category",
                "기타",
            ),
            answer=result.get(
                "answer",
                "",
            ),
            sources=sources,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )