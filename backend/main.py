
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.api_v13_support import _quoted_sources
from backend.graph.legal_graph_v13 import run_legal_chat


app = FastAPI(
    title="Law Action Assistant API",
    version="2.0.0",
)


# =========================================================
# CORS
# =========================================================

cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

cors_env = os.getenv("CORS_ORIGINS", "")

if cors_env:
    for origin in cors_env.split(","):
        origin = origin.strip()

        if origin and origin not in cors_origins:
            cors_origins.append(origin)


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Legal-Validation-Status",
        "X-Legal-Evidence-Mode",
    ],
)


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():
    return {
        "service": "Law Action Assistant API",
        "status": "running",
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# =========================================================
# Chat API - LangGraph v13
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest, response: Response):
    try:
        result = run_legal_chat(
            question=request.question,
            session_id=request.session_id,
        )

        response.headers["X-Legal-Validation-Status"] = str(
            result.get("answer_validation_status", "unknown")
        )
        response.headers["X-Legal-Evidence-Mode"] = str(
            result.get("evidence_selection_mode", "unknown")
        )

        sources = _quoted_sources(result)

        return ChatResponse(
            category=result.get("category", "기타"),
            answer=result.get("answer", ""),
            sources=sources,
        )

    except Exception as e:
        error_message = str(e)

        # Groq API 사용량 / Rate Limit 초과
        if (
            "rate_limit_exceeded" in error_message
            or "Rate limit reached" in error_message
            or "Error code: 429" in error_message
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "현재 AI 사용량이 일시적으로 많습니다. "
                    "잠시 후 다시 시도해주세요."
                ),
            )

        # 그 외 서버 오류
        raise HTTPException(
            status_code=500,
            detail=(
                "답변을 생성하는 중 오류가 발생했습니다. "
                "잠시 후 다시 시도해주세요."
            ),
        )
