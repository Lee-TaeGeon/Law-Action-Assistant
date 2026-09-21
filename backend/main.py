from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.chat import ChatRequest, ChatResponse, LawSource
from backend.graph.legal_graph import run_legal_chat


app = FastAPI(
    title="Law Action Assistant API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Law Action Assistant API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    try:
        result = run_legal_chat(
            question=request.question,
            session_id=request.session_id,
        )

        sources = [
            LawSource(
                    law_name=source["law_name"],
                    article=source.get("article"),
                    content=source["content"],
        )
            for source in result.get("sources", [])
        ]

        return ChatResponse(
            category=result.get("category", "기타"),
            answer=result.get("answer", ""),
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )