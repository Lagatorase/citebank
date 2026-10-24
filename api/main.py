"""REST API around the RAG pipeline.

  uvicorn api.main:app --reload       then open http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from banklens.config import get_settings
from banklens.pipeline import RAGPipeline

# the pipeline holds the models in memory, so I create it once when the server starts
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = RAGPipeline(get_settings())
    yield
    state.clear()


app = FastAPI(title="BankLens", description="Cited answers from DBS, OCBC and UOB annual reports", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["What was DBS's net profit for 2025?"])
    top_k: int | None = Field(None, ge=1, le=20)


class SearchRequest(AskRequest):
    pass


@app.get("/health")
def health():
    # lets a load balancer or docker check that the index and models are loaded
    rag: RAGPipeline = state["rag"]
    return {"status": "ok", "chunks": len(rag.store.chunks), "banks": sorted(rag.store.banks()),
            "years": sorted(rag.store.years()), "llm": rag.llm.name, "embedder": rag.store.embedder_name}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return state["rag"].ask(req.question, req.top_k).to_dict()
    except RuntimeError as exc:
        # this is usually Ollama not running, which the caller should see as "service unavailable"
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search")
def search(req: SearchRequest):
    # retrieval only, which is useful for debugging why an answer went wrong
    hits = state["rag"].retrieve(req.question, req.top_k)
    return [{"bank": h.chunk.bank, "year": h.chunk.year, "page": h.chunk.page, "section": h.chunk.section,
             "kind": h.chunk.kind, "score": h.score, "via": h.source, "text": h.chunk.text} for h in hits]
