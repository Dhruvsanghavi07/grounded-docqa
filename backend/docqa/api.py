"""REST API.

GET  /health          liveness + index size
POST /ask             {"question": "...", "k": 5, "mode": "hybrid"} -> answer with sources
GET  /metrics         request count, p50/p95 latency, grounded rate, fallback rate
Every route is also served under /api. If DOCQA_STATIC points at the built web UI,
it is served at / so one container hosts both.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .index import HybridIndex
from .llm import make_generator
from .pipeline import QAPipeline

log = logging.getLogger("docqa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


class Stats:
    def __init__(self, window: int = 1000):
        self.latencies: deque[float] = deque(maxlen=window)
        self.asked = self.grounded = self.fell_back = 0

    def record(self, ms: float, grounded: bool, fell_back: bool) -> None:
        self.latencies.append(ms)
        self.asked += 1
        self.grounded += grounded
        self.fell_back += fell_back

    def snapshot(self) -> dict:
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        return {
            "requests": self.asked,
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "grounded_rate": round(self.grounded / self.asked, 3) if self.asked else None,
            "fallback_rate": round(self.fell_back / self.asked, 3) if self.asked else None,
        }


def create_app(index: HybridIndex | None = None, generator=None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        idx = index or HybridIndex.load(os.environ.get("DOCQA_INDEX", "data/index.pkl"))
        state["pipeline"] = QAPipeline(idx, generator or make_generator())
        state["stats"] = Stats()
        log.info("loaded index with %d chunks, generator=%s", len(idx.chunks), state["pipeline"].generator.name)
        yield

    app = FastAPI(title="Grounded DocQA", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("DOCQA_CORS", "*").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        log.info(
            "%s %s %d %.1fms", request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000
        )
        return response

    api = APIRouter()

    @api.get("/health")
    def health():
        p = state["pipeline"]
        return {"status": "ok", "chunks": len(p.index.chunks), "generator": p.generator.name}

    @api.post("/ask")
    def ask(req: AskRequest):
        p: QAPipeline = state["pipeline"]
        t0 = time.perf_counter()
        try:
            ans = p.ask(req.question, k=req.k, mode=req.mode)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        state["stats"].record((time.perf_counter() - t0) * 1000, ans.grounded, ans.fell_back)
        return ans.to_dict()

    @api.get("/metrics")
    def metrics():
        return state["stats"].snapshot()

    # Same routes at / (for curl and tests) and /api (what the web UI calls).
    app.include_router(api)
    app.include_router(api, prefix="/api")
    static = os.environ.get("DOCQA_STATIC")
    if static and os.path.isdir(static):
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    return app


app = None
if os.environ.get("DOCQA_SERVE"):
    app = create_app()
