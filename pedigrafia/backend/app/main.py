"""Aplicação FastAPI do Pedigrafia Digital.

Sem banco de dados, sem login, sem histórico — por decisão de escopo do MVP.
Sessões são temporárias, com TTL curto e apagamento automático.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .config import get_settings
from .storage import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pedigrafia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.start_reaper()
    logger.info("pedigrafia iniciado (segmentador=%s)", get_settings().segmenter)
    yield
    store.stop_reaper()
    store.purge_all()


app = FastAPI(
    title="Pedigrafia Digital",
    version="1.0.0",
    description=(
        "Pedigrafia calibrada por marcador fiducial de 50 × 50 mm. "
        "Toda geometria trafega em milímetros; o PDF sai em escala 1:1 real."
    ),
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Rate limiting básico por IP, em memória (MVP de processo único)."""
    s = get_settings()
    if request.url.path.startswith("/api") and request.method in ("POST", "PUT"):
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = _hits[client]
        while window and now - window[0] > s.rate_limit_window_seconds:
            window.popleft()
        if len(window) >= s.rate_limit_requests:
            return JSONResponse(status_code=429, content={
                "code": "rate_limited",
                "message": "Muitas requisições. Aguarde alguns segundos.",
            })
        window.append(now)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "name": "Pedigrafia Digital",
        "version": "1.0.0",
        "docs": "/docs",
        "notice": ("Escala física derivada exclusivamente do marcador de 50 × 50 mm. "
                   "Nenhuma dimensão é inferida por numeração de calçado."),
    }
