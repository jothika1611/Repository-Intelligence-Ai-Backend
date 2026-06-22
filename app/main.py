import os
# Prevent OpenBLAS memory allocation errors on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.qdrant_client import qdrant_service

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Qdrant database...")
    await qdrant_service.initialize_collection()
    yield
    # Shutdown actions
    logger.info("Closing Qdrant connection...")
    await qdrant_service.close()

app = FastAPI(
    title="Repository Intelligence Agent",
    description="A RAG system for analyzing GitHub repositories.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.repositories import router as repositories_router
from app.api.chat import router as chat_router

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(repositories_router)
app.include_router(chat_router)
