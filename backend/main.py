import os
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.api.routes_rag import router as rag_router
from backend.api.routes_voice import router as voice_router
from backend.api.routes_health import router as health_router
from backend.api.routes_benchmark import router as benchmark_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Voice-Enabled Adaptive RAG Engine (HH Goa 2026)"
)

# Custom HTTP Middleware guaranteeing CORS headers on every response, OPTIONS preflight, and error
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin", "*")
    if request.method == "OPTIONS":
        response = Response(status_code=200)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    try:
        response = await call_next(request)
    except Exception as exc:
        response = JSONResponse(status_code=500, content={"detail": str(exc)})

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Also register standard CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(rag_router)
app.include_router(voice_router)
app.include_router(benchmark_router)


@app.get("/")
@app.get("/health")
async def root_health():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


# Mount frontend build static directory if available
FRONTEND_DIST = Path("./frontend/dist")
if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


@app.on_event("startup")
async def on_startup():
    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"  Gemini Model:    {settings.GEMINI_MODEL}")
    print(f"  Sarvam STT:      {settings.SARVAM_STT_MODEL}")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"  Qdrant Storage:  {settings.QDRANT_STORAGE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
