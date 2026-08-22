import time
from pathlib import Path
from fastapi import FastAPI, Request
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

from fastapi.responses import JSONResponse

# 1. Official FastAPI CORS Middleware (Registered immediately after app creation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kineticai-hhgoa.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.onrender\.com|http://localhost:.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 2. Safe Request & Error Logging Middleware (No secrets/tokens logged)
@app.middleware("http")
async def request_logger_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    method = request.method
    path = request.url.path
    origin = request.headers.get("origin", "no-origin")
    print(f">>> {method} {path} (Origin: {origin})")

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"<<< {method} {path} STATUS: {response.status_code} ({duration_ms:.1f}ms)")
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        print(f"!!! ERROR {method} {path} ({duration_ms:.1f}ms): {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"},
            headers={"Access-Control-Allow-Origin": origin if origin != "no-origin" else "*"}
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": origin}
    )

# 3. Register API Routers
app.include_router(health_router)
app.include_router(rag_router)
app.include_router(voice_router)
app.include_router(benchmark_router)


@app.get("/")
@app.head("/")
@app.get("/health")
@app.head("/health")
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
    try:
        import psutil
        current_rss = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        current_rss = 0.0

    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"  [MEMORY] Process RSS at startup: {current_rss} MB")
    print(f"  Gemini Model:    {settings.GEMINI_MODEL}")
    print(f"  Sarvam STT:      {settings.SARVAM_STT_MODEL}")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"  Qdrant Storage:  {settings.QDRANT_STORAGE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
