from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.core.config import get_settings
from src.core.telemetry import setup_telemetry, get_prometheus_metrics
from src.api.v1.endpoints import router as api_v1_router
from src.api.dependencies import get_mede_workflow, get_data_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize telemetry and Prometheus
    setup_telemetry()
    
    # Pre-warm heavy singletons so first request doesn't eat the cold-start cost
    get_data_engine()
    get_mede_workflow()
    
    yield
    # Shutdown logic would go here if needed

def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Marketing Econometrics & Decision Engine (MEDE) API",
        lifespan=lifespan
    )
    
    # Configure CORS for API clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Should be restricted in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount the v1 router
    app.include_router(api_v1_router, prefix=settings.API_V1_STR)
    
    @app.get("/health", tags=["System"])
    async def health_check():
        return JSONResponse(
            content={"status": "healthy", "version": settings.VERSION}
        )
        
    @app.get("/metrics", tags=["System"])
    async def metrics():
        return PlainTextResponse(get_prometheus_metrics())
        
    return app

app = create_app()
