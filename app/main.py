from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.exceptions import register_exception_handlers
from app.middleware.rate_limit import RateLimitMiddleware, configure_rate_limits, api_rate_limiter
from app.api.routes.health import router as health_router
from app.api.routes.fireflies import router as fireflies_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.chat import router as chat_router
from app.api.routes.webhook import router as webhook_router
from app.api.routes.auth import router as auth_router
from app.api.routes.users import router as users_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.api.routes.ghostwriter import router as ghostwriter_router
from app.api.routes.infographic import router as infographic_router
from app.api.routes.mcp import router as mcp_router
from app.services.background_ingestion import background_ingestion
from app.database.connection import init_db, close_db
from app.cache.redis_client import init_redis, close_redis

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 DANI Engine starting up...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"⚠️ Database initialization failed: {e}")
    
    # Initialize Redis
    try:
        await init_redis()
        logger.info("✅ Redis initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization failed: {e}")
    
    # Initialize MCP (Model Context Protocol) for external tools
    if settings.MCP_ENABLED:
        try:
            from app.mcp.registry import initialize_mcp
            logger.info("🔌 Initializing MCP...")
            success = await initialize_mcp()
            if success:
                logger.info("MCP initialized")
            else:
                logger.warning("MCP initialization skipped (disabled or no credentials)")
        except Exception as e:
            logger.warning(f"MCP initialization failed: {e}")
    
    # Start background ingestion (non-blocking) - only if enabled
    if settings.BACKGROUND_INGESTION_ENABLED:
        await background_ingestion.start()
    else:
        logger.info("⏸️ Background ingestion disabled (set BACKGROUND_INGESTION_ENABLED=true to enable)")
    
    # Pre-warm LLM to avoid cold start on first request
    if settings.LLM_WARMUP_ENABLED:
        try:
            from app.llm.ollama import OllamaClient
            logger.info("🔥 Warming up LLM...")
            llm = OllamaClient()
            # Simple warmup prompt - first inference loads model into memory
            await llm.generate("Hello", stream=False)
            await llm.close()
            logger.info("✅ LLM warmed up")
        except Exception as e:
            logger.warning(f"⚠️ LLM warmup failed (first request may be slow): {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 DANI Engine shutting down...")
    if settings.BACKGROUND_INGESTION_ENABLED:
        await background_ingestion.stop()
    
    # Shutdown MCP
    if settings.MCP_ENABLED:
        try:
            from app.mcp.registry import shutdown_mcp
            await shutdown_mcp()
            logger.info("✅ MCP shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️ MCP shutdown failed: {e}")
    
    # Close database
    try:
        await close_db()
        logger.info("✅ Database closed")
    except Exception as e:
        logger.warning(f"⚠️ Database close failed: {e}")
    
    # Close Redis
    try:
        await close_redis()
        logger.info("✅ Redis closed")
    except Exception as e:
        logger.warning(f"⚠️ Redis close failed: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENV == "development" else None,
    redoc_url="/redoc" if settings.ENV == "development" else None,
    openapi_url="/openapi.json" if settings.ENV == "development" else None,
)

# Register custom exception handlers for standardized error responses
register_exception_handlers(app)

# Rate limiting middleware (applied before other middleware)
rate_limit_config = configure_rate_limits(
    chat_per_minute=60,
    retrieval_per_minute=60,
    ingestion_per_minute=10,
    auth_per_minute=30,
)
api_rate_limiter.config = rate_limit_config
app.add_middleware(RateLimitMiddleware, rate_limiter=api_rate_limiter)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://34.136.51.101:3000", 
        "*",  # Allow all origins for flexibility (can restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health_router,
    prefix=settings.API_V1_PREFIX,
    tags=["Health"],
)

app.include_router(
    fireflies_router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    ingest_router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    retrieval_router,
    prefix=settings.API_V1_PREFIX,
)


app.include_router(
    chat_router, 
    prefix=settings.API_V1_PREFIX
)

app.include_router(
    webhook_router,
    prefix=settings.API_V1_PREFIX,
)

# Auth & User routes
app.include_router(
    auth_router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"],
)

app.include_router(
    users_router,
    prefix=f"{settings.API_V1_PREFIX}/users",
    tags=["Users"],
)

app.include_router(
    conversations_router,
    prefix=f"{settings.API_V1_PREFIX}/conversations",
    tags=["Conversations"],
)

app.include_router(
    documents_router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    ghostwriter_router,
    prefix=settings.API_V1_PREFIX,
    tags=["Ghostwriter"],
)

app.include_router(
    infographic_router,
    prefix=settings.API_V1_PREFIX,
    tags=["Infographic"],
)

app.include_router(
    mcp_router,
    prefix=settings.API_V1_PREFIX,
    tags=["MCP Tools"],
)


@app.get("/")
def root():
    return {"message": "DANI Engine is running"}
