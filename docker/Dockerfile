# Multi-stage Dockerfile for DANI Backend (Development & Production)
# Build with: docker build --build-arg BUILD_ENV=production -f docker/Dockerfile .

# ============================================================================
# Stage 1: Builder - Install dependencies and download NLTK data
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data to a specific location
ENV NLTK_DATA=/opt/nltk_data
RUN mkdir -p ${NLTK_DATA} && \
    python -c "import nltk; nltk.download('punkt_tab', download_dir='${NLTK_DATA}', quiet=True)"

# ============================================================================
# Stage 2: Final - Runtime image
# ============================================================================
FROM python:3.11-slim AS final

# Build argument to control environment (development or production)
ARG BUILD_ENV=development
ENV BUILD_ENV=${BUILD_ENV}

WORKDIR /app

# Install runtime dependencies (curl for health checks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r dani && useradd -r -g dani dani

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/opt/nltk_data

# Copy application code with proper ownership
COPY --chown=dani:dani app/ ./app/
COPY --chown=dani:dani scripts/ ./scripts/
COPY --chown=dani:dani alembic/ ./alembic/
COPY --chown=dani:dani alembic.ini ./

# Add metadata labels
LABEL org.opencontainers.image.title="DANI Engine API"
LABEL org.opencontainers.image.description="AI-powered document analysis backend"
LABEL org.opencontainers.image.version="1.0.0"
LABEL maintainer="PRODG TEAM"

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Switch to non-root user
USER dani

# Conditional CMD based on BUILD_ENV
# Development: use --reload for hot-reloading
# Production: use multiple workers, no reload
CMD if [ "$BUILD_ENV" = "production" ]; then \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-4}; \
    else \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload; \
    fi
