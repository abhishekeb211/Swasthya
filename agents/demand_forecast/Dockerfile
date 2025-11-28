# =============================================================================
# DEMAND FORECAST AGENT - DOCKERFILE
# =============================================================================
# Multi-stage build for optimized production image
# Base: Python 3.10 (required for Prophet compatibility)
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: Builder
# Install dependencies and compile Prophet (which has C++ components)
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
# Prophet requires: gcc, g++, make for compiling Stan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 2: Runtime
# Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

# Labels for container metadata
LABEL maintainer="Hospital AI Platform Team" \
    version="1.0.0" \
    description="Demand Forecast Agent - Patient volume prediction using Prophet" \
    org.opencontainers.image.source="https://github.com/hospital-ai/demand-forecast-agent"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application settings
    SERVICE_NAME=demand-forecast-agent \
    API_HOST=0.0.0.0 \
    API_PORT=8001 \
    LOG_LEVEL=INFO \
    # Prophet/Stan settings
    CMDSTAN_VERSION=2.33.1 \
    # Disable Stan's verbose output
    STAN_NUM_THREADS=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appgroup /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command: Run API server
# Can be overridden for FL client mode
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]

# =============================================================================
# ALTERNATIVE ENTRY POINTS
# =============================================================================
# Run as FL client:
#   docker run demand-forecast-agent python fl_client.py --server fl-server:8087
#
# Run training only:
#   docker run demand-forecast-agent python train.py --synthetic
#
# Interactive shell:
#   docker run -it demand-forecast-agent /bin/bash
# =============================================================================
