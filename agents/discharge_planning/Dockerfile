# =============================================================================
# Discharge Planning Agent - Docker Image
# =============================================================================
# Multi-stage build for optimized production image
#
# Build: docker build -t discharge-planning-agent:latest .
# Run:   docker run -p 8004:8004 discharge-planning-agent:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Image
# -----------------------------------------------------------------------------
FROM python:3.10-slim as production

# Labels for container metadata
LABEL maintainer="Hospital AI Team"
LABEL version="1.0.0"
LABEL description="Discharge Planning Agent - Hybrid ML + Rules for discharge readiness assessment"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Service configuration
    API_HOST=0.0.0.0 \
    API_PORT=8004 \
    LOG_LEVEL=INFO \
    # XGBoost needs OpenMP
    OMP_NUM_THREADS=4

# Install runtime dependencies
# libgomp1 is required for XGBoost's OpenMP support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appgroup /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose the service port
EXPOSE 8004

# Health check
# Checks the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Default command
# Uses uvicorn to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]

# =============================================================================
# Build Arguments (optional customization)
# =============================================================================
# To build with custom settings:
# docker build --build-arg MODEL_PATH=/custom/path -t discharge-agent .
ARG MODEL_PATH=/app/models/discharge_readiness.json
ENV MODEL_PATH=${MODEL_PATH}
