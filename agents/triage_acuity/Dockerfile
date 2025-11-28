# =============================================================================
# TRIAGE & ACUITY AGENT - DOCKERFILE
# =============================================================================
# Multi-stage build for optimized production image

# -----------------------------------------------------------------------------
# STAGE 1: Build Stage
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# -----------------------------------------------------------------------------
# STAGE 2: Production Stage
# -----------------------------------------------------------------------------
FROM python:3.10-slim as production

# Labels
LABEL maintainer="Hospital AI Team" \
      service="triage-acuity-agent" \
      version="1.0.0" \
      description="AI-powered Emergency Department triage system"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Service configuration
    SERVICE_NAME=triage-acuity-agent \
    API_HOST=0.0.0.0 \
    API_PORT=8005 \
    LOG_LEVEL=INFO

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for model storage
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the application
# Using exec form for proper signal handling
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8005"]


# -----------------------------------------------------------------------------
# STAGE 3: Development Stage (optional)
# -----------------------------------------------------------------------------
FROM production as development

# Switch to root temporarily for dev tools
USER root

# Install development dependencies
RUN pip install \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    httpx>=0.25.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    mypy>=1.7.0

# Switch back to appuser
USER appuser

# Override CMD for development (with reload)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
