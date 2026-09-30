# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies required for numerical libraries (numpy, scipy, pymc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration
COPY pyproject.toml ./

# Upgrade pip, build and install dependencies into a local wheels directory
RUN pip install --upgrade pip build && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels .

# Stage 2: Runtime
FROM python:3.11-slim

# Create a non-root user and group
RUN groupadd -r mede && useradd -r -g mede mede

WORKDIR /app

# Install runtime dependencies for scientific computing (e.g. libgomp for some C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder
COPY --from=builder /app/wheels /wheels

# Install dependencies from wheels without index
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source code and data
COPY src/ /app/src/
COPY data/ /app/data/

# Ensure the non-root user owns the directory
RUN chown -R mede:mede /app

# Switch to non-root user
USER mede

# Expose FastAPI application port and Prometheus metrics port
EXPOSE 8000
EXPOSE 9090

# Run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
