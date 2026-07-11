# ---------------------------------------------------------------------------
# Dockerfile for proxy_server.py (the public OpenAI-compatible proxy).
#
# This image only runs the lightweight FastAPI proxy — it does NOT contain
# llama.cpp / GGUF / GPU dependencies. The actual model inference happens on
# the Kaggle worker (run_model_with_proxy.py), which connects outbound to
# whatever host runs this container. No GPU is needed here.
#
# Build:
#   docker build -t gguf-proxy .
#
# Run (persist config + logs on the host so edits/restarts survive):
#   docker run -d \
#     --name gguf-proxy \
#     -p 8000:8000 \
#     -v $(pwd)/proxy_server_config.json:/app/proxy_server_config.json \
#     -v $(pwd)/logs:/app/logs \
#     --restart unless-stopped \
#     gguf-proxy
#
# On first run (if no config file is mounted), the container will generate a
# default proxy_server_config.json inside /app — copy it out, edit the API
# keys/secrets, then re-mount it as shown above.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr so logs show up immediately
# with `docker logs`, and avoid writing .pyc files into the image layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS packages needed to build/verify wheels (curl also doubles as the
# HEALTHCHECK client below). Kept minimal on purpose.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer whenever
# only application code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code + config. The config file gets baked in as a fallback
# default; mount your own edited copy over it at runtime (see docker run
# example above) to persist real API keys/secrets across rebuilds.
COPY proxy_server.py .
COPY proxy_server_config.json .

# Run as a non-root user for defense in depth.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Basic container-level health check against the proxy's own /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# uvicorn is invoked directly (rather than `python proxy_server.py`) so we
# can pass production-friendly flags (worker count, proxy headers) without
# touching the app code. Keep --workers 1: job/worker state in
# proxy_server.py is in-memory and must not be split across processes.
CMD ["uvicorn", "proxy_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*"]
