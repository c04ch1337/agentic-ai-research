    FROM python:3.11-slim AS builder

    ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PIP_NO_CACHE_DIR=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy requirements first for caching
    COPY requirements.txt .
    RUN python -m pip install --upgrade pip && \
        pip wheel --wheel-dir /wheels -r requirements.txt
    
    # ---- runtime ----
    FROM python:3.11-slim
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        APP_HOME=/app \
        CHROMA_DB_PATH=/app/chroma_db \
        DATA_PATH=/app/data \
        SESSIONS_PATH=/app/sessions \
        PYTHONPATH=/app
    
    WORKDIR ${APP_HOME}
    
    # Create non-root user
    RUN useradd -m appuser && mkdir -p /app/data /app/sessions /app/chroma_db /app/logs && \
        chown -R appuser:appuser /app
    
    # System deps minimal
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy wheels and install
    COPY --from=builder /wheels /wheels
    RUN pip install --no-cache-dir /wheels/*
    
    # Copy source
    COPY . .
    
    # Entrypoint
    RUN printf '%s\n' '#!/bin/sh' \
    'set -e' \
    'case "$1" in' \
    '  web) exec python -m agentic_ai.web_interface ;;' \
    '  interactive) exec python -m agentic_ai interactive ;;' \
    '  research) shift; exec python -m agentic_ai research "$@" ;;' \
    '  upload) shift; exec python -m agentic_ai upload "$@" ;;' \
    '  *) exec python -m agentic_ai "$@" ;;' \
    'esac' > /usr/local/bin/entrypoint && chmod +x /usr/local/bin/entrypoint
    
    USER appuser
    
    EXPOSE 8080
    HEALTHCHECK --interval=30s --timeout=3s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health').read()" || exit 1
    ENTRYPOINT ["entrypoint"]
    CMD ["web"]    