FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/data /app/sessions /app/chroma_db /app/logs

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CHROMA_DB_PATH=/app/chroma_db
ENV DATA_PATH=/app/data
ENV SESSIONS_PATH=/app/sessions

# Expose port
EXPOSE 8080

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "web" ]; then\n\
    python -m agentic_ai.web_interface\n\
elif [ "$1" = "interactive" ]; then\n\
    python -m agentic_ai interactive\n\
elif [ "$1" = "research" ]; then\n\
    shift\n\
    python -m agentic_ai research "$@"\n\
elif [ "$1" = "upload" ]; then\n\
    shift\n\
    python -m agentic_ai upload "$@"\n\
else\n\
    python -m agentic_ai "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["web"]