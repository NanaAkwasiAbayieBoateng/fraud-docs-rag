# Fraud Docs RAG — FastAPI app image.
#
# The app talks to Ollama over HTTP (OLLAMA_HOST); Ollama itself is NOT
# baked into this image. Run both together with docker-compose.yml, or
# point OLLAMA_HOST at an Ollama already running on the host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this layer stays cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, the sample docs, and the entrypoint. A prebuilt
# data/index.json is copied too when one exists locally; on a fresh
# clone it does not (it is gitignored), and the entrypoint builds it
# from the sample docs on first start instead.
COPY rag_app/ ./rag_app/
COPY data/ ./data/
COPY docker-entrypoint.sh .

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["sh", "docker-entrypoint.sh"]
