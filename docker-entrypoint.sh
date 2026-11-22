#!/bin/sh
# Build the search index on first start if it is missing (a fresh clone
# has none — data/index.json is a generated artifact, not committed),
# then hand over to the API server.
set -e
if [ ! -f data/index.json ]; then
  echo "No index found - building one from data/sample_docs ..."
  python -m rag_app.ingest --docs data/sample_docs --out data/index.json
fi
exec uvicorn rag_app.main:app --host 0.0.0.0 --port 8000
