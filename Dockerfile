FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[anthropic,local-embeddings]"

ENV MEMENTO_DB_PATH=/data/memento.db
VOLUME /data

ENTRYPOINT ["memento-mcp"]
