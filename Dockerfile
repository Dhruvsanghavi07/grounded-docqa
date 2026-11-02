# Stage 1: build the React UI
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: fetch the corpus and build the search index
FROM python:3.12-slim AS index
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/pyproject.toml ./backend/
COPY backend/docqa ./backend/docqa
RUN pip install --no-cache-dir ./backend
COPY scripts ./scripts
RUN ./scripts/fetch_corpus.sh backend/data/corpus && cd backend && docqa ingest data/corpus --out data/index.pkl

# Stage 3: small runtime image with the API, the index and the built UI
FROM python:3.12-slim
WORKDIR /app/backend
COPY backend/pyproject.toml ./
COPY backend/docqa ./docqa
RUN pip install --no-cache-dir . && useradd --create-home app
COPY --from=index /app/backend/data/index.pkl ./data/index.pkl
COPY --from=ui /ui/dist ./static
ENV DOCQA_SERVE=1 DOCQA_INDEX=data/index.pkl DOCQA_STATIC=static
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "docqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
