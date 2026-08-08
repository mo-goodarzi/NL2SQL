FROM python:3.11-slim

WORKDIR /app

# System deps for pandas/sqlite wheels build if no manylinux wheel is available
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/

# data/ is mounted as a volume at runtime (see docker-compose.yml); it is not
# baked into the image because the SQLite db is large and regenerable.
RUN mkdir -p data

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]

# Overridden per-service in docker-compose.yml (api vs ui).
CMD ["api"]
