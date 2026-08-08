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

# Locally (docker-compose.yml), data/ is mounted as a volume and this COPY is
# shadowed by the mount. For a platform with no volume step of its own (e.g.
# Railway), the image needs the database baked in -- data/olist.db is tracked
# via Git LFS specifically so this COPY has something to find.
COPY data/olist.db data/olist.db

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]

# Overridden per-service in docker-compose.yml (api vs ui).
CMD ["api"]
