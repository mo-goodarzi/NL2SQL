FROM python:3.11-slim

WORKDIR /app

# System deps for pandas/sqlite wheels build if no manylinux wheel is
# available, plus curl for the Git-LFS fallback fetch below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
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

# Some CI/build platforms (confirmed: Railway, for a private repo) check out
# the repo without resolving Git LFS smudging, so the COPY above can end up
# with the ~130-byte pointer stub instead of the real file. Detect that (an
# LFS pointer starts with the literal text "version", a real SQLite file
# starts with the "SQLite format 3" magic header) and fall back to fetching
# the real object through the LFS Batch API -- the same protocol the git-lfs
# client itself uses over HTTPS: authenticate once to get a short-lived,
# unauthenticated presigned S3 URL, then download from that directly.
# media.githubusercontent.com (tried first, before this) expects Basic-auth
# git-over-HTTPS conventions rather than a Bearer header and 404s otherwise.
# GITHUB_TOKEN only needs to be supplied when this branch actually runs --
# building locally with git-lfs installed never hits it.
ARG GITHUB_TOKEN
RUN if head -c 7 data/olist.db | grep -q "version"; then \
        echo "data/olist.db is an unresolved Git LFS pointer -- fetching the real object via the LFS batch API..." && \
        curl -fsS -u "x-access-token:${GITHUB_TOKEN}" \
             -H "Accept: application/vnd.git-lfs+json" \
             -H "Content-Type: application/vnd.git-lfs+json" \
             -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"e03a8941e6a09d041294edf7a201514725faa0b4b6eb1f894ef25cd2819714db","size":196939776}]}' \
             https://github.com/mo-goodarzi/NL2SQL.git/info/lfs/objects/batch \
             -o /tmp/lfs_batch.json && \
        cat /tmp/lfs_batch.json && \
        download_url=$(sed -n 's/.*"download":{"href":"\([^"]*\)".*/\1/p' /tmp/lfs_batch.json) && \
        [ -n "$download_url" ] && \
        curl -fL "$download_url" -o data/olist.db; \
    fi && \
    size=$(wc -c < data/olist.db) && \
    echo "data/olist.db: ${size} bytes" && \
    [ "$size" -gt 1000000 ] || (echo "data/olist.db is too small to be the real database -- aborting build" && exit 1)

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]

# Overridden per-service in docker-compose.yml (api vs ui).
CMD ["api"]
