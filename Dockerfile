FROM python:3.11-slim

WORKDIR /app

# System deps for pandas/sqlite wheels build if no manylinux wheel is
# available, plus git/git-lfs for the LFS fallback fetch below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs \
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

# Some CI/build platforms (confirmed: Railway) check out the repo without
# resolving Git LFS smudging, so the COPY above can end up with the ~130-byte
# pointer stub instead of the real file. Detect that (an LFS pointer starts
# with the literal text "version") and fall back to a plain, anonymous
# `git clone` + `git lfs pull` of this public repo -- no credentials needed.
# A normal local build (git-lfs installed, real file already in place) never
# hits this branch.
RUN if head -c 7 data/olist.db | grep -q "version"; then \
        echo "data/olist.db is an unresolved Git LFS pointer -- fetching via git-lfs..." && \
        git clone --depth 1 --filter=blob:none \
            https://github.com/mo-goodarzi/NL2SQL.git /tmp/lfs_repo && \
        (cd /tmp/lfs_repo && git lfs pull) && \
        cp /tmp/lfs_repo/data/olist.db data/olist.db && \
        rm -rf /tmp/lfs_repo; \
    fi && \
    size=$(wc -c < data/olist.db) && \
    echo "data/olist.db: ${size} bytes" && \
    [ "$size" -gt 1000000 ] || (echo "data/olist.db is too small to be the real database -- aborting build" && exit 1)

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]

# Overridden per-service in docker-compose.yml (api vs ui).
CMD ["api"]
