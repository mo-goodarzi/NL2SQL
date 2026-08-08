#!/bin/sh
set -e

# Build the SQLite database from the raw CSVs on first run, if it isn't
# already present on the mounted data volume.
if [ ! -f "data/olist.db" ]; then
    if [ -d "data/raw" ] && [ -n "$(ls -A data/raw 2>/dev/null)" ]; then
        echo "No database found -- building it from data/raw ..."
        python scripts/load_data.py
    else
        echo "WARNING: no database at data/olist.db and no CSVs in data/raw."
        echo "Mount a populated ./data directory or run scripts/load_data.py."
    fi
fi

case "$1" in
    api)
        # $PORT is set dynamically by platforms like Railway; falls back to
        # the fixed port docker-compose.yml expects for local use.
        exec uvicorn app.api:app --host 0.0.0.0 --port "${PORT:-8000}"
        ;;
    ui)
        exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
        ;;
    *)
        exec "$@"
        ;;
esac
