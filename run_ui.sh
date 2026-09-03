#!/usr/bin/env bash
# Start the web UI for the LiDAR failure analyzer.
set -euo pipefail
cd "$(dirname "$0")"
exec ./venv/bin/python -m uvicorn web.server:app --host 127.0.0.1 --port 8000 "$@"
