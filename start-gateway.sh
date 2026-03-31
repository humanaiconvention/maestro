#!/usr/bin/env bash
# Start the Maestro API Gateway with config from infra/.env
# Run from the maestro/ directory
# Usage: bash start-gateway.sh

set -a
source "$(dirname "$0")/infra/.env"
set +a

exec python -m uvicorn apps.gateway.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
