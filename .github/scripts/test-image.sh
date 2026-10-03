#!/usr/bin/env bash
# Tests a built image: runs the test suite inside the container, then starts it with an
# API key and checks health, access protection, analysis and the MCP endpoint from outside.
# Usage: .github/scripts/test-image.sh <image>
set -euo pipefail

IMAGE="${1:?usage: $0 <image>}"
NAME="image-test-$$"
KEY="ci-test-key-$RANDOM$RANDOM"
PORT=18000

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "::group::Test suite inside the image"
docker run --rm -u root \
  -v "$PWD/tests:/app/tests:ro" \
  -v "$PWD/requirements-dev.txt:/tmp/requirements-dev.txt:ro" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$IMAGE" sh -c "pip install -q -r /tmp/requirements-dev.txt && python -m pytest -q -p no:cacheprovider"
echo "::endgroup::"

echo "::group::Smoke test of the running container"
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" -e API_KEY="$KEY" "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health"
echo

status=$(curl -s -o /dev/null -w '%{http_code}' -F "file=@README.md" "http://127.0.0.1:$PORT/analyze")
[ "$status" = "401" ] || { echo "expected 401 without key, got $status"; exit 1; }

status=$(curl -s -o /dev/null -w '%{http_code}' -H "X-API-Key: $KEY" -F "file=@README.md" "http://127.0.0.1:$PORT/analyze")
[ "$status" = "200" ] || { echo "expected 200 with key, got $status"; exit 1; }

status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/mcp" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"ci","version":"1"}}}')
[ "$status" = "200" ] || { echo "expected 200 from /mcp initialize, got $status"; exit 1; }

health=$(docker inspect --format '{{.State.Health.Status}}' "$NAME")
echo "container health: $health"
echo "::endgroup::"
echo "Image test passed: $IMAGE"
