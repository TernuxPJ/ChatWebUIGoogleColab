#!/usr/bin/env bash
# Warm up all configured Unicollect Ollama models (Mini + 1.2) and pin KV cache in RAM.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/warm_all.py "$@"
