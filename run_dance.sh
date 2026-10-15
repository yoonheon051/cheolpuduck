#!/usr/bin/env bash
# 박자에 맞춰 춤추는 microduck (CPU MuJoCo). 인자는 dance.py 로 그대로 전달.
# 예: ./run_dance.sh --bpm 120   |   ./run_dance.sh --bpm 100 --headless 20
exec "$(cd "$(dirname "$0")" && pwd)/.venv/bin/python" "$(cd "$(dirname "$0")" && pwd)/dance.py" "$@"
