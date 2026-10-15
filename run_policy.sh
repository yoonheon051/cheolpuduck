#!/usr/bin/env bash
# ONNX 정책을 CPU MuJoCo에서 키보드로 구동 (GPU 불필요)
# 사용: ./run_policy.sh            -> walking+standing+sitstand+roulade 핫스왑
#       ./run_policy.sh walk       -> walking만
# 추가 인자는 infer_policy.py로 그대로 전달 (예: --no-bam, --debug)
P="$(cd "$(dirname "$0")" && pwd)/policies"
cd "$(dirname "$0")/microduck_rl"
mode="${1:-all}"; [ $# -gt 0 ] && shift
case "$mode" in
  walk) args=(--walking "$P/alpha_walking.onnx") ;;
  *)    args=(--walking "$P/alpha_walking.onnx" --standing "$P/alpha_stand.onnx" \
              --sitstand "$P/alpha_sitstand.onnx" --roulade "$P/roulade.onnx") ;;
esac
exec ../.venv/bin/python scripts/infer_policy.py "${args[@]}" --new-cmd-obs "$@"
