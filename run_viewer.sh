#!/usr/bin/env bash
# 정적 MuJoCo 뷰어: 로봇 모델/키프레임(INIT/STAND/SIT/FOLD)을 마우스로 둘러보기
cd "$(dirname "$0")/microduck_rl"
exec ../.venv/bin/python -m mujoco.viewer --mjcf "${1:-src/mjlab_microduck/robot/microduck/scene.xml}"
