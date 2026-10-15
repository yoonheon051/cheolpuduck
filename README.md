# Microduck × MuJoCo 워크스페이스

GPU 없이 CPU MuJoCo에서 microduck을 구동하는 환경 (학습은 CUDA GPU 필요 → 여기선 추론/뷰만).

```
microduck_rl/   pollen-robotics/microduck_rl (MJCF 모델, infer_policy.py)
policies/       HF kyoungsim/microduck 의 ONNX 정책 (walking/stand/sitstand/roulade)
.venv/          mujoco 3.x, onnxruntime, BAM(락된 커밋 62bd8ce) — uv로 생성
run_viewer.sh   모델만 둘러보기 (키프레임 INIT/STAND/SIT/FOLD)
run_policy.sh   정책 구동 + 키보드 조작
```

## 사용
```bash
./run_viewer.sh                 # 정적 뷰어
./run_policy.sh walk            # 걷기 정책만
./run_policy.sh                 # 4개 정책 핫스왑 (아래 키)
./run_policy.sh walk --no-bam   # BAM 액추에이터 대신 XML PD
```
키보드 (터미널에 포커스): 속도 명령 키, `Y` 앉기/서기, `R` roulade. 정확한 키맵은
`microduck_rl/scripts/infer_policy.py` 상단/README 참고. `--debug`, `--save-csv`, `--record` 지원.

## 참고
- 모든 정책은 61차원 관측이라 `--new-cmd-obs` 필수 (스크립트에 이미 포함).
- 다른 정책은 `policies/`에 넣고 `run_policy.sh`의 경로만 바꾸면 됨.
- 모델 라이선스는 `microduck_rl/LICENSE-MODELS.md` (CC-BY-NC-SA) 확인.
- uv는 `~/.local/bin/uv`에 설치됨 (PATH에 없으면 `export PATH=$HOME/.local/bin:$PATH`).
