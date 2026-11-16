# Microduck 작업 노트

작성일: 2026-10-15 (Claude Code 대화 정리)

## 1. 환경 점검
- GPU: NVIDIA GeForce RTX 5080 Laptop (VRAM 16GB, 전력 한도 80W) + Intel Arrow Lake-U 내장
- 드라이버 595.99.02 / CUDA 13.2, `nvidia-smi` 정상, 커널 모듈 정상 로드
- MuJoCo 시뮬레이션 자체는 CPU로 동작. 학습(mjlab/MuJoCo Warp)은 이 GPU 사용 가능
- 마이크: `arecord -l` 기준 card 1 (sof-hda-dsp) DMIC 인식됨

## 2. 워크스페이스 (`~/microduck_ws`)
```
microduck_rl/   pollen-robotics/microduck_rl (MJCF 모델, scripts/infer_policy.py)
policies/       HF kyoungsim/microduck ONNX 정책 (alpha_walking/alpha_stand/alpha_sitstand/roulade)
.venv/          mujoco, onnxruntime, BAM (uv로 생성)
run_viewer.sh   정적 모델 뷰어
run_policy.sh   정책 구동 + 키보드 조작
dance.py        박자에 맞춰 춤추는 데모 (이번에 추가)
run_dance.sh    dance.py 실행 래퍼
README.md       워크스페이스 설명 (수정 예정)
NOTES.md        이 파일
```

## 3. 실행 방법
```bash
cd ~/microduck_ws
./run_viewer.sh                 # 정적 뷰어 (확인 완료, 창 정상)
./run_policy.sh walk            # 걷기 정책만
./run_policy.sh                 # 4개 정책 핫스왑
./run_policy.sh walk --lin-vel-x 0.2   # 처음부터 전진 명령
./run_dance.sh --bpm 120        # 춤 데모
```
- 키보드는 **터미널 창에 포커스**를 두고 입력 (MuJoCo 창 아님). stdin이 TTY가 아니면 키 입력 불가.
- 종료는 `Q` 또는 뷰어 창 닫기.

## 4. 키맵 (`infer_policy.py`의 `handle_key` 기준)
속도는 누르면 최대/최소값으로 바로 설정된다 (증가 방식 아님).

| 키 | 기본 모드 |
|---|---|
| `↑` / `↓` | 전진 / 후진 (±0.3 m/s) |
| `←` / `→` | 좌 / 우 옆걸음 (±0.2 m/s) |
| `A` / `E` | 좌 / 우 회전 (1.5 rad/s) |
| `Space` | 속도 명령 0 (정지) |
| `Y` | 앉기/서기 (전체 모드, sitstand) |
| `R` | roulade (전체 모드) |
| `P` | 랜덤 푸시 |
| `T` | 정책 추론 on/off |
| `Q` | 종료 |
| `H` | 머리 모드 토글: `↑↓` head_pitch, `←→` head_yaw, `A/E` head_roll, `Z/S` neck_pitch, `Space` 리셋 |
| `B` | 몸체 자세 모드 토글: `↑↓` z, `←→` pitch, `A/E` roll, `Z/S` yaw, `Space` 리셋 |

`G`(ground pick), `K`/`L`(킥)은 해당 정책 옵션을 줘야 동작 (현재 스크립트엔 없음).

## 5. 로봇/정책 핵심 규약 (microduck_rl/AGENTS.md 요약)
- 관측 61D = 기본 48 + 명령 13D `[twist(3), head_pose(4), body_pose(6)]`. 모든 정책이 공유하므로 `--new-cmd-obs` 필수.
- 관절 14개: 0–4 왼다리, 5–8 목/머리(neck_pitch, head_pitch, head_yaw, head_roll), 9–13 오른다리.
- 학습 범위: head_pitch/neck ±1.1, head_yaw ±1.4, head_roll ±0.31, 몸체 z ±30mm, 몸체 각도 ±30°.
- body_cmd 순서(new_cmd_obs): `[x, y, z, roll, pitch, yaw]`.
- 50Hz 제어, 시뮬레이션 timestep 0.005s, decimation 4.
- 액추에이터는 BAM(XL330 전압 제어 모델). `--no-bam`은 학습과 다른 액추에이터.
- 정책은 필터 없이(unfiltered) 학습됨. 학습 시작 전에는 항상 64 env × 5 iter 스모크 테스트.

## 6. 깃허브 업로드 / 라이선스 (법률 자문 아님)
| 대상 | 라이선스 | 비고 |
|---|---|---|
| microduck_rl 코드, MJCF(.xml), 설정 | Apache 2.0 (Pollen Robotics) | 라이선스/저작권 고지 유지, 수정 표시 |
| microduck_rl의 STL 3D 모델 | CC BY-NC-SA 4.0 | 비영리만, 크레딧+링크, 수정 표기, 동일 라이선스 |
| `policies/*.onnx` | **미확인** (HF `kyoungsim/microduck`) | 모델 페이지에서 확인 전엔 올리지 말 것 |
| `.venv/` (360MB) | 패키지별 | 올리지 않음 |
| README, run_*.sh, dance.py | 직접 작성 | 문제 없음 |

결론: **내가 만든 코드와 README만 올리고**, microduck_rl과 정책은 클론/다운로드 안내만 한다 (또는 fork/submodule).
`.gitignore`에 `.venv/`, `policies/`, `microduck_rl/` 추가 권장. 아직 `git init`/업로드는 하지 않음.

## 7. 춤추는 오리 프로젝트
목표: 노래가 들리면 춤추는 오리. 실물 로봇은 **2027년 1월경** 수령 예정이라 시뮬레이션 우선.

접근 옵션
1. 기존 정책 전환(걷기/서기/앉기/roulade) — 표현 제한적
2. **머리·몸체 pose 명령을 박자에 맞춰 흔들기 (선택)** — 학습 불필요, 가장 빠름
3. 모션 레퍼런스 모방 새 정책 학습 — 자유도 최고, 시간 많이 듦

### 현재 구현 (`dance.py`)
- `infer_policy.py`를 import해서 사용 (Pollen repo 미수정), `alpha_stand.onnx` 사용
- `Dancer.pose(beat)`: 박자 위상 → 머리 4D / 몸체 6D 명령
  - 매 박 고개 끄덕, 2박마다 무릎 굽힘(z), 좌우 체중 이동(roll), 앞뒤 까딱(pitch), 머리 갸웃(yaw)
- 현재는 고정 BPM 메트로놈 (`--bpm`, `--amp`), `--headless N`으로 낙상 점검

### 헤드리스 테스트 (20초)
| BPM / amp | 최대 기울기 | 최저 몸통 z | 결과 |
|---|---|---|---|
| 100 / 0.4 | 6.9° | 114mm | OK |
| 120 / 0.6 | 10.1° | 114mm | OK |
| 140 / 1.0 | 16.7° | 113mm | OK |

화면으로 직접 본 건 아님 → 동작이 자연스러운지는 육안 확인 필요.

## 8. 다음 할 일
- [ ] `./run_dance.sh --bpm 120`으로 직접 보고 동작 피드백
- [ ] 오디오 파일 모드: `librosa` 설치 → 비트 분석 + 재생(ffplay 있음) + 비트 시각 동기화
- [ ] 마이크 실시간 모드: `sounddevice` + 실시간 비트 추정 (정확도/지연 한계)
- [ ] 동작 레퍼토리 확장 (곡 구간/에너지에 따라 동작 전환, 걷기/앉기 섞기)
- [ ] 필요 시 춤 정책 직접 학습 (RTX 5080, `uv run train`)
- [ ] README 수정 (키맵 표, 설치 방법, 정책별 설명, GPU 문구 "GPU 없이" 정정, 출처/라이선스)
- [ ] `.gitignore` 작성 후 깃허브 업로드
- [ ] HF `kyoungsim/microduck` 정책 라이선스 확인

## 9. 알아둘 점
- 이 대화에서 `pkill -f`로 자기 셸이 종료되는 일이 있었음 → 프로세스는 PID나 `pgrep -f "[패턴]"`으로 처리.
- 백그라운드로 띄운 프로세스는 TTY가 없어 키보드 조작 불가. 조작은 직접 터미널에서 실행.
