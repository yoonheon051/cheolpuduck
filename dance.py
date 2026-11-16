#!/usr/bin/env python3
"""박자에 맞춰 춤추는 microduck (MuJoCo, CPU).

서 있는 정책(standing)의 머리/몸체 pose 명령 슬롯을 박자 위상(phase)에 맞춰 흔든다.
Pollen의 microduck_rl/scripts/infer_policy.py 를 import 해서 쓰며 해당 repo는 수정하지 않는다.

사용:
    ./run_dance.sh --bpm 120                  # 고정 BPM 메트로놈
    ./run_dance.sh --bpm 120 --headless 20    # 뷰어 없이 20초 돌려 넘어지는지 점검
    ./run_dance.sh --music song.mp3           # 음악을 틀고 비트·세기에 맞춰 춤
    ./run_dance.sh --music song.mp3 --headless 999   # 곡 전체를 뷰어·소리 없이 돌려 낙상 점검
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()  # 아래 chdir 전에 저장 (--music 상대 경로용)
RL = os.path.join(ROOT, "microduck_rl")
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)  # infer_policy 의 상대 경로(scene.xml 등) 때문

import mujoco  # noqa: E402
import infer_policy as ip  # noqa: E402

sys.path.insert(0, ROOT)
from choreo import Dancer  # noqa: E402

POLICIES = os.path.join(ROOT, "policies")
DECIMATION = 4
TIMESTEP = 0.005
CONTROL_DT = DECIMATION * TIMESTEP  # 50 Hz


def build(args):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
        ip.MICRODUCK_XML, bam_model, TIMESTEP, 0.1, ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=None,
        standing_onnx_path=os.path.join(POLICIES, "alpha_stand.onnx"),
        new_cmd_obs=True,
    )
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    data.qpos[adr + 2] = 0.125
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    for i, qi in enumerate(policy.joint_qpos_indices):
        data.qpos[qi] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    return model, data, bam_ctrl, policy, adr


def control_step(model, data, bam_ctrl, policy, dancer, beat, energy=1.0, gain=1.0):
    head, body = dancer.pose(beat, energy, gain)
    policy.head_offset[:] = head
    policy.body_cmd[:] = body
    policy._update_command()
    policy.apply_action(policy.infer())
    for _ in range(DECIMATION):
        bam_ctrl.update()
        mujoco.mj_step(model, data)


def make_cue(args):
    """시각 t(초) -> (beat, energy, gain) 함수와 곡 길이(메트로놈이면 None)."""
    if not args.music:
        return (lambda t: (t * args.bpm / 60.0, 1.0, 1.0)), None

    import music
    print(f"[music] 분석 중: {args.music}")
    bm = music.analyze(args.music)
    print(f"[music] {bm.duration:.0f}s, {bm.tempo:.0f} BPM{' (half-time)' if bm.half_time else ''}, "
          f"비트 {len(bm.beats)}개")
    if len(bm.beats) < 2:
        sys.exit("[music] 비트를 찾지 못했습니다. 리듬이 분명한 곡으로 시도하거나 --bpm 을 쓰세요.")
    first = float(bm.beats[0])

    def cue(t):
        intro = min(max((t - (first - 1.0)) / 1.0, 0.0), 1.0)  # 첫 비트까지 1초 동안 페이드인
        outro = min(max((bm.duration - t) / 2.0, 0.0), 1.0)    # 끝나기 2초 전부터 페이드아웃
        return bm.beat_at(t), bm.energy_at(t), min(intro, outro)

    return cue, bm.duration


def run_headless(args, model, data, bam_ctrl, policy, adr, dancer, cue, duration):
    secs = args.headless if duration is None else min(args.headless, duration + 1.0)
    min_z, max_tilt = 9.0, 0.0
    for k in range(int(secs / CONTROL_DT)):
        control_step(model, data, bam_ctrl, policy, dancer, *cue(k * CONTROL_DT))
        q = data.qpos[adr + 3:adr + 7]
        up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)  # 몸통 z축의 월드 z 성분 (1=직립)
        max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(up_z, -1, 1))))
        min_z = min(min_z, float(data.qpos[adr + 2]))
    what = os.path.basename(args.music) if args.music else f"{args.bpm:.0f} BPM"
    print(f"{secs:.0f}s @ {what} amp={args.amp}: "
          f"min trunk z={min_z*1000:.0f} mm, max tilt={max_tilt:.1f} deg, "
          f"final z={data.qpos[adr+2]*1000:.0f} mm -> "
          f"{'OK' if max_tilt < 45 and min_z > 0.06 else 'FELL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=120.0, help="메트로놈 템포 (--music 없을 때)")
    ap.add_argument("--amp", type=float, default=0.6, help="동작 크기 0~1 (기본 0.6)")
    ap.add_argument("--music", help="음악 파일 (mp3/wav/ogg/flac). 비트와 곡의 세기에 맞춰 춤춘다")
    ap.add_argument("--latency", type=float, default=0.1,
                    help="소리가 동작보다 늦게 들리면 키우는 값(초, 기본 0.1)")
    ap.add_argument("--mute", action="store_true", help="--music 을 소리 없이 (동작만)")
    ap.add_argument("--headless", type=float, default=0.0,
                    help="뷰어 없이 N초 시뮬레이션 후 넘어졌는지 보고 종료 (--music 이면 최대 곡 길이)")
    args = ap.parse_args()
    if args.music:
        args.music = os.path.abspath(os.path.join(CWD, args.music))
        if not os.path.isfile(args.music):
            sys.exit(f"음악 파일이 없습니다: {args.music}")

    cue, duration = make_cue(args)
    model, data, bam_ctrl, policy, adr = build(args)
    dancer = Dancer(args.amp, ip.BODY_CMD_MAX_Z, ip.BODY_CMD_MAX_ANGLE)

    if args.headless > 0:
        run_headless(args, model, data, bam_ctrl, policy, adr, dancer, cue, duration)
        return

    import mujoco.viewer
    player = None
    if args.music:
        import music
        player = music.Player(args.music, audio=not args.mute)
    t_start = time.monotonic()
    ended = False
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running():
                t0 = time.monotonic()
                t = (player.time() - args.latency) if player else (t0 - t_start)
                control_step(model, data, bam_ctrl, policy, dancer, *cue(t))
                if duration is not None and not ended and t > duration:
                    ended = True
                    print("[music] 곡이 끝났습니다. 창을 닫으면 종료합니다.")
                viewer.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - t0)))
    finally:
        if player:
            player.stop()


if __name__ == "__main__":
    main()
