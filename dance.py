#!/usr/bin/env python3
"""박자에 맞춰 춤추는 microduck (MuJoCo, CPU).

서 있는 정책(standing)의 머리/몸체 pose 명령 슬롯을 박자 위상(phase)에 맞춰 흔든다.
Pollen의 microduck_rl/scripts/infer_policy.py 를 import 해서 쓰며 해당 repo는 수정하지 않는다.

사용:
    ./run_dance.sh --bpm 120              # 고정 BPM 메트로놈
    ./run_dance.sh --bpm 120 --headless 20   # 뷰어 없이 20초 돌려 넘어지는지 점검
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(ROOT, "microduck_rl")
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)  # infer_policy 의 상대 경로(scene.xml 등) 때문

import mujoco  # noqa: E402
import infer_policy as ip  # noqa: E402

POLICIES = os.path.join(ROOT, "policies")
DECIMATION = 4
TIMESTEP = 0.005
CONTROL_DT = DECIMATION * TIMESTEP  # 50 Hz


class Dancer:
    """박자 위상 phase(beat 단위, 연속값) -> head_offset(4) / body_cmd(6) 를 만든다.

    amp: 0~1 로 모든 동작 크기를 일괄 조절 (학습 범위를 넘지 않도록 아래에서 한 번 더 clip).
    """

    def __init__(self, amp=0.6):
        self.amp = amp

    def pose(self, beat):
        a = self.amp
        s = math.sin
        bar = beat / 4.0  # 4박 = 1마디

        # 머리: 매 박 고개 끄덕(pitch), 2마디마다 좌우로 갸웃(yaw), 마디마다 roll 스웨이
        head = np.zeros(4, dtype=np.float32)  # neck_pitch, head_pitch, head_yaw, head_roll
        head[1] = 0.5 * a * s(2 * math.pi * beat)
        head[2] = 0.7 * a * s(math.pi * bar)
        head[3] = 0.25 * a * s(math.pi * beat)

        # 몸체: 2박에 한 번 무릎 굽혔다 펴기(z), 좌우 체중 이동(roll), 앞뒤 까딱(pitch)
        body = np.zeros(6, dtype=np.float32)  # x, y, z, roll, pitch, yaw
        body[2] = ip.BODY_CMD_MAX_Z * a * s(math.pi * beat)
        body[3] = ip.BODY_CMD_MAX_ANGLE * 0.6 * a * s(math.pi * beat / 2)
        body[4] = ip.BODY_CMD_MAX_ANGLE * 0.3 * a * s(math.pi * beat)

        head = np.clip(head, -1.1, 1.1)
        head[3] = np.clip(head[3], -0.3, 0.3)  # head_roll 학습 범위 ±0.31
        body[2] = np.clip(body[2], -ip.BODY_CMD_MAX_Z, ip.BODY_CMD_MAX_Z)
        body[3:6] = np.clip(body[3:6], -ip.BODY_CMD_MAX_ANGLE, ip.BODY_CMD_MAX_ANGLE)
        return head, body


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


def control_step(model, data, bam_ctrl, policy, dancer, beat):
    head, body = dancer.pose(beat)
    policy.head_offset[:] = head
    policy.body_cmd[:] = body
    policy._update_command()
    policy.apply_action(policy.infer())
    for _ in range(DECIMATION):
        bam_ctrl.update()
        mujoco.mj_step(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--amp", type=float, default=0.6, help="동작 크기 0~1 (기본 0.6)")
    ap.add_argument("--headless", type=float, default=0.0,
                    help="뷰어 없이 N초 시뮬레이션 후 넘어졌는지 보고 종료")
    args = ap.parse_args()

    model, data, bam_ctrl, policy, adr = build(args)
    dancer = Dancer(args.amp)
    beats_per_step = args.bpm / 60.0 * CONTROL_DT

    if args.headless > 0:
        n = int(args.headless / CONTROL_DT)
        beat, min_z, max_tilt = 0.0, 9.0, 0.0
        for _ in range(n):
            control_step(model, data, bam_ctrl, policy, dancer, beat)
            beat += beats_per_step
            q = data.qpos[adr + 3:adr + 7]
            up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)  # 몸통 z축의 월드 z 성분 (1=직립)
            max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(up_z, -1, 1))))
            min_z = min(min_z, float(data.qpos[adr + 2]))
        print(f"{args.headless:.0f}s @ {args.bpm:.0f} BPM amp={args.amp}: "
              f"min trunk z={min_z*1000:.0f} mm, max tilt={max_tilt:.1f} deg, "
              f"final z={data.qpos[adr+2]*1000:.0f} mm -> "
              f"{'OK' if max_tilt < 45 and min_z > 0.06 else 'FELL'}")
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        beat = 0.0
        while viewer.is_running():
            t0 = time.time()
            control_step(model, data, bam_ctrl, policy, dancer, beat)
            beat += beats_per_step
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))


if __name__ == "__main__":
    main()
