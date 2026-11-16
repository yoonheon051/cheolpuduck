"""춤 동작: 박자 위상과 곡의 에너지 -> 머리(4D) / 몸체(6D) pose 명령.

MuJoCo·infer_policy 없이 import 할 수 있게 dance.py 에서 분리했다.
"""
import math

import numpy as np

HEAD_LIMIT = 1.1        # neck_pitch / head_pitch 학습 범위 ±1.1
HEAD_ROLL_LIMIT = 0.3   # head_roll 학습 범위 ±0.31


def smoothstep(lo, hi, x):
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Dancer:
    """박자 위상 beat(연속값) + 에너지 -> head_offset(4) / body_cmd(6).

    amp: 0~1 로 모든 동작 크기를 일괄 조절.
    energy: 0~1. 곡이 조용하면 고개만 까딱이고, 커질수록 좌우 흔들기 -> 무릎 바운스가 더해진다.
            energy=1, gain=1 이면 예전 메트로놈 춤과 똑같다. 어떤 값이든 진폭은 그보다 커지지 않는다.
    gain: 0~1 전체 페이드 (곡 시작 전/끝난 뒤 0).
    """

    def __init__(self, amp=0.6, max_z=0.03, max_angle=math.radians(30)):
        self.amp = amp
        self.max_z = max_z
        self.max_angle = max_angle

    @staticmethod
    def layers(energy):
        """에너지 -> (끄덕, 흔들기, 바운스) 가중치, 각각 0~1."""
        e = min(max(energy, 0.0), 1.0)
        nod = 0.4 + 0.6 * e
        sway = smoothstep(0.25, 0.6, e)
        bounce = smoothstep(0.5, 0.85, e)
        return nod, sway, bounce

    def pose(self, beat, energy=1.0, gain=1.0):
        a = self.amp * min(max(gain, 0.0), 1.0)
        nod, sway, bounce = self.layers(energy)
        s = math.sin
        bar = beat / 4.0  # 4박 = 1마디

        # 머리: 매 박 고개 끄덕(pitch), 2마디마다 좌우로 갸웃(yaw), 2박 주기 roll 스웨이
        head = np.zeros(4, dtype=np.float32)  # neck_pitch, head_pitch, head_yaw, head_roll
        head[1] = 0.5 * a * nod * s(2 * math.pi * beat)
        head[2] = 0.7 * a * sway * s(math.pi * bar)
        head[3] = 0.25 * a * sway * s(math.pi * beat)

        # 몸체: 2박에 한 번 무릎 굽혔다 펴기(z), 좌우 체중 이동(roll), 앞뒤 까딱(pitch)
        body = np.zeros(6, dtype=np.float32)  # x, y, z, roll, pitch, yaw
        body[2] = self.max_z * a * bounce * s(math.pi * beat)
        body[3] = self.max_angle * 0.6 * a * sway * s(math.pi * beat / 2)
        body[4] = self.max_angle * 0.3 * a * bounce * s(math.pi * beat)

        head = np.clip(head, -HEAD_LIMIT, HEAD_LIMIT)
        head[3] = np.clip(head[3], -HEAD_ROLL_LIMIT, HEAD_ROLL_LIMIT)
        body[2] = np.clip(body[2], -self.max_z, self.max_z)
        body[3:6] = np.clip(body[3:6], -self.max_angle, self.max_angle)
        return head, body
