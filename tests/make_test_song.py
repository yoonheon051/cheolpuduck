#!/usr/bin/env python3
"""테스트용 곡 만들기: 일정 BPM 킥+하이햇, 앞부분은 조용하고 뒷부분은 크게.

    python tests/make_test_song.py out.wav --bpm 128 --seconds 60 --quiet 20
"""
import argparse

import numpy as np
import soundfile as sf

SR = 22050


def make(bpm, seconds, quiet):
    n = int(seconds * SR)
    y = np.zeros(n)
    t = np.arange(n) / SR
    y += 0.03 * np.sin(2 * np.pi * 220 * t)  # 깔리는 음
    kn = int(0.15 * SR)
    kt = np.arange(kn) / SR
    kick = np.sin(2 * np.pi * (60 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 20)
    hn = int(0.04 * SR)
    hat = np.random.default_rng(0).standard_normal(hn) * np.exp(-np.arange(hn) / SR * 80) * 0.3
    beat = 60.0 / bpm
    for i, bt in enumerate(np.arange(0.3, seconds - 0.2, beat)):
        k = int(bt * SR)
        y[k:k + kn] += kick[:max(0, min(kn, n - k))]
        h = int((bt + beat / 2) * SR)
        y[h:h + hn] += hat[:max(0, min(hn, n - h))]
    y[:int(quiet * SR)] *= 0.12
    return (0.6 * y / np.abs(y).max()).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bpm", type=float, default=128)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--quiet", type=float, default=20, help="앞에서 몇 초를 조용하게 할지")
    a = ap.parse_args()
    sf.write(a.out, make(a.bpm, a.seconds, a.quiet), SR)
    print(f"wrote {a.out}")
