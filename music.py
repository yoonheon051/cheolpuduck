"""음악 파일 분석(비트·에너지)과 재생.

MuJoCo 없이도 import 할 수 있게 분리해 둔 모듈.
- analyze(path)  : librosa 로 비트 시각과 비트별 에너지(0~1)를 뽑아 BeatMap 으로 돌려준다.
- BeatMap        : 시각 t(초) -> 연속 박자 위상 beat, 에너지 energy
- Player         : ffplay 로 음악을 재생하고, 재생 시작 기준 경과 시간을 준다.
"""
import shutil
import subprocess
import time
from dataclasses import dataclass

import numpy as np

HALF_TIME_ABOVE_BPM = 140.0  # 이보다 빠른 곡은 한 박 걸러 춤춘다 (테스트한 최대 템포 140 BPM)
ENERGY_SMOOTH_BEATS = 4      # 에너지를 몇 박 평균으로 부드럽게 할지 (1마디)


@dataclass
class BeatMap:
    beats: np.ndarray   # 비트 시각(초), 오름차순
    energy: np.ndarray  # 비트별 에너지 0~1 (beats 와 길이 같음)
    tempo: float        # 춤에 쓰는 템포(BPM). half-time 적용 후 값
    duration: float     # 곡 길이(초)
    half_time: bool = False

    def beat_at(self, t):
        """시각 t(초)의 연속 박자 위상. 첫 비트에서 0, 다음 비트에서 1 ...

        첫 비트 이전과 마지막 비트 이후는 가장 가까운 비트 간격으로 외삽한다.
        """
        b = self.beats
        if len(b) < 2:
            return t * self.tempo / 60.0
        if t <= b[0]:
            return (t - b[0]) / (b[1] - b[0])
        if t >= b[-1]:
            return (len(b) - 1) + (t - b[-1]) / (b[-1] - b[-2])
        i = int(np.searchsorted(b, t, side="right")) - 1
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def energy_at(self, t):
        if len(self.beats) == 0:
            return 0.0
        return float(np.interp(t, self.beats, self.energy))


def _smooth(x, n):
    if n <= 1 or len(x) < 2:
        return x
    pad = n // 2
    xp = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")


def beatmap_from(beats, rms, rms_times, duration, tempo, half_time_above=HALF_TIME_ABOVE_BPM):
    """비트 시각과 RMS 곡선으로 BeatMap 을 만든다 (librosa 없이 테스트 가능하게 분리)."""
    beats = np.asarray(beats, dtype=float)
    half = False
    if tempo > half_time_above and len(beats) >= 4:
        beats, tempo, half = beats[::2], tempo / 2.0, True

    edges = np.append(beats, duration)
    e = np.zeros(len(beats))
    for i in range(len(beats)):
        m = (rms_times >= edges[i]) & (rms_times < edges[i + 1])
        e[i] = rms[m].mean() if m.any() else 0.0

    if len(e):
        lo, hi = np.percentile(e, 10), np.percentile(e, 95)
        e = np.clip((e - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.ones_like(e)
        e = _smooth(e, ENERGY_SMOOTH_BEATS)
    return BeatMap(beats=beats, energy=e, tempo=float(tempo), duration=float(duration), half_time=half)


def analyze(path, half_time_above=HALF_TIME_ABOVE_BPM):
    import librosa  # 무거워서 필요할 때만 import

    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = len(y) / sr
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(frames, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    return beatmap_from(beats, rms, rms_times, duration, tempo, half_time_above)


class Player:
    """ffplay 로 음악 재생. ffplay 가 없으면 소리 없이 시계만 돈다."""

    def __init__(self, path, audio=True):
        self.proc = None
        exe = shutil.which("ffplay") if audio else None
        if audio and exe is None:
            print("[music] ffplay 를 찾지 못해 소리 없이 춤춥니다 (sudo apt install ffmpeg).")
        if exe:
            self.proc = subprocess.Popen(
                [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                stdin=subprocess.DEVNULL)
        self.t0 = time.monotonic()

    def time(self):
        return time.monotonic() - self.t0

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
