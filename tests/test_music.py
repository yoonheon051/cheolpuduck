"""music.py / choreo.py 테스트 (MuJoCo 불필요).

    .venv/bin/python -m pytest tests/        # 또는
    .venv/bin/python tests/test_music.py
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import music  # noqa: E402
from choreo import Dancer  # noqa: E402
from make_test_song import SR, make  # noqa: E402


def test_beat_at_interpolates_and_extrapolates():
    bm = music.BeatMap(beats=np.array([1.0, 1.5, 2.0, 2.5]), energy_t=np.array([0.0, 3.0]),
                       energy=np.ones(2), tempo=120, duration=3)
    assert bm.beat_at(1.0) == 0
    assert math.isclose(bm.beat_at(1.25), 0.5)
    assert math.isclose(bm.beat_at(2.5), 3)
    assert math.isclose(bm.beat_at(0.5), -1)   # 첫 비트 이전 외삽
    assert math.isclose(bm.beat_at(3.0), 4)    # 마지막 이후 외삽


def test_beatmap_half_time_for_fast_songs():
    beats = np.arange(0, 10, 60 / 170)
    rms_t = np.linspace(0, 10, 500)
    bm = music.beatmap_from(beats, np.ones(500), rms_t, 10, 170)
    assert bm.half_time and math.isclose(bm.tempo, 85)
    assert len(bm.beats) == len(beats[::2])
    bm2 = music.beatmap_from(beats, np.ones(500), rms_t, 10, 120)
    assert not bm2.half_time


def test_energy_follows_loudness():
    beats = np.arange(0, 20, 0.5)
    rms_t = np.linspace(0, 20, 2000)
    rms = np.where(rms_t < 10, 0.05, 0.5)
    bm = music.beatmap_from(beats, rms, rms_t, 20, 120)
    assert bm.energy_at(3) < 0.2 and bm.energy_at(17) > 0.8
    assert bm.energy.min() >= 0 and bm.energy.max() <= 1


def test_uniformly_loud_song_stays_energetic():
    # 처음부터 끝까지 큰 곡: 1~2dB 흔들림을 0~1 로 부풀리지 않아야 한다
    rms_t = np.linspace(0, 30, 3000)
    rms = 0.5 * (1 + 0.1 * np.sin(rms_t * 3))
    bm = music.beatmap_from(np.arange(0, 30, 0.5), rms, rms_t, 30, 120)
    assert bm.energy[100:-100].min() > 0.8


def test_dancer_never_exceeds_metronome_amplitude():
    d = Dancer(amp=1.0)
    ref = np.zeros(10)
    for beat in np.linspace(0, 16, 801):
        h, b = d.pose(beat, 1.0, 1.0)
        ref = np.maximum(ref, np.abs(np.concatenate([h, b])))
    for energy in (0, 0.3, 0.6, 0.9, 1.0):
        for gain in (0, 0.5, 1):
            for beat in np.linspace(0, 16, 401):
                h, b = d.pose(beat, energy, gain)
                assert np.all(np.abs(np.concatenate([h, b])) <= ref + 1e-6)


def test_dancer_layers_grow_with_energy():
    quiet = Dancer.layers(0.0)
    loud = Dancer.layers(1.0)
    assert quiet == (0.4, 0.0, 0.0)
    assert loud == (1.0, 1.0, 1.0)
    d = Dancer(amp=1.0)
    _, body_q = d.pose(0.5, 0.0)
    _, body_l = d.pose(0.5, 1.0)
    assert body_q[2] == 0 and abs(body_l[2]) > 0  # 조용할 땐 무릎 바운스 없음
    h0, b0 = d.pose(0.7, 1.0, gain=0.0)
    assert not h0.any() and not b0.any()          # gain 0 이면 정지 자세


def test_analyze_test_song():
    import soundfile as sf
    y = make(120, 30, quiet=15)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "song.wav")
        sf.write(path, y, SR)
        bm = music.analyze(path)
    assert 110 <= bm.tempo <= 130, bm.tempo
    gaps = np.diff(bm.beats)
    assert abs(np.median(gaps) - 0.5) < 0.03, np.median(gaps)
    assert bm.energy_at(5) < bm.energy_at(25), (bm.energy_at(5), bm.energy_at(25))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
