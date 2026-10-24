import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.audio_encoder import AudioEncoder  # noqa: E402
from preprocessing.audio_extractor import extract_audio  # noqa: E402
from preprocessing.audio_features import extract_log_mel  # noqa: E402
from preprocessing.audio_segmenter import segment_audio_features  # noqa: E402

SR = 16000
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def make_wav(path, seconds, kind="tone"):
    n = int(seconds * SR)
    if kind == "silent":
        wave = np.zeros(n, dtype=np.float32)
    else:
        t = np.arange(n) / SR
        wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), wave, SR)
    return path


def make_video(path, seconds, with_audio=True):
    command = [FFMPEG, "-y", "-v", "error",
               "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=64x64:rate=10"]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    command += ["-c:v", "mpeg4"]
    if with_audio:
        command += ["-c:a", "aac"]
    command += [str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return path


# ---------- Extraction: missing / corrupt / variable duration ----------

def test_video_without_audio_raises_clear_error(tmp_path):
    video = make_video(tmp_path / "no_audio.mp4", 2, with_audio=False)
    with pytest.raises(RuntimeError):
        extract_audio(video, tmp_path / "out.wav")


def test_missing_video_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "does_not_exist.mp4", tmp_path / "out.wav")


def test_corrupt_video_file(tmp_path):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a real video file" * 100)
    with pytest.raises(RuntimeError):
        extract_audio(bad, tmp_path / "out.wav")


@pytest.mark.parametrize("seconds", [1, 3, 7])
def test_variable_duration_extraction(tmp_path, seconds):
    video = make_video(tmp_path / f"v{seconds}.mp4", seconds)
    out = extract_audio(video, tmp_path / f"a{seconds}.wav")
    audio, sr = sf.read(str(out))
    assert sr == SR
    assert audio.ndim == 1  # mono
    assert abs(len(audio) / sr - seconds) < 0.5


# ---------- Log-Mel features ----------

def test_logmel_shape_and_no_nan(tmp_path):
    wav = make_wav(tmp_path / "tone.wav", 5)
    mel = extract_log_mel(wav, tmp_path / "tone.npy")
    assert mel.shape[0] == 64
    assert np.isfinite(mel).all()


def test_silent_audio_has_no_nan(tmp_path):
    wav = make_wav(tmp_path / "silent.wav", 5, kind="silent")
    mel = extract_log_mel(wav, tmp_path / "silent.npy")
    assert np.isfinite(mel).all()


def test_empty_audio_file_fails(tmp_path):
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(0, dtype=np.float32), SR)
    with pytest.raises(Exception):
        extract_log_mel(empty, tmp_path / "empty.npy")


def test_missing_audio_file_for_features(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_log_mel(tmp_path / "nope.wav", tmp_path / "nope.npy")


# ---------- Windowing ----------

def test_segmenter_window_count_and_shapes(tmp_path):
    wav = make_wav(tmp_path / "long.wav", 12)
    extract_log_mel(wav, tmp_path / "long.npy")
    count = segment_audio_features(tmp_path / "long.npy", tmp_path / "windows")
    assert count == 3  # 12 seconds = 5 + 5 + 2
    files = sorted((tmp_path / "windows").glob("window_*.npy"))
    assert len(files) == 3
    for f in files:
        assert np.load(f).shape[0] == 64


def test_segmenter_short_audio_gives_one_window(tmp_path):
    wav = make_wav(tmp_path / "short.wav", 2)
    extract_log_mel(wav, tmp_path / "short.npy")
    count = segment_audio_features(tmp_path / "short.npy", tmp_path / "windows")
    assert count == 1


# ---------- Audio encoder ----------

def test_encoder_output_shapes():
    model = AudioEncoder().eval()
    emb, logit = model(torch.randn(4, 64, 313))
    assert emb.shape == (4, 256)
    assert logit.shape == (4,)


@pytest.mark.parametrize("frames", [10, 100, 313, 1000])
def test_encoder_variable_length_input(frames):
    model = AudioEncoder().eval()
    emb, logit = model(torch.randn(2, 64, frames))
    assert emb.shape == (2, 256)
    assert torch.isfinite(logit).all()


def test_encoder_silent_input_no_nan():
    model = AudioEncoder().eval()
    emb, logit = model(torch.zeros(2, 64, 313))
    assert torch.isfinite(emb).all()
    assert torch.isfinite(logit).all()


def test_encoder_single_sample_batch():
    model = AudioEncoder().eval()
    emb, logit = model(torch.randn(1, 64, 313))
    assert emb.shape == (1, 256)