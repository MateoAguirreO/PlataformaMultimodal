"""Extraccion de features desde media CRUDA (mp4 + wav de la entrevista),
modo DEMO: Demucs (GPU) + eGeMAPS para audio, frames a 1fps + py-feat (GPU)
para rostro.

Validado contra pid 61 del dataset de entrenamiento en esta misma sesion:
  - Audio (Demucs vocals + segmentacion 5s/50% + eGeMAPSv02): r=1.0 vs.
    features_ansiedad_egemaps.csv, ~5% diferencia relativa promedio.
  - Rostro a 1fps (vs. frame_skip=3 / ~10fps de entrenamiento): r=0.9977 vs.
    dataset_au_features.csv, 2.9% diferencia relativa promedio en las AUs.

Esto NO es identico a la extraccion de entrenamiento (esa es frame_skip=3,
~10fps, ~30 min/video en GPU) -- es la aproximacion rapida para demo (~3.5 min
video + ~20s audio en GPU). Para subir la fidelidad, bajar VIDEO_FPS_DEMO
hacia el pipeline de entrenamiento real (ver pipeline_multimodal_xai/ en el
repo de tesis) a costa de mas tiempo.

Requiere: demucs, ffmpeg en PATH, opensmile, py-feat, torch con CUDA (si no
hay GPU, cae a CPU automaticamente pero MUY lento -- ver notas de la sesion:
~30 min/video en GPU vs 1.5-2h+ en CPU a la fidelidad de entrenamiento).
"""
import re
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import opensmile
import pandas as pd

SR_TARGET = 16000
PRE_EMPHASIS = 0.97
VAD_TOP_DB = 20
SEG_DURATION = 5
OVERLAP = 0.50
MIN_SEG_FRAC = 0.1
VIDEO_FPS_DEMO = 1  # 1 frame/seg -- ~9x mas rapido que el frame_skip=3 de entrenamiento

_smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                          feature_level=opensmile.FeatureLevel.Functionals)
_detector = None


def _segment_audio(audio, sr, segment_duration=SEG_DURATION, overlap=OVERLAP,
                    min_seg_frac=MIN_SEG_FRAC):
    """Identica a segment_audio() de pipeline_feat_depresion.py (repo de tesis)."""
    n_samples = int(segment_duration * sr)
    step = max(int(n_samples * (1 - overlap)), 1)
    min_length = int(n_samples * min_seg_frac)
    if len(audio) == 0:
        return []
    if len(audio) <= n_samples:
        return [np.pad(audio, (0, n_samples - len(audio)))]
    segments, start = [], 0
    while start < len(audio):
        seg = audio[start:start + n_samples]
        if len(seg) < min_length:
            if not segments:
                segments.append(np.pad(seg, (0, n_samples - len(seg))))
            break
        if len(seg) < n_samples:
            seg = np.pad(seg, (0, n_samples - len(seg)))
        segments.append(seg)
        start += step
    return segments


def denoise_audio(wav_path: Path, workdir: Path) -> Path:
    """Demucs (GPU) two-stems vocals + downmix a mono. ~18s en GPU para una
    entrevista de 5-6 min (vs ~2.4 min en CPU)."""
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-d", "cuda",
         "-o", str(workdir), str(wav_path)],
        check=True, capture_output=True, text=True,
    )
    vocals = workdir / "htdemucs" / wav_path.stem / "vocals.wav"
    mono = workdir / f"{wav_path.stem}_vocals_mono.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(vocals), "-ac", "1", str(mono)],
        check=True, capture_output=True, text=True,
    )
    return mono


def audio_to_segments(wav_path: Path) -> list[dict]:
    """wav denoised -> lista de dicts (1 por segmento) con las 88 features
    eGeMAPSv02 crudas (nombres tal cual los da opensmile)."""
    y, sr = librosa.load(str(wav_path), sr=SR_TARGET, mono=True)
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])
    y_trimmed, _ = librosa.effects.trim(y, top_db=VAD_TOP_DB)
    y_proc = y_trimmed if len(y_trimmed) > sr * 0.1 else y
    segments = _segment_audio(y_proc, sr)
    if not segments:
        raise ValueError("audio_to_segments: no se pudo generar ningun segmento (audio vacio?)")
    return [_smile.process_signal(seg, sr).reset_index(drop=True).iloc[0].to_dict()
            for seg in segments]


def _extract_frames(mp4_path: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", f"fps={fps}", "-q:v", "2",
         str(out_dir / "frame_%06d.jpg")],
        check=True, capture_output=True, text=True,
    )
    return sorted(out_dir.glob("*.jpg"))


def _get_detector():
    global _detector
    if _detector is None:
        import torch
        from feat import Detectorv1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _detector = Detectorv1(face_model="retinaface", au_model="xgb",
                                emotion_model="resmasknet", device=device)
    return _detector


def video_to_features(mp4_path: Path, workdir: Path, fps: float = VIDEO_FPS_DEMO) -> dict:
    """mp4 -> dict con mean/std de AU/emocion/pose (nombres tal cual
    dataset_au_features.csv: AU01_mean, AU01_std, ..., anger_mean, ...,
    Pitch_mean, ...). ~3.5 min a 1fps en GPU para una entrevista de 5-6 min."""
    frames = _extract_frames(mp4_path, workdir / "frames", fps)
    if not frames:
        raise ValueError("video_to_features: ffmpeg no genero ningun frame")

    detector = _get_detector()
    fex = detector.detect([str(f) for f in frames], data_type="image",
                          batch_size=32, face_detection_threshold=0.9, progress_bar=False)
    fex_df = pd.DataFrame(fex)

    au_cols = sorted(c for c in fex_df.columns if re.fullmatch(r"AU\d\d", c))
    emotion_cols = [c for c in fex_df.columns if c.lower() in
                    ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")]
    pose_cols = [c for c in fex_df.columns if c in ("Pitch", "Roll", "Yaw", "X", "Y", "Z")]

    valid = fex_df[fex_df[au_cols].notna().any(axis=1)].reset_index(drop=True)
    if valid.empty:
        raise ValueError("video_to_features: py-feat no detecto rostro en ningun frame")

    row = {}
    for col in au_cols + emotion_cols + pose_cols:
        vals = pd.to_numeric(valid[col], errors="coerce")
        row[f"{col}_mean"] = float(np.nanmean(vals)) if vals.notna().any() else np.nan
        row[f"{col}_std"] = float(np.nanstd(vals)) if vals.notna().any() else np.nan
    return row
