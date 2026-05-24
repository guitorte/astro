"""Transcribe video audio to text via faster-whisper.

Model size and device come from env (WHISPER_MODEL, WHISPER_DEVICE).
'medium' is the default — good PT-BR quality, reasonable speed on CPU
and fast on GPU. Use 'large-v3' if accuracy on heavy regional accents
matters and you have GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        device = config.WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _MODEL = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type)
    return _MODEL


def transcribe(video_path: Path, out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or config.TRANSCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.txt"
    if out_path.exists():
        return out_path

    model = _get_model()
    segments, _info = model.transcribe(
        str(video_path),
        language=config.WHISPER_LANGUAGE,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    out_path.write_text(text, encoding="utf-8")
    return out_path
