"""Pipeline configuration.

Paths resolve against the repository root. Data lives under tiktok/data/
and is gitignored — only the coded CSVs and findings are committed.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIKTOK_DIR = PROJECT_ROOT / "tiktok"

DATA_DIR = TIKTOK_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
RAW_METADATA_DIR = DATA_DIR / "raw_metadata"

ACCOUNTS_CSV = TIKTOK_DIR / "02_accounts.csv"
VIDEOS_CSV = TIKTOK_DIR / "03_videos.csv"
SCHEMA_MD = TIKTOK_DIR / "SCHEMA.md"

REQUEST_DELAY_SEC = float(os.environ.get("TIKTOK_REQUEST_DELAY", "3.0"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_LANGUAGE = "pt"
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

YTDLP_BASE_OPTS: dict = {
    "quiet": True,
    "no_warnings": True,
    "extractor_retries": 3,
    "socket_timeout": 30,
    "sleep_interval": 1,
    "max_sleep_interval": 3,
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, VIDEOS_DIR, TRANSCRIPTS_DIR, RAW_METADATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
