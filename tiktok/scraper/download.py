"""Download TikTok videos as mp4 via yt-dlp.

Files land under tiktok/data/videos/ named by TikTok video id. The file
exists check is the resume mechanism — re-running the pipeline does not
re-download.
"""
from __future__ import annotations

from pathlib import Path

import yt_dlp

from . import config


def download_video(url: str, out_dir: Path | None = None) -> Path | None:
    out_dir = out_dir or config.VIDEOS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")

    opts = {
        **config.YTDLP_BASE_OPTS,
        "outtmpl": template,
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "overwrites": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_id = info.get("id")
    if not video_id:
        return None
    path = out_dir / f"{video_id}.mp4"
    return path if path.exists() else None
