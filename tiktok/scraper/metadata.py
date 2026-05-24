"""Per-video metadata extraction.

Maps yt-dlp's TikTok extractor output to the quantitative columns of
tiktok/03_videos.csv. The saves count is best-effort: yt-dlp exposes it
inconsistently. When absent, the column is left blank and `notes` is
marked, so analysis code can distinguish "zero saves" from "unknown".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp

from . import config


@dataclass
class VideoMetadata:
    video_url: str
    account_handle: str
    length_seconds: int | None
    views: int | None
    likes: int | None
    comments: int | None
    shares: int | None
    saves: int | None
    upload_date: str | None
    caption: str | None

    @property
    def save_rate(self) -> float | None:
        if self.saves is None or not self.views:
            return None
        return round(self.saves / self.views, 4)


def fetch_video(url: str, raw_dump_dir: Path | None = None) -> VideoMetadata:
    opts = {**config.YTDLP_BASE_OPTS, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if raw_dump_dir:
        raw_dump_dir.mkdir(parents=True, exist_ok=True)
        video_id = info.get("id", "unknown")
        (raw_dump_dir / f"{video_id}.json").write_text(
            json.dumps(info, default=str, indent=2)
        )

    handle = info.get("uploader_id") or info.get("uploader") or ""
    return VideoMetadata(
        video_url=info.get("webpage_url", url),
        account_handle=f"@{handle.lstrip('@')}",
        length_seconds=info.get("duration"),
        views=info.get("view_count"),
        likes=info.get("like_count"),
        comments=info.get("comment_count"),
        shares=info.get("repost_count"),
        saves=_extract_saves(info),
        upload_date=info.get("upload_date"),
        caption=info.get("description") or info.get("title"),
    )


def _extract_saves(info: dict) -> int | None:
    for key in ("save_count", "collect_count"):
        if info.get(key) is not None:
            return info[key]
    stats = info.get("stats") or {}
    if isinstance(stats, dict):
        for key in ("collectCount", "saveCount"):
            if key in stats:
                return stats[key]
    return None


def to_csv_row(
    md: VideoMetadata,
    sample_type: str,
    sample_date: str,
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a 03_videos.csv row. Qualitative fields come from `classification`
    (the classify.py output) or are left blank if not yet classified.
    """
    c = classification or {}
    return {
        "video_url": md.video_url,
        "account_handle": md.account_handle,
        "sample_type": sample_type,
        "length_seconds": md.length_seconds if md.length_seconds is not None else "",
        "views": md.views if md.views is not None else "",
        "likes": md.likes if md.likes is not None else "",
        "comments": md.comments if md.comments is not None else "",
        "shares": md.shares if md.shares is not None else "",
        "saves": md.saves if md.saves is not None else "",
        "save_rate": md.save_rate if md.save_rate is not None else "",
        "hook_type": c.get("hook_type", ""),
        "first_15s_topic": c.get("first_15s_topic", ""),
        "format": c.get("format", ""),
        "sub_niche": c.get("sub_niche", ""),
        "specificity": c.get("specificity", ""),
        "cta_type": c.get("cta_type", ""),
        "has_text_overlay": c.get("has_text_overlay", ""),
        "has_voiceover": c.get("has_voiceover", ""),
        "caption_question": "yes" if md.caption and "?" in md.caption else "no",
        "sample_date": sample_date,
        "notes": c.get("notes", "metadata_only" if not classification else ""),
    }
