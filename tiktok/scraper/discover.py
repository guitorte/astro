"""Account discovery via TikTok hashtag pages.

Uses yt-dlp's flat-playlist extractor against /tag/<tag> URLs. TikTok
increasingly gates these pages behind a login wall — discovery may
return zero entries on some networks. Fallback: pass a manual list of
handles via --handles to the pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import yt_dlp

from . import config


@dataclass(frozen=True)
class DiscoveryHit:
    handle: str
    discovery_method: str
    sample_video_url: str


def _flat_playlist(url: str) -> list[dict]:
    opts = {**config.YTDLP_BASE_OPTS, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("entries", []) if info else []


def discover_hashtag(tag: str) -> list[DiscoveryHit]:
    url = f"https://www.tiktok.com/tag/{tag.lstrip('#')}"
    seen: dict[str, DiscoveryHit] = {}
    for e in _flat_playlist(url):
        handle = e.get("uploader") or e.get("uploader_id") or ""
        video_url = e.get("url") or e.get("webpage_url") or ""
        if not handle or handle in seen:
            continue
        seen[handle] = DiscoveryHit(
            handle=f"@{handle.lstrip('@')}",
            discovery_method=f"hashtag:{tag}",
            sample_video_url=video_url,
        )
    return list(seen.values())


def discover_many(hashtags: Iterable[str]) -> list[DiscoveryHit]:
    out: dict[str, DiscoveryHit] = {}
    for tag in hashtags:
        for hit in discover_hashtag(tag):
            out.setdefault(hit.handle, hit)
        time.sleep(config.REQUEST_DELAY_SEC)
    return list(out.values())


def list_account_video_urls(handle: str, max_videos: int = 20) -> list[str]:
    """Return URLs of the N most recent videos on an account page (flat)."""
    url = f"https://www.tiktok.com/@{handle.lstrip('@')}"
    opts = {
        **config.YTDLP_BASE_OPTS,
        "extract_flat": True,
        "playlist_items": f"1:{max_videos}",
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries", []) if info else []
    return [e.get("url") or e.get("webpage_url") for e in entries if e and (e.get("url") or e.get("webpage_url"))]
