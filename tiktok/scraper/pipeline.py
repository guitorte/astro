"""End-to-end orchestrator and CLI.

Stages:
  discover   hashtags  -> tiktok/data/discovery.json
  metadata   handles   -> 03_videos.csv (quantitative cols only)
  sample     videos    -> select best / worst / 3 random per account
  download   videos    -> tiktok/data/videos/*.mp4
  transcribe videos    -> tiktok/data/transcripts/*.txt
  classify   videos    -> 03_videos.csv (qualitative cols filled)
  pilot                -> run all stages on N accounts end-to-end

Run `python -m tiktok.scraper.pipeline --help` for usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from . import config, discover, download, metadata, transcribe


VIDEO_CSV_FIELDS = [
    "video_url", "account_handle", "sample_type", "length_seconds",
    "views", "likes", "comments", "shares", "saves", "save_rate",
    "hook_type", "first_15s_topic", "format", "sub_niche", "specificity",
    "cta_type", "has_text_overlay", "has_voiceover", "caption_question",
    "sample_date", "notes",
]


def _append_video_row(row: dict) -> None:
    needs_header = not config.VIDEOS_CSV.exists() or config.VIDEOS_CSV.stat().st_size == 0
    with config.VIDEOS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_CSV_FIELDS)
        if needs_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in VIDEO_CSV_FIELDS})


def cmd_discover(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    hits = discover.discover_many(args.hashtags)
    out = config.DATA_DIR / "discovery.json"
    out.write_text(json.dumps([asdict(h) for h in hits], indent=2, ensure_ascii=False))
    print(f"Discovered {len(hits)} accounts -> {out}")


def cmd_metadata(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    sample_date = date.today().isoformat()
    for handle in args.handles:
        urls = discover.list_account_video_urls(handle, max_videos=args.max_videos)
        print(f"{handle}: {len(urls)} videos")
        for url in urls:
            try:
                md = metadata.fetch_video(url, raw_dump_dir=config.RAW_METADATA_DIR)
                row = metadata.to_csv_row(md, sample_type="random_1", sample_date=sample_date)
                _append_video_row(row)
            except Exception as e:
                print(f"  ! {url}: {e}")
            time.sleep(config.REQUEST_DELAY_SEC)


def cmd_sample(args: argparse.Namespace) -> None:
    """Re-tag sample_type in 03_videos.csv: per-account pick best/worst/3 random."""
    rows = list(_read_videos_csv())
    by_account: dict[str, list[dict]] = {}
    for r in rows:
        by_account.setdefault(r["account_handle"], []).append(r)

    for handle, items in by_account.items():
        if len(items) < 5:
            print(f"{handle}: only {len(items)} videos, skipping sampling")
            continue
        items_with_views = [i for i in items if str(i.get("views", "")).strip().isdigit()]
        if not items_with_views:
            continue
        items_with_views.sort(key=lambda x: int(x["views"]))
        items_with_views[-1]["sample_type"] = "best"
        items_with_views[0]["sample_type"] = "worst"
        middle = [i for i in items_with_views[1:-1]]
        random.shuffle(middle)
        for i, item in enumerate(middle[:3]):
            item["sample_type"] = f"random_{i+1}"

    _rewrite_videos_csv(rows)
    print(f"Re-tagged sample_type for {len(by_account)} accounts.")


def cmd_download(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    rows = list(_read_videos_csv())
    sampled = [r for r in rows if r.get("sample_type") in {"best", "worst", "random_1", "random_2", "random_3"}]
    print(f"Downloading {len(sampled)} sampled videos.")
    for r in sampled:
        try:
            path = download.download_video(r["video_url"])
            print(f"  {r['video_url']} -> {path}")
        except Exception as e:
            print(f"  ! {r['video_url']}: {e}")
        time.sleep(config.REQUEST_DELAY_SEC)


def cmd_transcribe(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    for video_path in sorted(config.VIDEOS_DIR.glob("*.mp4")):
        try:
            out = transcribe.transcribe(video_path)
            print(f"  {video_path.name} -> {out.name}")
        except Exception as e:
            print(f"  ! {video_path.name}: {e}")


def cmd_classify(args: argparse.Namespace) -> None:
    from . import classify as classifier
    rows = list(_read_videos_csv())
    sampled = [r for r in rows if r.get("sample_type") in {"best", "worst", "random_1", "random_2", "random_3"}]
    for r in sampled:
        if r.get("hook_type"):
            continue
        video_id = r["video_url"].rstrip("/").split("/")[-1]
        transcript_path = config.TRANSCRIPTS_DIR / f"{video_id}.txt"
        if not transcript_path.exists():
            print(f"  ! no transcript for {video_id}, skipping")
            continue
        transcript = transcript_path.read_text(encoding="utf-8")
        try:
            result = classifier.classify(classifier.ClassifyInput(
                transcript=transcript,
                caption=None,
                length_seconds=int(r["length_seconds"]) if r.get("length_seconds") else None,
                views=int(r["views"]) if r.get("views", "").isdigit() else None,
                likes=int(r["likes"]) if r.get("likes", "").isdigit() else None,
            ))
            r.update({k: v for k, v in result.items() if k in VIDEO_CSV_FIELDS})
            print(f"  classified {video_id}: {result['hook_type']} / {result['sub_niche']}")
        except Exception as e:
            print(f"  ! {video_id}: {e}")
    _rewrite_videos_csv(rows)


def cmd_pilot(args: argparse.Namespace) -> None:
    """End-to-end smoke test on N accounts."""
    print(f"--- PILOT ({args.n} accounts, hashtags={args.hashtags}) ---")
    cmd_discover(argparse.Namespace(hashtags=args.hashtags))

    discovery_path = config.DATA_DIR / "discovery.json"
    hits = json.loads(discovery_path.read_text())[: args.n]
    handles = [h["handle"] for h in hits]
    print(f"Piloting on handles: {handles}")

    cmd_metadata(argparse.Namespace(handles=handles, max_videos=args.max_videos))
    cmd_sample(argparse.Namespace())
    cmd_download(argparse.Namespace())
    cmd_transcribe(argparse.Namespace())
    if args.classify:
        cmd_classify(argparse.Namespace())
    print("--- PILOT DONE ---")


def _read_videos_csv() -> list[dict]:
    if not config.VIDEOS_CSV.exists():
        return []
    with config.VIDEOS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rewrite_videos_csv(rows: list[dict]) -> None:
    with config.VIDEOS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in VIDEO_CSV_FIELDS})


def main() -> None:
    p = argparse.ArgumentParser(prog="tiktok-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("discover", help="sweep hashtags -> discovery.json")
    pd.add_argument("--hashtags", nargs="+", required=True)
    pd.set_defaults(func=cmd_discover)

    pm = sub.add_parser("metadata", help="fetch per-video metadata for given handles")
    pm.add_argument("--handles", nargs="+", required=True)
    pm.add_argument("--max-videos", type=int, default=20)
    pm.set_defaults(func=cmd_metadata)

    ps = sub.add_parser("sample", help="tag sample_type (best/worst/random_*) in CSV")
    ps.set_defaults(func=cmd_sample)

    pdn = sub.add_parser("download", help="download sampled videos")
    pdn.set_defaults(func=cmd_download)

    pt = sub.add_parser("transcribe", help="transcribe downloaded videos with faster-whisper")
    pt.set_defaults(func=cmd_transcribe)

    pc = sub.add_parser("classify", help="classify videos via Claude API")
    pc.set_defaults(func=cmd_classify)

    pp = sub.add_parser("pilot", help="run all stages on N accounts as a smoke test")
    pp.add_argument("--hashtags", nargs="+", required=True)
    pp.add_argument("--n", type=int, default=5)
    pp.add_argument("--max-videos", type=int, default=20)
    pp.add_argument("--classify", action="store_true")
    pp.set_defaults(func=cmd_pilot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
