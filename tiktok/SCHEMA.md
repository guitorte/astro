# Schema reference

Authoritative value lists for the CSV columns. Keep this file open while coding so categories stay stable.

## 02_accounts.csv

- `handle` — `@handle` form, no URL.
- `url` — full profile URL.
- `tier` — `T1` (0–5k), `T2` (5k–50k), `T3` (50k–500k), `T4` (500k+). Use follower count on the sample_date.
- `sub_niche` — one of: `western_general`, `vedic`, `hellenistic`, `signs`, `ascendant`, `synastry`, `transits`, `houses`, `mundane`, `astro_psych`, `mixed`. If a new sub-niche appears, do not invent — open `SCHEMA.md`, add it deliberately, and update this list.
- `account_age_months` — months since the first visible post. Integer.
- `followers` — at sample_date.
- `avg_views_last_5` — arithmetic mean of views on the five most recent videos.
- `post_frequency_per_week` — videos posted in the last 28 days divided by 4. One decimal.
- `primary_format` — one of: `talking_head`, `text_only`, `b_roll`, `mixed`. Use `mixed` only when no single format dominates ≥ 60% of the last 20 videos.
- `language_register` — one of: `formal`, `casual`, `mystical`, `humorous`, `academic`. Pick the dominant register.
- `regional_signal` — Brazilian region cue in accent or cultural references: `NE`, `SE`, `S`, `N`, `CO`, or `unmarked`.
- `monetization_visible` — one of: `none`, `readings`, `digital_product`, `course`, `brand`, `affiliate`, `multiple`.
- `sample_save_rate` — save-rate of the best-performing sampled video, as a sanity check on tier-level engagement.
- `discovery_method` — `search:<term>`, `hashtag:<tag>`, or `snowball:<source_handle>`.
- `sample_date` — `YYYY-MM-DD`.
- `notes` — anything you'd want a future coder to know.

## 03_videos.csv

- `video_url` — full URL.
- `account_handle` — joins to `02_accounts.csv`.
- `sample_type` — one of: `best`, `worst`, `random_1`, `random_2`, `random_3`.
- `length_seconds` — integer.
- `views`, `likes`, `comments`, `shares`, `saves` — integers as visible. Leave blank if not visible; do not estimate.
- `save_rate` — `saves / views` to 4 decimals. Blank if saves not visible.
- `hook_type` — one of: `question`, `hot_take`, `claim`, `story`, `list`, `demo`, `pattern_interrupt`. Decided by the dominant element in the first 1.5 seconds.
- `first_15s_topic` — short free-text description of what is established in the opening 15 seconds.
- `format` — `talking_head`, `text_only`, `b_roll`, `mixed`.
- `sub_niche` — same value list as `02_accounts.csv`.
- `specificity`:
  - `generic` — applies to all signs / all charts ("os 12 signos no amor").
  - `specific` — narrows by one dimension ("ascendente em câncer no amor").
  - `hyper_specific` — narrows by two or more dimensions ("ascendente em câncer com lua em capricórnio no amor").
- `cta_type` — `none`, `follow`, `comment`, `save`, `link_bio`, `dm`.
- `has_text_overlay` — `yes` / `no`.
- `has_voiceover` — `yes` / `no` (vs. on-camera or music-only).
- `caption_question` — `yes` / `no` for whether the caption ends in a question or contains a `?`.
- `sample_date` — `YYYY-MM-DD`.
- `notes` — judgment calls, ambiguities, anything off-schema.
