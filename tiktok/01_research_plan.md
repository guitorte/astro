# Research plan

## Sample frame

**Population.** TikTok accounts whose primary topic is astrology — broadly defined to include Western, Vedic, Hellenistic, sign-based, tarot crossover, mundane/predictive, and astro-psychology — and whose primary content language is Brazilian Portuguese.

**Inclusion criteria.**
- Posted at least 10 videos in the last 6 months (active).
- Primary content language is PT-BR, not occasional dubs or subtitled imports.
- Public account.

**Exclusion criteria.**
- Pure tarot or oracle work with no astrological component.
- Inactive accounts (fewer than 10 posts in the last 6 months).
- Personal accounts that occasionally post astrology among unrelated content.

**Discovery method.**
- TikTok search for: "astrologia", "mapa astral", "signos", "ascendente", "casa astrológica", "trânsitos", "astrologia védica", "sinastria".
- Hashtag pass on `#astrologia`, `#mapaastral`, `#signos`, `#ascendente`, `#astrologiabrasileira`, recording every account whose video appears in the top results for at least two of the searches/hashtags.
- Snowball: from comment sections of identified accounts, log every commenter whose own account meets the inclusion criteria.
- Stop discovery when three consecutive search or hashtag passes yield no new accounts.

**Target N.** At least 8 accounts in each tier, ≥ 40 in total:
- T1: 0 – 5k followers
- T2: 5k – 50k
- T3: 50k – 500k
- T4: 500k+

The tier split matters because the lessons that transfer to a new account come mostly from T1 and T2 — accounts that grew recently without leveraging an existing audience.

## Per-account coding

One row per account in `02_accounts.csv`. The schema columns are defined in the CSV header. Apply consistently; if a value doesn't fit any existing category, leave a `?` and a note rather than inventing a category mid-coding.

## Per-video coding

Sample five videos per account in `03_videos.csv`:
- The single **best-performing** video by views.
- The **worst-performing** of their last 20 videos.
- Three **random** videos from the last 20.

This deliberately mixes survivorship-biased samples (the best) with baseline samples (worst and random) so the comparison has meaning. Coding only top videos tells you what success looks like in retrospect; it does not tell you what works on average.

## Analysis questions (the only ones we will answer)

Q1. Which sub-niches show the highest save-rate (saves / views), controlling for tier?
Q2. Which formats correlate with above-median save-rate?
Q3. What is the median video length of above-median save-rate videos, by tier?
Q4. Which hook types appear in the top decile of save-rate?
Q5. What questions and complaints recur in comments across multiple accounts? (Demand signal — use the `astro-comment-analyzer` agent.)
Q6. Are there sub-niches with high comment-side demand and few accounts supplying them? (Gap signal.)

Refuse the temptation to answer questions the data was not collected to support.

## Schedule

| Week | Milestone |
|---|---|
| 1 | Commit `00_priors.md`. Build sample frame. Run discovery passes. |
| 2 | Code accounts.csv (≥ 40 rows). |
| 3 | Code videos.csv (≥ 200 rows). |
| 4 | Comment harvest. Analysis. Red-team pass on draft positioning. |

## Kill criteria

Stop or pivot if any of the following holds:

- After 2 weeks of discovery, fewer than 25 accounts meet inclusion criteria. The niche may be too sparse on TikTok and the platform question is moot — evaluate Instagram or YouTube first.
- Top-quartile save-rates across all sub-niches sit below 1%. This signals platform-level low intent for the topic; format won't save you.
- After launching, four consecutive weeks with median video completion-rate below 25% on your own content. The bottleneck is execution, not the niche. Stop posting and re-evaluate.

## What this plan deliberately does NOT do

- It does not assume the assistant can scrape TikTok directly. Collection is manual; agents help with coding, comment analysis, and red-teaming.
- It does not measure vibes. Every claim in `04_findings.md` must cite at least one CSV column.
- It does not lock positioning before week 4. The temptation to commit early is the failure mode this plan exists to prevent.
