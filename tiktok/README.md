# TikTok Astrology — Brazilian Portuguese (side project)

A research-driven plan for launching a Brazilian Portuguese TikTok account on astrology. The structure here is designed to fight confirmation bias and produce evidence that is specific, measurable, and falsifiable.

## Working language

Research and planning artifacts in this folder are written in US English to keep iteration with the assistant fast and to match the maintainer's working defaults. Audience-facing artifacts (hooks, scripts, captions, on-screen text) are written in Brazilian Portuguese and live under `tiktok/scripts/` once positioning is locked.

If you'd rather have the planning artifacts in PT-BR, say so and they will be translated in place. The schemas in the CSVs are language-neutral.

## Workflow

1. Fill in `00_priors.md` and commit it **before** any sampling, scraping, or competitor analysis. Locking predictions up front is what makes confirmation bias auditable later.
2. Read `01_research_plan.md` — sample frame, discovery method, analysis questions, kill criteria, schedule.
3. Collect accounts in `02_accounts.csv`, stratified across follower tiers per the plan.
4. Code videos in `03_videos.csv` (five per account: best, worst, three random from the last twenty).
5. Write findings in `04_findings.md`. Every claim cites a column in one of the CSVs. No vibes.
6. Derive positioning in `05_positioning.md` only after findings are settled, and run the red-team agent against it before locking it in.

## Subagents

`.claude/agents/` contains three assistants you invoke through Claude Code's Agent tool:

- **astro-red-team** — adversarial review of a hypothesis, finding, or positioning concept. Use it before locking any decision.
- **astro-comment-analyzer** — turn a raw paste of PT-BR TikTok comments into a structured demand-signal table.
- **astro-content-coder** — code one video against the `03_videos.csv` schema, consistently, so coding doesn't drift across 200 rows.

Invoke them by asking Claude Code, for example: "Use the astro-red-team agent on `tiktok/05_positioning.md`."

## Kill criteria

This project has explicit stop conditions in `01_research_plan.md`. If you hit one, pivot or stop — do not push harder.
