# Roblox Up-and-Coming top-10 forecaster

Entry for make.fun `roblox-up-and-coming-forecast-01`: forecast Roblox's
Up-and-Coming chart (top 10, place by place) over the 72 hours after entries
close, as one `forecast.json` of per-place probabilities plus a top-10 guess.

## Method

`main.py` (Python 3.13, standard library only, fully deterministic) reads
`/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json`:

1. **Recent-level histogram** — each game's rank history, recency-weighted
   (8h half-life), smoothed with a Gaussian kernel. The chart's top-10
   membership is sticky (8–10 of 10 survive between reads), so the recent
   level carries most of the signal.
2. **Pooled rank-transition model** — from the history, learn how a game's
   rank moves over 24h given its current rank band and its 24h player-growth
   decile (24h log-playing growth predicts next-day rank gain, r ≈ 0.33).
   Step the current distribution forward to +24h/+48h/+72h and average over
   the scored window.
3. **Blend 50/50**, age out games that pass the 28-day chart rule, cap each
   place at (1 − unnamed share) and leave the rest unnamed so games the
   model never saw still get a share.

## Backtest (practice window, 2026-11-07T23:00Z → 2026-11-10T23:00Z)

| Model | Log score | Top-10 hits | Agreement |
|---|---|---|---|
| House persistence baseline | −43.51 | 6.4 | 53.5 |
| This entry | −35.43 | 6.4 | 54.0 |

Second window (train ≤ 2026-11-10T23:00Z, score 2026-11-11 → 2026-11-12):
−35.53 vs baseline −45.74. Runtime ~0.5s on the practice input, output is
byte-identical across runs.

## Files

- `main.py` — the entry (entrypoint: `python3 /bot/main.py`)
- `LICENSE` — MIT
