"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json (or $MAKEFUN_INPUT_DIR),
writes /output/forecast.json (or $MAKEFUN_OUTPUT_DIR).

Method (stdlib only, fully deterministic):
  1. Recent-level histogram: each game's rank history, recency-weighted
     (8h half-life), smoothed with a Gaussian kernel.
  2. Pooled rank-transition model: from the history, learn how a game's rank
     moves over 24h given its current rank band and its 24h player-growth
     decile; step the current distribution forward to +24h/+48h/+72h and
     average over the 72h scored window.
  3. Blend 50/50, age out games that pass 28 days (28-day chart rule),
     cap each place at (1 - unnamed share), leave the rest unnamed.
"""
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HALF_LIFE_H = 8.0
KERNEL_SIGMA = 2.5
BLEND_TRANS = 0.5
UNNAMED = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28)
MIN_P = 0.002
MAX_ROWS_PLACE = 100
MAX_RANK = 100
AGE_DAYS = 28.0
GROWTH_WIN_H = 24.0
TRANS_H = 24.0


def _hours(a, b):
    return (b - a).total_seconds() / 3600.0


def _band(rk):
    if rk <= 1:
        return 0
    if rk <= 3:
        return 1
    if rk <= 6:
        return 2
    if rk <= 10:
        return 3
    if rk <= 15:
        return 4
    if rk <= 25:
        return 5
    if rk <= 50:
        return 6
    return 7


def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            chart = {}
            for g in r.get("up_and_coming") or []:
                chart[g["universe_id"]] = (g["rank"], g.get("playing") or 0)
            details = {}
            if r.get("details"):
                for uid, d in r["details"].items():
                    if d:
                        details[int(uid)] = d
            reads.append({"at": _parse_ts(r["at"]), "chart": chart, "details": details})
    reads.sort(key=lambda r: r["at"])
    return reads


def smooth(counts, sigma):
    tot = sum(counts)
    if tot == 0:
        return None
    base = [c / tot for c in counts]
    out = [0.0] * len(base)
    for k in range(1, len(base)):
        if base[k] == 0:
            continue
        lo = max(1, int(k - 4 * sigma))
        hi = min(MAX_RANK, int(k + 4 * sigma) + 1)
        s = 0.0
        tmp = {}
        for j in range(lo, hi + 1):
            v = math.exp(-0.5 * ((j - k) / sigma) ** 2)
            tmp[j] = v
            s += v
        for j, v in tmp.items():
            out[j] += base[k] * v / s
    return out


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = task.get("places", 10)
    output_name = task.get("output", "forecast.json")
    reads = load_history(INPUT_DIR)
    if not reads:
        forecast = {"probabilities": [], "top10": []}
    else:
        forecast = build(reads, places)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True) + "\n")


def build(reads, places):
    t_now = reads[-1]["at"]
    series = {}
    first_seen = {}
    created = {}
    for r in reads:
        for uid, (rk, pl) in r["chart"].items():
            series.setdefault(uid, []).append((r["at"], rk, pl))
            first_seen.setdefault(uid, r["at"])
        for uid, d in r["details"].items():
            if d.get("created"):
                try:
                    created[uid] = _parse_ts(d["created"])
                except Exception:
                    pass

    def w(t):
        return 0.5 ** (_hours(t, t_now) / HALF_LIFE_H)

    def growth_at(uid, t):
        pts = series.get(uid, [])

        def playing_near(target, tol_h):
            best = None
            bd = 1e18
            for tt, _rk, pl in pts:
                d = abs(_hours(tt, target))
                if d < bd and pl:
                    best, bd = pl, d
            return best if bd <= tol_h else None

        p_now = playing_near(t, 6)
        p_old = playing_near(t - timedelta(hours=GROWTH_WIN_H), 8)
        if p_now and p_old and p_old > 0 and p_now > 0:
            return (math.log(p_now) - math.log(p_old)) / GROWTH_WIN_H
        return None

    # pooled 24h transitions conditioned on (rank band, growth decile)
    all_g = []
    samples = []
    for i, r in enumerate(reads):
        t = r["at"]
        j = None
        bd = 1e18
        for k, r2 in enumerate(reads):
            d = abs(_hours(r2["at"], t + timedelta(hours=TRANS_H)))
            if d < bd:
                j, bd = k, d
        if j is None or bd > 8 or j == i:
            continue
        for uid, (rk, _pl) in r["chart"].items():
            g = growth_at(uid, t)
            if g is None or uid not in reads[j]["chart"]:
                continue
            all_g.append(g)
            samples.append((_band(rk), g, reads[j]["chart"][uid][0]))
    all_g.sort()
    dec = [all_g[int(len(all_g) * q / 10)] for q in range(1, 10)] if all_g else [0.0] * 9

    def decile(g):
        d = 0
        for th in dec:
            if g > th:
                d += 1
        return d

    trans = defaultdict(lambda: [0.0] * (MAX_RANK + 1))
    trans_band = defaultdict(lambda: [0.0] * (MAX_RANK + 1))
    trans_all = [0.0] * (MAX_RANK + 1)
    for band, g, rk2 in samples:
        d = decile(g)
        trans[(band, d)][min(rk2, MAX_RANK)] += 1.0
        trans_band[band][min(rk2, MAX_RANK)] += 1.0
        trans_all[min(rk2, MAX_RANK)] += 1.0
    trans_s = {k: smooth(v, KERNEL_SIGMA) for k, v in trans.items()}
    trans_band_s = {k: smooth(v, KERNEL_SIGMA) for k, v in trans_band.items()}
    trans_all_s = smooth(trans_all, KERNEL_SIGMA)

    def trans_dist(band, g):
        if g is None:
            return trans_band_s.get(band) or trans_all_s
        dist = trans_s.get((band, decile(g)))
        if dist is None:
            dist = trans_band_s.get(band) or trans_all_s
        return dist

    def step(d0, g):
        out = [0.0] * (MAX_RANK + 1)
        for k in range(1, MAX_RANK + 1):
            if d0[k] == 0:
                continue
            td = trans_dist(_band(k), g)
            if td is None:
                continue
            for j in range(1, MAX_RANK + 1):
                if td[j]:
                    out[j] += d0[k] * td[j]
        return out

    dist = {}
    for uid, pts in series.items():
        r0 = pts[-1][1]
        hist = [0.0] * (MAX_RANK + 1)
        wsum = 0.0
        for t, rk, _ in pts:
            wi = w(t)
            hist[min(rk, MAX_RANK)] += wi
            wsum += wi
        hist_s = smooth([x / wsum for x in hist], KERNEL_SIGMA) if wsum else None
        g = growth_at(uid, t_now)
        cur = [0.0] * (MAX_RANK + 1)
        cur[min(r0, MAX_RANK)] = 1.0
        cur = smooth(cur, KERNEL_SIGMA)
        d24 = step(cur, g)
        d48 = step(d24, g)
        d72 = step(d48, g)
        acc = [0.0] * (MAX_RANK + 1)
        for arr, wt in ((cur, 12.0), (d24, 24.0), (d48, 24.0), (d72, 12.0)):
            for j in range(1, MAX_RANK + 1):
                acc[j] += arr[j] * wt / 72.0
        final = [0.0] * (MAX_RANK + 1)
        for j in range(1, MAX_RANK + 1):
            hv = hist_s[j] if hist_s else 0.0
            final[j] = BLEND_TRANS * acc[j] + (1 - BLEND_TRANS) * hv
        born = created.get(uid) or first_seen.get(uid)
        if born:
            hours_left = _hours(t_now, born + timedelta(days=AGE_DAYS))
            frac = max(0.0, min(1.0, hours_left / 72.0))
            final = [x * frac for x in final]
        dist[uid] = final

    probabilities = []
    top10_score = {}
    for place in range(1, places + 1):
        rows = []
        for uid, probs in dist.items():
            val = probs[place] if place < len(probs) else 0.0
            if val >= MIN_P:
                rows.append((uid, val))
        rows.sort(key=lambda kv: (-kv[1], kv[0]))
        rows = rows[:MAX_ROWS_PLACE]
        total = sum(v for _, v in rows)
        cap = 1.0 - UNNAMED[place - 1]
        scale = cap / total if total > cap else 1.0
        for uid, val in rows:
            probabilities.append({"universe_id": uid, "place": place, "p": round(val * scale, 6)})
            top10_score[uid] = top10_score.get(uid, 0.0) + val * scale
    last_rank = {u: rk for u, (rk, _) in reads[-1]["chart"].items()}
    top10 = sorted(top10_score, key=lambda u: (-top10_score[u], last_rank.get(u, 999)))[:places]
    for u, (_rk, _pl) in sorted(reads[-1]["chart"].items(), key=lambda kv: kv[1][0]):
        if len(top10) >= places:
            break
        if u not in top10:
            top10.append(u)
    return {"probabilities": probabilities, "top10": top10}


if __name__ == "__main__":
    main()
