"""gta6-twitch-ccu-forecast-01 entry: a launch-shape ensemble forecaster.

The idea: a game's release-day viewer curve, hour by hour, looks like the
release days of past launches whose run-up to the checkpoint looked the
same. So the program

  1. builds, for the case's own category and for every comparable launch
     in the case, an hourly series relative to that launch's day_start
     (hour 0 is the release day's first hour, negative hours before it);
  2. weights each comparable by how closely its observed pre-checkpoint
     trajectory (relative hours -13..0, each hour in units of its own
     hour-1 print, in log space) matches the case's own trajectory, with
     a light penalty for a very different hour-1 scale;
  3. forecasts each hour 2..24 as the weighted empirical quantiles of the
     comparables' log-ratios ln(viewers_h / viewers_1) at that hour,
     scaled by the case's own hour-1 print.

Everything is a deterministic function of the case file: no randomness,
no network, Python standard library only. Any failure falls back to the
house persistence shape (flat at hour 1, random-walk spread), so a case
always gets a valid forecast.

Run as the grader does: python3 -P -s /bot/main.py, reading the case in
$MAKEFUN_INPUT_DIR (else /input) and writing $MAKEFUN_OUTPUT_DIR (else
/output). -P keeps this file's directory off sys.path; nothing beside
main.py is imported, so that does not matter here.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (for the fallback spread).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Relative hours of the run-up window the shape match compares: the last
# 13 hours before the release day plus hour 1 itself. For a rolling
# midnight-local unlock this window holds the first regional unlocks
# (New Zealand at -13) through the checkpoint, so it separates launches
# that were already live at hour 1 from ones that unlock later in the day.
WINDOW = list(range(-13, 1))
# Shape-match widths: how far (in log RMS over the window, and in log
# hour-1 scale) a comparable may sit from the case before its weight
# fades. Every comparable also keeps a small floor weight so the tails
# of the ensemble never rest on one or two launches.
SIGMA_SHAPE = 1.1
SIGMA_SCALE = 2.5
MIN_WEIGHT = 0.01
MIN_COMMON = 6
# Adjacent-hour pooling: hour h's quantile estimate also draws on the
# log-ratios of hours h-1 and h+1 at this weight. A launch-day curve is
# smooth, so neighbouring hours carry nearly the same distribution, and
# pooling roughly doubles the effective sample behind each hour's tails.
POOL = 0.35
# Spread floor: neither side of an hour's band may sit closer to the
# median than FLOOR_K times the house baseline's random-walk half-width
# at that level (0.15 * sqrt(h - 1) in log space). The ensemble can be
# overconfident where similar launches happened to agree; the floor
# keeps the 5% and 95% quantiles honest on a day that breaks the pattern.
FLOOR_K = 0.7
# Calibration, fitted by leave-one-out validation over the kit's
# comparables: the raw ensemble median overstates how far a release day
# travels from its first hour, and its spread runs slightly wide once
# the floor is in. ALPHA shrinks the median log-ratio toward 0 (the
# persistence forecast) and BETA scales the remaining spread around it.
ALPHA = 0.9
BETA = 0.88


def parse_ts(ts: str) -> float:
    """ISO 8601 UTC timestamp -> epoch seconds."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def hour_series(day_start: str, rows, key: str = "from") -> dict[int, float]:
    """Mean viewers per relative hour from rows stamped with `key`."""
    start = parse_ts(day_start)
    buckets: dict[int, list[float]] = {}
    for row in rows:
        try:
            rel = (parse_ts(row[key]) - start) / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        hour = int(round(rel))
        if abs(rel - hour) > 0.01:
            continue
        v = row.get("viewers")
        if v is None:
            continue
        buckets.setdefault(hour, []).append(float(v))
    return {h: sum(vs) / len(vs) for h, vs in buckets.items()}


def target_series(case: dict) -> dict[int, float]:
    """The case's own hourly series: history_hourly plus hour 1's print."""
    series = hour_series(case["day_start"], case.get("history_hourly") or [])
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            series[0] = float(row["viewers"])
    if 0 not in series:
        # Hour 1 may also be the last history_hourly row.
        hist = hour_series(case["day_start"], case.get("history") or [], key="at")
        series.update({h: v for h, v in hist.items() if h not in series})
    return series


def comparable_series(comp: dict) -> dict[int, float] | None:
    """A comparable launch's hourly series, or None if it is not a launch
    day we can learn from (no hour-1 print, or too sparse a day)."""
    if "before release" in (comp.get("name") or ""):
        return None
    series = hour_series(comp["day_start"], comp.get("hourly") or [])
    if series.get(0, 0) <= 0:
        return None
    day_hours = sum(1 for h in range(1, 24) if h in series)
    if day_hours < 12:
        return None
    return series


def window_vec(series: dict[int, float]) -> dict[int, float]:
    """The run-up trajectory in units of the hour-1 print, log space."""
    p1 = max(series.get(0, 1.0), 1.0)
    return {h: math.log(max(series[h], 1.0) / p1) for h in WINDOW if h in series}


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted quantile of (value, weight) pairs, linear interpolation
    over the weighted empirical CDF."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        pairs = [(v, 1.0) for v, _ in pairs]
        total = float(len(pairs))
    acc = 0.0
    prev_c, prev_v = 0.0, pairs[0][0]
    for v, w in pairs:
        acc += w
        c = acc / total
        if c >= tau:
            if c == prev_c:
                return v
            return prev_v + (v - prev_v) * (tau - prev_c) / (c - prev_c)
        prev_c, prev_v = c, v
    return pairs[-1][0]


def ensemble_weights(target: dict[int, float],
                     library: list[dict[int, float]]) -> list[tuple[dict[int, float], float]]:
    tvec = window_vec(target)
    tp1 = max(target.get(0, 1.0), 1.0)
    items = []
    for series in library:
        lvec = window_vec(series)
        common = [h for h in WINDOW if h in tvec and h in lvec]
        if len(common) < MIN_COMMON:
            continue
        mse = sum((tvec[h] - lvec[h]) ** 2 for h in common) / len(common)
        d_shape = math.sqrt(mse)
        d_scale = abs(math.log(max(series[0], 1.0)) - math.log(tp1))
        w = math.exp(-0.5 * (d_shape / SIGMA_SHAPE) ** 2
                     - 0.5 * (d_scale / SIGMA_SCALE) ** 2) + MIN_WEIGHT
        items.append((series, w))
    return items


def ensemble_forecast(case: dict) -> dict[int, list[float]]:
    target = target_series(case)
    p1 = max(target.get(0, 1.0), 1.0)
    library = [s for s in (comparable_series(c) for c in case.get("comparables") or [])
               if s is not None]
    items = ensemble_weights(target, library)
    if not items:
        raise ValueError("no comparable launch matched the run-up window")
    out = {}
    for h in range(1, 24):
        pairs = []
        for s, w in items:
            for hh, pw in ((h - 1, POOL), (h, 1.0), (h + 1, POOL)):
                if 1 <= hh <= 23 and hh in s:
                    pairs.append((math.log(s[hh] / s[0]), w * pw))
        if not pairs:
            raise ValueError(f"no comparable has hour {h + 1}")
        logs = [weighted_quantile(pairs, tau) for tau in LEVELS]
        raw_med = logs[2]
        med = ALPHA * raw_med
        logs = [med + (q - raw_med) * BETA for q in logs]
        half = 0.15 * math.sqrt(h)  # the baseline's half-width unit at hour h + 1
        for i, z in enumerate(Z):
            if z < 0:
                logs[i] = min(logs[i], med + FLOOR_K * z * half)
            elif z > 0:
                logs[i] = max(logs[i], med + FLOOR_K * z * half)
        out[h + 1] = [p1 * math.exp(q) for q in logs]
    return out


def baseline_forecast(p1: float) -> dict[int, list[float]]:
    """The house persistence shape, used only if the ensemble cannot run."""
    p = max(1.0, p1)
    return {h: [p * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            for h in range(2, 25)}


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def clean_row(values: list[float]) -> list[float | int]:
    """Clamp into (0, 1e9], round, and enforce non-decreasing order."""
    row = []
    prev = 0.0
    for v in values:
        v = min(max(v, 0.01), 1e9)
        v = float(round2(v))
        v = max(v, prev)
        row.append(int(v) if v.is_integer() else v)
        prev = v
    return row


def forecast_case(case: dict, name: str) -> dict:
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first and first.get("viewers") is not None else 1.0
    try:
        hours = ensemble_forecast(case)
    except Exception:
        hours = baseline_forecast(p1)
    rows = [{"hour": row["hour"], "q": clean_row(hours[row["hour"]])}
            for row in case["forecast"]]
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
