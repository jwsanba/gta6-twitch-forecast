# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-shape ensemble forecaster for Grand Theft Auto VI's release-day
Twitch viewers. For each hour 2–24 of the release day (UTC) it writes five
quantiles (5%, 25%, 50%, 75%, 95%) of that hour's viewer count in the
"Grand Theft Auto VI" Twitch category.

The model: a launch's release-day curve looks like the release days of
past launches whose run-up to the checkpoint looked the same. The program
weights every comparable launch in the case by how closely its observed
pre-checkpoint trajectory (the last 13 hours before the day plus hour 1,
in units of its own hour-1 print, log space) matches the case's own, then
reads the weighted empirical quantiles of the comparables' hourly
log-ratios to hour 1. Adjacent hours are pooled when estimating the
quantiles, the spread is floored at a fraction of the house baseline's
random-walk width, and the median and spread carry a small calibration
shrink fitted by leave-one-out validation over the kit's comparables.
Pure Python 3.13 standard library, offline, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
