# GTA VI Trophy Rarity Forecast

[![Bounty](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![Prize](https://img.shields.io/badge/prize-%24250%20USDC-green)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![Status](https://img.shields.io/badge/status-entered-orange)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at **2026-11-26T00:00:00Z** — one week after launch — as five quantiles
(5 / 25 / 50 / 75 / 95%) per trophy.

Bounty: [gta6-trophy-rarity-forecast-01](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
($250 USDC, top 20 paid, only for entries strictly below the best house
entry). Entry: `main.py` (Python 3.13, standard library only, no network).

## How it works

All modelling happens on the **logit scale** (`logit(p) = ln(p/(1-p))`,
rates clamped to 0.01–99.99%), because on that scale a miss from 1% to 2%
costs about what a miss from 33% to 50% does — exactly how the bounty
scores (mean pinball loss on the logit scale).

1. **Log-time trend on all history.** Per trophy, least-squares fit of
   `logit(rate) ~ a + b * ln(1 + hours since 2026-11-19T00:00:00Z)` over
   every read in the case history. Only *distinct refresh states* enter
   the fit: Exophase refreshes rarely, so consecutive reads repeating the
   same page count once — a long flat stretch must not outweigh a real
   move. The fitted slope is shrunk toward a **prior slope** from the
   trophy's tier and description (early-story bronze rises fast, platinum
   and 100%-completion gold barely move, grindy silver/bronze in between).
2. **Dilution-aware count model.** Exophase rates are shares of *tracked*
   players, so when `players_tracked` jumps, every rate falls mechanically
   (new players have earned nothing yet). A second forecast extrapolates
   `log(earn count)` and `log(players_tracked)` separately and divides
   them; the two forecasts are averaged on the logit scale whenever the
   count model has enough distinct points.
3. **Honest wide spreads.** The total logit sigma combines a refresh-jump
   term (0.55 — roughly a halving/doubling of a rate), a random-walk term
   growing with horizon (0.45·√days), the fit residual, and the slope
   uncertainty projected to the target; clamped to [0.70, 1.70]. On launch
   night a single Exophase refresh moved every rate at once, outside the
   house baseline's narrow 90-minute band (practice loss ≈ 1.51) — a
   forecast that pretends refreshes cannot happen pays for it at every
   trophy.

Robustness: missing/null rates are skipped, flat histories fall back to
the (damped) prior slope, 0%/100% edge cases are handled by the logit
clamp, and a trophy held by 100% of tracked players is centred near 97%
(new players dilute it). The median's logit move from the last rate is
clamped to [−2.5, +4.5]. Fully deterministic — no randomness anywhere,
two runs are byte-identical.

## Local validation (19 Nov 2026)

```sh
KIT=~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/gta6-trophy-rarity-forecast-01-kit
MAKEFUN_INPUT_DIR=$KIT/input MAKEFUN_OUTPUT_DIR=/tmp/fc1 python3 main.py
```

- Practice cases `p00` / `p01`: both outputs valid — exactly one key
  (`trophies`), 55 rows, 5 finite non-decreasing quantiles in [0, 100]
  each, 2,791 / 2,793 bytes (limit 65,536).
- Determinism: two runs produce byte-identical outputs (`diff` clean).
- Runtime: ~0.09 s for both practice cases (limit 8 s/case).
- Sanity per tier (p00 medians): platinum 0.69% (last 0.63%), gold
  0.66–1.75%, silver 0.75–0.92%, bronze 0.85–3.22% except the 100%
  trophy centred at 97%. Chapter-1 bronze 1.25% → median 1.56%;
  100%-completion gold 0.63% → 0.64% (nearly flat, as intended).
- Synthetic backtest (48 h of synthetic history: log-time trends +
  player growth, answer = trend continued then a refresh that halves
  rates via dilution): pinball **0.366 vs house baseline 0.667** on the
  same answer. The practice histories themselves are completely flat
  (638 players, every rate unchanged), so they check shape/determinism
  only — they carry no trend signal, and the final's 2.5 days of real
  reads are where the trend + count models earn their keep.

## Run it

```sh
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Defaults are `/input` and `/output` (the grader's paths). The program
only creates/overwrites its own `<id>.out` files; it never deletes
anything in the output directory.

## License

MIT — see [LICENSE](LICENSE).
