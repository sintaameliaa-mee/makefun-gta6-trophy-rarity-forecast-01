#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, its earn rate on
Exophase at a case's target time, as five quantiles (5/25/50/75/95%).

Method (stdlib only, no network, fully deterministic):
  * Work on the logit scale, logit(p) = ln(p / (1 - p)), p = rate / 100,
    with rates clamped to [0.01, 99.99] (the page prints 2 decimals).
  * Time is x = ln(1 + hours since 2026-11-19T00:00:00Z).  Earn rates grow
    roughly log-linearly in time after launch: early players finish the
    easy story trophies fast, then the curve flattens as the remaining
    pool is the slower / less engaged tail.
  * Per trophy, fit logit(rate) ~ a + b*x by least squares on ALL of the
    case history (distinct refresh states only: consecutive reads that
    repeat the same page are one state, so a long flat stretch does not
    outweigh a real move).  The slope is shrunk toward a prior slope
    chosen from the trophy's tier and description (early-story bronze
    rises fast, platinum / 100%-completion gold barely moves, grindy
    silver/bronze sit in between).
  * Dilution: Exophase rates are shares of *tracked* players, so when
    players_tracked jumps, every rate falls mechanically (the new players
    have earned nothing yet).  A second forecast is built from earn
    COUNTS: count = rate * players_tracked / 100, log(count) fitted
    against x and extrapolated, divided by players_tracked extrapolated
    the same way.  The two forecasts are averaged on the logit scale
    whenever the count model has enough distinct points.
  * Spread: honest and wide.  Pinball loss punishes a narrow band every
    time an Exophase refresh moves the rates (on launch night one refresh
    moved every rate at once, outside the house baseline's 90-minute
    band).  Total logit sigma combines a refresh-jump term (0.55, roughly
    a halving/doubling of a rate), a random-walk term growing with the
    horizon (0.45 * sqrt(days)), the fit residual, and the slope
    uncertainty projected to the target.  Clamped to [0.70, 1.70].

Usage: MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> python3 main.py
Reads every *.in file in the input dir (else /input/), writes one
<id>.out per case to the output dir (else /output/).  Never deletes
anything in the output dir; it only creates/overwrites its own .out
files.  Runs in well under a second per case on 1 CPU.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Logit-scale slope priors: logit points per unit of ln(1 + hours).
# A slope of 0.8 over the final horizon (x grows ~1.0 from checkpoint to
# target) lifts a 1% rate to ~2.2%; a slope of 0.1 barely moves it.


def prior_slope(trophy):
    """Prior log-time slope from tier + description cues."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "bronze").lower()
    if tier == "platinum":
        return 0.10
    if "100% completion" in desc or "100% complet" in desc:
        return 0.05
    if "chapter 1" in desc:
        return 1.00
    if "chapter 2" in desc:
        return 0.90
    if "chapter 3" in desc:
        return 0.80
    if "chapter 4" in desc:
        return 0.70
    if "final mission" in desc:
        return 0.55 if tier == "gold" else 0.50
    # Grindy / completionist trophies rise slowly.
    grindy = ("250 stunts" in desc or "gold medal" in desc
              or "100 locations" in desc or "collectible card" in desc
              or "5 bank vaults" in desc or "8 mission strands" in desc
              or "100% completion" in desc)
    if grindy:
        return 0.20
    slow_grind = ("50 challenges" in desc or "250,000" in desc
                  or "100,000" in desc or "100 vehicles" in desc
                  or "20 different races" in desc or "all garage" in desc
                  or "10 vehicles" in desc or "every spot" in desc
                  or "every location" in desc or "each gang" in desc)
    if slow_grind:
        return 0.35
    # Easy one-off activities and story missions rise fast.
    if tier == "gold":
        return 0.40
    if tier == "silver":
        return 0.30
    return 0.55  # default bronze


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp (trailing Z allowed)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def x_of(moment):
    """Log-time coordinate: ln(1 + hours since launch midnight)."""
    return math.log1p(max(0.0, hours_since_epoch(moment)))


def clamp_rate(rate):
    return min(99.99, max(0.01, float(rate)))


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(eta):
    eta = min(9.21, max(-9.21, eta))
    return 100.0 / (1.0 + math.exp(-eta))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_var, slope_var, x_mean, n)."""
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if n < 2 or sxx <= 1e-12:
        return y_mean, 0.0, 0.0, 0.0, x_mean, n
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = y_mean - b * x_mean
    if n > 2:
        resid_var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        slope_var = resid_var / sxx
    else:
        resid_var = 0.0
        slope_var = 0.0
    return a, b, resid_var, slope_var, x_mean, n


def distinct_states(points):
    """Collapse consecutive points with the same value to their first time.

    `points` is a list of (x, value) in time order.  Exophase refreshes
    rarely, so most reads repeat the previous page; counting each repeat
    would let a long flat stretch dominate the fit.
    """
    out = []
    last_value = None
    for x, value in points:
        if last_value is None or value != last_value:
            out.append((x, value))
            last_value = value
    return out


def forecast_trophy(trophy, history, idx, x_target, horizon_days):
    """Return the five quantiles for one trophy."""
    # Series of (x, rate) and (x, players) in time order.
    rate_points = []
    player_points = []
    count_points = []
    last_rate = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        rate = float(rates[idx])
        last_rate = rate
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        x = x_of(moment)
        rate_points.append((x, rate))
        players = read.get("players_tracked")
        if players:
            players = float(players)
            player_points.append((x, players))
            count_points.append((x, math.log(rate / 100.0 * players + 0.5)))

    if last_rate is None:
        # No history at all for this trophy: flat, maximally unsure.
        last_rate = 1.0
        return [0.02, 0.30, 1.0, 3.30, 50.0]

    last_logit = logit_pct(last_rate)

    # --- Trend forecast on the logit scale, all history. ---
    states = distinct_states(rate_points)
    xs = [x for x, _ in states]
    ys = [logit_pct(r) for _, r in states]
    a, b_fit, resid_var, slope_var, x_mean, n = ols(xs, ys)
    prior = prior_slope(trophy)
    # Shrink the fitted slope toward the prior; the prior counts as
    # K_PRIOR pseudo-observations.  With flat history (one distinct
    # state) the fit says nothing, so the slope is mostly the prior
    # damped by the lack of evidence.
    k_prior = 3.0
    if n <= 1:
        b = prior * 0.5
        resid_sd = 0.15
        slope_se = 0.30
        # Anchor at the LAST read that carried this rate, not at when the
        # flat state was first seen: the state persisted until then, so
        # only the log-time after the last read can move the forecast.
        x_last_read = rate_points[-1][0] if rate_points else xs[-1]
        eta_trend = last_logit + b * (x_target - x_last_read)
    else:
        b = (n * b_fit + k_prior * prior) / (n + k_prior)
        resid_sd = math.sqrt(max(0.0, resid_var))
        slope_se = math.sqrt(max(0.0, slope_var))
        # Re-anchor the intercept at the shrunk slope so the line still
        # passes through the fitted mean point.
        y_mean = sum(ys) / n
        eta_trend = y_mean + b * (x_target - x_mean)

    # --- Count forecast (dilution-aware), when the data supports it. ---
    eta = eta_trend
    p_states = distinct_states(player_points)
    c_states = distinct_states(count_points)
    if len(p_states) >= 2 and len(c_states) >= 3:
        pa, pb, _, _, _, _ = ols([x for x, _ in p_states],
                                 [math.log(v) for _, v in p_states])
        players_target = math.exp(pa + pb * x_target)
        ca, cb, _, _, _, _ = ols([x for x, _ in c_states],
                                 [v for _, v in c_states])
        count_target = math.exp(ca + cb * x_target) - 0.5
        if players_target > 0 and count_target > 0:
            rate_count = 100.0 * count_target / players_target
            eta_count = logit_pct(rate_count)
            # Average the two views on the logit scale.
            eta = 0.5 * eta_trend + 0.5 * eta_count

    # --- Median, clamped so one wild fit cannot explode the forecast. ---
    if last_rate >= 99.99:
        # A trophy everybody holds today cannot stay at 100% once the
        # tracked pool grows: new players dilute it.  Centre near 97%.
        median_logit = logit_pct(97.0)
    else:
        median_logit = min(last_logit + 4.5,
                           max(last_logit - 2.5, eta))
        median_logit = min(math.log(99.9 / 0.1),
                           max(math.log(0.02 / 99.98), median_logit))

    # --- Spread: refresh jump + horizon random walk + fit uncertainty. ---
    sigma2 = (0.55 ** 2                      # Exophase refresh jump
              + (0.45 * math.sqrt(horizon_days)) ** 2
              + resid_sd ** 2
              + (slope_se * (x_target - x_mean)) ** 2)
    sigma = min(1.70, max(0.70, math.sqrt(sigma2)))

    quantiles = []
    for z in Z:
        q = inv_logit_pct(median_logit + z * sigma)
        quantiles.append(round(min(99.99, max(0.01, q)), 4))
    # Enforce non-decreasing after rounding.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def fallback_forecast(case):
    """Last-rate flat forecast with a wide spread (never invalid)."""
    history = case.get("history") or []
    last_rates = None
    for read in reversed(history):
        rates = read.get("rates") or []
        if rates:
            last_rates = rates
            break
    rows = []
    for trophy in case.get("trophies") or []:
        tid = trophy["id"]
        idx = tid - 1
        rate = 1.0
        if last_rates and idx < len(last_rates) and last_rates[idx] is not None:
            rate = float(last_rates[idx])
        centre = logit_pct(rate)
        qs = [round(min(99.99, max(0.01, inv_logit_pct(centre + z * 1.0))), 4)
              for z in Z]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    """Forecast one case dict -> {"trophies": [{"id": n, "q": [...]}]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    try:
        target = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        return fallback_forecast(case)
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    x_target = x_of(target)
    rows = []
    for idx, trophy in enumerate(trophies):
        try:
            qs = forecast_trophy(trophy, history, idx, x_target, horizon_days)
        except Exception:
            qs = fallback_forecast(case)["trophies"][idx]["q"]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main(argv=None):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    case_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".in"))
    if not case_files:
        print("no .in files in %s" % input_dir, file=sys.stderr)
        return 1
    for name in case_files:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            forecast = forecast_case(case)
        except Exception as exc:  # never crash: emit a valid forecast
            print("case %s: %r; using fallback" % (case_id, exc),
                  file=sys.stderr)
            forecast = fallback_forecast(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
