#!/usr/bin/env python3
"""
CEREBRO SISTER ENGINE DIAGNOSTIC — Predictions vs Actuals
========================================================
Runs sister alone across all calibration episodes. Plots pred vs actual,
identifies failure patterns: systematic bias (early/late), outliers, country effects.
Output: cerebro_data/sister_diagnostic.json + sister_diagnostic.png
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "cerebro_data" / "sister_diagnostic.json"
OUTPUT_PNG = SCRIPT_DIR / "cerebro_data" / "sister_diagnostic.png"
MIN_TRAIN = 3


def _load_calibration_episodes():
    """Same episode pool as cerebro_calibration run_calibration."""
    from cerebro_calibration import (
        _load_episodes,
        _load_oecd_clocks,
        _build_oecd_episodes,
        OECD_EVENT_YEARS,
        MIN_TRAIN_OPERATIONAL,
    )
    raw_us, _ = _load_episodes(score_threshold=1.5)
    raw = list(raw_us)
    oecd_clocks = _load_oecd_clocks()
    for country, clock_df in oecd_clocks.items():
        event_years = OECD_EVENT_YEARS.get(country, [])
        if not event_years:
            event_years = [yr + 5 for yr in range(1970, 2020, 10)]
        uk_threshold = 1.3 if country == "UK" else 1.5
        raw_oecd, _ = _build_oecd_episodes(country, clock_df, event_years, score_threshold=uk_threshold)
        if len(raw_oecd) >= MIN_TRAIN_OPERATIONAL + 2:
            raw.extend(raw_oecd)
    return raw


def _sister_walkforward(episodes: list) -> list:
    """Run sister only, per-country, same logic as calibration."""
    from cerebro_eval_utils import past_only_pool
    from cerebro_sister_engine import sister_predict

    by_country = {}
    for ep in episodes:
        c = ep.get("country", "US")
        by_country.setdefault(c, []).append(ep)

    results = []
    for country, country_eps in by_country.items():
        if len(country_eps) < MIN_TRAIN + 2:
            continue
        sorted_ep = sorted(country_eps, key=lambda e: e.get("saddle_year", 0))
        for ep in sorted_ep:
            t = ep.get("saddle_year")
            if t is None:
                continue
            pool = past_only_pool(country_eps, t)
            if len(pool) < MIN_TRAIN:
                continue
            event_yr = ep.get("event_year", t + 5)
            try:
                pred = sister_predict(
                    t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0), pool
                )
            except Exception:
                continue
            delta_actual = event_yr - t
            delta_pred = pred["peak_year"] - t
            error = pred["peak_year"] - event_yr
            results.append({
                "saddle_year": t,
                "event_year": event_yr,
                "country": country,
                "pred_peak_year": pred["peak_year"],
                "delta_actual": delta_actual,
                "delta_pred": delta_pred,
                "error": error,
                "abs_error": abs(error),
                "position": ep.get("position"),
                "velocity": ep.get("velocity"),
                "acceleration": ep.get("acceleration"),
                "n_train": pred.get("n_train"),
                "residual_iqr": pred.get("residual_iqr"),
            })
    return results


def run_diagnostic() -> dict:
    episodes = _load_calibration_episodes()
    results = _sister_walkforward(episodes)
    if len(results) < 5:
        return {"error": "Insufficient episodes", "n": len(results)}

    errors = [r["error"] for r in results]
    abs_errors = [r["abs_error"] for r in results]
    mean_error = sum(errors) / len(errors)
    mae = sum(abs_errors) / len(abs_errors)

    # Bias: positive = sister predicts too late, negative = too early
    bias_early = sum(1 for e in errors if e < -2)
    bias_late = sum(1 for e in errors if e > 2)
    bias_neutral = len(errors) - bias_early - bias_late

    # Worst episodes by |error|
    worst = sorted(results, key=lambda r: r["abs_error"], reverse=True)[:10]

    # By country
    by_country = {}
    for r in results:
        c = r["country"]
        if c not in by_country:
            by_country[c] = {"mae": [], "errors": []}
        by_country[c]["mae"].append(r["abs_error"])
        by_country[c]["errors"].append(r["error"])
    country_mae = {c: round(sum(v["mae"]) / len(v["mae"]), 2) for c, v in by_country.items()}
    country_bias = {c: round(sum(v["errors"]) / len(v["errors"]), 2) for c, v in by_country.items()}

    # By era (saddle_year)
    pre_1990 = [r for r in results if r["saddle_year"] < 1990]
    post_1990 = [r for r in results if r["saddle_year"] >= 1990]
    mae_pre = round(sum(r["abs_error"] for r in pre_1990) / len(pre_1990), 2) if pre_1990 else None
    mae_post = round(sum(r["abs_error"] for r in post_1990) / len(post_1990), 2) if post_1990 else None
    bias_pre = round(sum(r["error"] for r in pre_1990) / len(pre_1990), 2) if pre_1990 else None
    bias_post = round(sum(r["error"] for r in post_1990) / len(post_1990), 2) if post_1990 else None

    diag = {
        "n_episodes": len(results),
        "mae": round(mae, 2),
        "mean_error_bias": round(mean_error, 2),
        "bias_early_count": bias_early,
        "bias_late_count": bias_late,
        "bias_neutral_count": bias_neutral,
        "interpretation": "sister predicts too late" if mean_error > 0.5 else ("sister predicts too early" if mean_error < -0.5 else "no strong systematic bias"),
        "worst_10": [
            {
                "saddle_year": r["saddle_year"],
                "country": r["country"],
                "event_year": r["event_year"],
                "pred_peak_year": r["pred_peak_year"],
                "error": r["error"],
            }
            for r in worst
        ],
        "by_country_mae": country_mae,
        "by_country_bias": country_bias,
        "by_era": {
            "pre_1990": {"n": len(pre_1990), "mae": mae_pre, "bias": bias_pre},
            "post_1990": {"n": len(post_1990), "mae": mae_post, "bias": bias_post},
        },
        "episodes": results,
    }
    return diag


def plot_diagnostic(diag: dict) -> None:
    """Scatter: pred_peak_year vs event_year, with y=x reference."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    episodes = diag.get("episodes", [])
    if not episodes:
        return

    actuals = [e["event_year"] for e in episodes]
    preds = [e["pred_peak_year"] for e in episodes]
    countries = [e.get("country", "US") for e in episodes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pred vs actual scatter
    ax = axes[0]
    colors = {"US": "C0", "CA": "C1", "FR": "C2", "JP": "C3", "SE": "C4", "AU": "C5"}
    for c in set(countries):
        mask = [i for i, x in enumerate(countries) if x == c]
        ax.scatter(
            [actuals[i] for i in mask],
            [preds[i] for i in mask],
            label=c, alpha=0.7, s=60,
            c=colors.get(c, "gray"),
        )
    lo = min(min(actuals), min(preds))
    hi = max(max(actuals), max(preds))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x (perfect)")
    ax.set_xlabel("Actual event year")
    ax.set_ylabel("Sister predicted peak year")
    ax.set_title("Sister: Predicted vs Actual")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Right: error distribution
    ax = axes[1]
    errors = [e["error"] for e in episodes]
    ax.hist(errors, bins=15, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("Error (pred - actual years)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error distribution (MAE={diag.get('mae', 0):.2f}, bias={diag.get('mean_error_bias', 0):.2f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=120)
    plt.close()
    print(f"  → {OUTPUT_PNG}")


def main():
    print("Sister Engine Diagnostic")
    print("=" * 50)
    diag = run_diagnostic()
    if "error" in diag:
        print(f"Error: {diag['error']}")
        return 1

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    # Don't dump full episodes to JSON (large); keep summary + worst
    out = {k: v for k, v in diag.items() if k != "episodes"}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"n_episodes: {diag['n_episodes']}")
    print(f"MAE: {diag['mae']}")
    print(f"Mean error (bias): {diag['mean_error_bias']} — {diag['interpretation']}")
    print(f"Bias: early={diag['bias_early_count']}, late={diag['bias_late_count']}, neutral={diag['bias_neutral_count']}")
    print(f"By country MAE: {diag['by_country_mae']}")
    print(f"By country bias: {diag['by_country_bias']}")
    print(f"By era: pre-1990 {diag['by_era']['pre_1990']}, post-1990 {diag['by_era']['post_1990']}")
    print(f"Worst 3: {[(r['saddle_year'], r['country'], r['error']) for r in diag['worst_10'][:3]]}")
    print(f"  → {OUTPUT_JSON}")
    plot_diagnostic(diag)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
