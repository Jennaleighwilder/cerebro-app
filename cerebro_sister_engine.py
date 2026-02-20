#!/usr/bin/env python3
"""
CEREBRO SISTER ENGINE — Time-varying peak estimator (era-split GPR)
==================================================================
Era-split Gaussian Process: pre-1990 vs post-1990 models. Handles non-stationarity.
Fallback to Huber when era has < 5 episodes.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "sister_latest.json"
MIN_TRAIN = 5
POST_1990_WEIGHT = 3.0
OUTLIER_STD_THRESH = 2.5


def _feature_vec_base(ep: dict) -> np.ndarray:
    """Base: [position, velocity, acceleration, |velocity|, sign(velocity), sign(acceleration)]."""
    pos = float(ep.get("position", 0))
    vel = float(ep.get("velocity", 0))
    acc = float(ep.get("acceleration", 0))
    vel_abs = abs(vel)
    sign_vel = 1.0 if vel > 0 else (-1.0 if vel < 0 else 0.0)
    sign_acc = 1.0 if acc > 0 else (-1.0 if acc < 0 else 0.0)
    return np.array([pos, vel, acc, vel_abs, sign_vel, sign_acc], dtype=np.float64)


def _velocity_sign_change_count(pool: list, ep_index: int, window: int = 10) -> float:
    """Count velocity sign changes in last `window` episodes before ep_index."""
    start = max(0, ep_index - window)
    window_eps = pool[start:ep_index]
    if len(window_eps) < 2:
        return 0.0
    vels = [float(e.get("velocity", 0)) for e in window_eps]
    count = 0
    for i in range(1, len(vels)):
        if (vels[i - 1] > 0 and vels[i] < 0) or (vels[i - 1] < 0 and vels[i] > 0):
            count += 1
    return float(count)


def _acceleration_magnitude(pool: list, ep_index: int, window: int = 5) -> float:
    """Mean absolute acceleration over last `window` episodes."""
    start = max(0, ep_index - window)
    window_eps = pool[start:ep_index]
    if not window_eps:
        return 0.0
    accs = [abs(float(e.get("acceleration", 0))) for e in window_eps]
    return float(np.mean(accs))


def _feature_vec_extended(ep: dict, pool: list, ep_index: int, year: int | None = None) -> np.ndarray:
    """Base + era_index, velocity_sign_change_count, acceleration_magnitude."""
    base = _feature_vec_base(ep)
    era_index = (year - 1974) / 50.0 if year is not None else 0.5
    vsc = _velocity_sign_change_count(pool, ep_index)
    acc_mag = _acceleration_magnitude(pool, ep_index)
    return np.concatenate([base, [era_index, vsc, acc_mag]])


def _fit_huber(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    """Fallback: Huber regression."""
    try:
        from sklearn.linear_model import HuberRegressor
        reg = HuberRegressor(max_iter=200, epsilon=1.0)
        reg.fit(X, y, sample_weight=sample_weight)
        return reg.predict, reg
    except ImportError:
        n, d = X.shape
        lam = 0.01
        Xb = np.hstack([np.ones((n, 1)), X])
        yw = y if sample_weight is None else y * np.sqrt(sample_weight)
        Xw = Xb if sample_weight is None else Xb * np.sqrt(sample_weight)[:, None]
        w = np.linalg.solve(Xw.T @ Xw + lam * np.eye(d + 1), Xw.T @ yw)

        def predict(Xnew):
            Xb_new = np.hstack([np.ones((Xnew.shape[0], 1)), Xnew])
            return Xb_new @ w
        return predict, None


def _fit_gpr(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    """Gaussian Process with RBF + WhiteKernel. Fixed kernel to avoid bound warnings."""
    try:
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        kernel = RBF(length_scale=2.0, length_scale_bounds=(0.5, 50)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 2))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X, y)
        return gpr.predict, gpr
    except Exception:
        return _fit_huber(X, y, sample_weight)


def _cap_outliers(y: np.ndarray, thresh: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Cap y at mean ± thresh*std for training. Return (y_capped, y_original)."""
    mean, std = float(np.mean(y)), float(np.std(y))
    if std < 1e-8:
        return y.copy(), y.copy()
    lo, hi = mean - thresh * std, mean + thresh * std
    y_capped = np.clip(y, lo, hi)
    return y_capped, y.copy()


def _fit_era_model(episodes: list, use_gpr: bool = True) -> tuple[callable, np.ndarray, float]:
    """Fit model on episodes. Returns (predict_fn, preds, residual_iqr)."""
    pool = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    n = len(pool)
    X = np.array([_feature_vec_extended(e, pool, i, e.get("saddle_year")) for i, e in enumerate(pool)])
    y_raw = np.array([e.get("event_year", 0) - e.get("saddle_year", 0) for e in pool], dtype=float)
    y, _ = _cap_outliers(y_raw, OUTLIER_STD_THRESH)
    weights = np.array([
        POST_1990_WEIGHT if e.get("saddle_year", 0) >= 1990 else 1.0 for e in pool
    ], dtype=float)

    if use_gpr and n >= 5:
        predict_fn, _ = _fit_gpr(X, y, weights)
    else:
        predict_fn, _ = _fit_huber(X, y, weights)

    preds = predict_fn(X)
    if hasattr(preds, "reshape"):
        preds = np.asarray(preds).ravel()
    residuals = np.abs(y_raw - preds)
    iqr = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
    return predict_fn, preds, iqr


def sister_train_predict_fn(analogue_episodes: list[dict]):
    """Train on pool, return (predict_delta_fn, residual_iqr). Uses era-split when possible."""
    if len(analogue_episodes) < MIN_TRAIN:
        return lambda p, v, a: 5.0, 5.0
    pool = sorted(analogue_episodes, key=lambda e: e.get("saddle_year", 0))
    predict_fn, preds, iqr = _fit_era_model(pool, use_gpr=True)

    def predict_delta(pos: float, vel: float, acc: float) -> float:
        ep = {"position": pos, "velocity": vel, "acceleration": acc}
        x = _feature_vec_extended(ep, pool, len(pool), year=None).reshape(1, -1)
        return float(np.clip(predict_fn(x)[0], 1, 15))

    return predict_delta, iqr


def sister_predict(
    now_year: int,
    position: float,
    velocity: float,
    acceleration: float,
    analogue_episodes: list[dict],
) -> dict:
    """
    Sister peak using era-split GPR. Pre-1990 vs post-1990 models.
    """
    if len(analogue_episodes) < MIN_TRAIN:
        return {
            "peak_year": now_year + 5,
            "window_start": now_year + 3,
            "window_end": now_year + 10,
            "confidence_pct": 50,
            "method": "sister_gpr",
            "n_train": 0,
            "residual_iqr": 5.0,
        }

    pool = sorted(analogue_episodes, key=lambda e: e.get("saddle_year", 0))
    pre1990 = [e for e in pool if e.get("event_year", 0) < 1990]
    post1990 = [e for e in pool if e.get("event_year", 0) >= 1990]
    use_post = now_year >= 1990
    era_pool = post1990 if use_post else pre1990

    if len(era_pool) < MIN_TRAIN:
        era_pool = pool
    predict_fn, preds, residual_iqr = _fit_era_model(era_pool, use_gpr=len(era_pool) >= 5)
    residuals = np.abs(
        np.array([e.get("event_year", 0) - e.get("saddle_year", 0) for e in era_pool]) - preds
    )
    q10 = float(np.percentile(residuals, 10))
    q90 = float(np.percentile(residuals, 90))

    ep = {"position": position, "velocity": velocity, "acceleration": acceleration}
    x = _feature_vec_extended(ep, pool, len(pool), year=now_year).reshape(1, -1)
    delta_pred = float(np.clip(predict_fn(x)[0], 1, 15))
    # Bias correction: GPR tends to predict too early; add 1 year when post-1990
    if now_year >= 1990 and len(post1990) >= 10:
        delta_pred = min(15, delta_pred + 1.0)

    peak_year = now_year + int(round(delta_pred))
    window_start = now_year + int(round(max(1, delta_pred - (q90 - q10) / 2)))
    window_end = now_year + int(round(min(15, delta_pred + (q90 - q10) / 2)))
    if window_start >= window_end:
        window_start = peak_year - 2
        window_end = peak_year + 2

    conf = 70
    conf -= min(30, int(residual_iqr * 4))
    n_eff = len(analogue_episodes)
    if n_eff < 5:
        conf = int(conf * 0.6)
    conf = max(40, min(95, conf))

    return {
        "peak_year": peak_year,
        "window_start": window_start,
        "window_end": window_end,
        "confidence_pct": conf,
        "method": "sister_gpr",
        "n_train": n_eff,
        "residual_iqr": round(residual_iqr, 2),
    }


def run_sister_engine() -> dict:
    """Run sister for latest state."""
    from cerebro_calibration import _load_episodes
    from cerebro_eval_utils import past_only_pool

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 2:
        return {"error": "Insufficient episodes", "peak_year": 0, "window_start": 0, "window_end": 0, "confidence_pct": 50}

    latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
    pool = past_only_pool(episodes, latest["saddle_year"])
    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past", "peak_year": 0, "window_start": 0, "window_end": 0, "confidence_pct": 50}

    return sister_predict(
        latest["saddle_year"],
        latest.get("position", 0),
        latest.get("velocity", 0),
        latest.get("acceleration", 0),
        pool,
    )


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_sister_engine()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Sister: peak={r.get('peak_year')}, conf={r.get('confidence_pct')}% → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
