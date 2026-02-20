# CEREBRO — Formal Technical Paper Structure

No hype. Pure math and evaluation.

---

## 1. Theoretical Motivation (Second Derivative Dynamics)

- Pressure as latent state; velocity = first derivative, acceleration = second derivative
- Saddle points: critical transition zones where |v| is small and sign(a) opposes sign(v)
- Why second derivative: momentum reversal precedes regime change

---

## 2. State-Space Geometry

- Normalized position ∈ [-1, 1] (or 0–10 scale)
- Velocity and acceleration in derivative space
- Euclidean distance in weighted state space: d² = (Δp)² + w_v(Δv)² + w_a(Δa)²

---

## 3. Analogue Mapping Method

- Historical episodes: (saddle_year, event_year, position, velocity, acceleration)
- Distance-weighted analogue selection
- Peak year = saddle_year + weighted_median(Δt_i)
- Interval: weighted quantiles (p10, p90) for 80% window

---

## 4. Distance Weighting Derivation

- vel_weight, acc_weight from empirical fit or grid search
- Inverse-distance weighting: w_i = 1 / (1 + d_i)

---

## 5. Backtest Methodology

- Walk-forward: no future leakage
- Rolling-origin: full decade-by-decade validation
- Cross-national: train US → test UK; train OECD → test non-OECD; train 1900–1970 → test 1970–present

---

## 6. Walk-Forward Design

- Train window expands; test window fixed
- Metrics: mean_mae, median_error, std_error, coverage_50, coverage_80, stability_score

---

## 7. Sensitivity Analysis

- Parameter stability map: grid sweep V_THRESH, DIST_VEL_WEIGHT, DIST_ACC_WEIGHT
- mae_surface_variance, coverage_surface_variance
- Low variance = robust; high variance = fragile overfit

---

## 8. Baseline Comparison

- Linear regression on (p, v, a)
- ARIMA on Δt series
- Mean Δt analogue (naive)
- Random saddle timing
- CEREBRO must clearly beat all

---

## 9. Failure Modes

- Insufficient analogues
- Data integrity degradation (freshness, missing ratio, variance shift)
- Regime shift outside training distribution

---

## 10. Limitations

- US-centric training data; cross-national transfer requires country-level clocks
- Labeled events are hand-curated; ground truth is noisy
- No causal claims; correlational analogue mapping only
