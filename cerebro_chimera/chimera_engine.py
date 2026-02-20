#!/usr/bin/env python3
"""
CHIMERA ENGINE — Orchestrator
============================
Load state, run reconstruction, simulation, stress, coupling, evolution,
entropy, validation, failure modes, archive. Write chimera_master.json.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_master.json"
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def _load_json(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _run_safe(module: str, func: str) -> dict:
    """Run a chimera module's main function, return result or empty dict."""
    try:
        mod = __import__(module, fromlist=[func])
        fn = getattr(mod, func)
        return fn()
    except Exception as e:
        return {"error": str(e)}


def run_chimera() -> dict:
    """Execute full CHIMERA pipeline: online learning, then reconstruction, etc."""
    DATA_DIR.mkdir(exist_ok=True)

    # 0. Online learning (weights, conformal, regime HMM, drift)
    online = {}
    try:
        from cerebro_chimera import chimera_online
        online = chimera_online.run_online_learning()
        # Append to history
        history_path = DATA_DIR / "chimera_history.jsonl"
        with open(history_path, "a") as f:
            import datetime
            record = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"), "online": online}
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        online = {"error": str(e)}

    # 1. Reconstruction
    from cerebro_chimera import chimera_reconstruction
    rec = chimera_reconstruction.run_reconstruction()
    if not rec.get("error"):
        with open(DATA_DIR / "chimera_reconstruction.json", "w") as f:
            json.dump(rec, f, indent=2)

    # 2. Forward simulation
    from cerebro_chimera import chimera_simulator
    sim = chimera_simulator.run_forward_simulation()
    if not sim.get("error"):
        with open(DATA_DIR / "chimera_forward_simulation.json", "w") as f:
            json.dump(sim, f, indent=2)

    # 3. Stressfield
    from cerebro_chimera import chimera_stressfield
    stress = chimera_stressfield.run_stressfield()
    if not stress.get("error"):
        with open(DATA_DIR / "chimera_stress_matrix.json", "w") as f:
            json.dump(stress, f, indent=2)

    # 4. Coupling
    from cerebro_chimera import chimera_coupling
    coupling = chimera_coupling.run_coupling()
    if not coupling.get("error"):
        with open(DATA_DIR / "chimera_coupling.json", "w") as f:
            json.dump(coupling, f, indent=2)

    # 5. Evolution
    from cerebro_chimera import chimera_evolution
    evolution = chimera_evolution.run_evolution()
    if not evolution.get("error"):
        with open(DATA_DIR / "chimera_evolution.json", "w") as f:
            json.dump(evolution, f, indent=2)

    # 6. Entropy
    from cerebro_chimera import chimera_entropy
    entropy = chimera_entropy.run_entropy()
    with open(DATA_DIR / "chimera_entropy.json", "w") as f:
        json.dump(entropy, f, indent=2)

    # 7. Failure modes
    from cerebro_chimera import chimera_failure_modes
    failure = chimera_failure_modes.run_failure_modes()
    with open(DATA_DIR / "chimera_failure.json", "w") as f:
        json.dump(failure, f, indent=2)

    # 8. Validation
    from cerebro_chimera import chimera_validation
    validation = chimera_validation.run_validation()
    if not validation.get("error"):
        with open(DATA_DIR / "chimera_validation.json", "w") as f:
            json.dump(validation, f, indent=2)

    # 9. Archive + store signature
    from cerebro_chimera import chimera_archive
    archive = chimera_archive.run_archive()
    with open(DATA_DIR / "chimera_archive_signature.json", "w") as f:
        json.dump(archive, f, indent=2)
    from cerebro_chimera import chimera_store
    chimera_store.write_signature()

    # Learned params, drift, conformal, next_regime for master
    params = chimera_store.load_params()
    drift_report = online.get("drift", {}) if online else {}
    cal = online.get("calibration", {}) if online else {}
    regime_hmm = online.get("regime_hmm", {}) if online else {}

    # Master summary
    master = {
        "version": 1,
        "reconstruction": {
            "mae_mean": rec.get("mae_mean"),
            "coverage_80_mean": rec.get("coverage_80_mean"),
            "n_used": rec.get("n_used"),
        } if not rec.get("error") else {"error": rec.get("error")},
        "forward_simulation": {
            "event_probability_5yr": sim.get("event_probability_5yr"),
            "entropy_time_distribution": sim.get("entropy_time_distribution"),
        } if not sim.get("error") else {"error": sim.get("error")},
        "stress_stability": stress.get("mean_stability") if not stress.get("error") else None,
        "coupling": {
            "systemic_instability_index": coupling.get("systemic_instability_index"),
        } if not coupling.get("error") else {"error": coupling.get("error")},
        "evolution": {
            "model_structure_drift": evolution.get("model_structure_drift"),
            "drift_magnitude_score": evolution.get("drift_magnitude_score"),
        } if not evolution.get("error") else {"error": evolution.get("error")},
        "entropy": entropy,
        "failure_mode": failure.get("failure_mode"),
        "failure_severity": failure.get("severity"),
        "validation_skill_score": validation.get("skill_score") if not validation.get("error") else None,
        "archive_signature": archive.get("run_signature"),
        "adaptive": {
            "vel_weight": params.get("vel_weight"),
            "acc_weight": params.get("acc_weight"),
            "tau": params.get("tau"),
            "rolling_mae": params.get("rolling_mae"),
            "n_updates": params.get("n_updates"),
            "drift_mode": drift_report.get("drift_mode"),
            "conformal_q80": cal.get("conformal_q80"),
            "coverage_last_50": cal.get("coverage_last_50"),
            "next_regime": regime_hmm.get("most_likely_next_regime"),
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(master, f, indent=2)

    return master


def _safety_gate_clamp() -> bool:
    """Return True if any safety gate triggers (confidence must be clamped to 55)."""
    failure = _load_json("chimera_failure.json")
    shift = _load_json("distribution_shift.json")
    stress = _load_json("chimera_stress_matrix.json")
    rec = _load_json("chimera_reconstruction.json")

    if failure.get("severity", 0) > 0.8:
        return True
    if shift.get("ood_level") == "SEVERE":
        return True
    coverage = rec.get("coverage_80_mean", 1.0)
    if coverage is not None and coverage < 0.6:
        return True
    if stress.get("mean_stability", 1.0) is not None and stress.get("mean_stability") < 0.5:
        return True
    return False


def main():
    master = run_chimera()
    print(f"Chimera master → {OUTPUT_PATH}")
    if _safety_gate_clamp():
        print("WARNING: STRUCTURAL INSTABILITY — Honeycomb confidence should be clamped to ≤55")
    return 0


if __name__ == "__main__":
    sys.exit(main())
