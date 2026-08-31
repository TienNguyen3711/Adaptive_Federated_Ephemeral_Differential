import json
import os
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

# Two-tailed 95% critical values for Student's t at small df (df = n-1).
# Same table as run_fedgan_variance.py / run_fedgan_mi_all_variance.py.
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
    30: 2.042,
}

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import (
    detect_stay_points, score_trajectory,
)
from Adaptive_Federated_Ephemeral_Differential.temporal_obfuscation import (
    TemporalObfuscationParams, obfuscate_temporal,
)

N_USERS = 30          # matches run_real.py FULL-mode / analyze_dwell_times.py
BASE_SEED = 42
TIME_THRESH_S = 300.0
DIST_THRESH_M = 200.0
REPETITIONS = 8        # matches this codebase's established n=8 CI convention


OLD_PARAMS = TemporalObfuscationParams(
    jitter_scale_base_s=30.0, pause_low_s=5.0, pause_high_s=90.0,
)
NEW_PARAMS = TemporalObfuscationParams()  # redesigned defaults


def _ci95(vals):
    arr = np.array(vals, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    t_crit = _T_CRIT_975.get(n - 1, 1.96)
    return mean, std, t_crit * se


def _load(dataset):
    if dataset == "geolife":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import (
            load_flat_with_timestamps,
        )
        return load_flat_with_timestamps(n_users=N_USERS, min_points=20, seed=BASE_SEED)
    elif dataset == "tdrive":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import (
            load_flat_with_timestamps,
        )
        return load_flat_with_timestamps(n_taxis=N_USERS, min_points=20, seed=BASE_SEED)
    else:
        from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import (
            load_flat_with_timestamps,
        )
        return load_flat_with_timestamps(n_trips=N_USERS * 50, min_points=20, seed=BASE_SEED)


def _ground_truth_stays(trajs):
    """For each trajectory, detect real stay points and rank them exactly as
    infer_home_work() does. Returns one record per trajectory that has at
    least one qualifying stay point, holding the trajectory's timestamps,
    SA-DP scores, and its top-2 ranked (home, work; S=5) stay points -- the
    population Temporal Obfuscation targets. Grouped by trajectory (rather
    than flattened per stay-point) so each trajectory is obfuscated only
    once per repetition below, not once per stay-point.
    """
    records = []
    for latlon_ts in trajs:
        if len(latlon_ts) < 3:
            continue
        latlon = [(p[0], p[1]) for p in latlon_ts]
        ts = [p[2] for p in latlon_ts]
        stays = detect_stay_points(
            latlon, timestamps=ts,
            dist_thresh_m=DIST_THRESH_M, time_thresh_s=TIME_THRESH_S,
        )
        if not stays:
            continue
        ranked = sorted(stays, key=lambda s: (s.visit_count, s.duration), reverse=True)
        scores = score_trajectory(
            latlon, timestamps=ts,
            dist_thresh_m=DIST_THRESH_M, time_thresh_s=TIME_THRESH_S,
        )
        records.append({"ts": ts, "scores": scores, "stays": ranked[:2]})
    return records


def run_dataset(dataset):
    print(f"\nLoading {dataset} (n={N_USERS}, seed={BASE_SEED})...", flush=True)
    trajs = _load(dataset)
    records = _ground_truth_stays(trajs)
    n_stay_points = sum(len(r["stays"]) for r in records)
    print(f"  {len(trajs)} trajectories -> {len(records)} with a qualifying "
          f"stay-point -> {n_stay_points} home/work-cluster (S=5) attack "
          f"targets", flush=True)

    if not records:
        return None

    true_durations = [sp.duration for r in records for sp in r["stays"]]
    mean_true = float(np.mean(true_durations))

    before_mae_per_rep = []
    after_mae_per_rep = []

    for rep in range(REPETITIONS):
        rng_before = np.random.default_rng(BASE_SEED * 1000 + rep * 2)
        rng_after = np.random.default_rng(BASE_SEED * 1000 + rep * 2 + 1)

        before_errors = []
        after_errors = []
        for r in records:
            ts, scores, stays = r["ts"], r["scores"], r["stays"]

            ts_before = obfuscate_temporal(ts, OLD_PARAMS, rng=rng_before)
            ts_after = obfuscate_temporal(
                ts, NEW_PARAMS, rng=rng_after, sensitivity_scores=scores
            )
            for sp in stays:
                i0, i1 = sp.point_indices[0], sp.point_indices[-1]
                before_errors.append(abs((ts_before[i1] - ts_before[i0]) - sp.duration))
                after_errors.append(abs((ts_after[i1] - ts_after[i0]) - sp.duration))

        before_mae_per_rep.append(float(np.mean(before_errors)))
        after_mae_per_rep.append(float(np.mean(after_errors)))

    before_mean, before_std, before_ci = _ci95(before_mae_per_rep)
    after_mean, after_std, after_ci = _ci95(after_mae_per_rep)

    result = {
        "dataset": dataset,
        "n_stay_points": n_stay_points,
        "mean_true_dwell_s": round(mean_true, 1),
        "before": {
            "mae_s": round(before_mean, 1), "std_s": round(before_std, 1),
            "ci95_s": round(before_ci, 1),
            "mae_pct_of_true": round(100 * before_mean / mean_true, 2),
        },
        "after": {
            "mae_s": round(after_mean, 1), "std_s": round(after_std, 1),
            "ci95_s": round(after_ci, 1),
            "mae_pct_of_true": round(100 * after_mean / mean_true, 2),
        },
        "improvement_factor": round(after_mean / before_mean, 2) if before_mean > 0 else None,
    }
    print(f"  Before: MAE={result['before']['mae_s']}s ± {result['before']['ci95_s']}s "
          f"({result['before']['mae_pct_of_true']}% of mean true dwell "
          f"{result['mean_true_dwell_s']}s)")
    print(f"  After:  MAE={result['after']['mae_s']}s ± {result['after']['ci95_s']}s "
          f"({result['after']['mae_pct_of_true']}% of mean true dwell "
          f"{result['mean_true_dwell_s']}s)")
    print(f"  Improvement: {result['improvement_factor']}x MAE increase")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  Temporal Obfuscation dwell-duration recovery attack")
    print("  (before: original defaults; after: RR2/RR5 redesign)")
    print("=" * 70)

    t0 = time.perf_counter()
    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        r = run_dataset(dataset)
        if r is not None:
            all_results[dataset] = r
    elapsed = time.perf_counter() - t0

    out_path = os.path.join(os.path.dirname(__file__), "results", "temporal_attack.json")
    with open(out_path, "w") as f:
        json.dump({
            "seed": BASE_SEED, "n_users_per_dataset": N_USERS,
            "repetitions": REPETITIONS,
            "old_params": vars(OLD_PARAMS), "new_params": vars(NEW_PARAMS),
            "results": all_results,
        }, f, indent=2)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Saved -> {out_path}")
