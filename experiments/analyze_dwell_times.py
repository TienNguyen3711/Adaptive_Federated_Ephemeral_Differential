import json
import os
import sys

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import detect_stay_points

# Same "FULL" real-data configuration already used for Table VIII
# (run_real.py, non-alldata branch): n_users/n_taxis/n_trips = 30, seed 42.
N_USERS = 30
SEED = 42
TIME_THRESH_S = 300.0
DIST_THRESH_M = 200.0


def _load_geolife():
    from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import (
        load_flat_with_timestamps,
    )
    return load_flat_with_timestamps(n_users=N_USERS, min_points=20, seed=SEED)


def _load_tdrive():
    from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import (
        load_flat_with_timestamps,
    )
    return load_flat_with_timestamps(n_taxis=N_USERS, min_points=20, seed=SEED)


def _load_porto():
    from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import (
        load_flat_with_timestamps,
    )
    return load_flat_with_timestamps(n_trips=N_USERS * 50, min_points=20, seed=SEED)


def _stats(values):
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "n":      int(arr.size),
        "mean_s":   round(float(np.mean(arr)), 1),
        "median_s": round(float(np.median(arr)), 1),
        "p10_s":    round(float(np.percentile(arr, 10)), 1),
        "p90_s":    round(float(np.percentile(arr, 90)), 1),
        "min_s":    round(float(np.min(arr)), 1),
        "max_s":    round(float(np.max(arr)), 1),
    }


def analyze_dataset(name, trajs):
    home_durations = []
    work_durations = []
    other_durations = []
    n_with_stay_points = 0

    for traj_ts in trajs:
        if len(traj_ts) < 3:
            continue
        latlon = [(p[0], p[1]) for p in traj_ts]
        ts = [p[2] for p in traj_ts]
        stays = detect_stay_points(
            latlon, timestamps=ts,
            dist_thresh_m=DIST_THRESH_M, time_thresh_s=TIME_THRESH_S,
        )
        if not stays:
            continue
        n_with_stay_points += 1
        ranked = sorted(stays, key=lambda s: (s.visit_count, s.duration), reverse=True)
        for rank, sp in enumerate(ranked):
            if rank == 0:
                home_durations.append(sp.duration)
            elif rank == 1:
                work_durations.append(sp.duration)
            else:
                other_durations.append(sp.duration)

    hw_durations = home_durations + work_durations
    return {
        "dataset":       name,
        "n_trajectories": len(trajs),
        "n_with_stay_points": n_with_stay_points,
        "home_work_dwell_s": _stats(hw_durations),
        "home_only_dwell_s": _stats(home_durations),
        "work_only_dwell_s": _stats(work_durations),
        "other_stay_dwell_s": _stats(other_durations),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  Real dwell-time distribution: home/work-cluster stay-points")
    print("  (grounds Temporal Obfuscation's jitter_scale_base_s redesign)")
    print("=" * 70)

    loaders = [
        ("geolife", _load_geolife),
        ("tdrive",  _load_tdrive),
        ("porto",   _load_porto),
    ]

    all_results = {}
    for name, loader_fn in loaders:
        print(f"\nLoading {name} (n={N_USERS}, seed={SEED})...")
        trajs = loader_fn()
        result = analyze_dataset(name, trajs)
        all_results[name] = result

        print(f"  Trajectories: {result['n_trajectories']}  "
              f"(with stay-points: {result['n_with_stay_points']})")
        hw = result["home_work_dwell_s"]
        if hw:
            print(f"  Home/work dwell (s): n={hw['n']}  mean={hw['mean_s']}  "
                  f"median={hw['median_s']}  p10={hw['p10_s']}  p90={hw['p90_s']}  "
                  f"min={hw['min_s']}  max={hw['max_s']}")
            print(f"    -> mean {hw['mean_s']/60:.1f} min, "
                  f"median {hw['median_s']/60:.1f} min, "
                  f"p90 {hw['p90_s']/3600:.2f} hr")
        else:
            print("  No home/work stay-points detected.")

    print("\n" + "=" * 70)
    print("  Pooled home/work dwell summary (per dataset, seconds):")
    for name, r in all_results.items():
        hw = r["home_work_dwell_s"]
        if hw:
            print(f"    {name:10s}  mean={hw['mean_s']:8.1f}s  median={hw['median_s']:8.1f}s  "
                  f"p10={hw['p10_s']:8.1f}s  p90={hw['p90_s']:8.1f}s")

    out_path = os.path.join(os.path.dirname(__file__), "results", "dwell_time_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED, "n_users_per_dataset": N_USERS,
            "time_thresh_s": TIME_THRESH_S, "dist_thresh_m": DIST_THRESH_M,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")
