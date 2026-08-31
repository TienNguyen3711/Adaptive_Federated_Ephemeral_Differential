import json
import math
import os
import sys

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import (
    score_trajectory, METERS_PER_DEGREE,
)

N_USERS = 30
SEED = 42
DELTA_F_M = 500.0          # sensitivity_m used throughout the paper
N_DIRECTIONS = 4
MAX_TARGET_POINTS_PER_TRAJ = 2   # cap re-scoring cost per trajectory
MAX_TRAJ_LEN = 200              # matches this paper's existing GeoLife length cap
                                 # (real trajectories average ~936 pts, up to 34k)
MAX_TRAJS_SAMPLED = 150


def _perturb(lat, lng, dist_m, bearing_rad):
    dlat = (dist_m * math.cos(bearing_rad)) / METERS_PER_DEGREE
    dlng = (dist_m * math.sin(bearing_rad)) / (
        METERS_PER_DEGREE * math.cos(math.radians(lat)) + 1e-9
    )
    return lat + dlat, lng + dlng


def _load(dataset):
    if dataset == "geolife":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import load_flat_with_profile_split
        return load_flat_with_profile_split(n_users=N_USERS, min_points=20, seed=SEED)
    elif dataset == "tdrive":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import load_flat_with_profile_split
        return load_flat_with_profile_split(n_taxis=N_USERS, min_points=20, seed=SEED)
    else:
        from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import load_flat_with_profile_split
        return load_flat_with_profile_split(n_taxis=N_USERS, trips_per_taxi=100, min_points=20, seed=SEED)


def run_dataset(dataset):
    print(f"\nLoading {dataset} (n={N_USERS}, seed={SEED})...", flush=True)
    trajs, _ts, profiles = _load(dataset)

    n_high_sensitivity_points = 0
    n_score_changed = 0
    n_crossed_threshold = 0
    max_abs_change = 0.0

    rng = np.random.default_rng(SEED)
    sample_idx = list(range(min(MAX_TRAJS_SAMPLED, len(trajs))))

    for ti in sample_idx:
        traj = trajs[ti]
        profile = profiles[ti]
        if len(traj) < 5:
            continue
        if len(traj) > MAX_TRAJ_LEN:
            traj = traj[:MAX_TRAJ_LEN]
        scores = score_trajectory(traj, profile=profile)
        high_idx = [i for i, s in enumerate(scores) if s >= 3.0]
        if not high_idx:
            continue
        targets = rng.choice(
            high_idx, size=min(MAX_TARGET_POINTS_PER_TRAJ, len(high_idx)), replace=False
        )
        for i in targets:
            n_high_sensitivity_points += 1
            lat, lng = traj[i]
            changed_here = False
            crossed_here = False
            local_max_change = 0.0
            for d in range(N_DIRECTIONS):
                bearing = 2 * math.pi * d / N_DIRECTIONS
                plat, plng = _perturb(lat, lng, DELTA_F_M, bearing)
                perturbed_traj = list(traj)
                perturbed_traj[i] = (plat, plng)
                # profile is unchanged: it was built entirely from segments
                # OTHER than `traj`, so it cannot depend on this point.
                new_scores = score_trajectory(perturbed_traj, profile=profile)
                new_s = new_scores[i]
                old_s = scores[i]
                if abs(new_s - old_s) > 1e-9:
                    changed_here = True
                    local_max_change = max(local_max_change, abs(new_s - old_s))
                if (new_s >= 3.0) != (old_s >= 3.0):
                    crossed_here = True
            if changed_here:
                n_score_changed += 1
            if crossed_here:
                n_crossed_threshold += 1
            max_abs_change = max(max_abs_change, local_max_change)

    result = {
        "dataset": dataset,
        "n_eval_trajectories": len(sample_idx),
        "n_high_sensitivity_points_tested": n_high_sensitivity_points,
        "n_score_changed_by_any_direction": n_score_changed,
        "pct_score_changed": round(100 * n_score_changed / max(1, n_high_sensitivity_points), 2),
        "n_crossed_S3_threshold": n_crossed_threshold,
        "pct_crossed_S3_threshold": round(100 * n_crossed_threshold / max(1, n_high_sensitivity_points), 2),
        "max_abs_score_change_observed": round(max_abs_change, 4),
    }
    print(f"  Tested {n_high_sensitivity_points} high-sensitivity (S>=3) points "
          f"across {len(sample_idx)} eval trajectories (each capped at "
          f"{MAX_TRAJ_LEN} points), scored against a frozen historical profile")
    print(f"  Score changed under >=1 of {N_DIRECTIONS} Delta_f-perturbations: "
          f"{n_score_changed} ({result['pct_score_changed']}%)")
    print(f"  Crossed the S>=3 threshold specifically: "
          f"{n_crossed_threshold} ({result['pct_crossed_S3_threshold']}%)")
    print(f"  Max |S_i change| observed: {max_abs_change:.4f}")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  Score stability under Delta_f-bounded perturbation (peer review C4)")
    print("=" * 70)

    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        r = run_dataset(dataset)
        all_results[dataset] = r

    out_path = os.path.join(os.path.dirname(__file__), "results", "score_stability_profiled.json")
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED, "n_users_per_dataset": N_USERS,
            "delta_f_m": DELTA_F_M, "n_directions": N_DIRECTIONS,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("(pre-Phase-5 same-trajectory-scoring results kept at score_stability.json for comparison)")
