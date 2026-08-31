import json
import math
import os
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import (
    score_trajectory, apply_adaptive_laplace, compute_budget_analysis,
)
from Adaptive_Federated_Ephemeral_Differential.experiments.exp_sadp import (
    mean_absolute_displacement, trip_distance_error, range_query_mae,
    apply_planar_laplace,
)

SEED = 42
EPS_VALUES = [0.5, 1.0, 2.0]
SENSITIVITY_M = 500.0
N_SAMPLE = 150
MAX_TRAJ_LEN = 200


def _load(dataset):
    if dataset == "geolife":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import load_flat_with_profile_split
        return load_flat_with_profile_split(n_users=30, min_points=20, seed=SEED)
    elif dataset == "tdrive":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import load_flat_with_profile_split
        return load_flat_with_profile_split(n_taxis=30, min_points=20, seed=SEED)
    else:
        from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import load_flat_with_profile_split
        return load_flat_with_profile_split(n_taxis=30, trips_per_taxi=100, min_points=20, seed=SEED)


def run_dataset(dataset):
    print(f"\nLoading {dataset} (profile/eval split, seed={SEED})...", flush=True)
    t0 = time.perf_counter()
    segs, _ts, profiles = _load(dataset)
    print(f"  {len(segs)} eval trajectories in {time.perf_counter()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(SEED)
    sample = list(zip(segs[:N_SAMPLE], profiles[:N_SAMPLE]))

    results_per_eps = []
    for eps in EPS_VALUES:
        savings_list = []
        theorem1_ok = True
        made_sadp_hi, made_uni_hi, made_pl_hi = [], [], []
        made_sadp_lo, made_uni_lo, made_pl_lo = [], [], []
        tdist_sadp, tdist_uni, tdist_pl = [], [], []
        rq_sadp, rq_uni, rq_pl = [], [], []
        n_high_total, n_pts_total = 0, 0

        for traj, profile in sample:
            if len(traj) < 5:
                continue
            traj = traj[:MAX_TRAJ_LEN]
            scores = score_trajectory(traj, profile=profile)
            n_high_total += sum(1 for s in scores if s >= 3.0)
            n_pts_total += len(scores)
            analysis = compute_budget_analysis(scores, eps, SENSITIVITY_M)
            savings_list.append(analysis["savings_pct"])
            if not analysis["theorem1_holds"]:
                theorem1_ok = False

            n_sadp = apply_adaptive_laplace(traj, scores, eps, SENSITIVITY_M, rng=rng)
            n_uni = apply_adaptive_laplace(traj, [1.0] * len(traj), eps, SENSITIVITY_M, rng=rng)
            n_pl = apply_planar_laplace(traj, eps, SENSITIVITY_M, rng=rng)

            hi_idx = [i for i, s in enumerate(scores) if s >= 3.0]
            s1_idx = [i for i, s in enumerate(scores) if s <= 1.05]

            def _ms(orig, nois, idx):
                if not idx:
                    return 0.0
                return float(np.mean([
                    math.sqrt((orig[i][0]-nois[i][0])**2 + (orig[i][1]-nois[i][1])**2) * 111_000
                    for i in idx
                ]))

            made_sadp_hi.append(_ms(traj, n_sadp, hi_idx))
            made_uni_hi.append(_ms(traj, n_uni, hi_idx))
            made_pl_hi.append(_ms(traj, n_pl, hi_idx))
            made_sadp_lo.append(_ms(traj, n_sadp, s1_idx))
            made_uni_lo.append(_ms(traj, n_uni, s1_idx))
            made_pl_lo.append(_ms(traj, n_pl, s1_idx))

            if s1_idx:
                s1_orig = [traj[i] for i in s1_idx]
                s1_sadp = [n_sadp[i] for i in s1_idx]
                s1_uni = [n_uni[i] for i in s1_idx]
                s1_pl = [n_pl[i] for i in s1_idx]
                tdist_sadp.append(trip_distance_error(s1_orig, s1_sadp))
                tdist_uni.append(trip_distance_error(s1_orig, s1_uni))
                tdist_pl.append(trip_distance_error(s1_orig, s1_pl))
            rq_sadp.append(range_query_mae(traj, n_sadp, n_queries=20, rng=rng))
            rq_uni.append(range_query_mae(traj, n_uni, n_queries=20, rng=rng))
            rq_pl.append(range_query_mae(traj, n_pl, n_queries=20, rng=rng))

        def _m(lst):
            return round(float(np.mean(lst)) if lst else 0.0, 3)

        results_per_eps.append({
            "epsilon_base": eps,
            "n_trajectories": len(sample),
            "pct_points_high_sensitivity": round(100 * n_high_total / max(1, n_pts_total), 2),
            "mean_budget_savings_pct": _m(savings_list),
            "theorem1_holds": theorem1_ok,
            "sadp_made_high_m": _m(made_sadp_hi),
            "uniform_made_high_m": _m(made_uni_hi),
            "planar_made_high_m": _m(made_pl_hi),
            "sadp_made_s1_m": _m(made_sadp_lo),
            "uniform_made_s1_m": _m(made_uni_lo),
            "planar_made_s1_m": _m(made_pl_lo),
            "sadp_trip_dist_err_pct": _m(tdist_sadp),
            "uniform_trip_dist_err_pct": _m(tdist_uni),
            "planar_trip_dist_err_pct": _m(tdist_pl),
            "sadp_rq_mae_pct": _m(rq_sadp),
            "uniform_rq_mae_pct": _m(rq_uni),
            "planar_rq_mae_pct": _m(rq_pl),
        })
        print(f"  eps={eps}: savings={results_per_eps[-1]['mean_budget_savings_pct']:.1f}%  "
              f"high-sens pts={results_per_eps[-1]['pct_points_high_sensitivity']:.1f}%  "
              f"Theorem1={theorem1_ok}", flush=True)

    return {"dataset": dataset, "n_eval_trajectories": len(sample), "sadp_comparison": results_per_eps}


if __name__ == "__main__":
    print("=" * 70)
    print("  SA-DP budget savings / utility under historical-profile scoring (Phase 5)")
    print("=" * 70)

    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        all_results[dataset] = run_dataset(dataset)

    out_path = os.path.join(os.path.dirname(__file__), "results", "sadp_profiled_seed42.json")
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "eps_values": EPS_VALUES, "results": all_results}, f, indent=2)
    print(f"\nSaved -> {out_path}")
