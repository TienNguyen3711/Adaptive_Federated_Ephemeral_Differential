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
from Adaptive_Federated_Ephemeral_Differential.experiments.synthetic_poi import (
    generate_synthetic_pois_per_trajectory,
)

SEED = 42
EPSILON_BASE = 1.0
SENSITIVITY_M = 500.0
N_POI_PER_TRAJ = 6

# Matches run_real.py's alldata configuration exactly (same loader calls,
# same n_users/n_sample), so results are directly comparable to the
# existing inferred-cluster-mode numbers in Tables III/IV.
_DATASET_CONFIG = {
    "geolife": {"n_users": 182, "n_sample": 1000},
    "tdrive":  {"n_users": 500, "n_sample": 500},
    "porto":   {"n_users": 100, "n_sample": 2000},
}


def _load_flat(dataset, n_users):
    if dataset == "geolife":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import load_flat_with_timestamps
        ts = load_flat_with_timestamps(n_users=n_users, min_points=20, seed=SEED)
    elif dataset == "tdrive":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import load_flat_with_timestamps
        ts = load_flat_with_timestamps(n_taxis=n_users, min_points=20, seed=SEED)
    else:
        from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import load_flat_with_timestamps
        ts = load_flat_with_timestamps(n_trips=n_users * 50, min_points=20, seed=SEED)
    return [[(p[0], p[1]) for p in seg] for seg in ts]


def _haversine_m(p1, p2):
    R = 6_371_000
    la1, lo1 = math.radians(p1[0]), math.radians(p1[1])
    la2, lo2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


def run_dataset(dataset: str) -> dict:
    cfg = _DATASET_CONFIG[dataset]
    print(f"\nLoading {dataset} (n_users/taxis/trips-basis={cfg['n_users']}, "
          f"seed={SEED})...", flush=True)
    flat_trajs = _load_flat(dataset, cfg["n_users"])
    sample = flat_trajs[: cfg["n_sample"]]
    sample = [t for t in sample if len(t) >= 5]
    print(f"  {len(sample)} trajectories in sample", flush=True)

    cat_counts = {}
    n_pois_total = 0
    rng = np.random.default_rng(SEED)
    savings_list = []
    made_hi, made_lo = [], []
    theorem1_ok = True

    t0 = time.perf_counter()
    for traj_idx, traj in enumerate(sample):
        # Per-trajectory synthetic POIs, seeded off the dataset seed + index
        # for reproducibility without needing a dataset-wide shared pool
        # (see synthetic_poi.py docstring: a dataset-wide pool small enough
        # to keep scoring tractable is too sparse for most individual
        # trajectories to have any nearby POI at all).
        pois = generate_synthetic_pois_per_trajectory(
            traj, n_poi=N_POI_PER_TRAJ, seed=SEED * 100_000 + traj_idx
        )
        n_pois_total += len(pois)
        for p in pois:
            cat_counts[p.category] = cat_counts.get(p.category, 0) + 1

        scores = score_trajectory(traj, poi_records=pois)
        analysis = compute_budget_analysis(scores, EPSILON_BASE, SENSITIVITY_M)
        savings_list.append(analysis["savings_pct"])
        if not analysis["theorem1_holds"]:
            theorem1_ok = False

        noised = apply_adaptive_laplace(traj, scores, EPSILON_BASE, SENSITIVITY_M, rng=rng)
        hi_idx = [i for i, s in enumerate(scores) if s >= 3.0]
        lo_idx = [i for i, s in enumerate(scores) if s <= 1.05]

        def _made(idx):
            if not idx:
                return None
            return float(np.mean([_haversine_m(traj[i], noised[i]) for i in idx]))

        m_hi, m_lo = _made(hi_idx), _made(lo_idx)
        if m_hi is not None:
            made_hi.append(m_hi)
        if m_lo is not None:
            made_lo.append(m_lo)
    elapsed = time.perf_counter() - t0

    made_hi_m = float(np.mean(made_hi)) if made_hi else None
    made_lo_m = float(np.mean(made_lo)) if made_lo else None
    targeting_ratio = (made_hi_m / made_lo_m) if (made_hi_m and made_lo_m) else None

    result = {
        "dataset": dataset,
        "n_trajectories": len(sample),
        "n_poi_total": n_pois_total,
        "n_poi_per_trajectory": N_POI_PER_TRAJ,
        "poi_category_counts": cat_counts,
        "epsilon_base": EPSILON_BASE,
        "mean_budget_savings_pct": round(float(np.mean(savings_list)), 3),
        "theorem1_holds": theorem1_ok,
        "made_high_m": round(made_hi_m, 3) if made_hi_m else None,
        "made_low_m": round(made_lo_m, 3) if made_lo_m else None,
        "targeting_ratio": round(targeting_ratio, 3) if targeting_ratio else None,
        "n_high_sensitivity_points": len(made_hi),
        "n_low_sensitivity_points": len(made_lo),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  Budget savings: {result['mean_budget_savings_pct']}%  "
          f"MADE(S>=3)={result['made_high_m']}m  MADE(S=1)={result['made_low_m']}m  "
          f"TR={result['targeting_ratio']}  [{elapsed:.1f}s]", flush=True)
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  Experiment 7: SA-DP with synthetic POI-category data (RR10)")
    print("=" * 70)

    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        all_results[dataset] = run_dataset(dataset)

    out_path = os.path.join(os.path.dirname(__file__), "results", "poi_semantic_seed42.json")
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "n_poi_per_trajectory": N_POI_PER_TRAJ, "results": all_results}, f, indent=2)
    print(f"\nSaved -> {out_path}")
