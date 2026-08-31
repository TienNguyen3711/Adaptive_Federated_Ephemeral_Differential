"""
exp_poi_semantic_real_osm.py — Experiment 6 re-run against REAL
OpenStreetMap POI data (peer-review Phase 7 evidence item), closing the
gap the paper's own Limitation 1 and exp_poi_semantic.py's docstring
already disclose: the existing POI-category scoring path (tab:poi_semantic)
was only ever exercised against a disclosed *synthetic* POI layer
(6 category-labelled points per trajectory, placed near the trajectory's
own points but carrying no real-world semantic meaning), never against
genuine semantic ground truth.

Uses the real Overpass API POI tiles cached by build_osm_poi_cache.py:
Beijing (39.6-40.3N, 115.9-117.0E, covering both GeoLife and T-Drive --
both are Beijing-based datasets, so one real tile serves both) and Porto
(40.95-41.35N, 8.75-8.45W). Mirrors exp_poi_semantic.py's methodology
exactly (same seed, epsilon, sample sizes, MADE/Targeting-Ratio
definitions) so the two are directly, honestly comparable.
"""
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
    POIRecord, METERS_PER_DEGREE,
)

SEED = 42
EPSILON_BASE = 1.0
SENSITIVITY_M = 500.0
NEARBY_RADIUS_M = 300.0   # bounding-box prefilter margin around each
                           # trajectory; comfortably exceeds every OSM
                           # category's own proximity radius_m (80-180m,
                           # see build_osm_poi_cache.py), so no POI that
                           # could affect scoring is dropped by the filter

# Matches exp_poi_semantic.py's own _DATASET_CONFIG exactly, so results
# are directly comparable to the synthetic-POI table.
_DATASET_CONFIG = {
    "geolife": {"n_users": 182, "n_sample": 1000, "osm_city": "beijing"},
    "tdrive":  {"n_users": 500, "n_sample": 500, "osm_city": "beijing"},
    "porto":   {"n_users": 100, "n_sample": 2000, "osm_city": "porto"},
}

_OSM_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "osm_cache")


def _load_osm_pois(city: str):
    path = os.path.join(_OSM_CACHE_DIR, f"osm_pois_{city}.json")
    with open(path) as f:
        raw = json.load(f)
    return [POIRecord(lat=r["lat"], lng=r["lng"], category=r["category"],
                       name=r["name"], radius_m=r["radius_m"]) for r in raw]


def _nearby_pois(traj, all_pois, radius_m=NEARBY_RADIUS_M):
    lats = [p[0] for p in traj]
    lngs = [p[1] for p in traj]
    ref_lat = sum(lats) / len(lats)
    lat_pad = radius_m / METERS_PER_DEGREE
    lng_pad = radius_m / (METERS_PER_DEGREE * math.cos(math.radians(ref_lat)) + 1e-9)
    lat_min, lat_max = min(lats) - lat_pad, max(lats) + lat_pad
    lng_min, lng_max = min(lngs) - lng_pad, max(lngs) + lng_pad
    return [p for p in all_pois
            if lat_min <= p.lat <= lat_max and lng_min <= p.lng <= lng_max]


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
    all_pois = _load_osm_pois(cfg["osm_city"])
    print(f"\nLoading {dataset} (n_users/taxis/trips-basis={cfg['n_users']}, "
          f"seed={SEED}, real OSM city={cfg['osm_city']}, "
          f"{len(all_pois)} real POIs)...", flush=True)
    flat_trajs = _load_flat(dataset, cfg["n_users"])
    sample = flat_trajs[: cfg["n_sample"]]
    sample = [t for t in sample if len(t) >= 5]
    print(f"  {len(sample)} trajectories in sample", flush=True)

    cat_counts = {}
    n_pois_total = 0
    n_trajs_with_poi = 0
    rng = np.random.default_rng(SEED)
    savings_list = []
    made_hi, made_lo = [], []
    theorem1_ok = True

    t0 = time.perf_counter()
    for traj in sample:
        pois = _nearby_pois(traj, all_pois)
        n_pois_total += len(pois)
        if pois:
            n_trajs_with_poi += 1
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
        "osm_city": cfg["osm_city"],
        "n_real_pois_in_city": len(all_pois),
        "n_trajectories": len(sample),
        "n_trajectories_with_nearby_poi": n_trajs_with_poi,
        "pct_trajectories_with_nearby_poi": round(100 * n_trajs_with_poi / max(1, len(sample)), 1),
        "n_poi_total_matched": n_pois_total,
        "mean_poi_per_trajectory": round(n_pois_total / max(1, len(sample)), 2),
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
    print(f"  {n_trajs_with_poi}/{len(sample)} trajectories ({result['pct_trajectories_with_nearby_poi']}%) "
          f"have >=1 real POI within {NEARBY_RADIUS_M:.0f}m", flush=True)
    print(f"  Budget savings: {result['mean_budget_savings_pct']}%  "
          f"MADE(S>=3)={result['made_high_m']}m  MADE(S=1)={result['made_low_m']}m  "
          f"TR={result['targeting_ratio']}  [{elapsed:.1f}s]", flush=True)
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  Experiment 6 (real OSM POI evidence, Phase 7)")
    print("=" * 70)

    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        all_results[dataset] = run_dataset(dataset)

    out_path = os.path.join(os.path.dirname(__file__), "results", "poi_semantic_real_osm_seed42.json")
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "nearby_radius_m": NEARBY_RADIUS_M, "results": all_results}, f, indent=2)
    print(f"\nSaved -> {out_path}")
