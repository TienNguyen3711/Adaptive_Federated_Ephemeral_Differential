import json
import math
import os
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
    30: 2.042,
}

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import (
    build_profile, score_trajectory, apply_adaptive_laplace, METERS_PER_DEGREE,
)

SEED = 42
N_USERS = 40
MIN_BG_POINTS = 30
MIN_TARGET_POINTS = 20
MAX_BG_POINTS = 600     # caps grid-frequency clustering + build_profile cost;
MAX_TARGET_POINTS = 400  # caps score_trajectory/apply_adaptive_laplace cost
                          # (O(n*profile_size) per trajectory) -- matches this
                          # codebase's established per-trajectory length-cap
                          # convention (e.g. run_real.py's max_traj_len=200)
GRID_CELL_M = 150.0
MIN_HOME_WORK_SEP_M = 500.0
REID_RADIUS_M = 200.0
EPSILON_BASE = 1.0
SENSITIVITY_M = 500.0
CONDITIONS = ["raw", "uniform_dp", "sadp"]


def _ci95(vals):
    arr = np.array(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    t_crit = _T_CRIT_975.get(n - 1, 1.96)
    return mean, std, t_crit * se


def _haversine_m(p1, p2):
    R = 6_371_000
    la1, lo1 = math.radians(p1[0]), math.radians(p1[1])
    la2, lo2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


def _grid_key(lat, lng, ref_lat, cell_m=GRID_CELL_M):
    lat_m = lat * METERS_PER_DEGREE
    lng_m = lng * METERS_PER_DEGREE * math.cos(math.radians(ref_lat))
    return (round(lat_m / cell_m), round(lng_m / cell_m))


def _cell_center(key, ref_lat, cell_m=GRID_CELL_M):
    iy, ix = key
    lat = (iy * cell_m) / METERS_PER_DEGREE
    lng = (ix * cell_m) / (METERS_PER_DEGREE * math.cos(math.radians(ref_lat)) + 1e-9)
    return lat, lng


def infer_home_work_by_frequency(points, ref_lat):
    """Independent re-id attack primitive -- see module docstring for why
    this is not circular with SA-DP's own scoring machinery."""
    if not points:
        return None, None
    counts = {}
    for lat, lng in points:
        k = _grid_key(lat, lng, ref_lat)
        counts[k] = counts.get(k, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    home = _cell_center(ranked[0][0], ref_lat)
    work = None
    for k, _ in ranked[1:]:
        cand = _cell_center(k, ref_lat)
        if _haversine_m(home, cand) >= MIN_HOME_WORK_SEP_M:
            work = cand
            break
    return home, work


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset user loading: profile (raw, held-out) / target (release surface)
# ─────────────────────────────────────────────────────────────────────────────

def _load_users(dataset, n_users, seed):
    import random
    rng = random.Random(seed)
    users = []

    if dataset == "geolife":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import (
            iter_user_trajectories, _DATA_ROOT,
        )
        all_ids = sorted(os.listdir(_DATA_ROOT))
        selected = rng.sample(all_ids, min(n_users, len(all_ids)))
        gen = iter_user_trajectories(user_ids=selected, min_points=20, gap_seconds=1800)
    elif dataset == "tdrive":
        import glob
        from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import (
            iter_taxi_trajectories, _DATA_ROOT,
        )
        all_files = sorted(glob.glob(os.path.join(_DATA_ROOT, "*.txt")))
        all_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
        selected = rng.sample(all_ids, min(n_users, len(all_ids)))
        gen = iter_taxi_trajectories(taxi_ids=selected, min_points=20, gap_seconds=1800)
    else:
        from collections import defaultdict
        from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import (
            iter_trips, _DATA_ROOT,
        )
        taxi_trips = defaultdict(list)
        max_read = n_users * 40
        for _, taxi_id, pts in iter_trips(_DATA_ROOT, n_trips=max_read, min_points=20):
            if len(taxi_trips[taxi_id]) < 40:
                taxi_trips[taxi_id].append(pts)
        eligible = [t for t, trips in taxi_trips.items() if len(trips) >= 2]
        selected = rng.sample(eligible, min(n_users, len(eligible)))
        gen = ((tid, taxi_trips[tid]) for tid in selected)

    for uid, segs in gen:
        if len(segs) < 2:
            continue
        n_profile = max(1, len(segs) // 2)
        profile_segs, target_segs = segs[:n_profile], segs[n_profile:]
        if not target_segs:
            continue
        bg_pts = [p for seg in profile_segs for p in seg][:MAX_BG_POINTS]
        tgt_pts = [p for seg in target_segs for p in seg][:MAX_TARGET_POINTS]
        if len(bg_pts) < MIN_BG_POINTS or len(tgt_pts) < MIN_TARGET_POINTS:
            continue
        profile_segs_capped, remaining = [], MAX_BG_POINTS
        for seg in profile_segs:
            if remaining <= 0:
                break
            profile_segs_capped.append(seg[:remaining])
            remaining -= len(seg)
        profile = build_profile(profile_segs_capped)
        users.append({
            "user_id": uid, "bg_pts": bg_pts, "target_pts": tgt_pts, "profile": profile,
        })
    return users


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset):
    print(f"\nLoading {dataset} (n_users<={N_USERS}, seed={SEED})...", flush=True)
    users = _load_users(dataset, N_USERS, SEED)
    rng = np.random.default_rng(SEED)
    for u in users:
        ref_lat = float(np.mean([p[0] for p in u["target_pts"]]))
        u["ref_lat"] = ref_lat
        u["true_home"], u["true_work"] = infer_home_work_by_frequency(u["target_pts"], ref_lat)
    users = [u for u in users if u["true_home"] is not None]
    print(f"  {len(users)} users with a valid ground-truth home "
          f"(from >= {MIN_TARGET_POINTS} release-surface pts each)", flush=True)
    if len(users) < 5:
        return None

    guesses = {c: {} for c in CONDITIONS}   # condition -> user_id -> (lat,lng)
    made_ext_hi, made_ext_lo = [], []

    t0 = time.perf_counter()
    for u in users:
        ref_lat = u["ref_lat"]
        tgt = u["target_pts"]

        raw_home, _ = infer_home_work_by_frequency(tgt, ref_lat)

        scores_uniform = [1.0] * len(tgt)
        noised_uniform = apply_adaptive_laplace(
            tgt, scores_uniform, EPSILON_BASE, SENSITIVITY_M, rng=rng)
        uni_home, _ = infer_home_work_by_frequency(noised_uniform, ref_lat)

        scores_sadp = score_trajectory(tgt, profile=u["profile"])
        noised_sadp = apply_adaptive_laplace(
            tgt, scores_sadp, EPSILON_BASE, SENSITIVITY_M, rng=rng)
        sadp_home, _ = infer_home_work_by_frequency(noised_sadp, ref_lat)

        if raw_home is not None:
            guesses["raw"][u["user_id"]] = raw_home
        if uni_home is not None:
            guesses["uniform_dp"][u["user_id"]] = uni_home
        if sadp_home is not None:
            guesses["sadp"][u["user_id"]] = sadp_home

        for orig, noi in zip(tgt, noised_sadp):
            d_home = _haversine_m(orig, u["true_home"])
            is_ext_sensitive = d_home <= REID_RADIUS_M
            if u["true_work"] is not None:
                d_work = _haversine_m(orig, u["true_work"])
                is_ext_sensitive = is_ext_sensitive or (d_work <= REID_RADIUS_M)
            disp = _haversine_m(orig, noi)
            (made_ext_hi if is_ext_sensitive else made_ext_lo).append(disp)
    elapsed = time.perf_counter() - t0

    true_homes = {u["user_id"]: u["true_home"] for u in users}

    cond_results = {}
    for c in CONDITIONS:
        disp_own = []
        hits = 0
        top1_correct = 0
        n_c = 0
        for uid, guess in guesses[c].items():
            d_own = _haversine_m(guess, true_homes[uid])
            disp_own.append(d_own)
            if d_own <= REID_RADIUS_M:
                hits += 1
            best_uid, best_d = None, math.inf
            for other_uid, other_home in true_homes.items():
                d = _haversine_m(guess, other_home)
                if d < best_d:
                    best_d, best_uid = d, other_uid
            if best_uid == uid:
                top1_correct += 1
            n_c += 1
        mean_d, std_d, ci_d = _ci95(disp_own)
        arr = np.array(disp_own, dtype=float)
        median_d = float(np.median(arr)) if len(arr) else 0.0
        q25 = float(np.percentile(arr, 25)) if len(arr) else 0.0
        q75 = float(np.percentile(arr, 75)) if len(arr) else 0.0
        cond_results[c] = {
            "n_users": n_c,
            "median_displacement_m": round(median_d, 1),
            "iqr_displacement_m": [round(q25, 1), round(q75, 1)],
            "mean_displacement_m": round(mean_d, 1),
            "ci95_displacement_m": round(ci_d, 1),
            "reid_success_rate_pct": round(100 * hits / max(1, n_c), 1),
            "population_top1_linkage_acc_pct": round(100 * top1_correct / max(1, n_c), 1),
        }

    made_hi_m = float(np.mean(made_ext_hi)) if made_ext_hi else None
    made_lo_m = float(np.mean(made_ext_lo)) if made_ext_lo else None
    ext_targeting_ratio = (made_hi_m / made_lo_m) if (made_hi_m and made_lo_m) else None

    result = {
        "dataset": dataset,
        "n_users": len(users),
        "elapsed_s": round(elapsed, 1),
        "conditions": cond_results,
        "external_targeting_ratio": {
            "made_near_true_home_work_m": round(made_hi_m, 2) if made_hi_m else None,
            "made_elsewhere_m": round(made_lo_m, 2) if made_lo_m else None,
            "n_near": len(made_ext_hi),
            "n_elsewhere": len(made_ext_lo),
            "targeting_ratio": round(ext_targeting_ratio, 3) if ext_targeting_ratio else None,
        },
    }

    print(f"  [{elapsed:.1f}s] n_users={len(users)}")
    for c in CONDITIONS:
        r = cond_results[c]
        print(f"    {c:<12} median_disp={r['median_displacement_m']:.0f}m "
              f"(IQR {r['iqr_displacement_m'][0]:.0f}-{r['iqr_displacement_m'][1]:.0f})  "
              f"mean={r['mean_displacement_m']:.0f}+/-{r['ci95_displacement_m']:.0f}m  "
              f"reid_success={r['reid_success_rate_pct']:.1f}%  "
              f"top1_linkage={r['population_top1_linkage_acc_pct']:.1f}%")
    print(f"  External Targeting Ratio (SA-DP, ground-truth-anchored): "
          f"{result['external_targeting_ratio']['targeting_ratio']}  "
          f"(MADE near-home/work={result['external_targeting_ratio']['made_near_true_home_work_m']}m, "
          f"elsewhere={result['external_targeting_ratio']['made_elsewhere_m']}m, "
          f"n={result['external_targeting_ratio']['n_near']}/{result['external_targeting_ratio']['n_elsewhere']})")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  External re-identification attack (Phase 7, grid-frequency,")
    print("  independent of SA-DP's own scoring) + externally-anchored")
    print("  Targeting Ratio")
    print("=" * 70)

    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        r = run_dataset(dataset)
        if r is not None:
            all_results[dataset] = r

    out_path = os.path.join(os.path.dirname(__file__), "results", "external_reid_seed42.json")
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED, "n_users_target": N_USERS,
            "grid_cell_m": GRID_CELL_M, "reid_radius_m": REID_RADIUS_M,
            "epsilon_base": EPSILON_BASE, "sensitivity_m": SENSITIVITY_M,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")
