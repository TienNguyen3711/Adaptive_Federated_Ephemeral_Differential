import json
import os
import secrets
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import (
    detect_stay_points, score_trajectory,
)
from Adaptive_Federated_Ephemeral_Differential.temporal_obfuscation import (
    TemporalObfuscationParams, obfuscate_temporal, derive_place_ids_for_trajectory,
)

N_USERS = 30           # matches exp_temporal.py / run_real.py FULL-mode
BASE_SEED = 42
TIME_THRESH_S = 300.0
DIST_THRESH_M = 200.0
MAX_RELEASES = 32
RELEASE_COUNTS = [1, 2, 4, 8, 16, 32]   # report MAE of the averaged estimate at each N

PARAMS = TemporalObfuscationParams()   # current (post round-2-fix) defaults


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
    """Same population as exp_temporal.py: top-2 (home, work; S=5) stay
    points per trajectory, plus this experiment's extra ingredient --
    each record's place_ids (derive_place_ids_for_trajectory), needed for
    the place-keyed condition.
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
        place_ids = derive_place_ids_for_trajectory(
            latlon, ts, dist_thresh_m=DIST_THRESH_M, time_thresh_s=TIME_THRESH_S,
        )
        records.append({
            "ts": ts, "scores": scores, "stays": ranked[:2], "place_ids": place_ids,
        })
    return records


def run_dataset(dataset, place_key: bytes):
    print(f"\nLoading {dataset} (n={N_USERS}, seed={BASE_SEED})...", flush=True)
    trajs = _load(dataset)
    records = _ground_truth_stays(trajs)
    n_stay_points = sum(len(r["stays"]) for r in records)
    print(f"  {len(trajs)} trajectories -> {len(records)} with a qualifying "
          f"stay-point -> {n_stay_points} home/work-cluster (S=5) attack "
          f"targets, {MAX_RELEASES} simulated releases each", flush=True)

    if not records:
        return None

    true_durations = [sp.duration for r in records for sp in r["stays"]]
    mean_true = float(np.mean(true_durations))

    # per_target_estimates[condition][k] = list of per-release point estimates
    # for target k = (record index, stay index), one list of length MAX_RELEASES
    targets = [(ri, si) for ri, r in enumerate(records) for si in range(len(r["stays"]))]
    target_index = {t: k for k, t in enumerate(targets)}
    estimates_unprotected = {k: [] for k in range(len(targets))}
    estimates_protected = {k: [] for k in range(len(targets))}
    true_by_target = []
    for ri, si in targets:
        true_by_target.append(records[ri]["stays"][si].duration)

    t0 = time.perf_counter()
    for rel in range(MAX_RELEASES):
        rng_unprot = np.random.default_rng(BASE_SEED * 10_000 + rel * 2)
        rng_prot = np.random.default_rng(BASE_SEED * 10_000 + rel * 2 + 1)

        for ri, r in enumerate(records):
            ts, scores, stays, place_ids = r["ts"], r["scores"], r["stays"], r["place_ids"]

            ts_unprot = obfuscate_temporal(ts, PARAMS, rng=rng_unprot, sensitivity_scores=scores)
            ts_prot = obfuscate_temporal(
                ts, PARAMS, rng=rng_prot, sensitivity_scores=scores,
                place_ids=place_ids, place_key=place_key,
            )
            for si, sp in enumerate(stays):
                i0, i1 = sp.point_indices[0], sp.point_indices[-1]
                k = target_index[(ri, si)]
                estimates_unprotected[k].append(ts_unprot[i1] - ts_unprot[i0])
                estimates_protected[k].append(ts_prot[i1] - ts_prot[i0])
    elapsed = time.perf_counter() - t0
    print(f"  Simulated in {elapsed:.1f}s", flush=True)

    curve_unprotected = []
    curve_protected = []
    for n in RELEASE_COUNTS:
        errs_u, errs_p = [], []
        for k in range(len(targets)):
            true_d = true_by_target[k]
            avg_u = float(np.mean(estimates_unprotected[k][:n]))
            avg_p = float(np.mean(estimates_protected[k][:n]))
            errs_u.append(abs(avg_u - true_d))
            errs_p.append(abs(avg_p - true_d))
        curve_unprotected.append(round(float(np.mean(errs_u)), 1))
        curve_protected.append(round(float(np.mean(errs_p)), 1))

    result = {
        "dataset": dataset,
        "n_targets": len(targets),
        "mean_true_dwell_s": round(mean_true, 1),
        "release_counts": RELEASE_COUNTS,
        "mae_unprotected_by_n": curve_unprotected,
        "mae_protected_by_n": curve_protected,
    }
    print(f"  Unprotected MAE by N: {curve_unprotected}")
    print(f"  Place-keyed MAE by N: {curve_protected}")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  Cross-release averaging attack on Temporal Obfuscation")
    print("  (unprotected vs. place-keyed, round-3 C6)")
    print("=" * 70)

    place_key = secrets.token_bytes(32)   # simulated persisted per-user secret

    t0 = time.perf_counter()
    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        r = run_dataset(dataset, place_key)
        if r is not None:
            all_results[dataset] = r
    elapsed = time.perf_counter() - t0

    out_path = os.path.join(os.path.dirname(__file__), "results", "repeated_release_attack.json")
    with open(out_path, "w") as f:
        json.dump({
            "seed": BASE_SEED, "n_users_per_dataset": N_USERS,
            "max_releases": MAX_RELEASES, "release_counts": RELEASE_COUNTS,
            "params": vars(PARAMS),
            "results": all_results,
        }, f, indent=2)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Saved -> {out_path}")
