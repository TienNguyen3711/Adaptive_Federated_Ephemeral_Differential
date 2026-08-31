import json
import math
import os
import random
import sys
from typing import List, Tuple

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import (
    METERS_PER_DEGREE, S_MAX, S_MIN,
    score_trajectory, apply_adaptive_laplace,
    compute_budget_analysis, POIRecord,
    _laplace,
)
from Adaptive_Federated_Ephemeral_Differential.experiments.exp_sadp import (
    mean_absolute_displacement, range_query_mae,
    apply_planar_laplace,
)
from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import load_flat


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_m(p1, p2):
    R = 6_371_000
    la1, lo1 = math.radians(p1[0]), math.radians(p1[1])
    la2, lo2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
# DA-DP — Density-Adaptive DP (density-based adaptive budget)
# ─────────────────────────────────────────────────────────────────────────────

def da_dp_scores(
    trajectory: List[Tuple[float, float]],
    window: int = 5,
) -> List[float]:
    """
    Compute per-point sensitivity proxy from local trajectory density.

    Dense regions (small inter-point distances → potential stay locations)
    get higher sensitivity scores; sparse segments (fast movement) get lower.
    This is analogous to SA-DP but uses structural signals instead of POI
    data. Author-constructed; not a reproduction of any prior paper.
    """
    n = len(trajectory)
    if n == 0:
        return []

    # Compute step distances
    dists = [0.0] + [
        _haversine_m(trajectory[i-1], trajectory[i]) for i in range(1, n)
    ]

    # Local mean distance in a sliding window → inversely proportional to density
    scores = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        local_dists = [dists[j] for j in range(lo, hi) if dists[j] > 0]
        if not local_dists:
            scores.append(float(S_MAX))
            continue

        mean_dist = sum(local_dists) / len(local_dists)
        # Normalise: very small mean distance → high sensitivity score
        # threshold: 50 m → S=5; 500 m+ → S=1
        s_raw = 5.0 * math.exp(-mean_dist / 100.0)
        scores.append(max(float(S_MIN), min(float(S_MAX), s_raw)))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# SS-DP — Segment-State DP (DWELL/TRANSIT/TRANSITION heuristic)
# ─────────────────────────────────────────────────────────────────────────────

# State types with sensitivity weights (author-defined, not from any cited paper)
_SS_DP_STATE_WEIGHT = {
    "dwell":      5.0,   # stopped/slow → high sensitivity → low ε
    "transition": 3.0,   # accelerating/decelerating → medium sensitivity
    "transit":    1.0,   # fast moving → low sensitivity → high ε
}


def _classify_states(
    trajectory: List[Tuple[float, float]],
    slow_thresh_m: float = 50.0,   # dist < this → DWELL
    fast_thresh_m: float = 200.0,  # dist > this → TRANSIT
) -> List[str]:
    """
    Classify each trajectory point as DWELL / TRANSIT / TRANSITION
    based on inter-point distance. Author-constructed structural
    heuristic — not a reproduction of any specific prior paper's
    mobility-state detection mechanism.
    """
    n = len(trajectory)
    if n == 0:
        return []

    # Step distances (metres)
    step = [0.0] + [
        _haversine_m(trajectory[i-1], trajectory[i]) for i in range(1, n)
    ]

    states = []
    for i in range(n):
        d = step[i]
        if d < slow_thresh_m:
            states.append("dwell")
        elif d > fast_thresh_m:
            states.append("transit")
        else:
            states.append("transition")

    # Propagate dwell segments: if a point is surrounded by dwells, it is a dwell
    smoothed = list(states)
    for i in range(1, n - 1):
        if states[i-1] == "dwell" and states[i+1] == "dwell":
            smoothed[i] = "dwell"

    return smoothed


def ss_dp_scores(
    trajectory: List[Tuple[float, float]],
) -> List[float]:
    """
    Assign sensitivity scores based on DWELL/TRANSIT/TRANSITION state
    classification. DWELL → S=5, TRANSITION → S=3, TRANSIT → S=1.
    Author-constructed heuristic representing the "mobility-state-adaptive"
    design point in the comparison, not a reproduction of any cited work.
    """
    states = _classify_states(trajectory)
    return [float(_SS_DP_STATE_WEIGHT[s]) for s in states]


def apply_baseline_laplace(
    trajectory: List[Tuple[float, float]],
    scores: List[float],
    epsilon_base: float,
    sensitivity_m: float = 500.0,
    rng=None,
) -> List[Tuple[float, float]]:
    """Apply Laplace noise with DA-DP/SS-DP scores (same mechanism as SA-DP)."""
    if rng is None:
        rng = np.random.default_rng(0)
    delta_f = sensitivity_m / METERS_PER_DEGREE
    result = []
    for (lat, lng), s in zip(trajectory, scores):
        eps_i = epsilon_base / max(s, 1e-9)
        scale = delta_f / eps_i
        result.append((
            round(lat + _laplace(scale, rng), 7),
            round(lng + _laplace(scale, rng), 7),
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MADE split by sensitivity level  (for Table 4 format)
# ─────────────────────────────────────────────────────────────────────────────

def made_by_level(
    original: List[Tuple[float, float]],
    noisy: List[Tuple[float, float]],
    scores: List[float],
) -> dict:
    """
    Return MADE in metres, split exactly as in paper Table 4:
      high  → S ≥ 3  (medical/religious/home/work/legal)
      low   → S ≤ 1.5  (transit/other — exactly S=1 in practice)
    Points with 1.5 < S < 3 (commercial, S≈2) are excluded from both buckets
    so that MADE(low) corresponds purely to transit waypoints where
    both Uniform and SA-DP apply ε_base/1 (identical noise scale).
    """
    high_disps, low_disps = [], []
    for (la, lo), (ln, ln2), s in zip(original, noisy, scores):
        d = _haversine_m((la, lo), (ln, ln2))
        if s >= 3.0:
            high_disps.append(d)
        elif s <= 1.5:
            low_disps.append(d)
        # 1.5 < s < 3  → skip (commercial tier, not reported separately)
    return {
        "made_high": round(sum(high_disps) / len(high_disps), 1) if high_disps else 0.0,
        "made_low":  round(sum(low_disps)  / len(low_disps),  1) if low_disps  else 0.0,
        "n_high": len(high_disps),
        "n_low":  len(low_disps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison  (runs on GeoLife flat trajectories)
# ─────────────────────────────────────────────────────────────────────────────

def run_taldp_comparison(
    epsilon_base: float = 1.0,
    n_trajs: int = 500,
    seed: int = 42,
):
    rng = random.Random(seed)
    print(f"Loading GeoLife flat trajectories (n≤{n_trajs}, ε_base={epsilon_base})...")
    trajs = load_flat(n_users=177, min_points=20, seed=seed)
    trajs = [t[:200] for t in trajs if len(t) >= 20]  # cap at 200 pts
    rng.shuffle(trajs)
    trajs = trajs[:n_trajs]
    print(f"  Loaded {len(trajs)} trajectories.")

    results = {m: {"made_high": [], "made_low": [],
                   "made_high_own": [], "made_low_own": [],
                   "savings_pct": [], "rq_mae": []}
               for m in ("sa_dp", "uniform", "planar", "da_dp", "ss_dp")}

    for traj in trajs:
        # ── SA-DP scores (POI-based, using inferred home/work) ──────────────
        sadp_scores = score_trajectory(traj)

        # ── DA-DP scores (structural density) ────────────────────────────────
        da_dp_score_list = da_dp_scores(traj)

        # ── SS-DP scores (segment state) ──────────────────────────────────────
        ss_dp_score_list = ss_dp_scores(traj)

        # ── Uniform scores (all S=1) ─────────────────────────────────────────
        uniform_scores = [1.0] * len(traj)

        # ── Apply noise ──────────────────────────────────────────────────────
        sadp_noisy  = apply_adaptive_laplace(traj, sadp_scores,       epsilon_base)
        da_dp_noisy = apply_baseline_laplace(traj, da_dp_score_list,  epsilon_base)
        ss_dp_noisy = apply_baseline_laplace(traj, ss_dp_score_list,  epsilon_base)
        uniform_noisy = apply_adaptive_laplace(traj, uniform_scores,  epsilon_base)
        planar_noisy  = apply_planar_laplace(traj, epsilon_base)

        # ── Budget savings ───────────────────────────────────────────────────
        ba_sadp  = compute_budget_analysis(sadp_scores,       epsilon_base)
        ba_da_dp = compute_budget_analysis(da_dp_score_list,  epsilon_base)
        ba_ss_dp = compute_budget_analysis(ss_dp_score_list,  epsilon_base)

        # ── MADE by level ────────────────────────────────────────────────────
        # Report MADE under two sensitivity partitions:
        #  (1) SA-DP's labels, as a common cross-method reference point;
        #  (2) each method's OWN labels (scores_for_made), so the Targeting
        #      Ratio is not always evaluated against SA-DP's own partition —
        #      the ratio DA-DP/SS-DP are otherwise implicitly judged on is
        #      close to tautological for SA-DP, since it's exactly what
        #      SA-DP was designed to optimise.
        for method, noisy, ba, scores_for_made in [
            ("sa_dp",  sadp_noisy,    ba_sadp,   sadp_scores),
            ("uniform",  uniform_noisy, None,    uniform_scores),
            ("planar",   planar_noisy,  None,    uniform_scores),
            ("da_dp",  da_dp_noisy,   ba_da_dp,  da_dp_score_list),
            ("ss_dp",  ss_dp_noisy,   ba_ss_dp,  ss_dp_score_list),
        ]:
            m_ref = made_by_level(traj, noisy, sadp_scores)
            results[method]["made_high"].append(m_ref["made_high"])
            results[method]["made_low"].append(m_ref["made_low"])

            m_own = made_by_level(traj, noisy, scores_for_made)
            results[method]["made_high_own"].append(m_own["made_high"])
            results[method]["made_low_own"].append(m_own["made_low"])

            if ba:
                results[method]["savings_pct"].append(ba["savings_pct"])
            else:
                results[method]["savings_pct"].append(0.0)
            # range_query_mae() already returns a percentage (×100 internally);
            # do not scale again here.
            results[method]["rq_mae"].append(
                range_query_mae(traj, noisy)
            )

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary = {}
    for method, vals in results.items():
        made_high_m     = round(sum(vals["made_high"])     / len(vals["made_high"]),     1)
        made_low_m      = round(sum(vals["made_low"])      / len(vals["made_low"]),      1)
        made_high_own_m = round(sum(vals["made_high_own"]) / len(vals["made_high_own"]), 1)
        made_low_own_m  = round(sum(vals["made_low_own"])  / len(vals["made_low_own"]),  1)
        summary[method] = {
            "made_high_m":       made_high_m,
            "made_low_m":        made_low_m,
            "targeting_ratio":   round(made_high_m / made_low_m, 2) if made_low_m else None,
            # Own-partition MADE/TR: evaluated against this method's OWN
            # sensitivity labels rather than always SA-DP's (fixes the
            # circular evaluation flagged in peer review R5). Not meaningful
            # for uniform/planar, which have no per-point sensitivity
            # partition of their own (every point is nominally S=1).
            "made_high_own_m":   made_high_own_m,
            "made_low_own_m":    made_low_own_m,
            "targeting_ratio_own": (round(made_high_own_m / made_low_own_m, 2)
                                     if made_low_own_m and method in ("sa_dp", "da_dp", "ss_dp")
                                     else None),
            "savings_pct":       round(sum(vals["savings_pct"]) / len(vals["savings_pct"]), 2),
            "rq_mae_pct":        round(sum(vals["rq_mae"])    / len(vals["rq_mae"]),    2),
            "n_trajs":           len(trajs),
        }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  DA-DP / SS-DP vs SA-DP vs Uniform Laplace — GeoLife, ε_base=1.0")
    print("=" * 65)

    summary = run_taldp_comparison(epsilon_base=1.0, n_trajs=500, seed=42)

    out_path = os.path.join(
        os.path.dirname(__file__), "results", "taldp_comparison_seed42.json"
    )
    with open(out_path, "w") as f:
        json.dump({"epsilon_base": 1.0, "seed": 42, "results": summary}, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Print Table 4 extension ───────────────────────────────────────────────
    print("\n=== TABLE 4 EXTENSION — DA-DP/SS-DP Comparison (GeoLife, ε_base=1.0) ===")
    print(f"{'Method':<20}  {'MADE(S≥3) m':>11}  {'MADE(S=1) m':>11}  "
          f"{'Savings %':>9}  {'RQ-MAE %':>8}")
    print("-" * 65)
    order = [
        ("sa_dp",    "SA-DP (ours)"),
        ("ss_dp",    "SS-DP (segment)"),
        ("da_dp",    "DA-DP (density)"),
        ("uniform",  "Uniform Laplace"),
        ("planar",   "Planar Laplace"),
    ]
    for key, label in order:
        r = summary[key]
        print(f"{label:<20}  {r['made_high_m']:>11.1f}  {r['made_low_m']:>11.1f}  "
              f"{r['savings_pct']:>9.2f}  {r['rq_mae_pct']:>8.2f}")

    print("\n=== TARGETING RATIO: SA-DP-referenced vs. each method's OWN partition ===")
    print(f"{'Method':<20}  {'TR (SA-DP labels)':>18}  {'TR (own labels)':>16}")
    print("-" * 60)
    for key, label in order:
        r = summary[key]
        own = r["targeting_ratio_own"]
        own_str = f"{own:.2f}" if own is not None else "n/a"
        print(f"{label:<20}  {r['targeting_ratio']:>18.2f}  {own_str:>16}")

    print("\nInterpretation:")
    print("  MADE(S≥3): displacement at high-sensitivity locations (higher = more private)")
    print("  MADE(S=1): displacement at transit waypoints (lower = better utility)")
    print("  Savings %: budget saved vs uniform baseline (higher = more efficient)")
    print("  RQ-MAE  : range query accuracy (lower = better utility)")
    print("  TR (SA-DP labels): Targeting Ratio using SA-DP's sensitivity partition")
    print("                     for all methods (cross-method reference point).")
    print("  TR (own labels)  : Targeting Ratio using each method's OWN sensitivity")
    print("                     partition (n/a for uniform/planar, which have none).")
