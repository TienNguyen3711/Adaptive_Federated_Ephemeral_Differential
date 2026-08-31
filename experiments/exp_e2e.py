import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Resolve package path when run as a script
_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from paper2.adaptive_dp import score_trajectory, compute_budget_analysis
from paper2.afed_pipeline import (
    AFEDConfig,
    AFEDPipeline,
    ProcessingResult,
    make_test_trajectory,
    run_e2e_benchmark,
)


# ─────────────────────────────────────────────────────────────────────────────
# Uniform-DP baseline timing (uniform Laplace + single-key PBKDF2 + AES)
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_dp_baseline_timing(n_points: int, epsilon: float, repetitions: int,
                                 sensitivity_m: float = 500.0, seed: int = 0) -> Dict:
    """
    Simulate uniform-DP baseline: uniform Laplace (Layer 4) + single-key PBKDF2
    derivation (Layer 6) + AES-256-GCM encryption.

    Layers 1–3 and 5 are common to both baseline and AFED-PPTE and therefore
    cancel in the comparison; only the differentiating components are timed.
    """
    import hashlib
    import secrets
    import struct

    dp_times, kd_times, enc_times = [], [], []
    rng = np.random.default_rng(seed)

    for _ in range(repetitions):
        # Uniform Laplace (same scale for every point)
        t0 = time.perf_counter()
        scale = sensitivity_m / (epsilon * 111_000)
        _ = rng.laplace(0, scale, (n_points, 2))
        dp_times.append((time.perf_counter() - t0) * 1000)

        # PBKDF2 key derivation (Layer 6 — single pre-shared key)
        t0 = time.perf_counter()
        salt = secrets.token_bytes(32)
        k_store = b"UNIFORM_BASELINE_KEY_PLACEHOLDER"
        key = hashlib.pbkdf2_hmac("sha256", k_store + b"user", salt, 600_000, dklen=32)
        kd_times.append((time.perf_counter() - t0) * 1000)

        # AES-GCM (simulated via PBKDF2 stream — same as AFED fallback)
        t0 = time.perf_counter()
        payload = struct.pack(">I", n_points) + bytes(n_points * 16)
        nonce = secrets.token_bytes(12)
        stream = hashlib.pbkdf2_hmac("sha256", key + nonce, b"afed-ctr", 1, dklen=len(payload))
        _ = bytes(a ^ b for a, b in zip(payload, stream))
        enc_times.append((time.perf_counter() - t0) * 1000)

    def _m(lst):
        return round(sum(lst) / len(lst), 3)

    return {
        "dp_ms":       _m(dp_times),
        "kd_ms":       _m(kd_times),
        "encrypt_ms":  _m(enc_times),
        "total_ms":    _m([dp_times[i] + kd_times[i] + enc_times[i]
                           for i in range(repetitions)]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6: end-to-end benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment6(
    trajectory_lengths: List[int] = None,
    epsilon_values: List[float] = None,
    dka_configs: List[Tuple[int, int]] = None,
    repetitions: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Full end-to-end AFED-PPTE vs uniform-DP baseline comparison.

    Returns dict with:
        "afed_results"      : per-config timing and budget from AFED-PPTE
        "uniform_baseline"  : per (length, epsilon) uniform-DP baseline timing
        "overhead"          : AFED overhead ratio vs uniform baseline
        "budget_savings"    : mean SA-DP budget savings across all configs
    """
    if trajectory_lengths is None:
        trajectory_lengths = [10, 50, 100, 500, 1000]
    if epsilon_values is None:
        epsilon_values = [0.5, 1.0, 2.0]
    if dka_configs is None:
        dka_configs = [(3, 5), (5, 10)]

    afed_results = run_e2e_benchmark(
        trajectory_lengths=trajectory_lengths,
        epsilon_values=epsilon_values,
        dka_configs=dka_configs,
        repetitions=repetitions,
        seed=seed,
    )

    # Uniform-DP baseline (single pre-shared key, no Shamir overhead)
    uniform_baseline = []
    for n_pts in trajectory_lengths:
        for eps in epsilon_values:
            b = _uniform_dp_baseline_timing(n_pts, eps, min(repetitions, 10), seed=seed)
            b["n_points"] = n_pts
            b["epsilon_base"] = eps
            uniform_baseline.append(b)

    # Compute overhead ratios (AFED total vs uniform baseline total)
    overhead = []
    baseline_map = {(b["n_points"], b["epsilon_base"]): b["total_ms"]
                    for b in uniform_baseline}

    for r in afed_results:
        base_ms = baseline_map.get((r["n_points"], r["epsilon_base"]), None)
        afed_ms = r.get("total_ms_mean", None)
        if base_ms and afed_ms:
            overhead.append({
                "n_points":          r["n_points"],
                "epsilon_base":      r["epsilon_base"],
                "dka_t":             r["dka_t"],
                "dka_n":             r["dka_n"],
                "afed_total_ms":     afed_ms,
                "baseline_total_ms": base_ms,
                "overhead_ratio":    round(afed_ms / max(1e-9, base_ms), 4),
                "overhead_pct":      round((afed_ms / max(1e-9, base_ms) - 1.0) * 100, 2),
            })

    mean_savings = round(
        float(np.mean([r["mean_budget_savings"] for r in afed_results])) * 100, 2
    )

    return {
        "afed_results":            afed_results,
        "uniform_baseline":        uniform_baseline,
        "overhead":                overhead,
        "mean_budget_savings_pct": mean_savings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_overhead_table(overhead: List[Dict]):
    print(f"\n  {'L':>6}  {'ε':>5}  {'(t,n)':>7}  {'AFED(ms)':>10}  "
          f"{'Base(ms)':>10}  {'overhead%':>10}")
    print("  " + "-" * 58)
    for r in overhead:
        print(f"  {r['n_points']:>6}  {r['epsilon_base']:>5.1f}  "
              f"({r['dka_t']},{r['dka_n']}):>7  "
              f"{r['afed_total_ms']:>10.2f}  "
              f"{r['ppte_total_ms']:>10.2f}  "
              f"{r['overhead_pct']:>9.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AFED-PPTE Experiment 6: End-to-End Pipeline Evaluation"
    )
    parser.add_argument("--reps",    type=int, default=20,
                        help="Repetitions per configuration (default: 20)")
    parser.add_argument("--quick",   action="store_true",
                        help="Quick mode: smaller parameter grid + fewer reps")
    parser.add_argument("--output",  default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    if args.quick:
        traj_lens   = [10, 100, 500]
        eps_vals    = [1.0]
        dka_cfgs    = [(3, 5)]
        reps        = 5
    else:
        traj_lens   = [10, 50, 100, 500, 1000]
        eps_vals    = [0.5, 1.0, 2.0]
        dka_cfgs    = [(3, 5), (5, 10)]
        reps        = args.reps

    print("=== Experiment 6: End-to-End AFED-PPTE Pipeline ===")
    t0     = time.perf_counter()
    result = run_experiment6(
        trajectory_lengths=traj_lens,
        epsilon_values=eps_vals,
        dka_configs=dka_cfgs,
        repetitions=reps,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s")

    print(f"\n  Mean SA-DP budget savings: {result['mean_budget_savings_pct']:.1f}%")

    print("\n  Per-stage timing (ε=1.0, t=3/n=5, mean ms):")
    relevant = [r for r in result["afed_results"]
                if r["epsilon_base"] == 1.0 and r["dka_t"] == 3]
    if relevant:
        print(f"  {'L':>6}  {'score':>8}  {'noise':>8}  {'dka':>8}  "
              f"{'encrypt':>8}  {'total':>8}  {'savings%':>9}")
        print("  " + "-" * 68)
        for r in relevant:
            print(f"  {r['n_points']:>6}  "
                  f"{r.get('sadp_score_ms_mean', 0):>8.2f}  "
                  f"{r.get('sadp_noise_ms_mean', 0):>8.2f}  "
                  f"{r.get('dka_derive_ms_mean', 0):>8.2f}  "
                  f"{r.get('encrypt_ms_mean', 0):>8.2f}  "
                  f"{r.get('total_ms_mean', 0):>8.2f}  "
                  f"{r.get('mean_budget_savings', 0) * 100:>8.1f}%")

    print("\n  AFED vs uniform-DP overhead (ε=1.0, t=3/n=5):")
    relevant_oh = [r for r in result["overhead"]
                   if r["epsilon_base"] == 1.0 and r["dka_t"] == 3]
    if relevant_oh:
        print(f"  {'L':>6}  {'AFED(ms)':>10}  {'Base(ms)':>10}  {'overhead%':>10}")
        print("  " + "-" * 44)
        for r in relevant_oh:
            print(f"  {r['n_points']:>6}  "
                  f"{r['afed_total_ms']:>10.2f}  "
                  f"{r['baseline_total_ms']:>10.2f}  "
                  f"{r['overhead_pct']:>9.1f}%")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        summary = {
            "mean_budget_savings_pct": result["mean_budget_savings_pct"],
            "overhead_sample":         result["overhead"][:4] if result["overhead"] else [],
            "uniform_baseline_sample": result["uniform_baseline"][:3],
        }
        print("\n--- JSON summary ---")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
