"""
exp_dka.py — Experiment 3: Decentralised Key Agreement (DKA) Benchmark

AFED-PPTE

Experiment 3: DKA Latency and Overhead
    - Measure split / reconstruct / serialise / deserialise latency
    - Vary (t, n) configurations: (2,3), (3,5), (5,10), (10,20)
    - Compare DKA overhead against PBKDF2 key derivation baseline
    - Verify correctness: reconstructed secret == original for all configs
    - Report: mean latency (ms), overhead ratio vs PBKDF2

Run:
    python -m paper2.experiments.exp_dka
    python -m paper2.experiments.exp_dka --output results/exp3.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Resolve package path when run as a script
_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from paper2.decentralised_key import (
    benchmark_reconstruction,
    split_secret,
    reconstruct_secret,
    serialise_share,
    deserialise_share,
    DKAKeyManager,
    SECRET_MAX_BYTES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_correctness(configs: List[Tuple[int, int]], repetitions: int = 10) -> Dict:
    """
    Verify that split → reconstruct recovers the exact secret for all configs,
    and that any subset of size < t yields a different value.

    Returns dict with per-config pass/fail and overall result.
    """
    import secrets as _secrets

    results = {}
    all_pass = True

    for (t, n) in configs:
        passes = 0
        threshold_security_holds = 0

        for _ in range(repetitions):
            secret = _secrets.token_bytes(32)
            shares = split_secret(secret, t, n)

            # Reconstruct from exactly t shares (choose a non-trivial subset)
            chosen = shares[:t]
            recovered = reconstruct_secret(chosen)
            if recovered == secret:
                passes += 1

            # Any t-1 shares should yield wrong value (probabilistic check)
            if t > 1:
                partial = reconstruct_secret(shares[:t - 1])
                if partial != secret:
                    threshold_security_holds += 1
            else:
                threshold_security_holds += 1  # t=1 is trivially secure

        results[f"t{t}_n{n}"] = {
            "t": t, "n": n,
            "reconstruct_correct": passes == repetitions,
            "threshold_security_holds": threshold_security_holds == repetitions,
            "passes": passes,
            "repetitions": repetitions,
        }
        if passes != repetitions:
            all_pass = False

    results["all_correct"] = all_pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Latency benchmark across (t, n) configurations
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment3(
    configs: List[Tuple[int, int]] = None,
    secret_sizes: List[int] = None,
    repetitions: int = 100,
) -> Dict:
    """
    Benchmark DKA split/reconstruct/serialise/deserialise latency.

    Returns dict with correctness results and per-config timing stats.
    """
    if configs is None:
        configs = [(2, 3), (3, 5), (5, 10), (10, 20)]
    if secret_sizes is None:
        secret_sizes = [16, 32, 64]

    # Correctness first
    correctness = verify_correctness(configs, repetitions=20)

    timing_results = []
    for (t, n) in configs:
        for size in secret_sizes:
            bm = benchmark_reconstruction(t, n, secret_size=size, repetitions=repetitions)
            # Add overhead ratio vs PBKDF2
            pbkdf2_mean = bm["pbkdf2_baseline"]["mean_ms"]
            recon_mean  = bm["reconstruct"]["mean_ms"]
            bm["recon_vs_pbkdf2_ratio"] = round(recon_mean / max(1e-9, pbkdf2_mean), 4)
            bm["total_dka_ms"] = round(
                bm["split"]["mean_ms"]
                + bm["reconstruct"]["mean_ms"]
                + bm["serialise"]["mean_ms"]
                + bm["deserialise"]["mean_ms"],
                3,
            )
            timing_results.append(bm)

    return {
        "correctness": correctness,
        "timing":      timing_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Key derivation benchmark (DKAKeyManager round-trip)
# ─────────────────────────────────────────────────────────────────────────────

def run_key_derivation_benchmark(
    configs: List[Tuple[int, int]] = None,
    repetitions: int = 50,
) -> List[Dict]:
    """
    Measure end-to-end DKAKeyManager latency:
      initialise → serialise shares → deserialise → reconstruct → derive_key
    """
    import secrets as _secrets

    if configs is None:
        configs = [(2, 3), (3, 5), (5, 10)]

    results = []
    for (t, n) in configs:
        total_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()

            mgr    = DKAKeyManager.initialise(t=t, n=n)
            blobs  = [serialise_share(idx, val) for idx, val in mgr.shares[:t]]
            shares = [deserialise_share(b) for b in blobs]
            mgr2   = DKAKeyManager.from_shares(shares)
            salt   = _secrets.token_bytes(32)
            _key   = mgr2.derive_encryption_key("user_bench", salt)

            total_times.append((time.perf_counter() - t0) * 1000)

        # Communication cost: n shares distributed, t shares collected
        share_bytes      = 68          # fixed per serialise_share()
        dist_bytes_total = n * share_bytes   # setup: distribute n shares
        recon_bytes      = t * share_bytes   # per-op: collect t shares

        results.append({
            "t": t, "n": n,
            "mean_e2e_ms":          round(sum(total_times) / repetitions, 3),
            "min_e2e_ms":           round(min(total_times), 3),
            "max_e2e_ms":           round(max(total_times), 3),
            "share_size_bytes":     share_bytes,
            "setup_comm_bytes":     dist_bytes_total,     # distribute n shares
            "recon_comm_bytes":     recon_bytes,          # collect t shares
            "setup_comm_kb":        round(dist_bytes_total / 1024, 3),
            "recon_comm_kb":        round(recon_bytes / 1024, 3),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Security comparison table (threat model analysis)
# ─────────────────────────────────────────────────────────────────────────────

def security_comparison_table() -> List[Dict]:
    """
    Qualitative + quantitative security comparison across key management schemes.

    Schemes compared:
      single_key   — server holds K in one place (env variable / HSM)
      pbkdf2_only  — password-derived key, single device
      shamir_t3n5  — Shamir (t=3, n=5), this work
      shamir_t5n10 — Shamir (t=5, n=10), higher threshold

    Dimensions:
      compromise_threshold : how many devices must be compromised to recover K
      info_theoretic       : is security unconditional (no crypto hardness assumed)?
      single_point_failure : does 1 compromised entity expose K?
      key_recovery_possible: can K be recovered with < t shares?
      overhead_ms          : latency overhead vs single key (from benchmark)
    """
    import secrets as _sec

    # Run quick latency benchmark for comparison
    def _lat(t, n, reps=20):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mgr = DKAKeyManager.initialise(t=t, n=n)
            blobs = [serialise_share(idx, val) for idx, val in mgr.shares[:t]]
            shares = [deserialise_share(b) for b in blobs]
            mgr2 = DKAKeyManager.from_shares(shares)
            mgr2.derive_encryption_key("u", _sec.token_bytes(32))
            times.append((time.perf_counter() - t0) * 1000)
        return round(sum(times) / reps, 2)

    # Single-key baseline latency (just PBKDF2)
    import hashlib
    single_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        salt = _sec.token_bytes(32)
        hashlib.pbkdf2_hmac("sha256", b"key", salt, 600_000, dklen=32)
        single_times.append((time.perf_counter() - t0) * 1000)
    single_lat = round(sum(single_times) / len(single_times), 2)

    lat_t3n5  = _lat(3, 5)
    lat_t5n10 = _lat(5, 10)

    table = [
        {
            "scheme":                "single_key",
            "description":          "K stored on one server / env var",
            "compromise_threshold": 1,
            "info_theoretic":       False,
            "single_point_failure": True,
            "key_recovery_lt_t":    True,   # trivially — 1 device = full key
            "total_latency_ms":     single_lat,
            "overhead_vs_single":   1.0,
            "comm_bytes_recon":     0,
        },
        {
            "scheme":                "pbkdf2_only",
            "description":          "Password-derived key, single holder",
            "compromise_threshold": 1,
            "info_theoretic":       False,
            "single_point_failure": True,
            "key_recovery_lt_t":    True,
            "total_latency_ms":     single_lat,
            "overhead_vs_single":   1.0,
            "comm_bytes_recon":     0,
        },
        {
            "scheme":                "shamir_t3n5",
            "description":          "Shamir (t=3, n=5) — this work",
            "compromise_threshold": 3,
            "info_theoretic":       True,
            "single_point_failure": False,
            "key_recovery_lt_t":    False,  # Theorem 3: I(K; <3 shares) = 0
            "total_latency_ms":     lat_t3n5,
            "overhead_vs_single":   round(lat_t3n5 / max(1e-9, single_lat), 4),
            "comm_bytes_recon":     3 * 68,   # 3 shares × 68 bytes
        },
        {
            "scheme":                "shamir_t5n10",
            "description":          "Shamir (t=5, n=10) — higher security",
            "compromise_threshold": 5,
            "info_theoretic":       True,
            "single_point_failure": False,
            "key_recovery_lt_t":    False,
            "total_latency_ms":     lat_t5n10,
            "overhead_vs_single":   round(lat_t5n10 / max(1e-9, single_lat), 4),
            "comm_bytes_recon":     5 * 68,
        },
    ]
    return table


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AFED-PPTE Experiment 3: DKA Benchmark")
    parser.add_argument("--reps",   type=int, default=100,
                        help="Repetitions per configuration")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    configs = [(2, 3), (3, 5), (5, 10), (10, 20)]

    print("=== Experiment 3: DKA Correctness ===")
    result = run_experiment3(configs=configs, repetitions=args.reps)

    print(f"  All correct: {result['correctness']['all_correct']}")
    for k, v in result["correctness"].items():
        if k == "all_correct":
            continue
        print(f"  {k}: reconstruct_ok={v['reconstruct_correct']}  "
              f"threshold_security={v['threshold_security_holds']}")

    print("\n=== Experiment 3: DKA Timing (32-byte secret) ===")
    for bm in result["timing"]:
        if bm["secret_size_bytes"] != 32:
            continue
        print(f"  (t={bm['t']}, n={bm['n']}) | "
              f"split={bm['split']['mean_ms']:.3f}ms  "
              f"recon={bm['reconstruct']['mean_ms']:.3f}ms  "
              f"ser={bm['serialise']['mean_ms']:.3f}ms  "
              f"total_dka={bm['total_dka_ms']:.3f}ms  "
              f"pbkdf2={bm['pbkdf2_baseline']['mean_ms']:.1f}ms  "
              f"ratio={bm['recon_vs_pbkdf2_ratio']:.4f}×")

    print("\n=== Experiment 3: DKAKeyManager E2E Latency + Communication ===")
    kd = run_key_derivation_benchmark(configs[:3], repetitions=max(10, args.reps // 10))
    for r in kd:
        print(f"  (t={r['t']}, n={r['n']}) "
              f"mean={r['mean_e2e_ms']:.1f}ms  "
              f"setup={r['setup_comm_kb']:.3f}KB  "
              f"recon={r['recon_comm_kb']:.3f}KB")

    print("\n=== Security Comparison Table ===")
    sec = security_comparison_table()
    print(f"  {'Scheme':>15}  {'Thresh':>7}  {'Info-theoretic':>15}  "
          f"{'SPF':>5}  {'Latency(ms)':>12}  {'Overhead':>9}  {'Comm(B)':>8}")
    print("  " + "-" * 80)
    for r in sec:
        print(f"  {r['scheme']:>15}  "
              f"{r['compromise_threshold']:>7}  "
              f"{'Yes' if r['info_theoretic'] else 'No':>15}  "
              f"{'Yes' if r['single_point_failure'] else 'No':>5}  "
              f"{r['total_latency_ms']:>12.1f}  "
              f"{r['overhead_vs_single']:>9.4f}×  "
              f"{r['comm_bytes_recon']:>8}")

    output = {"experiment3": result, "key_derivation": kd, "security_comparison": sec}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        print("\n--- JSON summary ---")
        print(json.dumps({"security_comparison": sec}, indent=2))


if __name__ == "__main__":
    main()
