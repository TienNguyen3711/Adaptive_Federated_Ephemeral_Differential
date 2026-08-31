"""
exp_mobile_benchmark.py — real on-device latency benchmark (peer-review
Phase 7 mobile-evidence item).

Run this ON THE ACTUAL PHONE, not on a desktop/laptop. It requires only the
`Adaptive_Federated_Ephemeral_Differential` package (this repo's codebase/
directory) + numpy — no GPS dataset files are needed, since it benchmarks
the *inference*-path latency on a synthetic trajectory, not accuracy.

Setup on an Android device via Termux (https://termux.dev):
    pkg install python
    pip install numpy
    # optional, for real AES-256-GCM instead of the HMAC-CTR fallback that
    # afed_pipeline.py already falls back to automatically if absent:
    pip install cryptography
    # copy (or `git clone`) this repo's codebase/ directory onto the device,
    # preserving the Adaptive_Federated_Ephemeral_Differential/ package layout
    cd codebase
    python3 -m Adaptive_Federated_Ephemeral_Differential.experiments.exp_mobile_benchmark

This times exactly the per-release inference path described in the paper's
Experiment 5 (End-to-End Pipeline Evaluation, Table tab:e2e): SA-DP scoring
+ noise, Fed-GAN inference against an already-initialised generator, DKA key
reconstruction from t shares, and AES-256-GCM encryption. Fed-GAN *training*
and DKA share *splitting* (Shamir polynomial evaluation, an enrollment-time,
not per-release, operation) are deliberately excluded, matching the paper's
own "encoding only, not federated training" scoping (see the Baseline
Architecture subsection). The Fed-GAN generator is left at its random
initialisation rather than actually trained — inference latency depends on
the matrix shapes the forward pass touches, not the weight values, and
training would need real GPS data plus far more time than this benchmark
needs.

Configuration mirrors Table tab:e2e's reference row exactly (epsilon_base =
1.0, DKA (t=3, n=5), Fed-GAN hidden_dim = 32 -- the paper's deployed
capacity, not the 512-dim capacity-ablation variant) so the reported
numbers are a direct, apples-to-apples mobile-vs-desktop comparison against
the existing 121-134 ms figure measured on an Apple M2 Pro laptop.
"""

import json
import os
import platform
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import build_profile
from Adaptive_Federated_Ephemeral_Differential.federated_gan import (
    TrajectoryGANParams, FedGANServer,
)
from Adaptive_Federated_Ephemeral_Differential.afed_pipeline import (
    AFEDConfig, AFEDPipeline, make_test_trajectory,
)

SEED = 42
N_POINTS = 100          # matches Table tab:e2e's mid-range n_points row
EPSILON_BASE = 1.0      # matches Table tab:e2e's reference epsilon
DKA_T, DKA_N = 3, 5     # matches Table tab:e2e's reference DKA config
HIDDEN_DIM = 32         # current deployed Fed-GAN capacity (Tables VI/VII)
N_WARMUP = 3
REPETITIONS = 30        # more than the desktop benchmark's reps, since
                         # mobile CPUs show higher run-to-run variance
                         # (thermal throttling, background scheduling)


def _stats(vals):
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_ms":   round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms":    round(float(np.percentile(arr, 95)), 3),
        "min_ms":    round(float(arr.min()), 3),
        "max_ms":    round(float(arr.max()), 3),
        "std_ms":    round(float(arr.std()), 3),
    }


def main():
    print("=" * 70)
    print("  AFED-PPTE on-device inference benchmark (Phase 7 mobile evidence)")
    print("=" * 70)
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor() or platform.machine()}")
    print(f"Python:    {sys.version.split()[0]}")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        print("AES-GCM:   cryptography package available (real AES-256-GCM)")
    except ImportError:
        print("AES-GCM:   cryptography package NOT found -- using the "
              "HMAC-CTR fallback already built into afed_pipeline.py")

    # ── One-time setup, NOT timed (mirrors real deployment: the Fed-GAN
    #    generator is trained once and the historical profile is built once
    #    from a user's prior sessions, then both are reused across every
    #    subsequent release) ────────────────────────────────────────────
    gan_params = TrajectoryGANParams(hidden_dim=HIDDEN_DIM, z_dim=16, seq_len=N_POINTS)
    gan_server = FedGANServer(gan_params)  # random init; latency is weight-shape-, not weight-value-, dependent

    cfg = AFEDConfig(
        epsilon_base=EPSILON_BASE, dka_t=DKA_T, dka_n=DKA_N, gan_params=gan_params,
    )
    pipeline = AFEDPipeline(cfg, pretrained_gan_server=gan_server)
    shares = pipeline.setup_dka()
    pipeline.load_shares(shares[:DKA_T])

    historical_segments = [make_test_trajectory(60, seed=100 + i) for i in range(4)]
    profile = build_profile(historical_segments)

    traj = make_test_trajectory(N_POINTS, seed=SEED)

    # ── Warm-up (import/cache/branch-prediction effects) — excluded ─────
    for _ in range(N_WARMUP):
        pipeline.process(traj, user_id="warmup", profile=profile,
                          rng=np.random.default_rng(0))

    # ── Timed repetitions ─────────────────────────────────────────────
    per_stage = {}
    wall_totals = []
    t_start = time.perf_counter()
    for rep in range(REPETITIONS):
        rep_rng = np.random.default_rng(SEED + rep)
        t0 = time.perf_counter()
        result = pipeline.process(traj, user_id=f"bench_{rep}", profile=profile, rng=rep_rng)
        wall_totals.append((time.perf_counter() - t0) * 1000)
        for k, v in result.timing_ms.items():
            per_stage.setdefault(k, []).append(v)
    elapsed_s = time.perf_counter() - t_start

    summary = {
        "device_platform":  platform.platform(),
        "device_processor": platform.processor() or platform.machine(),
        "python_version":   sys.version.split()[0],
        "n_points":         N_POINTS,
        "epsilon_base":     EPSILON_BASE,
        "dka_t":            DKA_T,
        "dka_n":            DKA_N,
        "hidden_dim":       HIDDEN_DIM,
        "repetitions":      REPETITIONS,
        "stages_ms":        {k: _stats(v) for k, v in per_stage.items()},
        "total_wall_clock_ms": _stats(wall_totals),
        "benchmark_elapsed_s": round(elapsed_s, 1),
    }

    print(f"\nCompleted {REPETITIONS} repetitions in {elapsed_s:.1f}s\n")
    print(f"{'Stage':<24}{'mean (ms)':>12}{'median (ms)':>14}{'p95 (ms)':>12}")
    print("-" * 62)
    for k, s in summary["stages_ms"].items():
        print(f"{k:<24}{s['mean_ms']:>12.3f}{s['median_ms']:>14.3f}{s['p95_ms']:>12.3f}")
    t = summary["total_wall_clock_ms"]
    print("-" * 62)
    print(f"{'TOTAL (wall clock)':<24}{t['mean_ms']:>12.3f}{t['median_ms']:>14.3f}{t['p95_ms']:>12.3f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results", "mobile_benchmark_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\nPlease report back: (1) this JSON file's contents, and (2) your exact")
    print("device model (Settings > About phone) and whether it was on battery or")
    print("charging, screen on, and otherwise idle during the run -- all of which")
    print("the paper will disclose alongside the numbers.")


if __name__ == "__main__":
    main()
