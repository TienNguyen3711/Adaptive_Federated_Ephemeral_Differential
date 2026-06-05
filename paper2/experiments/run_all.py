"""
run_all.py — AFED-PPTE Full Experiment Suite

Runs all 6 experiments with fixed seeds for reproducibility.
Results saved to experiments/results/ as JSON files.

Seeds
-----
    SEED_MAIN = 42   (primary seed — all stochastic elements)
    SEED_ALT  = 123  (secondary seed — cross-validation / sensitivity check)

Usage
-----
    python -m paper2.experiments.run_all              # full run
    python -m paper2.experiments.run_all --quick      # reduced params for debug
    python -m paper2.experiments.run_all --seed 123   # reproduce with alt seed
"""

import argparse
import json
import os
import sys
import time

# Resolve package path when run as a script
_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

from paper2.experiments.exp_sadp   import run_experiment1, run_experiment2, run_pareto_table
from paper2.experiments.exp_dka    import run_experiment3, run_key_derivation_benchmark, security_comparison_table
from paper2.experiments.exp_fedgan import run_experiment4, run_experiment5, run_baseline_comparison, measure_communication_overhead
from paper2.experiments.exp_e2e    import run_experiment6


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sets
# ─────────────────────────────────────────────────────────────────────────────

FULL_PARAMS = {
    # SA-DP (Exp 1 & 2)
    "sadp_n_traj":        200,
    "sadp_n_points":       50,
    "sadp_eps_values":    [0.1, 0.5, 1.0, 2.0, 5.0],
    # DKA (Exp 3)
    "dka_configs":        [(2,3), (3,5), (5,10), (10,20)],
    "dka_reps":           100,
    # Fed-GAN (Exp 4 & 5)
    "fedgan_n_clients":   [3, 5, 10],
    "fedgan_sigmas":      [0.5, 1.0, 1.5, 2.0],
    "fedgan_n_rounds":     20,
    "fedgan_samples":     100,
    "fedgan_reps":          5,
    # Baseline comparison (Exp 5 extended)
    "baseline_n_rounds":   20,
    "baseline_reps":        5,
    # E2E (Exp 6)
    "e2e_lengths":        [10, 50, 100, 500, 1000],
    "e2e_eps":            [0.5, 1.0, 2.0],
    "e2e_dka":            [(3,5), (5,10)],
    "e2e_reps":            20,
}

QUICK_PARAMS = {
    "sadp_n_traj":         30,
    "sadp_n_points":       20,
    "sadp_eps_values":    [0.5, 1.0, 2.0],
    "dka_configs":        [(2,3), (3,5), (5,10)],
    "dka_reps":            10,
    "fedgan_n_clients":   [3, 5],
    "fedgan_sigmas":      [0.5, 1.0, 1.5],
    "fedgan_n_rounds":      5,
    "fedgan_samples":      50,
    "fedgan_reps":          2,
    "baseline_n_rounds":    5,
    "baseline_reps":        2,
    "e2e_lengths":        [10, 100, 500],
    "e2e_eps":            [1.0],
    "e2e_dka":            [(3,5)],
    "e2e_reps":             5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(name: str, data: dict, seed: int):
    path = os.path.join(_RESULTS_DIR, f"{name}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → saved {path}")
    return path


def _banner(msg: str):
    bar = "=" * (len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_exp1_2(p: dict, seed: int) -> dict:
    _banner("Experiment 1 & 2: SA-DP Budget + Utility")

    t0 = time.perf_counter()
    exp1 = run_experiment1(
        epsilon_values=p["sadp_eps_values"],
        n_trajectories=p["sadp_n_traj"],
        n_points=p["sadp_n_points"],
        seed=seed,
    )
    print(f"  Exp1 done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    exp2 = run_experiment2(
        epsilon_values=p["sadp_eps_values"],
        n_trajectories=p["sadp_n_traj"],
        n_points=p["sadp_n_points"],
        seed=seed,
    )
    print(f"  Exp2 done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    pareto = run_pareto_table(
        epsilon_values=p["sadp_eps_values"],
        n_trajectories=p["sadp_n_traj"],
        n_points=p["sadp_n_points"],
        seed=seed,
    )
    print(f"  Pareto table done in {time.perf_counter()-t0:.1f}s")

    # Print key results
    print("\n  Theorem 1 holds across all configs:",
          all(r["theorem1_holds"] for r in exp1))
    savings = [r["mean_savings_pct"] for r in exp1 if r["scenario"] == "mixed"]
    print(f"  Mean budget savings (mixed POI): {sum(savings)/len(savings):.1f}%")
    pareto_wins = sum(1 for r in pareto if r["semantic_pareto"])
    print(f"  Semantic Pareto wins: {pareto_wins}/{len(pareto)}")

    result = {"experiment1": exp1, "experiment2": exp2, "pareto_table": pareto,
              "seed": seed}
    return result


def run_exp3(p: dict, seed: int) -> dict:
    _banner("Experiment 3: DKA Latency & Security")

    t0 = time.perf_counter()
    exp3 = run_experiment3(configs=p["dka_configs"], repetitions=p["dka_reps"])
    print(f"  Correctness: {exp3['correctness']['all_correct']}")

    kd = run_key_derivation_benchmark(
        configs=p["dka_configs"][:3],
        repetitions=max(10, p["dka_reps"] // 10),
    )
    sec = security_comparison_table()
    print(f"  DKA E2E done in {time.perf_counter()-t0:.1f}s")

    # Print key results
    for b in exp3["timing"]:
        if b["secret_size_bytes"] == 32 and b["t"] == 3:
            print(f"  (t=3,n=5) recon={b['reconstruct']['mean_ms']:.3f}ms  "
                  f"pbkdf2={b['pbkdf2_baseline']['mean_ms']:.0f}ms  "
                  f"ratio={b['recon_vs_pbkdf2_ratio']:.4f}×")
    print(f"  Shamir (t=3,n=5) overhead vs single-key: "
          f"{next(r['overhead_vs_single'] for r in sec if r['scheme']=='shamir_t3n5'):.4f}×")

    result = {"experiment3": exp3, "key_derivation": kd,
              "security_comparison": sec, "seed": seed}
    return result


def run_exp4_5(p: dict, seed: int) -> dict:
    _banner("Experiment 4 & 5: Fed-GAN Convergence + MI Resistance")

    t0 = time.perf_counter()
    exp4 = run_experiment4(
        noise_multipliers=p["fedgan_sigmas"],
        n_clients_list=p["fedgan_n_clients"],
        n_rounds=p["fedgan_n_rounds"],
        samples_per_client=p["fedgan_samples"],
        repetitions=p["fedgan_reps"],
        seed=seed,
    )
    print(f"  Exp4 done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    exp5 = run_experiment5(
        noise_multipliers=[0.0] + p["fedgan_sigmas"],
        n_clients=5,
        n_rounds=p["fedgan_n_rounds"],
        samples_per_client=p["fedgan_samples"],
        repetitions=p["fedgan_reps"],
        seed=seed,
    )
    print(f"  Exp5 done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    baseline = run_baseline_comparison(
        n_clients=5,
        n_rounds=p["baseline_n_rounds"],
        samples_per_client=p["fedgan_samples"],
        repetitions=p["baseline_reps"],
        seed=seed,
    )
    print(f"  Baseline comparison done in {time.perf_counter()-t0:.1f}s")

    comm = measure_communication_overhead()

    # Print key results
    print("\n  MI resistance (σ=1.0 vs σ=0.0 baseline):")
    for r in exp5:
        if r["noise_multiplier"] in (0.0, 1.0):
            print(f"    σ={r['noise_multiplier']:.1f}  MI acc={r['mean_mi_accuracy']:.3f}  "
                  f"adv={r['mean_mi_advantage']:+.3f}  ε={r['mean_epsilon']:.2f}")
    print("\n  Baseline comparison (KL trip length — lower = better utility):")
    for r in baseline:
        print(f"    {r['variant']:>15}  KL={r['mean_kl_trip']:.4f}  "
              f"MI adv={r['mean_mi_advantage']:+.4f}")
    print(f"\n  NOTE: MI comparison on synthetic data limited "
          f"(see exp_fedgan.py:run_baseline_comparison docstring)")

    result = {"experiment4": exp4, "experiment5": exp5,
              "baseline_comparison": baseline, "communication": comm,
              "seed": seed}
    return result


def run_exp6(p: dict, seed: int) -> dict:
    _banner("Experiment 6: End-to-End Pipeline")

    t0 = time.perf_counter()
    exp6 = run_experiment6(
        trajectory_lengths=p["e2e_lengths"],
        epsilon_values=p["e2e_eps"],
        dka_configs=p["e2e_dka"],
        repetitions=p["e2e_reps"],
        seed=seed,
    )
    print(f"  Exp6 done in {time.perf_counter()-t0:.1f}s")

    # Print key results
    print(f"\n  Mean SA-DP budget savings: {exp6['mean_budget_savings_pct']:.1f}%")
    ref = next((r for r in exp6["afed_results"]
                if r["n_points"] == 100 and r["epsilon_base"] == 1.0
                and r["dka_t"] == 3), None)
    if ref:
        print(f"  Pipeline (L=100, ε=1.0, t=3/n=5): "
              f"total={ref.get('total_ms_mean',0):.1f}ms  "
              f"savings={ref.get('mean_budget_savings',0)*100:.1f}%")
    oh_sample = next((r for r in exp6["overhead"]
                      if r["epsilon_base"] == 1.0 and r["dka_t"] == 3
                      and r["n_points"] == 100), None)
    if oh_sample:
        print(f"  AFED overhead vs uniform baseline: {oh_sample['overhead_pct']:.1f}%")

    result = {**exp6, "seed": seed}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AFED-PPTE — Run all experiments with fixed seeds"
    )
    parser.add_argument("--quick",  action="store_true",
                        help="Use reduced parameters for a quick debug run")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Primary random seed (default: 42)")
    parser.add_argument("--skip",   nargs="*", default=[],
                        choices=["exp12", "exp3", "exp45", "exp6"],
                        help="Skip specific experiments")
    args = parser.parse_args()

    p    = QUICK_PARAMS if args.quick else FULL_PARAMS
    seed = args.seed
    mode = "QUICK" if args.quick else "FULL"

    print(f"\n{'─'*60}")
    print(f"  AFED-PPTE Experiment Suite")
    print(f"  Mode: {mode}  |  Seed: {seed}")
    print(f"  Results → {_RESULTS_DIR}")
    print(f"{'─'*60}")

    wall_start = time.perf_counter()
    all_results = {"config": {"mode": mode, "seed": seed, "params": p}}

    # ── Exp 1 & 2 ─────────────────────────────────────────────────────────
    if "exp12" not in args.skip:
        r12 = run_exp1_2(p, seed)
        all_results["exp12"] = r12
        _save("exp12_sadp", r12, seed)

    # ── Exp 3 ──────────────────────────────────────────────────────────────
    if "exp3" not in args.skip:
        r3 = run_exp3(p, seed)
        all_results["exp3"] = r3
        _save("exp3_dka", r3, seed)

    # ── Exp 4 & 5 ─────────────────────────────────────────────────────────
    if "exp45" not in args.skip:
        r45 = run_exp4_5(p, seed)
        all_results["exp45"] = r45
        _save("exp45_fedgan", r45, seed)

    # ── Exp 6 ──────────────────────────────────────────────────────────────
    if "exp6" not in args.skip:
        r6 = run_exp6(p, seed)
        all_results["exp6"] = r6
        _save("exp6_e2e", r6, seed)

    # ── Master summary ─────────────────────────────────────────────────────
    all_results["total_wall_time_s"] = round(time.perf_counter() - wall_start, 1)
    _save("all_results", all_results, seed)

    print(f"\n{'─'*60}")
    print(f"  All experiments complete.")
    print(f"  Total wall time: {all_results['total_wall_time_s']:.0f}s")
    print(f"  Results saved in: {_RESULTS_DIR}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
