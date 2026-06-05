"""
run_real.py — AFED-PPTE Experiments on Real Datasets

Runs all 6 experiments on GeoLife / T-Drive / Porto Taxi datasets.
Results saved to experiments/results/ as JSON files.

Usage
-----
    python -m paper2.experiments.run_real --dataset geolife
    python -m paper2.experiments.run_real --dataset tdrive
    python -m paper2.experiments.run_real --dataset porto
    python -m paper2.experiments.run_real --dataset all --seed 42
    python -m paper2.experiments.run_real --dataset geolife --quick
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

from paper2.adaptive_dp import (
    score_trajectory, apply_adaptive_laplace, compute_budget_analysis,
    POIRecord,
)
from paper2.decentralised_key import DKAKeyManager, benchmark_reconstruction
from paper2.federated_gan import (
    TrajectoryGANParams, FedGANServer, FedGANClient,
    MembershipInferenceAttack, DPSGDTracker,
)
from paper2.experiments.exp_sadp import (
    mean_absolute_displacement, trip_distance_error,
    range_query_mae, apply_planar_laplace, run_pareto_table,
)
from paper2.experiments.exp_fedgan import (
    measure_generation_quality, _kl_divergence,
)
from paper2.experiments.exp_dka import security_comparison_table


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset(dataset: str, n_users: int, window_size: int,
                  stride: int, seed: int):
    """
    Load real trajectories for federated GAN experiments.
    Returns (client_ids, raw_windows, normalised_arrays).

    raw_windows[i]  : list of (lat,lon) tuples for client i (variable-length)
    norm_arrays[i]  : np.ndarray (N, window_size*2) normalised to [-1,1]
    """
    if dataset == "geolife":
        from paper2.data.loader_geolife import load_as_client_datasets, load_flat
        ids, windows_per_client = load_as_client_datasets(
            n_users=n_users, window_size=window_size, stride=stride, seed=seed
        )
        flat_trajs = load_flat(n_users=n_users, min_points=20, seed=seed)
    elif dataset == "tdrive":
        from paper2.data.loader_tdrive import load_as_client_datasets, load_flat
        ids, windows_per_client = load_as_client_datasets(
            n_taxis=n_users, window_size=window_size, stride=stride, seed=seed
        )
        flat_trajs = load_flat(n_taxis=n_users, min_points=20, seed=seed)
    elif dataset == "porto":
        from paper2.data.loader_porto import load_as_client_datasets, load_flat
        ids, windows_per_client = load_as_client_datasets(
            n_taxis=n_users, window_size=window_size, stride=stride,
            trips_per_taxi=100, seed=seed,
        )
        flat_trajs = load_flat(n_trips=n_users * 50, min_points=20, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Normalise windows globally per client for Fed-GAN input
    norm_arrays = []
    for windows in windows_per_client:
        arr = np.array([[p[0] for pt in w for p in [pt]] +
                        [p[1] for pt in w for p in [pt]]
                        for w in windows], dtype=np.float64)
        # Reshape: (N, window_size, 2) then flatten to (N, window_size*2)
        pts  = np.array([[[p[0], p[1]] for p in w] for w in windows])  # (N,T,2)
        flat = pts.reshape(len(windows), -1)   # (N, T*2)
        lo   = flat.min(axis=1, keepdims=True)
        hi   = flat.max(axis=1, keepdims=True)
        norm = 2.0 * (flat - lo) / (hi - lo + 1e-8) - 1.0
        norm_arrays.append(norm)

    return ids, windows_per_client, flat_trajs, norm_arrays


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 & 2 on real data — SA-DP
# ─────────────────────────────────────────────────────────────────────────────

def run_sadp_real(
    flat_trajs: List[List[Tuple[float, float]]],
    epsilon_values: List[float],
    sensitivity_m: float,
    n_sample: int,
    seed: int,
) -> Dict:
    """
    Run SA-DP budget analysis and utility comparison on real trajectories.
    Compares SA-DP vs Uniform vs Planar Laplace.
    """
    rng = np.random.default_rng(seed)
    sample = flat_trajs[:n_sample]

    results_per_eps = []
    for eps in epsilon_values:
        savings_list = []
        theorem1_ok  = True
        made_sadp_hi, made_uni_hi, made_pl_hi = [], [], []
        made_sadp_lo, made_uni_lo, made_pl_lo = [], [], []
        tdist_sadp,   tdist_uni,   tdist_pl   = [], [], []
        rq_sadp,      rq_uni,      rq_pl      = [], [], []

        for traj in sample:
            if len(traj) < 5:
                continue
            scores  = score_trajectory(traj)
            analysis = compute_budget_analysis(scores, eps, sensitivity_m)
            savings_list.append(analysis["savings_pct"])
            if not analysis["theorem1_holds"]:
                theorem1_ok = False

            n_sadp = apply_adaptive_laplace(traj, scores, eps, sensitivity_m)
            n_uni  = apply_adaptive_laplace(traj, [1.0]*len(traj), eps, sensitivity_m)
            n_pl   = apply_planar_laplace(traj, eps, sensitivity_m, rng=rng)

            hi_idx = [i for i, s in enumerate(scores) if s >= 3.0]
            s1_idx = [i for i, s in enumerate(scores) if s <= 1.05]

            def _ms(orig, nois, idx):
                if not idx:
                    return 0.0
                return float(np.mean([
                    math.sqrt((orig[i][0]-nois[i][0])**2 +
                              (orig[i][1]-nois[i][1])**2) * 111_000
                    for i in idx
                ]))

            made_sadp_hi.append(_ms(traj, n_sadp, hi_idx))
            made_uni_hi.append( _ms(traj, n_uni,  hi_idx))
            made_pl_hi.append(  _ms(traj, n_pl,   hi_idx))
            made_sadp_lo.append(_ms(traj, n_sadp, s1_idx))
            made_uni_lo.append( _ms(traj, n_uni,  s1_idx))
            made_pl_lo.append(  _ms(traj, n_pl,   s1_idx))

            # Trip distance error: only on S=1 points (low-sensitivity sub-path)
            # — full-path trip distance is meaningless for SA-DP because large
            #   noise at high-sensitivity points is intentional by design.
            if s1_idx:
                s1_orig  = [traj[i]   for i in s1_idx]
                s1_sadp  = [n_sadp[i] for i in s1_idx]
                s1_uni   = [n_uni[i]  for i in s1_idx]
                s1_pl    = [n_pl[i]   for i in s1_idx]
                tdist_sadp.append(trip_distance_error(s1_orig, s1_sadp))
                tdist_uni.append( trip_distance_error(s1_orig, s1_uni))
                tdist_pl.append(  trip_distance_error(s1_orig, s1_pl))
            rq_sadp.append(range_query_mae(traj, n_sadp, n_queries=20, rng=rng))
            rq_uni.append( range_query_mae(traj, n_uni,  n_queries=20, rng=rng))
            rq_pl.append(  range_query_mae(traj, n_pl,   n_queries=20, rng=rng))

        def _m(lst): return round(float(np.mean(lst)) if lst else 0.0, 3)

        results_per_eps.append({
            "epsilon_base":              eps,
            "n_trajectories":            len(sample),
            "mean_budget_savings_pct":   _m(savings_list),
            "theorem1_holds":            theorem1_ok,
            # MADE at sensitive locations (m)
            "sadp_made_high_m":          _m(made_sadp_hi),
            "uniform_made_high_m":       _m(made_uni_hi),
            "planar_made_high_m":        _m(made_pl_hi),
            # MADE at S=1 locations
            "sadp_made_s1_m":            _m(made_sadp_lo),
            "uniform_made_s1_m":         _m(made_uni_lo),
            "planar_made_s1_m":          _m(made_pl_lo),
            # Trip distance error
            "sadp_trip_dist_err_pct":    _m(tdist_sadp),
            "uniform_trip_dist_err_pct": _m(tdist_uni),
            "planar_trip_dist_err_pct":  _m(tdist_pl),
            # Range query MAE
            "sadp_rq_mae_pct":           _m(rq_sadp),
            "uniform_rq_mae_pct":        _m(rq_uni),
            "planar_rq_mae_pct":         _m(rq_pl),
        })

    return {"sadp_comparison": results_per_eps}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 & 5 on real data — Fed-GAN
# ─────────────────────────────────────────────────────────────────────────────

def run_fedgan_real(
    norm_arrays: List[np.ndarray],
    window_size: int,
    n_clients: int,
    n_rounds: int,
    noise_multipliers: List[float],
    seed: int,
) -> Dict:
    """
    Train Fed-GAN on real trajectory data.
    Reports convergence, KL divergence, Fréchet distance, MI accuracy.
    """
    rng_master = np.random.default_rng(seed)

    # Limit to n_clients clients with enough data
    eligible = [(i, d) for i, d in enumerate(norm_arrays) if len(d) >= 20][:n_clients]
    if len(eligible) < 2:
        return {"error": "Not enough clients with sufficient data"}

    client_data = [d for _, d in eligible]

    results = []
    for sigma in noise_multipliers:
        params = TrajectoryGANParams(
            noise_multiplier=sigma,
            seq_len=window_size,
            z_dim=16, hidden_dim=32,
        )
        server  = FedGANServer(params, rng=np.random.default_rng(seed))
        clients = [
            FedGANClient(i, client_data[i], params,
                         rng=np.random.default_rng(i + seed))
            for i in range(len(client_data))
        ]

        d_losses, g_losses = [], []
        for _ in range(n_rounds):
            gp = server.get_global_params()
            updates = [c.train_round(gp) for c in clients]
            stats = server.aggregate(updates)
            d_losses.append(stats["mean_d_loss"])
            g_losses.append(stats["mean_g_loss"])

        eps = max(c.privacy.compute_epsilon(1e-5) for c in clients)

        # Generation quality
        real_sample = np.concatenate([d[:20] for d in client_data])
        fake_sample = server.generate(len(real_sample))
        gq = measure_generation_quality(real_sample, fake_sample,
                                        seq_len=window_size, rng=rng_master)

        # MI attack (member = train, non-member = other half)
        held_out    = np.concatenate([d[20:40] for d in client_data
                                      if len(d) >= 40])
        if len(held_out) > 0:
            n_att = min(30, len(real_sample), len(held_out))
            mi    = MembershipInferenceAttack()
            mi_res = mi.fit_and_evaluate(
                server.gan, real_sample[:n_att], held_out[:n_att], rng=rng_master
            )
            mi_acc = mi_res["attack_accuracy"]
            mi_adv = mi_res["advantage"]
        else:
            mi_acc, mi_adv = 0.5, 0.0

        results.append({
            "noise_multiplier":  sigma,
            "n_clients":         len(client_data),
            "n_rounds":          n_rounds,
            "final_d_loss":      round(d_losses[-1], 4),
            "final_g_loss":      round(g_losses[-1], 4),
            "final_epsilon":     round(float(eps), 4),
            "kl_trip_length":    gq["kl_trip_length"],
            "kl_speed":          gq["kl_speed"],
            "mean_frechet":      gq["mean_frechet_dist"],
            "mi_accuracy":       round(mi_acc, 4),
            "mi_advantage":      round(mi_adv, 4),
        })

    return {"fedgan_real": results}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 on real data — DKA (dataset-independent, same timing)
# ─────────────────────────────────────────────────────────────────────────────

def run_dka_real(seed: int) -> Dict:
    """
    DKA latency is dataset-independent (pure cryptographic operation).
    Runs benchmark and security comparison table.
    """
    configs = [(2, 3), (3, 5), (5, 10), (10, 20)]
    timing  = [benchmark_reconstruction(t, n, repetitions=50)
               for (t, n) in configs]
    sec     = security_comparison_table()
    return {"dka_timing": timing, "security_comparison": sec}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6 on real data — E2E pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_e2e_real(
    flat_trajs: List[List[Tuple[float, float]]],
    epsilon_values: List[float],
    dka_configs: List[Tuple[int, int]],
    n_sample: int,
    repetitions: int,
    seed: int,
) -> Dict:
    """
    End-to-end pipeline timing on real trajectories.
    """
    import secrets as _sec
    import hashlib, struct

    rng = np.random.default_rng(seed)
    sample = flat_trajs[:n_sample]

    results = []
    for eps in epsilon_values:
        for (t, n) in dka_configs:
            from paper2.afed_pipeline import AFEDConfig, AFEDPipeline
            cfg      = AFEDConfig(epsilon_base=eps, dka_t=t, dka_n=n)
            pipeline = AFEDPipeline(cfg)
            shares   = pipeline.setup_dka()
            pipeline.load_shares(shares[:t])

            timings_all = []
            savings_all = []
            for traj in sample[:repetitions]:
                r = pipeline.process(traj, user_id="real_user")
                timings_all.append(r.timing_ms)
                savings_all.append(r.budget_savings)

            def _mean_stage(stage):
                vals = [t.get(stage, 0) for t in timings_all]
                return round(float(np.mean(vals)), 3)

            results.append({
                "epsilon_base":        eps,
                "dka_t": t, "dka_n": n,
                "n_trajectories":      len(sample[:repetitions]),
                "sadp_score_ms":       _mean_stage("sadp_score_ms"),
                "sadp_noise_ms":       _mean_stage("sadp_noise_ms"),
                "dka_derive_ms":       _mean_stage("dka_derive_ms"),
                "encrypt_ms":          _mean_stage("encrypt_ms"),
                "total_ms":            round(sum(
                    _mean_stage(s) for s in
                    ["sadp_score_ms","sadp_noise_ms","dka_derive_ms","encrypt_ms"]
                ), 3),
                "mean_budget_savings_pct": round(float(np.mean(savings_all))*100, 2),
            })

    return {"e2e_real": results}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _save(name: str, data: dict, dataset: str, seed: int):
    path = os.path.join(_RESULTS_DIR, f"{name}_{dataset}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → saved {path}")


def _banner(msg: str):
    bar = "=" * (len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}")


def run_dataset(dataset: str, args):
    seed    = args.seed
    alldata = getattr(args, "alldata", False)

    if args.quick:
        window_size  = 20
        n_users      = 10
        n_rounds     = 5
        n_sample     = 30
        reps         = 5
        eps_values   = [1.0]
        sigmas       = [1.0]
        max_traj_len = 100
    elif alldata:
        # Full dataset — all users/taxis, cap trajectory length to avoid O(N²) bottleneck
        window_size  = 50
        n_rounds     = 20
        reps         = 20
        eps_values   = [0.5, 1.0, 2.0]
        sigmas       = [0.5, 1.0, 1.5, 2.0]
        max_traj_len = 200   # cap per trajectory: prevents 34k-pt slowdowns
        # Dataset-specific user/sample counts
        if dataset == "geolife":
            n_users  = 182   # all users
            n_sample = 1000
        elif dataset == "tdrive":
            n_users  = 500   # representative sample of 10k taxis
            n_sample = 500
        else:   # porto
            n_users  = 100
            n_sample = 2000
    else:
        window_size  = 50
        n_users      = 30
        n_rounds     = 20
        n_sample     = 200
        reps         = 20
        eps_values   = [0.5, 1.0, 2.0]
        sigmas       = [0.5, 1.0, 1.5, 2.0]
        max_traj_len = None   # no cap

    mode   = "QUICK" if args.quick else ("ALL-DATA" if alldata else "FULL")
    suffix = f"{dataset}_alldata" if alldata else dataset
    print(f"\n{'─'*60}")
    print(f"  Dataset: {dataset.upper()}  |  Seed: {seed}  |  Mode: {mode}")
    print(f"{'─'*60}")

    # ── Load data ─────────────────────────────────────────────────────────
    _banner(f"Loading {dataset}")
    t0 = time.perf_counter()
    ids, windows, flat_trajs, norm_arrays = _load_dataset(
        dataset, n_users=n_users, window_size=window_size,
        stride=window_size // 2, seed=seed,
    )
    # Cap trajectory length to avoid O(N²) bottleneck in SA-DP scoring
    if max_traj_len is not None:
        flat_trajs = [t[:max_traj_len] for t in flat_trajs if len(t) >= 10]

    print(f"  Clients loaded: {len(ids)}")
    print(f"  Total flat trajectories: {len(flat_trajs)}")
    lengths = [len(t) for t in flat_trajs]
    print(f"  Traj length range: {min(lengths)}-{max(lengths)} pts  "
          f"(mean {sum(lengths)//len(lengths)})")
    if max_traj_len:
        print(f"  [capped at {max_traj_len} pts/traj]")
    print(f"  Load time: {time.perf_counter()-t0:.1f}s")

    all_results = {"dataset": dataset, "seed": seed}

    # ── SA-DP ─────────────────────────────────────────────────────────────
    _banner(f"SA-DP on {dataset}")
    t0 = time.perf_counter()
    r_sadp = run_sadp_real(flat_trajs, eps_values,
                           sensitivity_m=500.0,
                           n_sample=n_sample, seed=seed)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    for r in r_sadp["sadp_comparison"]:
        print(f"  ε={r['epsilon_base']:.1f}  savings={r['mean_budget_savings_pct']:.1f}%  "
              f"Theorem1={r['theorem1_holds']}  "
              f"MADE-hi(SA/Uni): {r['sadp_made_high_m']:.0f}/{r['uniform_made_high_m']:.0f}m")
    all_results["sadp"] = r_sadp
    _save("exp12_sadp", r_sadp, suffix, seed)

    # ── DKA ───────────────────────────────────────────────────────────────
    _banner("DKA Benchmark (dataset-independent)")
    t0 = time.perf_counter()
    r_dka = run_dka_real(seed)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    for b in r_dka["dka_timing"]:
        if b["t"] == 3:
            recon = b["reconstruct"]["mean_ms"]
            pbkdf = b["pbkdf2_baseline"]["mean_ms"]
            print(f"  (t=3,n=5) recon={recon:.3f}ms  "
                  f"overhead={recon/max(1e-9,pbkdf):.4f}×")
    all_results["dka"] = r_dka
    _save("exp3_dka", r_dka, suffix, seed)

    # ── Fed-GAN ───────────────────────────────────────────────────────────
    _banner(f"Fed-GAN on {dataset}")
    t0 = time.perf_counter()
    r_fedgan = run_fedgan_real(
        norm_arrays, window_size,
        n_clients=min(len(norm_arrays), 10),
        n_rounds=n_rounds,
        noise_multipliers=sigmas,
        seed=seed,
    )
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    if "fedgan_real" in r_fedgan:
        for r in r_fedgan["fedgan_real"]:
            print(f"  σ={r['noise_multiplier']:.1f}  "
                  f"ε={r['final_epsilon']:.2f}  "
                  f"KL={r['kl_trip_length']:.4f}  "
                  f"Fréchet={r['mean_frechet']:.4f}")
    all_results["fedgan"] = r_fedgan
    _save("exp45_fedgan", r_fedgan, suffix, seed)

    # ── E2E Pipeline ──────────────────────────────────────────────────────
    _banner(f"E2E Pipeline on {dataset}")
    t0 = time.perf_counter()
    r_e2e = run_e2e_real(
        flat_trajs, eps_values,
        dka_configs=[(3, 5)],
        n_sample=n_sample,
        repetitions=reps,
        seed=seed,
    )
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    for r in r_e2e["e2e_real"]:
        print(f"  ε={r['epsilon_base']:.1f}  total={r['total_ms']:.1f}ms  "
              f"savings={r['mean_budget_savings_pct']:.1f}%")
    all_results["e2e"] = r_e2e
    _save("exp6_e2e", r_e2e, suffix, seed)

    # ── Master save ───────────────────────────────────────────────────────
    suffix = f"{dataset}_alldata" if alldata else dataset
    _save("all_results", all_results, suffix, seed)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="AFED-PPTE — Experiments on real datasets"
    )
    parser.add_argument("--dataset", default="geolife",
                        choices=["geolife", "tdrive", "porto", "all"],
                        help="Which dataset to use")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--quick",   action="store_true",
                        help="Reduced parameters (10 users, 1 epsilon)")
    parser.add_argument("--alldata", action="store_true",
                        help="Use full dataset (182 GeoLife users, 500 T-Drive taxis, "
                             "2000 Porto trips) with max_traj_len=200 cap")
    args = parser.parse_args()

    wall = time.perf_counter()
    datasets = ["geolife", "tdrive", "porto"] if args.dataset == "all" \
               else [args.dataset]

    for ds in datasets:
        run_dataset(ds, args)

    print(f"\n{'─'*60}")
    print(f"  All done. Total time: {time.perf_counter()-wall:.0f}s")
    print(f"  Results in: {_RESULTS_DIR}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
