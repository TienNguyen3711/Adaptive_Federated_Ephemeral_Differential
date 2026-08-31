import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

# Resolve package path when run as a script
_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

from paper2.federated_gan import (
    TrajectoryGANParams,
    FedGANServer,
    FedGANClient,
    MembershipInferenceAttack,
    make_synthetic_client_data,
    fedgan_simulate,
    benchmark_fedgan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Convergence across noise multipliers
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment4(
    noise_multipliers: List[float] = None,
    n_clients_list: List[int] = None,
    n_rounds: int = 20,
    samples_per_client: int = 100,
    repetitions: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    Experiment 4: convergence curves and ε budget for each (σ, K) config.

    Returns list of result dicts including per-round loss trajectories.
    """
    if noise_multipliers is None:
        noise_multipliers = [0.5, 1.0, 1.5, 2.0]
    if n_clients_list is None:
        n_clients_list = [3, 5, 10]

    results = []
    rng_master = np.random.default_rng(seed)

    for n_clients in n_clients_list:
        for sigma in noise_multipliers:
            rep_results = []
            for rep in range(repetitions):
                params = TrajectoryGANParams(noise_multiplier=sigma)
                r = fedgan_simulate(
                    n_clients=n_clients,
                    n_rounds=n_rounds,
                    samples_per_client=samples_per_client,
                    params=params,
                    rng_seed=seed + rep * 37,
                )
                rep_results.append(r)

            # Aggregate convergence curves across repetitions
            d_losses = np.array([[s["mean_d_loss"] for s in r["round_stats"]]
                                 for r in rep_results])
            g_losses = np.array([[s["mean_g_loss"] for s in r["round_stats"]]
                                 for r in rep_results])
            epsilons = np.array([r["final_epsilon"] for r in rep_results])

            # ── Generation quality metrics (last repetition) ──────────────
            params_last = TrajectoryGANParams(noise_multiplier=sigma)
            last_result = fedgan_simulate(
                n_clients=n_clients, n_rounds=n_rounds,
                samples_per_client=samples_per_client,
                params=params_last, rng_seed=seed + 999,
            )
            from paper2.federated_gan import (
                FedGANServer, make_synthetic_client_data
            )
            _server = FedGANServer(params_last, rng=np.random.default_rng(seed))
            _datasets = make_synthetic_client_data(
                n_clients, samples_per_client, params_last.seq_len,
                np.random.default_rng(seed)
            )
            real_sample  = np.concatenate([d[:20] for d in _datasets])
            fake_sample  = np.array(last_result["round_stats"])  # placeholder
            # Re-generate using the trained server from last_result
            from paper2.federated_gan import (
                FedGANClient, FedGANServer as _FGS
            )
            _srv2 = _FGS(params_last, rng=np.random.default_rng(seed))
            _cli2 = [FedGANClient(i, _datasets[i], params_last,
                                  rng=np.random.default_rng(i+seed))
                     for i in range(n_clients)]
            for _ in range(n_rounds):
                _srv2.aggregate([c.train_round(_srv2.get_global_params())
                                 for c in _cli2])
            fake_sample = _srv2.generate(len(real_sample))

            gen_quality = measure_generation_quality(
                real_sample, fake_sample,
                seq_len=params_last.seq_len,
                rng=rng_master,
            )

            results.append({
                "n_clients":          n_clients,
                "noise_multiplier":   sigma,
                "n_rounds":           n_rounds,
                "repetitions":        repetitions,
                # Convergence curves (mean ± std across reps)
                "d_loss_mean": d_losses.mean(axis=0).round(4).tolist(),
                "d_loss_std":  d_losses.std(axis=0).round(4).tolist(),
                "g_loss_mean": g_losses.mean(axis=0).round(4).tolist(),
                "g_loss_std":  g_losses.std(axis=0).round(4).tolist(),
                # Final losses
                "final_d_loss":       round(float(d_losses[:, -1].mean()), 4),
                "final_g_loss":       round(float(g_losses[:, -1].mean()), 4),
                # Privacy budget
                "mean_final_epsilon": round(float(epsilons.mean()), 4),
                "std_final_epsilon":  round(float(epsilons.std()),  4),
                # Generation quality
                "kl_trip_length":     gen_quality["kl_trip_length"],
                "kl_speed":           gen_quality["kl_speed"],
                "mean_frechet_dist":  gen_quality["mean_frechet_dist"],
                # Timing
                "mean_time_s": round(
                    sum(r["total_time_s"] for r in rep_results) / repetitions, 2
                ),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Membership inference resistance
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment5(
    noise_multipliers: List[float] = None,
    n_clients: int = 5,
    n_rounds: int = 20,
    samples_per_client: int = 100,
    repetitions: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Experiment 5: MI attack accuracy vs DP noise level.

    For each σ, trains Fed-GAN then evaluates shadow-model MI attack.
    Also reports a centralised (no-DP) baseline to show the gap.
    """
    if noise_multipliers is None:
        noise_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]   # 0.0 = no DP (baseline)

    results = []
    rng_master = np.random.default_rng(seed)

    for sigma in noise_multipliers:
        mi_accs, mi_aucs, mi_advs, epsilons = [], [], [], []

        for rep in range(repetitions):
            rep_seed = int(rng_master.integers(0, 100_000))
            rng      = np.random.default_rng(rep_seed)

            # Use σ=0.01 to approximate "no DP" without division by zero in tracker
            effective_sigma = max(0.01, sigma)
            params = TrajectoryGANParams(noise_multiplier=effective_sigma)

            client_datasets = make_synthetic_client_data(
                n_clients, samples_per_client, params.seq_len, rng
            )

            server  = FedGANServer(params, rng=rng)
            clients = [
                FedGANClient(i, client_datasets[i], params, rng=np.random.default_rng(i + rep_seed))
                for i in range(n_clients)
            ]

            for _ in range(n_rounds):
                gp      = server.get_global_params()
                updates = [c.train_round(gp) for c in clients]
                server.aggregate(updates)

            eps = max(c.privacy.compute_epsilon(1e-5) for c in clients)
            epsilons.append(eps)

            # MI attack
            member_data = np.concatenate([cd[:20] for cd in client_datasets])
            fake_data   = server.generate(len(member_data))
            mi_attack   = MembershipInferenceAttack()
            mi_res = mi_attack.fit_and_evaluate(server.gan, member_data, fake_data, rng=rng)

            mi_accs.append(mi_res["attack_accuracy"])
            mi_aucs.append(mi_res["auc"])
            mi_advs.append(mi_res["advantage"])

        results.append({
            "noise_multiplier":     sigma,
            "n_clients":            n_clients,
            "n_rounds":             n_rounds,
            "mean_mi_accuracy":     round(float(np.mean(mi_accs)), 4),
            "std_mi_accuracy":      round(float(np.std(mi_accs)),  4),
            "mean_mi_auc":          round(float(np.mean(mi_aucs)), 4),
            "mean_mi_advantage":    round(float(np.mean(mi_advs)), 4),
            "mean_epsilon":         round(float(np.mean(epsilons)), 4),
            "privacy_utility_note": (
                "no-DP baseline" if sigma == 0.0
                else f"σ={sigma} DP-SGD"
            ),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utility metrics for generated trajectories
# ─────────────────────────────────────────────────────────────────────────────

def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """
    KL divergence KL(P || Q) estimated from samples via histogram binning.
    Adds 1e-10 smoothing to avoid log(0).
    """
    all_vals = np.concatenate([p, q])
    lo, hi   = all_vals.min(), all_vals.max()
    if hi - lo < 1e-10:
        return 0.0
    edges  = np.linspace(lo, hi, bins + 1)
    ph, _  = np.histogram(p, bins=edges, density=True)
    qh, _  = np.histogram(q, bins=edges, density=True)
    ph = ph + 1e-10;  ph /= ph.sum()
    qh = qh + 1e-10;  qh /= qh.sum()
    return float(np.sum(ph * np.log(ph / qh)))


def _trip_lengths(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Compute total Euclidean path length for each trajectory in data.
    data: (N, seq_len*2) normalised coordinate array.
    """
    pts = data.reshape(-1, seq_len, 2)   # (N, T, 2)
    diffs = np.diff(pts, axis=1)         # (N, T-1, 2)
    return np.sqrt((diffs**2).sum(axis=2)).sum(axis=1)  # (N,)


def _mean_speed(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Mean step-to-step speed (Euclidean) per trajectory."""
    pts   = data.reshape(-1, seq_len, 2)
    diffs = np.diff(pts, axis=1)
    return np.sqrt((diffs**2).sum(axis=2)).mean(axis=1)


def _discrete_frechet(p: np.ndarray, q: np.ndarray) -> float:
    """
    Discrete Fréchet distance between two trajectories p and q.
    p, q: (T, 2) arrays.
    Uses O(T²) DP — suitable for short sequences (T ≤ 100).
    """
    n, m  = len(p), len(q)
    ca    = np.full((n, m), -1.0)

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = float(np.linalg.norm(p[i] - q[j]))
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j-1), d)
        elif j == 0:
            ca[i, j] = max(_c(i-1, 0), d)
        else:
            ca[i, j] = max(min(_c(i-1,j), _c(i-1,j-1), _c(i,j-1)), d)
        return ca[i, j]

    return _c(n-1, m-1)


def measure_generation_quality(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    seq_len: int,
    n_frechet_pairs: int = 50,
    rng=None,
) -> Dict:
    """
    Measure quality of generated trajectories vs real ones.

    Metrics:
      kl_trip_length   : KL divergence of trip-length distributions
      kl_speed         : KL divergence of mean-speed distributions
      mean_frechet_dist: mean discrete Fréchet distance (sampled pairs)
      trip_len_real_mean / fake_mean: distribution means for inspection
    """
    rng = rng or np.random.default_rng(0)

    tl_real = _trip_lengths(real_data, seq_len)
    tl_fake = _trip_lengths(fake_data, seq_len)
    sp_real = _mean_speed(real_data,  seq_len)
    sp_fake = _mean_speed(fake_data,  seq_len)

    kl_tl = _kl_divergence(tl_real, tl_fake)
    kl_sp = _kl_divergence(sp_real, sp_fake)

    # Fréchet: sample n_frechet_pairs pairs (1 real, 1 fake each)
    n      = min(n_frechet_pairs, len(real_data), len(fake_data))
    r_idx  = rng.choice(len(real_data), n, replace=False)
    f_idx  = rng.choice(len(fake_data), n, replace=False)
    real_pts = real_data[r_idx].reshape(n, seq_len, 2)
    fake_pts = fake_data[f_idx].reshape(n, seq_len, 2)
    frechet_vals = [_discrete_frechet(real_pts[i], fake_pts[i]) for i in range(n)]

    return {
        "kl_trip_length":      round(kl_tl, 5),
        "kl_speed":            round(kl_sp, 5),
        "mean_frechet_dist":   round(float(np.mean(frechet_vals)), 5),
        "std_frechet_dist":    round(float(np.std(frechet_vals)),  5),
        "trip_len_real_mean":  round(float(tl_real.mean()), 5),
        "trip_len_fake_mean":  round(float(tl_fake.mean()), 5),
        "speed_real_mean":     round(float(sp_real.mean()), 5),
        "speed_fake_mean":     round(float(sp_fake.mean()), 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison: Fed-GAN vs Centralised / FedAvg-noDP / LDP
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_comparison(
    n_clients: int = 5,
    n_rounds: int = 20,
    samples_per_client: int = 100,
    repetitions: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    Four-way comparison for Experiment 5.

    NOTE — MI attack scope limitation (synthetic data):
      The shadow-model MI attack queries D's output confidence. In a correctly
      trained GAN, D distinguishes real/fake data (not member/non-member), so
      D(member_real) ≈ D(non-member_real) on synthetic datasets with small-scale
      models. Meaningful MI separation requires larger models with memorization
      behaviour, observable on real trajectory datasets (GeoLife/T-Drive).
      KL divergence and Fréchet distance remain valid utility metrics here.

    Variants:
      centralised  — all data on 1 server, no DP (σ ≈ 0, no federation)
      fedavg_nodp  — FedAvg federation, no DP-SGD (σ = 0.01)
      ldp          — federation + very high noise (σ = 10.0, LDP-like)
      fedgan_dp    — AFED-PPTE: federation + DP-SGD (σ = 1.1)

    MI attack design (correct):
      member_data    = first half of total dataset (IN training set)
      non_member_data = second half of total dataset (UNSEEN real data)
      → Tests whether D memorised individual samples vs unseen real ones.
      → All variants use same total data size for a fair comparison.

    For each variant reports:
      mi_accuracy, mi_advantage, mean_frechet_dist, kl_trip_length, epsilon
    """
    configs = [
        # (label, sigma, n_clients_for_federation)
        ("centralised",  0.01,  1),          # 1 client = all data centralised
        ("fedavg_nodp",  0.01,  n_clients),  # federation, no DP noise
        ("ldp",          10.0,  n_clients),  # federation + very high noise
        ("fedgan_dp",    1.1,   n_clients),  # AFED-PPTE
    ]

    results = []
    rng_master = np.random.default_rng(seed)

    for (label, sigma, n_cli) in configs:
        mi_accs, mi_advs, epsilons = [], [], []
        frechet_list, kl_list = [], []

        for rep in range(repetitions):
            rep_seed = int(rng_master.integers(0, 100_000))
            rng      = np.random.default_rng(rep_seed)

            params = TrajectoryGANParams(noise_multiplier=sigma)

            # Always generate n_clients × samples total so all variants
            # have the same total data pool — fair comparison.
            all_datasets = make_synthetic_client_data(
                n_clients, samples_per_client * 2, params.seq_len, rng
            )
            # First half → training (members); second half → held-out (non-members)
            train_datasets  = [d[:samples_per_client]  for d in all_datasets]
            held_out        = np.concatenate([d[samples_per_client:] for d in all_datasets])

            if label == "centralised":
                # Centralised: one client trains on ALL training data pooled
                pooled = np.concatenate(train_datasets)
                fed_datasets = [pooled]
            else:
                fed_datasets = train_datasets[:n_cli]

            server  = FedGANServer(params, rng=rng)
            clients = [
                FedGANClient(i, fed_datasets[i], params,
                             rng=np.random.default_rng(i + rep_seed))
                for i in range(len(fed_datasets))
            ]

            for _ in range(n_rounds):
                gp      = server.get_global_params()
                updates = [c.train_round(gp) for c in clients]
                server.aggregate(updates)

            eps = max(c.privacy.compute_epsilon(1e-5) for c in clients)
            epsilons.append(min(eps, 1e5))

            # ── Correct MI attack: member (training) vs non-member (held-out real) ──
            n_attack = min(50, len(held_out))
            member_data     = np.concatenate(train_datasets)[:n_attack]
            non_member_data = held_out[:n_attack]
            mi_attack = MembershipInferenceAttack()
            mi_res    = mi_attack.fit_and_evaluate(
                server.gan, member_data, non_member_data, rng=rng
            )
            mi_accs.append(mi_res["attack_accuracy"])
            mi_advs.append(mi_res["advantage"])

            # ── Generation quality: real (member) vs generated ──────────────────
            fake_data = server.generate(n_attack)
            gq = measure_generation_quality(
                member_data, fake_data,
                seq_len=params.seq_len, rng=rng,
            )
            frechet_list.append(gq["mean_frechet_dist"])
            kl_list.append(gq["kl_trip_length"])

        def _m(lst): return round(float(np.mean(lst)), 4)

        results.append({
            "variant":           label,
            "sigma":             sigma,
            "n_clients":         n_cli,
            "n_rounds":          n_rounds,
            "mean_mi_accuracy":  _m(mi_accs),
            "mean_mi_advantage": _m(mi_advs),
            "mean_frechet":      _m(frechet_list),
            "mean_kl_trip":      _m(kl_list),
            "mean_epsilon":      _m(epsilons),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Communication overhead measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_communication_overhead(
    params: Optional[TrajectoryGANParams] = None,
) -> Dict:
    """
    Measure model size (bytes) communicated per round per client.

    Fed-GAN sends gradient deltas, not raw data.  Each delta has the same
    shape as the model parameters.
    """
    if params is None:
        params = TrajectoryGANParams()

    from paper2.federated_gan import TrajectoryGAN
    import struct

    gan = TrajectoryGAN(params)
    total_params = sum(v.size for v in [*gan.G.values(), *gan.D.values()])
    bytes_per_round = total_params * 8   # float64

    return {
        "total_parameters": total_params,
        "bytes_per_client_per_round": bytes_per_round,
        "kb_per_client_per_round":    round(bytes_per_round / 1024, 2),
        "architecture": {
            "z_dim":      params.z_dim,
            "hidden_dim": params.hidden_dim,
            "seq_len":    params.seq_len,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AFED-PPTE Experiments 4 & 5: Fed-GAN Evaluation"
    )
    parser.add_argument("--n_rounds",  type=int, default=20,
                        help="Federated rounds per simulation")
    parser.add_argument("--reps",      type=int, default=3,
                        help="Repetitions per configuration")
    parser.add_argument("--output",    default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    print("=== Experiment 4: Fed-GAN Convergence ===")
    t0   = time.perf_counter()
    exp4 = run_experiment4(
        n_rounds=args.n_rounds,
        repetitions=args.reps,
    )
    print(f"  Completed in {time.perf_counter() - t0:.1f}s")

    header = (f"  {'K':>3}  {'σ':>5}  {'D-loss':>8}  {'G-loss':>8}  "
              f"{'ε':>8}  {'KL-len':>8}  {'KL-spd':>8}  {'Fréchet':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in exp4:
        print(f"  {r['n_clients']:>3}  {r['noise_multiplier']:>5.1f}  "
              f"{r['final_d_loss']:>8.4f}  {r['final_g_loss']:>8.4f}  "
              f"{r['mean_final_epsilon']:>8.4f}  "
              f"{r['kl_trip_length']:>8.5f}  "
              f"{r['kl_speed']:>8.5f}  "
              f"{r['mean_frechet_dist']:>8.5f}")

    print("\n=== Experiment 5: Membership Inference Resistance ===")
    t0   = time.perf_counter()
    exp5 = run_experiment5(n_rounds=args.n_rounds, repetitions=args.reps)
    print(f"  Completed in {time.perf_counter() - t0:.1f}s")

    print(f"  {'σ':>5}  {'MI acc':>8}  {'AUC':>6}  {'Advantage':>10}  {'ε':>8}")
    print("  " + "-" * 48)
    for r in exp5:
        print(f"  {r['noise_multiplier']:>5.1f}  "
              f"{r['mean_mi_accuracy']:>8.4f}  "
              f"{r['mean_mi_auc']:>6.4f}  "
              f"{r['mean_mi_advantage']:>10.4f}  "
              f"{r['mean_epsilon']:>8.4f}")

    print("\n=== Communication Overhead ===")
    comm = measure_communication_overhead()
    print(f"  Model parameters: {comm['total_parameters']}")
    print(f"  Bytes/client/round: {comm['bytes_per_client_per_round']}  "
          f"({comm['kb_per_client_per_round']:.2f} KB)")

    print("\n=== Baseline Comparison: Centralised / FedAvg-noDP / LDP / Fed-GAN ===")
    t0   = time.perf_counter()
    base = run_baseline_comparison(n_rounds=args.n_rounds, repetitions=args.reps)
    print(f"  Completed in {time.perf_counter() - t0:.1f}s")
    print(f"  {'Variant':>15}  {'MI acc':>8}  {'Advantage':>10}  "
          f"{'Fréchet':>9}  {'KL-len':>8}  {'ε':>10}")
    print("  " + "-" * 68)
    for r in base:
        print(f"  {r['variant']:>15}  "
              f"{r['mean_mi_accuracy']:>8.4f}  "
              f"{r['mean_mi_advantage']:>10.4f}  "
              f"{r['mean_frechet']:>9.5f}  "
              f"{r['mean_kl_trip']:>8.5f}  "
              f"{r['mean_epsilon']:>10.2f}")

    output = {
        "experiment4": exp4, "experiment5": exp5,
        "baseline_comparison": base, "communication": comm,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        print("\n--- JSON summary ---")
        print(json.dumps({"baseline_comparison": base}, indent=2))


if __name__ == "__main__":
    main()
