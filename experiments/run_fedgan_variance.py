import json
import os
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

# Two-tailed 95% critical values for Student's t at small df (df = n-1).
# Avoids a scipy dependency for a handful of constants in an otherwise
# NumPy-only codebase; falls back to the z-value (valid for large df).
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
    30: 2.042,
}

from Adaptive_Federated_Ephemeral_Differential.federated_gan import (
    TrajectoryGANParams, FedGANServer, FedGANClient, MembershipInferenceAttack,
)
from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import (
    load_as_client_datasets,
)


def load_once(n_users=30, window_size=50, seed=42):
    print(f"Loading GeoLife (n_users={n_users}, window_size={window_size})...", flush=True)
    t0 = time.perf_counter()
    ids, windows_per_client = load_as_client_datasets(
        n_users=n_users, window_size=window_size, stride=window_size // 2, seed=seed
    )
    norm_arrays = []
    for windows in windows_per_client:
        pts = np.array([[[p[0], p[1]] for p in w] for w in windows])
        flat = pts.reshape(len(windows), -1)
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        norm = 2.0 * (flat - lo) / (hi - lo + 1e-8) - 1.0
        norm_arrays.append(norm)
    print(f"  Loaded {len(norm_arrays)} clients in {time.perf_counter()-t0:.1f}s", flush=True)
    return norm_arrays


def one_trial(norm_arrays, sigma, n_clients, n_rounds, window_size, trial_seed):
    eligible = [d for d in norm_arrays if len(d) >= 40][:n_clients]
    if len(eligible) < 2:
        return None
    client_data = eligible

    params = TrajectoryGANParams(
        noise_multiplier=sigma, seq_len=window_size, z_dim=16, hidden_dim=32,
    )
    server = FedGANServer(params, rng=np.random.default_rng(trial_seed))
    clients = [
        FedGANClient(i, client_data[i], params,
                     rng=np.random.default_rng(i + trial_seed * 997))
        for i in range(len(client_data))
    ]

    d_losses = []
    for _ in range(n_rounds):
        gp = server.get_global_params()
        updates = [c.train_round(gp) for c in clients]
        stats = server.aggregate(updates)
        d_losses.append(stats["mean_d_loss"])

    eps = max(c.privacy.compute_epsilon(1e-5) for c in clients)

    real_sample = np.concatenate([d[:20] for d in client_data])
    held_out = np.concatenate([d[20:40] for d in client_data])
    n_att = min(30, len(real_sample), len(held_out))
    mi = MembershipInferenceAttack()
    attack_rng = np.random.default_rng(trial_seed * 7919 + 1)
    mi_res = mi.fit_and_evaluate(server.gan, real_sample[:n_att], held_out[:n_att], rng=attack_rng)

    return {
        "final_d_loss": round(float(d_losses[-1]), 4),
        "final_epsilon": round(float(eps), 4),
        "mi_accuracy": round(float(mi_res["attack_accuracy"]), 4),
        "mi_advantage": round(float(mi_res["advantage"]), 4),
        "mi_auc": round(float(mi_res["auc"]), 4),
        "mi_auc_advantage": round(float(mi_res["auc_advantage"]), 4),
    }


def _ci95(vals):
    """95% CI via Student's t (not a normal/z approximation): every call
    site in this codebase fixes n=8 trials (df=7), where t_{0.975,7}=2.3646
    is meaningfully wider than z_{0.975}=1.96 (peer review S1)."""
    arr = np.array(vals, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    t_crit = _T_CRIT_975.get(n - 1, 1.96)
    return mean, std, t_crit * se


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (default 42; use 123 for the RR6 cross-seed check)")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None,
                        help="Subset of sigma values to run (default: full sweep)")
    args = parser.parse_args()

    n_clients = 10
    n_rounds = 20
    window_size = 50
    repetitions = 8
    sigmas = args.sigmas if args.sigmas is not None else [0.5, 1.0, 1.5, 2.0, 4.0, 8.0]
    base_seed = args.seed

    norm_arrays = load_once(n_users=30, window_size=window_size, seed=base_seed)

    all_results = {}
    for sigma in sigmas:
        print(f"\nsigma={sigma}: running {repetitions} trials...", flush=True)
        trials = []
        for rep in range(repetitions):
            trial_seed = base_seed * 1000 + int(sigma * 100) * 10 + rep
            t0 = time.perf_counter()
            r = one_trial(norm_arrays, sigma, n_clients, n_rounds, window_size, trial_seed)
            if r is not None:
                trials.append(r)
                print(f"  rep {rep}: d_loss={r['final_d_loss']:.4f}  "
                      f"eps={r['final_epsilon']:.2f}  "
                      f"mi_adv={r['mi_advantage']:+.4f}  "
                      f"[{time.perf_counter()-t0:.1f}s]", flush=True)

        mi_advs = [t["mi_advantage"] for t in trials]
        auc_advs = [t["mi_auc_advantage"] for t in trials]
        d_losses = [t["final_d_loss"] for t in trials]
        epsilons = [t["final_epsilon"] for t in trials]

        mi_mean, mi_std, mi_ci = _ci95(mi_advs)
        auc_mean, auc_std, auc_ci = _ci95(auc_advs)
        d_mean, d_std, d_ci = _ci95(d_losses)
        eps_mean, eps_std, eps_ci = _ci95(epsilons)

        all_results[str(sigma)] = {
            "n_trials": len(trials),
            "trials": trials,
            "mi_advantage_mean": round(mi_mean, 4),
            "mi_advantage_std": round(mi_std, 4),
            "mi_advantage_ci95": round(mi_ci, 4),
            "mi_auc_advantage_mean": round(auc_mean, 4),
            "mi_auc_advantage_std": round(auc_std, 4),
            "mi_auc_advantage_ci95": round(auc_ci, 4),
            "d_loss_mean": round(d_mean, 4),
            "d_loss_std": round(d_std, 4),
            "d_loss_ci95": round(d_ci, 4),
            "epsilon_mean": round(eps_mean, 4),
            "epsilon_std": round(eps_std, 4),
            "epsilon_ci95": round(eps_ci, 4),
        }
        print(f"  => MI advantage (acc-based): {mi_mean:+.4f} +/- {mi_ci:.4f} (std={mi_std:.4f}, n={len(trials)})", flush=True)
        print(f"  => MI advantage (AUC-based, primary): {auc_mean:+.4f} +/- {auc_ci:.4f} (std={auc_std:.4f}, n={len(trials)})", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "results", f"fedgan_variance_seed{base_seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'sigma':>6}  {'AUC_adv_mean':>13}  {'+/-95%CI':>10}  {'acc_adv_mean':>13}  {'+/-95%CI':>10}  {'D-loss':>8}  {'eps':>8}")
    for sigma in sigmas:
        r = all_results[str(sigma)]
        print(f"{sigma:>6.1f}  {r['mi_auc_advantage_mean']:>+13.4f}  {r['mi_auc_advantage_ci95']:>10.4f}  "
              f"{r['mi_advantage_mean']:>+13.4f}  {r['mi_advantage_ci95']:>10.4f}  "
              f"{r['d_loss_mean']:>8.4f}  {r['epsilon_mean']:>8.2f}")


if __name__ == "__main__":
    main()
