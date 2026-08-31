import json
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

from Adaptive_Federated_Ephemeral_Differential.federated_gan import (
    TrajectoryGAN, TrajectoryGANParams, FedGANServer, FedGANClient,
    MembershipInferenceAttack,
)

SEED = 42
N_ROUNDS = 20
WINDOW_SIZE = 50
N_CLIENTS = 10
REPETITIONS = 8
HIDDEN_DIMS = [32, 512]   # 32 = current paper capacity (7,109 params);
                          # 512 = ablation capacity (112,229 params, >=1e5)
SIGMA_DP = 4.0            # this paper's recommended deployment threshold
SIGMA_NODP = 0.01         # approx. no-DP centralised control


def _param_count(hidden_dim, z_dim=16, seq_len=WINDOW_SIZE):
    p = TrajectoryGANParams(hidden_dim=hidden_dim, z_dim=z_dim, seq_len=seq_len)
    m = TrajectoryGAN(p, rng=np.random.default_rng(0))
    return sum(v.size for v in m.G.values()) + sum(v.size for v in m.D.values())


def normalise(windows_per_client):
    norm_arrays = []
    for windows in windows_per_client:
        pts = np.array([[[pt[0], pt[1]] for pt in w] for w in windows])
        flat = pts.reshape(len(windows), -1)
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        norm = 2.0 * (flat - lo) / (hi - lo + 1e-8) - 1.0
        norm_arrays.append(norm)
    return norm_arrays


def load_dataset(name, n_entities=30, window_size=WINDOW_SIZE, seed=SEED):
    print(f"Loading {name} (n={n_entities}, window={window_size})...", flush=True)
    t0 = time.perf_counter()
    if name == "geolife":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import load_as_client_datasets
        ids, windows = load_as_client_datasets(n_users=n_entities, window_size=window_size,
                                                stride=window_size // 2, seed=seed)
    elif name == "tdrive":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_tdrive import load_as_client_datasets
        ids, windows = load_as_client_datasets(n_taxis=n_entities, window_size=window_size,
                                                stride=window_size // 2, seed=seed)
    elif name == "porto":
        from Adaptive_Federated_Ephemeral_Differential.data.loader_porto import load_as_client_datasets
        ids, windows = load_as_client_datasets(n_taxis=n_entities, trips_per_taxi=100,
                                                window_size=window_size, stride=window_size // 2, seed=seed)
    else:
        raise ValueError(name)
    norm = normalise(windows)
    print(f"  Loaded {len(norm)} clients in {time.perf_counter()-t0:.1f}s", flush=True)
    return norm


def _ci95(vals):
    arr = np.array(vals, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    t_crit = _T_CRIT_975.get(n - 1, 1.96)
    return mean, std, t_crit * se


def train_and_attack(client_data, sigma, hidden_dim, trial_seed, centralised=False):
    """client_data: list of per-client arrays. If centralised, all data is
    pooled into a single client (matching run_centralized_baseline_real's
    design) so the no-DP control sees the same data volume as the
    federated condition, not ~10x more."""
    eligible = [d for d in client_data if len(d) >= 40]
    if len(eligible) < 1:
        return None
    if centralised:
        eligible = [np.concatenate(eligible)]

    params = TrajectoryGANParams(noise_multiplier=sigma, seq_len=WINDOW_SIZE,
                                  z_dim=16, hidden_dim=hidden_dim)
    server = FedGANServer(params, rng=np.random.default_rng(trial_seed))
    clients = [
        FedGANClient(i, eligible[i], params, rng=np.random.default_rng(i + trial_seed * 997))
        for i in range(len(eligible))
    ]

    d_losses = []
    for _ in range(N_ROUNDS):
        gp = server.get_global_params()
        updates = [c.train_round(gp) for c in clients]
        stats = server.aggregate(updates)
        d_losses.append(stats["mean_d_loss"])

    eps = max(c.privacy.compute_epsilon(1e-5) for c in clients)

    real_sample = np.concatenate([d[:20] for d in eligible])
    held_out = np.concatenate([d[20:40] for d in eligible])
    n_att = min(30, len(real_sample), len(held_out))
    mi = MembershipInferenceAttack()
    attack_rng = np.random.default_rng(trial_seed * 7919 + 1)
    mi_res = mi.fit_and_evaluate(server.gan, real_sample[:n_att], held_out[:n_att], rng=attack_rng)

    return {
        "final_d_loss": round(float(d_losses[-1]), 4),
        "final_epsilon": round(float(eps), 4),
        "mi_advantage": round(float(mi_res["advantage"]), 4),
        "mi_auc_advantage": round(float(mi_res["auc_advantage"]), 4),
    }


def run_config(client_data, sigma, hidden_dim, centralised, label):
    print(f"\n{label}: running {REPETITIONS} trials...", flush=True)
    trials = []
    for rep in range(REPETITIONS):
        trial_seed = SEED * 1000 + int(sigma * 100) * 10 + hidden_dim + rep
        t0 = time.perf_counter()
        r = train_and_attack(client_data, sigma, hidden_dim, trial_seed, centralised=centralised)
        if r is not None:
            trials.append(r)
            print(f"  rep {rep}: d_loss={r['final_d_loss']:.4f}  eps={r['final_epsilon']:.2f}  "
                  f"auc_adv={r['mi_auc_advantage']:+.4f}  [{time.perf_counter()-t0:.1f}s]", flush=True)
    auc_advs = [t["mi_auc_advantage"] for t in trials]
    acc_advs = [t["mi_advantage"] for t in trials]
    auc_mean, auc_std, auc_ci = _ci95(auc_advs)
    acc_mean, acc_std, acc_ci = _ci95(acc_advs)
    result = {
        "n_trials": len(trials), "trials": trials,
        "mi_auc_advantage_mean": round(auc_mean, 4), "mi_auc_advantage_ci95": round(auc_ci, 4),
        "mi_advantage_mean": round(acc_mean, 4), "mi_advantage_ci95": round(acc_ci, 4),
    }
    print(f"  => {label}: AUC-adv {auc_mean:+.4f} +/- {auc_ci:.4f}", flush=True)
    return result


def run_dataset(dataset):
    norm = load_dataset(dataset, n_entities=30, seed=SEED)
    federated_clients = [d for d in norm if len(d) >= 40][:N_CLIENTS]

    dataset_results = {}
    for hidden_dim in HIDDEN_DIMS:
        n_params = _param_count(hidden_dim)
        dp_result = run_config(
            federated_clients, SIGMA_DP, hidden_dim, centralised=False,
            label=f"{dataset} hidden_dim={hidden_dim} ({n_params} params) Fed-GAN sigma={SIGMA_DP}",
        )
        nodp_result = run_config(
            federated_clients, SIGMA_NODP, hidden_dim, centralised=True,
            label=f"{dataset} hidden_dim={hidden_dim} ({n_params} params) Centralised (no DP)",
        )
        dataset_results[f"hidden_dim_{hidden_dim}"] = {
            "n_params": n_params,
            "fedgan_dp_sigma4": dp_result,
            "centralised_no_dp": nodp_result,
        }
    return dataset_results


if __name__ == "__main__":
    print("=" * 70)
    print("  Fed-GAN capacity ablation (Phase 6): does capacity, not DP-SGD,")
    print("  explain the null MI result? hidden_dim=32 (7,109 params, current)")
    print("  vs hidden_dim=512 (112,229 params, >=1e5)")
    print("=" * 70)

    all_results = {}
    for dataset in ["geolife", "tdrive", "porto"]:
        all_results[dataset] = run_dataset(dataset)

    out_path = os.path.join(os.path.dirname(__file__), "results", "fedgan_capacity_ablation_seed42.json")
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED, "n_rounds": N_ROUNDS, "n_clients": N_CLIENTS,
            "repetitions": REPETITIONS, "hidden_dims": HIDDEN_DIMS,
            "sigma_dp": SIGMA_DP, "sigma_nodp": SIGMA_NODP,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")

    print("\n=== SUMMARY (AUC-based MI advantage, mean +/- 95% CI) ===")
    for dataset in ["geolife", "tdrive", "porto"]:
        for hidden_dim in HIDDEN_DIMS:
            r = all_results[dataset][f"hidden_dim_{hidden_dim}"]
            dp = r["fedgan_dp_sigma4"]
            nodp = r["centralised_no_dp"]
            print(f"{dataset:8s} hidden_dim={hidden_dim:4d} ({r['n_params']:6d} params)  "
                  f"DP(sigma=4): {dp['mi_auc_advantage_mean']:+.4f}+/-{dp['mi_auc_advantage_ci95']:.4f}   "
                  f"no-DP: {nodp['mi_auc_advantage_mean']:+.4f}+/-{nodp['mi_auc_advantage_ci95']:.4f}")
