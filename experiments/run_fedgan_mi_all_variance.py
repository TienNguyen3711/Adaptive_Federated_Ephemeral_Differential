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


def normalise(windows_per_client):
    norm_arrays = []
    for windows in windows_per_client:
        pts = np.array([[[p[0], p[1]] for p in w] for w in windows])
        flat = pts.reshape(len(windows), -1)
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        norm = 2.0 * (flat - lo) / (hi - lo + 1e-8) - 1.0
        norm_arrays.append(norm)
    return norm_arrays


def load_dataset(name, n_entities=30, window_size=50, seed=42):
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


def train_and_attack(client_data, sigma, n_rounds, window_size, trial_seed):
    """client_data: list of per-client arrays (>=1 client). Trains federated
    (or single-client/centralised if len==1) Fed-GAN and runs the fixed MI attack."""
    eligible = [d for d in client_data if len(d) >= 40]
    if len(eligible) < 1:
        return None

    params = TrajectoryGANParams(noise_multiplier=sigma, seq_len=window_size,
                                  z_dim=16, hidden_dim=32)
    server = FedGANServer(params, rng=np.random.default_rng(trial_seed))
    clients = [
        FedGANClient(i, eligible[i], params, rng=np.random.default_rng(i + trial_seed * 997))
        for i in range(len(eligible))
    ]

    d_losses = []
    for _ in range(n_rounds):
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


def load_geolife_reuse(base_seed, sigmas):
    """Reuse GeoLife's already-computed Fed-GAN sigma cells from
    fedgan_variance_seed{base_seed}.json instead of recomputing them here,
    when that file has every requested sigma. This script and
    run_fedgan_variance.py previously selected the 10-client federation via
    different filter/take orders (norm[:10] then filter here, vs. filter
    then [:10] there), which silently produced two different "seed 42,
    GeoLife" results for the same nominal configuration (peer review round
    3, M1) -- confirmed via the differing deterministic epsilon_g (30.06 vs.
    23.28 at sigma=1.0, which cannot be RNG variance since epsilon_g is a
    pure function of (sigma, q, T, delta)). GeoLife Fed-GAN cells are
    therefore sourced exclusively from run_fedgan_variance.py's file
    everywhere (Table VI and Table VII both), eliminating the gap rather
    than reconciling two independently-computed values.

    Returns None (caller falls back to computing fresh) if the source file
    doesn't cover every requested sigma -- e.g. seed 123's variance file
    only has the sigma=0.5 RR6 spot-check, not the full 1.0/4.0/8.0 sweep,
    so there is nothing to reuse and no duplicate-source gap to eliminate.
    """
    src_path = os.path.join(
        os.path.dirname(__file__), "results", f"fedgan_variance_seed{base_seed}.json"
    )
    with open(src_path) as f:
        src = json.load(f)
    if not all(str(sigma) in src for sigma in sigmas):
        return None
    reused = {}
    for sigma in sigmas:
        e = src[str(sigma)]
        reused[f"fedgan_sigma_{sigma}"] = {
            "n_trials": e["n_trials"],
            "mi_auc_advantage_mean": e["mi_auc_advantage_mean"],
            "mi_auc_advantage_ci95": e["mi_auc_advantage_ci95"],
            "mi_advantage_mean": e["mi_advantage_mean"],
            "mi_advantage_ci95": e["mi_advantage_ci95"],
            "reused_from": os.path.basename(src_path),
        }
    return reused


def run_config(client_data, sigma, n_rounds, window_size, repetitions, base_seed, label):
    print(f"\n{label}: running {repetitions} trials...", flush=True)
    trials = []
    for rep in range(repetitions):
        trial_seed = base_seed * 1000 + int(sigma * 100) * 10 + rep
        t0 = time.perf_counter()
        r = train_and_attack(client_data, sigma, n_rounds, window_size, trial_seed)
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
    print(f"  => {label}: AUC-adv {auc_mean:+.4f} +/- {auc_ci:.4f}  (acc-adv {acc_mean:+.4f} +/- {acc_ci:.4f})", flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (default 42; use 123 for the R10 cross-seed check)")
    args = parser.parse_args()

    n_rounds = 20
    window_size = 50
    repetitions = 8
    base_seed = args.seed
    sigmas = [1.0, 4.0, 8.0]

    all_results = {}

    for dataset in ["geolife", "tdrive", "porto"]:
        norm = load_dataset(dataset, n_entities=30, window_size=window_size, seed=base_seed)
        federated_clients = norm[:10]          # 10-client federation, matches paper setup
        # Pool the SAME 10 clients' data the federated condition uses (not all
        # 30 loaded clients) so the two conditions are matched on training-data
        # volume; otherwise the centralised baseline's null MI result could be
        # attributable to seeing ~3x more data rather than to model capacity
        # (peer review R7).
        centralised_pool  = [np.concatenate(federated_clients)]

        dataset_results = {}
        reused = load_geolife_reuse(base_seed, sigmas) if dataset == "geolife" else None
        if reused is not None:
            dataset_results.update(reused)
        else:
            for sigma in sigmas:
                key = f"fedgan_sigma_{sigma}"
                dataset_results[key] = run_config(
                    federated_clients, sigma, n_rounds, window_size, repetitions, base_seed,
                    label=f"{dataset} Fed-GAN sigma={sigma}",
                )
        dataset_results["centralised"] = run_config(
            centralised_pool, 0.01, n_rounds, window_size, repetitions, base_seed,
            label=f"{dataset} Centralised (no DP)",
        )
        all_results[dataset] = dataset_results

    out_path = os.path.join(os.path.dirname(__file__), "results", f"fedgan_mi_all_variance_seed{base_seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    print("\n=== SUMMARY (AUC-based advantage, mean +/- 95% CI) ===")
    print(f"{'config':>28}  {'GeoLife':>18}  {'T-Drive':>18}  {'Porto':>18}")
    rows = [("Fed-GAN sigma=1.0", "fedgan_sigma_1.0"),
            ("Fed-GAN sigma=4.0", "fedgan_sigma_4.0"),
            ("Fed-GAN sigma=8.0", "fedgan_sigma_8.0"),
            ("Centralised (no DP)", "centralised")]
    for label, key in rows:
        cells = []
        for dataset in ["geolife", "tdrive", "porto"]:
            r = all_results[dataset][key]
            cells.append(f"{r['mi_auc_advantage_mean']:+.4f}+/-{r['mi_auc_advantage_ci95']:.4f}")
        print(f"{label:>28}  {cells[0]:>18}  {cells[1]:>18}  {cells[2]:>18}")


if __name__ == "__main__":
    main()
