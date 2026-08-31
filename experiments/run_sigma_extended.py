import json
import os
import sys
import time

import numpy as np

_CODING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODING_DIR not in sys.path:
    sys.path.insert(0, _CODING_DIR)

_PKG = "Adaptive_Federated_Ephemeral_Differential"

from Adaptive_Federated_Ephemeral_Differential.federated_gan import (
    TrajectoryGANParams, FedGANServer, FedGANClient,
    MembershipInferenceAttack,
)
from Adaptive_Federated_Ephemeral_Differential.experiments.exp_fedgan import (
    measure_generation_quality,
)
from Adaptive_Federated_Ephemeral_Differential.data.loader_geolife import (
    load_as_client_datasets,
)


def run_sigma_extended(
    sigmas=(4.0, 8.0),
    n_clients=10,
    n_rounds=20,
    window_size=50,
    seed=42,
):
    """Train Fed-GAN for new σ values on GeoLife data, matching alldata setup."""
    print(f"Loading GeoLife data (n_clients={n_clients}, window_size={window_size})...")
    ids, windows_per_client = load_as_client_datasets(
        n_users=182, window_size=window_size, stride=window_size // 2, seed=seed
    )

    # Normalise per-client to [-1,1] — same as run_real.py
    norm_arrays = []
    for windows in windows_per_client:
        pts  = np.array([[[p[0], p[1]] for p in w] for w in windows])
        flat = pts.reshape(len(windows), -1)
        lo   = flat.min(axis=1, keepdims=True)
        hi   = flat.max(axis=1, keepdims=True)
        norm = 2.0 * (flat - lo) / (hi - lo + 1e-8) - 1.0
        norm_arrays.append(norm)

    eligible = [(i, d) for i, d in enumerate(norm_arrays) if len(d) >= 20][:n_clients]
    client_data = [d for _, d in eligible]
    print(f"  Using {len(client_data)} clients.")

    rng_master = np.random.default_rng(seed)
    results = []

    for sigma in sigmas:
        print(f"\n  σ = {sigma} ...")
        t0 = time.perf_counter()

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
        for rnd in range(n_rounds):
            gp      = server.get_global_params()
            updates = [c.train_round(gp) for c in clients]
            stats   = server.aggregate(updates)
            d_losses.append(stats["mean_d_loss"])
            g_losses.append(stats["mean_g_loss"])

        eps = max(c.privacy.compute_epsilon(1e-5) for c in clients)

        # Generation quality
        real_sample = np.concatenate([d[:20] for d in client_data])
        fake_sample = server.generate(len(real_sample))
        gq = measure_generation_quality(
            real_sample, fake_sample, seq_len=window_size, rng=rng_master
        )

        # MI attack
        held_out = np.concatenate(
            [d[20:40] for d in client_data if len(d) >= 40]
        )
        if len(held_out) > 0:
            n_att  = min(30, len(real_sample), len(held_out))
            mi     = MembershipInferenceAttack()
            mi_res = mi.fit_and_evaluate(
                server.gan, real_sample[:n_att], held_out[:n_att], rng=rng_master
            )
            mi_acc = mi_res["attack_accuracy"]
            mi_adv = mi_res["advantage"]
        else:
            mi_acc, mi_adv = 0.5, 0.0

        elapsed = time.perf_counter() - t0
        row = {
            "noise_multiplier": sigma,
            "n_clients":        len(client_data),
            "n_rounds":         n_rounds,
            "final_d_loss":     round(d_losses[-1], 4),
            "final_g_loss":     round(g_losses[-1], 4),
            "final_epsilon":    round(float(eps), 4),
            "kl_trip_length":   gq["kl_trip_length"],
            "kl_speed":         gq["kl_speed"],
            "mean_frechet":     gq["mean_frechet_dist"],
            "mi_accuracy":      round(mi_acc, 4),
            "mi_advantage":     round(mi_adv, 4),
            "elapsed_s":        round(elapsed, 2),
        }
        results.append(row)
        print(f"    ε_g={eps:.4f}  D-loss={d_losses[-1]:.4f}"
              f"  KL={gq['kl_trip_length']:.5f}"
              f"  Fréchet={gq['mean_frechet_dist']:.5f}"
              f"  MI_acc={mi_acc:.4f}  [{elapsed:.1f}s]")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  Fed-GAN σ ∈ {4.0, 8.0} — GeoLife (extending Table 6)")
    print("=" * 60)

    rows = run_sigma_extended(sigmas=[4.0, 8.0], n_clients=10, n_rounds=20, seed=42)

    out_path = os.path.join(
        os.path.dirname(__file__), "results", "fedgan_sigma_extended_seed42.json"
    )
    with open(out_path, "w") as f:
        json.dump({"fedgan_extended": rows, "seed": 42}, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Print paper-ready table rows ──────────────────────────────────────
    print("\n=== TABLE 6 NEW ROWS (paper2.tex) ===")
    print(f"{'σ':>4}  {'ε_g':>7}  {'D-loss':>7}  {'KL':>8}  {'Fréchet':>8}  MI_acc")
    for row in rows:
        print(f"{row['noise_multiplier']:>4.1f}  "
              f"{row['final_epsilon']:>7.4f}  "
              f"{row['final_d_loss']:>7.4f}  "
              f"{row['kl_trip_length']:>8.5f}  "
              f"{row['mean_frechet']:>8.5f}  "
              f"{row['mi_accuracy']:.4f}")
