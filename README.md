# AFED-PPTE

**Adaptive Federated Ephemeral Privacy-Preserving Trajectory Encoding**

Three orthogonal privacy components for trajectory encoding systems, addressing open problems in IoT-grade trajectory privacy:

| Component | Problem | Solution |
|---|---|---|
| **SA-DP** | Uniform DP budget wastes privacy at low-sensitivity locations | Semantic-adaptive Laplace noise: ε_i = ε_base / S_i |
| **DKA** | Single-server key storage creates full-compromise risk | Shamir (t,n)-threshold secret sharing |
| **Fed-GAN** | Centralised GAN training corpus exposes membership information | Federated training with DP-SGD gradient protection |

## Structure

```
paper2/
├── adaptive_dp.py          # SA-DP: semantic-adaptive differential privacy
├── decentralised_key.py    # DKA: Shamir threshold key agreement
├── federated_gan.py        # Fed-GAN: federated GAN with DP-SGD
├── afed_pipeline.py        # End-to-end integration pipeline
├── data/
│   ├── loader_geolife.py   # GeoLife Trajectories 1.3 loader
│   ├── loader_tdrive.py    # T-Drive Taxi Trajectories loader
│   └── loader_porto.py     # Porto Taxi dataset loader
└── experiments/
    ├── exp_sadp.py         # Experiments 1 & 2: SA-DP evaluation
    ├── exp_dka.py          # Experiment 3: DKA benchmark
    ├── exp_fedgan.py       # Experiments 4 & 5: Fed-GAN evaluation
    ├── exp_e2e.py          # Experiment 6: end-to-end pipeline
    ├── run_all.py          # Synthetic data runner (seed-fixed)
    ├── run_real.py         # Real dataset runner
    └── results/            # Experiment results (JSON, seed 42 & 123)
```

## Running Experiments

```bash
# Synthetic data (reproducible, seed 42)
python -m paper2.experiments.run_all --seed 42

# Real datasets
python -m paper2.experiments.run_real --dataset geolife --seed 42
python -m paper2.experiments.run_real --dataset all --alldata --seed 42
```

## Key Results (GeoLife / T-Drive / Porto, seed 42)

| Metric | GeoLife | T-Drive | Porto |
|---|---|---|---|
| SA-DP budget savings | 59.1% | 37.1% | 45.8% |
| Noise amplification at S≥3 | 3.75× | 4.09× | 3.82× |
| DKA reconstruct latency | 0.31ms | 0.31ms | 0.32ms |
| DKA overhead vs baseline | 0.0026× | 0.0026× | 0.0026× |
| E2E pipeline latency | 121ms | 124ms | 123ms |

Theorem 1 (budget efficiency bound) verified on 100% of trajectories across all three datasets.

## Requirements

- Python 3.9+
- numpy

No deep learning framework required. Pure NumPy implementation for IoT compatibility.
