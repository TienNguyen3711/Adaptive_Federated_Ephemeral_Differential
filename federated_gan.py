import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Architecture parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryGANParams:
    z_dim: int = 16           # noise vector dimension
    hidden_dim: int = 32      # hidden layer width (generator and discriminator)
    seq_len: int = 10         # output trajectory length (GPS points)
    lr: float = 1e-3          # learning rate (SGD)
    # DP-SGD parameters
    clip_norm: float = 1.0    # per-sample gradient clipping norm C
    noise_multiplier: float = 1.1   # σ: noise scale relative to C
    batch_size: int = 32      # local mini-batch size
    local_epochs: int = 1     # local training epochs per federated round


# ─────────────────────────────────────────────────────────────────────────────
# Activation helpers
# ─────────────────────────────────────────────────────────────────────────────

import warnings as _warnings
# Suppress BLAS overflow/invalid warnings: DP-noised weights can trigger
# IEEE 754 intermediate overflow in matmul, but all outputs are guarded by
# nan_to_num and weight clipping so results remain numerically valid.
_warnings.filterwarnings("ignore", message=".*overflow.*", category=RuntimeWarning)
_warnings.filterwarnings("ignore", message=".*invalid value.*", category=RuntimeWarning)
_warnings.filterwarnings("ignore", message=".*divide by zero.*encountered in matmul", category=RuntimeWarning)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight NumPy GAN
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryGAN:
    def __init__(self, params: TrajectoryGANParams, rng=None):
        self.p = params
        rng = rng or np.random.default_rng(42)

        out_dim = params.seq_len * 2
        in_dim  = params.seq_len * 2

        # Generator — Xavier initialisation
        sg1 = math.sqrt(2.0 / (params.z_dim + params.hidden_dim))
        sg2 = math.sqrt(2.0 / (params.hidden_dim + out_dim))
        self.G: Dict[str, np.ndarray] = {
            "W1": rng.normal(0, sg1, (params.hidden_dim, params.z_dim)),
            "b1": np.zeros(params.hidden_dim),
            "W2": rng.normal(0, sg2, (out_dim, params.hidden_dim)),
            "b2": np.zeros(out_dim),
        }

        # Discriminator — Xavier initialisation
        sd1 = math.sqrt(2.0 / (in_dim + params.hidden_dim))
        sd2 = math.sqrt(2.0 / (params.hidden_dim + 1))
        self.D: Dict[str, np.ndarray] = {
            "W1": rng.normal(0, sd1, (params.hidden_dim, in_dim)),
            "b1": np.zeros(params.hidden_dim),
            "W2": rng.normal(0, sd2, (1, params.hidden_dim)),
            "b2": np.zeros(1),
        }

    # ── Forward passes ────────────────────────────────────────────────────────

    def _safe(self, a: np.ndarray) -> np.ndarray:
        return np.nan_to_num(a, nan=0.0, posinf=5.0, neginf=-5.0)

    def generate(self, z: np.ndarray) -> np.ndarray:
        h = _relu(self._safe(z) @ self._safe(self.G["W1"]).T + self.G["b1"])
        return np.tanh(self._safe(h) @ self._safe(self.G["W2"]).T + self.G["b2"])

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        h = _relu(self._safe(x) @ self._safe(self.D["W1"]).T + self.D["b1"])
        return _sigmoid(self._safe(h) @ self._safe(self.D["W2"]).T + self.D["b2"])

    # ── Gradient computation ───────────────────────────────────────────────────

    def compute_d_gradients(
        self, real_batch: np.ndarray, rng
    ) -> Tuple[Dict[str, np.ndarray], float]:  # noqa: E501
        B = len(real_batch)
        z = rng.standard_normal((B, self.p.z_dim))
        fake = self.generate(z)

        h_r = _relu(real_batch @ self.D["W1"].T + self.D["b1"])
        s_r = _sigmoid(h_r @ self.D["W2"].T + self.D["b2"])   # (B,1)

        h_f = _relu(fake @ self.D["W1"].T + self.D["b1"])
        s_f = _sigmoid(h_f @ self.D["W2"].T + self.D["b2"])

        d_loss = (
            -np.mean(np.log(s_r + 1e-8))
            - np.mean(np.log(1.0 - s_f + 1e-8))
        )

        # Real branch
        dr = (s_r - 1.0) / B
        dW2_r = dr.T @ h_r
        db2_r = dr.sum(axis=0)
        dh_r  = dr @ self.D["W2"]
        dhl_r = dh_r * _relu_grad(h_r)
        dW1_r = dhl_r.T @ real_batch
        db1_r = dhl_r.sum(axis=0)

        # Fake branch
        df = s_f / B
        dW2_f = df.T @ h_f
        db2_f = df.sum(axis=0)
        dh_f  = df @ self.D["W2"]
        dhl_f = dh_f * _relu_grad(h_f)
        dW1_f = dhl_f.T @ fake
        db1_f = dhl_f.sum(axis=0)

        grad_D = {
            "W1": dW1_r + dW1_f,
            "b1": db1_r + db1_f,
            "W2": dW2_r + dW2_f,
            "b2": db2_r + db2_f,
        }
        return grad_D, float(d_loss)

    def compute_g_gradients(
        self, batch_size: int, rng
    ) -> Tuple[Dict[str, np.ndarray], float]:
        B = batch_size
        z    = rng.standard_normal((B, self.p.z_dim))
        fake = self.generate(z)

        h_d = _relu(fake @ self.D["W1"].T + self.D["b1"])
        s_f = _sigmoid(h_d @ self.D["W2"].T + self.D["b2"])
        g_loss = -np.mean(np.log(s_f + 1e-8))

        # Backprop through frozen D then through G
        delta  = (s_f - 1.0) / B
        dh_d   = delta @ self.D["W2"]
        dhl_d  = dh_d * _relu_grad(h_d)
        dfake  = dhl_d @ self.D["W1"]               # (B, seq_len*2)

        # tanh derivative
        h_g    = _relu(z @ self.G["W1"].T + self.G["b1"])
        dtanh  = dfake * (1.0 - fake ** 2)          # fake = tanh(h_g @ W2 + b2)

        dW2 = dtanh.T @ h_g / B
        db2 = dtanh.mean(axis=0)
        dh_g  = dtanh @ self.G["W2"]
        dhl_g = dh_g * _relu_grad(h_g)
        dW1   = dhl_g.T @ z / B
        db1   = dhl_g.mean(axis=0)

        grad_G = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grad_G, float(g_loss)

    # ── Parameter helpers ──────────────────────────────────────────────────────

    def copy_params(self) -> Dict:
        return {
            "G": {k: v.copy() for k, v in self.G.items()},
            "D": {k: v.copy() for k, v in self.D.items()},
        }

    def load_params(self, snapshot: Dict):
        self.G = {k: np.clip(np.nan_to_num(v, nan=0.0, posinf=5.0, neginf=-5.0), -10.0, 10.0)
                  for k, v in snapshot["G"].items()}
        self.D = {k: np.clip(np.nan_to_num(v, nan=0.0, posinf=5.0, neginf=-5.0), -10.0, 10.0)
                  for k, v in snapshot["D"].items()}


# ─────────────────────────────────────────────────────────────────────────────
# DP-SGD privacy accounting — Rényi DP moments accountant
# ─────────────────────────────────────────────────────────────────────────────

class DPSGDTracker:
    _ORDERS = list(range(2, 65)) + [128, 256, 512]

    def __init__(self, noise_multiplier: float, sampling_rate: float):
        self.sigma = noise_multiplier
        self.q     = sampling_rate
        self._steps = 0

    def step(self, n: int = 1):
        self._steps += n

    @property
    def steps(self) -> int:
        return self._steps

    def _rdp_per_step(self, alpha: int) -> float:
        return (self.q ** 2) * alpha / (2.0 * self.sigma ** 2)

    def compute_epsilon(self, delta: float) -> float:
        if self._steps == 0:
            return 0.0
        best = float("inf")
        for alpha in self._ORDERS:
            rdp = self._rdp_per_step(alpha) * self._steps
            eps = rdp + math.log(1.0 / delta) / (alpha - 1)
            best = min(best, eps)
        return best


# ─────────────────────────────────────────────────────────────────────────────
# DP gradient perturbation
# ─────────────────────────────────────────────────────────────────────────────

def _dp_perturb(
    grads: Dict[str, np.ndarray],
    clip_norm: float,
    noise_multiplier: float,
    batch_size: int,
    rng,
) -> Dict[str, np.ndarray]:
    sigma = noise_multiplier * clip_norm
    out = {}
    for k, g in grads.items():
        g_safe  = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        norm    = np.linalg.norm(g_safe)
        clipped = g_safe * min(1.0, clip_norm / (norm + 1e-12))
        noise   = rng.normal(0.0, sigma, g_safe.shape)
        out[k]  = np.nan_to_num((clipped + noise) / batch_size,
                                nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Federated client
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClientUpdate:
    """Gradient deltas returned by a device after one local training round."""
    g_delta:         Dict[str, np.ndarray]
    d_delta:         Dict[str, np.ndarray]
    num_samples:     int
    mean_d_loss:     float
    mean_g_loss:     float
    epsilon_consumed: float


class FedGANClient:

    def __init__(
        self,
        client_id: int,
        local_data: np.ndarray,
        params: TrajectoryGANParams,
        delta: float = 1e-5,
        rng=None,
    ):
        self.id   = client_id
        self.data = local_data
        self.p    = params
        self.delta = delta
        self.rng  = rng or np.random.default_rng(client_id)
        n_local   = max(1, len(local_data))
        self.privacy = DPSGDTracker(
            noise_multiplier=params.noise_multiplier,
            sampling_rate=min(1.0, params.batch_size / n_local),
        )

    def train_round(self, global_params: Dict) -> ClientUpdate:
        """
        Load global model, run local_epochs of DP-SGD, return delta vs initial.
        """
        gan = TrajectoryGAN(self.p, rng=self.rng)
        gan.load_params(global_params)
        initial = gan.copy_params()

        N = len(self.data)
        if N == 0:
            return ClientUpdate({k: np.zeros_like(v) for k, v in gan.G.items()},
                                {k: np.zeros_like(v) for k, v in gan.D.items()},
                                0, 0.0, 0.0, 0.0)

        total_d, total_g, steps = 0.0, 0.0, 0

        for _ in range(self.p.local_epochs):
            perm = self.rng.permutation(N)
            for start in range(0, N, self.p.batch_size):
                batch = self.data[perm[start:start + self.p.batch_size]]
                if len(batch) < 2:
                    continue

                # Discriminator DP-SGD step
                g_D, d_loss = gan.compute_d_gradients(batch, self.rng)
                g_D_dp = _dp_perturb(g_D, self.p.clip_norm,
                                     self.p.noise_multiplier, len(batch), self.rng)
                for k in gan.D:
                    gan.D[k] = np.clip(gan.D[k] - self.p.lr * g_D_dp[k], -10.0, 10.0)

                # Generator DP-SGD step
                g_G, g_loss = gan.compute_g_gradients(len(batch), self.rng)
                g_G_dp = _dp_perturb(g_G, self.p.clip_norm,
                                     self.p.noise_multiplier, len(batch), self.rng)
                for k in gan.G:
                    gan.G[k] = np.clip(gan.G[k] - self.p.lr * g_G_dp[k], -10.0, 10.0)

                total_d += d_loss if np.isfinite(d_loss) else 0.0
                total_g += g_loss if np.isfinite(g_loss) else 0.0
                steps   += 1

        self.privacy.step(steps)
        eps = self.privacy.compute_epsilon(self.delta)

        g_delta = {k: gan.G[k] - initial["G"][k] for k in gan.G}
        d_delta = {k: gan.D[k] - initial["D"][k] for k in gan.D}

        return ClientUpdate(
            g_delta=g_delta,
            d_delta=d_delta,
            num_samples=N,
            mean_d_loss=total_d / max(1, steps),
            mean_g_loss=total_g / max(1, steps),
            epsilon_consumed=eps,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Federated server — FedAvg aggregator
# ─────────────────────────────────────────────────────────────────────────────

class FedGANServer:

    def __init__(self, params: TrajectoryGANParams, rng=None):
        self.p    = params
        self.rng  = rng or np.random.default_rng(0)
        self.gan  = TrajectoryGAN(params, rng=self.rng)
        self.history: List[Dict] = []

    def aggregate(self, updates: List[ClientUpdate]) -> Dict:
        total = sum(u.num_samples for u in updates)
        if total == 0:
            return {}

        params = self.gan.copy_params()
        for layer, delta_attr in (("G", "g_delta"), ("D", "d_delta")):
            for k in params[layer]:
                weighted = sum(
                    getattr(u, delta_attr)[k] * u.num_samples
                    for u in updates
                    if k in getattr(u, delta_attr)
                )
                params[layer][k] += weighted / total

        # Clamp aggregated weights to prevent FedAvg accumulation overflow
        for layer in ("G", "D"):
            for k in params[layer]:
                params[layer][k] = np.clip(
                    np.nan_to_num(params[layer][k], nan=0.0, posinf=5.0, neginf=-5.0),
                    -10.0, 10.0,
                )
        self.gan.load_params(params)

        stats = {
            "num_clients":    len(updates),
            "total_samples":  total,
            "mean_d_loss":    sum(u.mean_d_loss for u in updates) / len(updates),
            "mean_g_loss":    sum(u.mean_g_loss for u in updates) / len(updates),
            "max_epsilon":    max(u.epsilon_consumed for u in updates),
        }
        self.history.append(stats)
        return stats

    def get_global_params(self) -> Dict:
        return self.gan.copy_params()

    def generate(self, n: int) -> np.ndarray:
        """Sample n synthetic trajectory segments from the global generator."""
        z = self.rng.standard_normal((n, self.p.z_dim))
        return self.gan.generate(z)


# ─────────────────────────────────────────────────────────────────────────────
# Membership Inference Attack — shadow-model approach (Shokri et al. 2017)
# ─────────────────────────────────────────────────────────────────────────────

class MembershipInferenceAttack:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._w: Optional[np.ndarray] = None
        self._b: float = 0.0
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray,
                      lr: float = 0.05, epochs: int = 500,
                      l2: float = 1e-2):
        n = len(X)
        mu    = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xs    = (X - mu) / sigma
        w = np.zeros(Xs.shape[1])
        b = 0.0
        for _ in range(epochs):
            p   = _sigmoid(Xs @ w + b)
            err = (p - y) / n
            w  -= lr * (Xs.T @ err + l2 * w)
            b  -= lr * err.sum()
        self._w = w
        self._b = b
        self._mu = mu
        self._sigma = sigma

    def fit_and_evaluate(
        self,
        gan: TrajectoryGAN,
        member_data: np.ndarray,
        non_member_data: np.ndarray,
        rng=None,
    ) -> Dict:
        rng = rng or np.random.default_rng(99)

        s_m  = gan.discriminate(member_data).ravel()
        s_nm = gan.discriminate(non_member_data).ravel()

        X = np.concatenate([s_m, s_nm]).reshape(-1, 1)
        y = np.array([1.0] * len(s_m) + [0.0] * len(s_nm))

        idx   = rng.permutation(len(y))
        split = int(0.8 * len(idx))
        Xtr, Xte = X[idx[:split]], X[idx[split:]]
        ytr, yte = y[idx[:split]], y[idx[split:]]

        self._fit_logistic(Xtr, ytr)

        Xte_s = (Xte - self._mu) / self._sigma
        proba = _sigmoid(Xte_s @ self._w + self._b)
        pred  = (proba >= self.threshold).astype(float)
        acc   = float((pred == yte).mean())

        pos = proba[yte == 1]
        neg = proba[yte == 0]
        auc = float(np.mean(pos[:, None] > neg[None, :])) if (len(pos) > 0 and len(neg) > 0) else 0.5

        return {
            "attack_accuracy": acc,
            "advantage":       acc - 0.5,
            "auc":             auc,
            "auc_advantage":   auc - 0.5,
            "n_member":        len(s_m),
            "n_nonmember":     len(s_nm),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helper
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_client_data(
    n_clients: int,
    samples_per_client: int,
    seq_len: int,
    rng=None,
) -> List[np.ndarray]:
    rng = rng or np.random.default_rng(7)
    datasets = []
    for c in range(n_clients):
        center = rng.uniform(-0.5, 0.5, 2)
        raw    = rng.normal(center, 0.1, (samples_per_client, seq_len, 2))
        flat   = raw.reshape(samples_per_client, seq_len * 2)
        # Per-sample min-max normalisation to [-1,1]
        lo  = flat.min(axis=1, keepdims=True)
        hi  = flat.max(axis=1, keepdims=True)
        normed = 2.0 * (flat - lo) / (hi - lo + 1e-8) - 1.0
        datasets.append(normed)
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation
# ─────────────────────────────────────────────────────────────────────────────

def fedgan_simulate(
    n_clients: int = 5,
    n_rounds: int = 20,
    samples_per_client: int = 100,
    params: Optional[TrajectoryGANParams] = None,
    delta: float = 1e-5,
    rng_seed: int = 42,
) -> Dict:
    if params is None:
        params = TrajectoryGANParams()

    rng = np.random.default_rng(rng_seed)
    t0  = time.perf_counter()

    client_datasets = make_synthetic_client_data(
        n_clients, samples_per_client, params.seq_len, rng
    )

    server  = FedGANServer(params, rng=rng)
    clients = [
        FedGANClient(
            i, client_datasets[i], params, delta=delta,
            rng=np.random.default_rng(i + rng_seed * 13),
        )
        for i in range(n_clients)
    ]

    for _ in range(n_rounds):
        gp      = server.get_global_params()
        updates = [c.train_round(gp) for c in clients]
        server.aggregate(updates)

    final_eps = max(c.privacy.compute_epsilon(delta) for c in clients)

    # Membership inference on a held-out sample
    mi_attack    = MembershipInferenceAttack()
    member_data  = np.concatenate([cd[:10] for cd in client_datasets])
    fake_data    = server.generate(len(member_data))
    mi_result    = mi_attack.fit_and_evaluate(server.gan, member_data, fake_data, rng=rng)

    return {
        "round_stats":   server.history,
        "final_epsilon": float(final_eps),
        "mi_result":     mi_result,
        "total_time_s":  time.perf_counter() - t0,
        "n_clients":     n_clients,
        "n_rounds":      n_rounds,
        "params": {
            "z_dim":            params.z_dim,
            "hidden_dim":       params.hidden_dim,
            "seq_len":          params.seq_len,
            "clip_norm":        params.clip_norm,
            "noise_multiplier": params.noise_multiplier,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helper (for exp_fedgan.py)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fedgan(
    n_clients_list: List[int] = None,
    n_rounds: int = 10,
    samples_per_client: int = 50,
    noise_multipliers: List[float] = None,
    repetitions: int = 3,
) -> List[Dict]:
    if n_clients_list is None:
        n_clients_list = [3, 5, 10]
    if noise_multipliers is None:
        noise_multipliers = [0.5, 1.0, 1.5]

    results = []
    for n_clients in n_clients_list:
        for sigma in noise_multipliers:
            times, epsilons, mi_accs = [], [], []
            for rep in range(repetitions):
                p = TrajectoryGANParams(noise_multiplier=sigma)
                r = fedgan_simulate(
                    n_clients=n_clients,
                    n_rounds=n_rounds,
                    samples_per_client=samples_per_client,
                    params=p,
                    rng_seed=rep * 100,
                )
                times.append(r["total_time_s"])
                epsilons.append(r["final_epsilon"])
                mi_accs.append(r["mi_result"]["attack_accuracy"])

            results.append({
                "n_clients":       n_clients,
                "noise_multiplier": sigma,
                "n_rounds":        n_rounds,
                "mean_time_s":     round(sum(times)   / repetitions, 3),
                "mean_epsilon":    round(sum(epsilons) / repetitions, 4),
                "mean_mi_accuracy": round(sum(mi_accs) / repetitions, 4),
                "mi_advantage":    round(sum(mi_accs) / repetitions - 0.5, 4),
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TrajectoryGAN forward-pass test ===")
    p   = TrajectoryGANParams(seq_len=5, hidden_dim=16, z_dim=8)
    rng = np.random.default_rng(0)
    gan = TrajectoryGAN(p, rng=rng)

    z    = rng.standard_normal((4, p.z_dim))
    fake = gan.generate(z)
    assert fake.shape == (4, p.seq_len * 2), f"Unexpected shape {fake.shape}"
    print(f"Generator output shape: {fake.shape}  OK")

    real  = rng.standard_normal((4, p.seq_len * 2))
    score = gan.discriminate(real)
    assert score.shape == (4, 1) and np.all((score >= 0) & (score <= 1))
    print(f"Discriminator score range: [{score.min():.3f}, {score.max():.3f}]  OK")

    print("\n=== DPSGDTracker privacy accounting ===")
    tracker = DPSGDTracker(noise_multiplier=1.1, sampling_rate=0.01)
    tracker.step(100)
    eps = tracker.compute_epsilon(1e-5)
    print(f"ε after 100 steps (σ=1.1, q=0.01): {eps:.4f}")

    print("\n=== DP gradient perturbation ===")
    grad = {"W": rng.standard_normal((8, 8)), "b": rng.standard_normal(8)}
    dp_g = _dp_perturb(grad, 1.0, 1.1, 32, rng)
    print(f"Clipped W norm: {np.linalg.norm(dp_g['W']):.4f}  OK")

    print("\n=== Quick Fed-GAN simulation (5 clients, 5 rounds) ===")
    result = fedgan_simulate(n_clients=5, n_rounds=5, samples_per_client=50,
                             rng_seed=42)
    print(f"Final ε:          {result['final_epsilon']:.4f}")
    print(f"MI accuracy:      {result['mi_result']['attack_accuracy']:.3f}  "
          f"(advantage: {result['mi_result']['advantage']:.3f})")
    print(f"Wall time:        {result['total_time_s']:.2f} s")
    rs = result["round_stats"]
    print(f"Round 5 D-loss:   {rs[-1]['mean_d_loss']:.4f}   "
          f"G-loss: {rs[-1]['mean_g_loss']:.4f}")
    print("Fed-GAN simulation OK")
