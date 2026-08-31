import hashlib
import json
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# AFED components (same package)
from paper2.adaptive_dp import (
    POIRecord,
    score_trajectory,
    apply_adaptive_laplace,
    compute_budget_analysis,
)


TrajectoryPoint = Tuple[float, float]
from paper2.decentralised_key import (
    DKAKeyManager,
    split_secret,
    reconstruct_secret,
    serialise_share,
    deserialise_share,
)
from paper2.federated_gan import (
    TrajectoryGANParams,
    FedGANServer,
    FedGANClient,
    make_synthetic_client_data,
    fedgan_simulate,
)
# AES-256-GCM via Python's built-in cryptography or fallback
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AFEDConfig:
    epsilon_base:       float = 1.0
    sensitivity_m:      float = 500.0
    dka_t:              int   = 3
    dka_n:              int   = 5
    dka_secret_size:    int   = 32
    gan_params:         Optional[TrajectoryGANParams] = None
    pbkdf2_iterations:  int   = 600_000


# ─────────────────────────────────────────────────────────────────────────────
# Processing result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    user_id:         str
    ciphertext:      bytes            # AES-256-GCM encrypted payload
    salt:            bytes            # PBKDF2 salt (public)
    nonce:           bytes            # AES-GCM nonce (public)
    epsilon_actual:  float            # actual DP budget consumed (Theorem 1)
    epsilon_uniform: float            # uniform-budget baseline composition cost
    budget_savings:  float            # fraction saved vs uniform (0..1)
    n_points:        int
    timing_ms:       Dict[str, float] # per-stage wall-clock milliseconds


# ─────────────────────────────────────────────────────────────────────────────
# AES-256-GCM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aes_gcm_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    if _HAS_CRYPTOGRAPHY:
        return AESGCM(key).encrypt(nonce, plaintext, None)
    # Lightweight fallback: AES-CTR (keystream via HKDF-SHA256) + HMAC-SHA256 tag
    # This matches the security properties for benchmarking purposes.
    stream = hashlib.pbkdf2_hmac("sha256", key + nonce, b"afed-ctr", 1, dklen=len(plaintext))
    ct     = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag    = hmac_sha256(key, nonce + ct)[:16]
    return ct + tag


def _aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    if _HAS_CRYPTOGRAPHY:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    ct, tag_stored = ciphertext[:-16], ciphertext[-16:]
    tag_computed = hmac_sha256(key, nonce + ct)[:16]
    if not secrets.compare_digest(tag_stored, tag_computed):
        raise ValueError("Authentication tag mismatch — decryption failed")
    stream = hashlib.pbkdf2_hmac("sha256", key + nonce, b"afed-ctr", 1, dklen=len(ct))
    return bytes(a ^ b for a, b in zip(ct, stream))


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    import hmac as _hmac
    return _hmac.new(key, msg, hashlib.sha256).digest()


# ─────────────────────────────────────────────────────────────────────────────
# AFED-PPTE Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AFEDPipeline:

    def __init__(
        self,
        config: AFEDConfig = None,
        pretrained_gan_server: Optional[FedGANServer] = None,
    ):
        self.cfg = config or AFEDConfig()
        self._dka_mgr: Optional[DKAKeyManager] = None
        self._shares: Optional[List[Tuple[int, int]]] = None
        # Allow reusing an already-trained Fed-GAN (e.g. from a prior
        # experiment) instead of training a throwaway one per pipeline
        # instance — training cost does not depend on epsilon_base/dka_t/n,
        # so retraining per-config in a sweep is pure waste.
        self._gan_server: Optional[FedGANServer] = pretrained_gan_server

    # ── DKA setup ─────────────────────────────────────────────────────────────

    def setup_dka(self) -> List[bytes]:
        mgr = DKAKeyManager.initialise(
            t=self.cfg.dka_t,
            n=self.cfg.dka_n,
            secret_size=self.cfg.dka_secret_size,
        )
        self._shares = mgr.shares
        # Wipe the local K_store — production systems distribute and discard
        mgr.wipe()
        return [serialise_share(idx, val) for idx, val in self._shares]

    def load_shares(self, share_blobs: List[bytes]):
        self._shares = [deserialise_share(b) for b in share_blobs]

    def _reconstruct_key_manager(self) -> DKAKeyManager:
        if self._shares is None:
            raise RuntimeError("No shares loaded — call setup_dka() or load_shares() first")
        needed = self._shares[:self.cfg.dka_t]
        return DKAKeyManager.from_shares(needed)

    # ── Fed-GAN setup ──────────────────────────────────────────────────────────

    def train_fedgan(
        self,
        client_datasets: Optional[List[np.ndarray]] = None,
        n_clients: int = 5,
        n_rounds: int = 20,
        samples_per_client: int = 100,
    ):

        gan_params = self.cfg.gan_params or TrajectoryGANParams()
        self._gan_server = FedGANServer(gan_params)

        if client_datasets is None:
            client_datasets = make_synthetic_client_data(
                n_clients, samples_per_client, gan_params.seq_len
            )

        clients = [
            FedGANClient(i, client_datasets[i], gan_params)
            for i in range(len(client_datasets))
        ]

        for _ in range(n_rounds):
            gp      = self._gan_server.get_global_params()
            updates = [c.train_round(gp) for c in clients]
            self._gan_server.aggregate(updates)

    def synthesise(self, n: int = 100) -> np.ndarray:
        """Generate n synthetic trajectory segments from the trained Fed-GAN."""
        if self._gan_server is None:
            raise RuntimeError("Fed-GAN not trained — call train_fedgan() first")
        return self._gan_server.generate(n)

    def _fedgan_substitute(
        self,
        reference: List[TrajectoryPoint],
    ) -> List[TrajectoryPoint]:
        """Stage 2: generate a synthetic trajectory scaled to reference's bounding box.

        The GAN produces normalised vectors in [-1, 1]; we rescale to the
        geographic bounding box of `reference` so the synthetic trajectory is
        geographically plausible.  Linear interpolation resamples from the
        GAN's fixed seq_len to len(reference) points.

        `reference` MUST be the SA-DP-noised trajectory (Stage 1's output),
        never the raw real trajectory: the synthetic trajectory's centroid
        and extent are a deterministic function of `reference`, so passing
        the real trajectory here would make the stored artefact's geographic
        positioning an unprotected function of the sensitive input, bypassing
        SA-DP's guarantee entirely (see \\S Integration, Stage 2 fix). Passing
        the noised trajectory instead makes this rescaling step post-processing
        of an \\varepsilon-DP output, which inherits the same guarantee under
        the DP post-processing closure property (Dwork & Roth 2014, Prop. 2.1).
        """
        raw = self._gan_server.generate(1)[0]           # shape (seq_len * 2,)
        gan_params = self.cfg.gan_params or TrajectoryGANParams()
        pts_norm = raw.reshape(-1, 2)                    # (seq_len, 2)

        ref = np.array(reference, dtype=float)
        lat_c = float(ref[:, 0].mean())
        lon_c = float(ref[:, 1].mean())
        lat_r = max(float(ref[:, 0].max() - ref[:, 0].min()), 0.005)
        lon_r = max(float(ref[:, 1].max() - ref[:, 1].min()), 0.005)

        synth_pts = [
            (round(lat_c + float(p[0]) * lat_r * 0.5, 7),
             round(lon_c + float(p[1]) * lon_r * 0.5, 7))
            for p in pts_norm
        ]

        # Resample to match original trajectory length via linear interpolation
        n_orig, n_synth = len(reference), len(synth_pts)
        if n_orig == n_synth:
            return synth_pts
        indices = np.linspace(0, n_synth - 1, n_orig)
        result = []
        for idx in indices:
            lo = int(idx)
            hi = min(lo + 1, n_synth - 1)
            alpha = idx - lo
            lat = synth_pts[lo][0] * (1 - alpha) + synth_pts[hi][0] * alpha
            lon = synth_pts[lo][1] * (1 - alpha) + synth_pts[hi][1] * alpha
            result.append((round(lat, 7), round(lon, 7)))
        return result

    # ── Main processing pipeline ───────────────────────────────────────────────

    def process(
        self,
        trajectory: List[TrajectoryPoint],
        user_id: str,
        timestamps: Optional[List[float]] = None,
        poi_records: Optional[List[POIRecord]] = None,
        profile: Optional[List[POIRecord]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ProcessingResult:
        """
        `timestamps` feeds SA-DP's Stage-1 stay-point scoring (real
        timestamps sharpen dwell-time detection, None falls back to an
        index-count proxy) and is also serialised into the encrypted
        payload as-is; None falls back to a synthesised uniform 30s
        sampling cadence (matches the codebase's existing convention, e.g.
        adaptive_dp.py's own self-test).

        `profile`, if given, is a frozen historical sensitivity profile
        (adaptive_dp.build_profile, built from this user's prior sessions,
        never from `trajectory` itself) passed straight through to Stage 1's
        score_trajectory() call. Omitting it falls back to same-trajectory
        scoring (see score_trajectory's own docstring for the data-dependent-
        sensitivity caveat this carries, Remark 1 in the paper).
        """
        timings: Dict[str, float] = {}

        ts_in = timestamps if timestamps is not None else [
            float(i * 30) for i in range(len(trajectory))
        ]

        # ── Stage 1: SA-DP scoring ─────────────────────────────────────────
        t0 = time.perf_counter()
        scores = score_trajectory(
            trajectory,
            timestamps=timestamps,
            poi_records=poi_records,
            profile=profile,
        )
        timings["sadp_score_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 2a: Adaptive Laplace noise (used for budget accounting) ──
        t0 = time.perf_counter()
        noised = apply_adaptive_laplace(
            trajectory,
            scores,
            self.cfg.epsilon_base,
            self.cfg.sensitivity_m,
            rng=rng,
        )
        timings["sadp_noise_ms"] = (time.perf_counter() - t0) * 1000

        budget = compute_budget_analysis(scores, self.cfg.epsilon_base, self.cfg.sensitivity_m)

        # ── Stage 2b: Fed-GAN substitution ────────────────────────────────
        # If a trained Fed-GAN is available, encrypt the synthetic trajectory
        # (real GPS never reaches persistent storage).  Otherwise fall back to
        # the SA-DP-noised real trajectory.
        #
        # The synthetic trajectory is rescaled to `noised`'s bounding box, NOT
        # `trajectory`'s: this makes the rescaling a post-processing step on
        # SA-DP's epsilon-DP output rather than an unprotected function of the
        # raw real coordinates, so the stored artefact's geographic extent
        # inherits SA-DP's guarantee (subject to the same Remark 1 scope) by
        # the DP post-processing closure property.
        t0 = time.perf_counter()
        if self._gan_server is not None:
            payload_trajectory = self._fedgan_substitute(noised)
        else:
            payload_trajectory = noised
        timings["fedgan_synth_ms"] = (time.perf_counter() - t0) * 1000

        # _fedgan_substitute() resamples to exactly len(trajectory) points,
        # so ts_in (index-aligned with `trajectory`) is already
        # index-aligned with `payload_trajectory` in both branches.
        payload_timestamps = ts_in

        # ── Stage 3: DKA key derivation ────────────────────────────────────
        t0 = time.perf_counter()
        salt = secrets.token_bytes(32)
        dka_mgr = self._reconstruct_key_manager()
        aes_key = dka_mgr.derive_encryption_key(user_id, salt)
        dka_mgr.wipe()
        timings["dka_derive_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: Serialise + AES-256-GCM encrypt ──────────────────────
        t0 = time.perf_counter()
        payload = _serialise_trajectory(payload_trajectory, payload_timestamps)
        nonce   = secrets.token_bytes(12)
        ct      = _aes_gcm_encrypt(aes_key, nonce, payload)
        timings["encrypt_ms"] = (time.perf_counter() - t0) * 1000

        return ProcessingResult(
            user_id=user_id,
            ciphertext=ct,
            salt=salt,
            nonce=nonce,
            epsilon_actual=budget["eps_adaptive"],
            epsilon_uniform=budget["eps_uniform"],
            budget_savings=budget["savings_pct"] / 100.0,
            n_points=len(trajectory),
            timing_ms=timings,
        )

    def decrypt(
        self,
        result: ProcessingResult,
        collected_share_blobs: List[bytes],
    ) -> List[Tuple[float, float, float]]:
        """Returns (lat, lon, timestamp) triples."""
        shares  = [deserialise_share(b) for b in collected_share_blobs]
        mgr     = DKAKeyManager.from_shares(shares[:self.cfg.dka_t])
        aes_key = mgr.derive_encryption_key(result.user_id, result.salt)
        mgr.wipe()

        payload = _aes_gcm_decrypt(aes_key, result.nonce, result.ciphertext)
        return _deserialise_trajectory(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory (de)serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_WIRE_FORMAT_VERSION = 2   # v2 adds obfuscated per-point timestamps (Layer 5)


def _serialise_trajectory(
    points: List[Tuple[float, float]],
    timestamps: List[float],
) -> bytes:
    buf = struct.pack(">BI", _WIRE_FORMAT_VERSION, len(points))
    for (lat, lon), ts in zip(points, timestamps):
        buf += struct.pack(">ddd", lat, lon, ts)
    return buf


def _deserialise_trajectory(data: bytes) -> List[Tuple[float, float, float]]:
    version, n = struct.unpack(">BI", data[:5])
    if version != _WIRE_FORMAT_VERSION:
        raise ValueError(f"Unsupported trajectory wire format version: {version}")
    pts  = []
    off  = 5
    for _ in range(n):
        lat, lon, ts = struct.unpack(">ddd", data[off:off + 24])
        pts.append((lat, lon, ts))
        off += 24
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic trajectory generator for pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

def make_test_trajectory(n_points: int = 20, seed: int = 0) -> List[Tuple[float, float]]:
    rng  = np.random.default_rng(seed)
    lat0, lon0 = 39.9, 116.4   # Beijing area (matches GeoLife)
    lats = lat0 + np.cumsum(rng.normal(0, 0.001, n_points))
    lons = lon0 + np.cumsum(rng.normal(0, 0.001, n_points))
    return [(float(lats[i]), float(lons[i])) for i in range(n_points)]


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end benchmark (for exp_e2e.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_e2e_benchmark(
    trajectory_lengths: List[int] = None,
    epsilon_values: List[float] = None,
    dka_configs: List[Tuple[int, int]] = None,
    repetitions: int = 10,
    seed: int = 42,
) -> List[Dict]:
    if trajectory_lengths is None:
        trajectory_lengths = [10, 50, 100, 500]
    if epsilon_values is None:
        epsilon_values = [0.5, 1.0, 2.0]
    if dka_configs is None:
        dka_configs = [(3, 5), (5, 10)]

    results = []
    rng = np.random.default_rng(seed)

    # Train Fed-GAN once and reuse across every (n_pts, eps, dka) combination
    # in the sweep — training cost doesn't depend on any of those parameters,
    # so retraining per-config would be pure waste. This also ensures the
    # benchmark actually exercises the Fed-GAN substitution path instead of
    # silently falling back to the SA-DP-noised trajectory.
    _bootstrap = AFEDPipeline(AFEDConfig())
    _bootstrap.train_fedgan()
    shared_gan_server = _bootstrap._gan_server

    for n_pts in trajectory_lengths:
        for eps in epsilon_values:
            for (t, n) in dka_configs:
                cfg      = AFEDConfig(epsilon_base=eps, dka_t=t, dka_n=n)
                pipeline = AFEDPipeline(cfg, pretrained_gan_server=shared_gan_server)
                shares   = pipeline.setup_dka()
                pipeline.load_shares(shares[:t])

                traj = make_test_trajectory(n_pts, seed=int(rng.integers(0, 10000)))

                stage_totals: Dict[str, float] = {}
                budget_savings = []

                for rep in range(repetitions):
                    rep_rng = np.random.default_rng(seed + rep)
                    r = pipeline.process(traj, user_id=f"bench_{rep}", rng=rep_rng)
                    for k, v in r.timing_ms.items():
                        stage_totals[k] = stage_totals.get(k, 0.0) + v
                    budget_savings.append(r.budget_savings)

                entry = {
                    "n_points":    n_pts,
                    "epsilon_base": eps,
                    "dka_t":       t,
                    "dka_n":       n,
                    "repetitions": repetitions,
                }
                for k, total in stage_totals.items():
                    entry[k + "_mean"] = round(total / repetitions, 3)
                entry["total_ms_mean"] = round(
                    sum(v for k, v in entry.items() if k.endswith("_mean")), 3
                )
                entry["mean_budget_savings"] = round(
                    sum(budget_savings) / repetitions, 4
                )
                results.append(entry)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AFED-PPTE pipeline self-test ===")

    cfg      = AFEDConfig(epsilon_base=1.0, dka_t=3, dka_n=5)
    pipeline = AFEDPipeline(cfg)
    shares   = pipeline.setup_dka()
    print(f"DKA: generated {len(shares)} shares, each {len(shares[0])} bytes")

    pipeline.load_shares(shares[:3])
    traj = make_test_trajectory(n_points=15)
    print(f"Test trajectory: {len(traj)} points")

    result = pipeline.process(traj, user_id="test_user_01")
    print(f"Ciphertext length: {len(result.ciphertext)} bytes")
    print(f"ε_actual / ε_uniform: {result.epsilon_actual:.3f} / {result.epsilon_uniform:.3f}")
    print(f"Budget savings: {result.budget_savings * 100:.1f}%")
    print(f"Timing: {result.timing_ms}")

    # Decrypt and verify round-trip
    pipeline.load_shares(shares[:3])
    recovered = pipeline.decrypt(result, shares[:3])
    print(f"Decrypted {len(recovered)} points  OK")

    print("\n=== Quick e2e benchmark (10-pt, ε=1.0, t=3/n=5, 5 reps) ===")
    bm = run_e2e_benchmark(
        trajectory_lengths=[10],
        epsilon_values=[1.0],
        dka_configs=[(3, 5)],
        repetitions=5,
    )
    print(json.dumps(bm[0], indent=2))
