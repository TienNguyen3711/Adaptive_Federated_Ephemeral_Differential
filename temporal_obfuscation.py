import hashlib
import hmac
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

def derive_place_id(lat: float, lng: float, grid_m: float = 100.0) -> bytes:
    meters_per_degree = 111_000
    cell_deg = grid_m / meters_per_degree
    ilat = round(lat / cell_deg)
    ilng = round(lng / cell_deg)
    return f"{ilat}:{ilng}".encode("utf-8")


def _place_rng(place_key: bytes, place_id: bytes) -> np.random.Generator:
    digest = hmac.new(place_key, place_id, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed)


def derive_place_ids_for_trajectory(
    trajectory: List[Tuple[float, float]],
    timestamps: Optional[List[float]] = None,
    dist_thresh_m: float = 200.0,
    time_thresh_s: float = 300.0,
    grid_m: float = 100.0,
) -> List[Optional[bytes]]:
    from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import detect_stay_points

    n = len(trajectory)
    place_ids: List[Optional[bytes]] = [None] * n
    stays = detect_stay_points(
        trajectory, timestamps, dist_thresh_m=dist_thresh_m, time_thresh_s=time_thresh_s
    )
    if not stays:
        return place_ids

    ranked = sorted(stays, key=lambda s: (s.visit_count, s.duration), reverse=True)
    for sp in ranked[:2]:   # home (rank 0), work (rank 1) -- S=5 in adaptive_dp
        base_id = derive_place_id(sp.lat, sp.lng, grid_m=grid_m)
        total = len(sp.point_indices)
        for local_idx, idx in enumerate(sp.point_indices):
            place_ids[idx] = base_id + b":" + str(local_idx).encode("ascii") + b"/" + str(total).encode("ascii")
    return place_ids


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TemporalObfuscationParams:
    epsilon:             float = 1.0     # DP budget for the timestamp-jitter step
    jitter_scale_base_s: float = 300.0   # Lap(0, jitter_scale_base_s * S_i / epsilon)
    min_gap_s:           float = 1.0     # monotonicity floor between consecutive points
    speed_low:           float = 0.7
    speed_high:          float = 1.3
    n_pauses:            int   = 2
    pause_low_s:         float = 180.0
    pause_high_s:        float = 1800.0


# ── Laplace sampler (same inverse-CDF construction as adaptive_dp._laplace) ──

def _laplace(scale: float, rng: np.random.Generator) -> float:
    u = float(rng.uniform(-0.5 + 1e-10, 0.5 - 1e-10))
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Laplace jitter
# ─────────────────────────────────────────────────────────────────────────────

def _laplace_jitter(
    timestamps: List[float],
    params: TemporalObfuscationParams,
    rng: np.random.Generator,
    sensitivity_scores: Optional[List[float]] = None,
    place_ids: Optional[List[Optional[bytes]]] = None,
    place_key: Optional[bytes] = None,
) -> List[float]:
    cluster_t0: dict = {}
    cluster_t1: dict = {}
    if place_ids is not None and place_key is not None:
        for i, pid in enumerate(place_ids):
            if pid is None:
                continue
            base_id, _, rest = pid.rpartition(b":")
            local_idx = int(rest.split(b"/")[0])
            if local_idx == 0:
                cluster_t0[base_id] = timestamps[i]
            cluster_t1[base_id] = timestamps[i]   # last write wins == highest local_idx seen

    out: List[float] = []
    prev: Optional[float] = None
    cluster_anchor: dict = {}
    cluster_departure_offset: dict = {}
    for i, t in enumerate(timestamps):
        s_i = sensitivity_scores[i] if sensitivity_scores is not None else 1.0
        scale = params.jitter_scale_base_s * s_i / params.epsilon
        pid = place_ids[i] if place_ids is not None else None
        if pid is not None and place_key is not None:
            base_id, _, rest = pid.rpartition(b":")
            local_idx_s, _, total_s = rest.partition(b"/")
            local_idx, total = int(local_idx_s), int(total_s)
            if local_idx == 0:
                noisy = t + _laplace(scale, _place_rng(place_key, pid))
                if prev is not None:
                    noisy = max(noisy, prev + params.min_gap_s)
                cluster_anchor[base_id] = noisy
                cluster_departure_offset[base_id] = max(
                    params.min_gap_s,
                    abs(_laplace(scale, _place_rng(place_key, base_id + b":departure_offset"))),
                )
            elif total <= 1:
                noisy = cluster_anchor[base_id]
            elif local_idx == total - 1:
                noisy = cluster_anchor[base_id] + cluster_departure_offset[base_id]
            else:
                t0, t1 = cluster_t0[base_id], cluster_t1[base_id]
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                noisy = cluster_anchor[base_id] + frac * cluster_departure_offset[base_id]
        else:
            noisy = t + _laplace(scale, rng)
            if prev is not None:
                noisy = max(noisy, prev + params.min_gap_s)
        out.append(noisy)
        prev = noisy
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Speed obfuscation (inter-point interval scaling)
# ─────────────────────────────────────────────────────────────────────────────

def _speed_obfuscate(
    timestamps: List[float],
    params: TemporalObfuscationParams,
    rng: np.random.Generator,
    place_ids: Optional[List[Optional[bytes]]] = None,
) -> List[float]:
    if len(timestamps) < 2:
        return list(timestamps)
    out = [timestamps[0]]
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        in_place = place_ids is not None and place_ids[i] is not None
        factor = 1.0 if in_place else float(rng.uniform(params.speed_low, params.speed_high))
        out.append(out[-1] + delta * factor)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pause injection
# ─────────────────────────────────────────────────────────────────────────────

def _inject_pauses(
    timestamps: List[float],
    params: TemporalObfuscationParams,
    rng: np.random.Generator,
    place_ids: Optional[List[Optional[bytes]]] = None,
) -> List[float]:
    n = len(timestamps)
    if n < 2:
        return list(timestamps)

    candidates = np.arange(1, n)
    if place_ids is not None:
        candidates = np.array([i for i in candidates if place_ids[i] is None])
    n_pauses = min(params.n_pauses, len(candidates))
    if n_pauses <= 0:
        return list(timestamps)
    gap_indices = sorted(
        rng.choice(candidates, size=n_pauses, replace=False).tolist()
    )

    out = list(timestamps)
    cumulative = 0.0
    gi = 0
    for i in range(1, n):
        if gi < len(gap_indices) and i == gap_indices[gi]:
            cumulative += float(rng.uniform(params.pause_low_s, params.pause_high_s))
            gi += 1
        out[i] = out[i] + cumulative
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def obfuscate_temporal(
    timestamps: List[float],
    params: Optional[TemporalObfuscationParams] = None,
    rng: Optional[np.random.Generator] = None,
    sensitivity_scores: Optional[List[float]] = None,
    place_ids: Optional[List[Optional[bytes]]] = None,
    place_key: Optional[bytes] = None,
) -> List[float]:
    params = params or TemporalObfuscationParams()
    rng = rng or np.random.default_rng()

    if not timestamps:
        return []

    if sensitivity_scores is not None and len(sensitivity_scores) != len(timestamps):
        raise ValueError("sensitivity_scores must be the same length as timestamps")
    if place_ids is not None and len(place_ids) != len(timestamps):
        raise ValueError("place_ids must be the same length as timestamps")

    ts = _laplace_jitter(list(timestamps), params, rng, sensitivity_scores, place_ids, place_key)
    ts = _speed_obfuscate(ts, params, rng, place_ids)
    ts = _inject_pauses(ts, params, rng, place_ids)
    return ts


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Temporal Obfuscation self-test ===")
    rng = np.random.default_rng(42)
    ts_in = [float(i * 30) for i in range(20)]   # 30s cadence
    ts_out = obfuscate_temporal(ts_in, rng=rng)
    deltas = [ts_out[i + 1] - ts_out[i] for i in range(len(ts_out) - 1)]
    print(f"Input:  {ts_in[:5]} ...")
    print(f"Output: {[round(t, 2) for t in ts_out[:5]]} ...")
    print(f"Monotonic: {all(d > 0 for d in deltas)}")
    print(f"Total added duration: {ts_out[-1] - ts_in[-1]:.1f}s (jitter + pauses)")

    print("\n=== Per-point sensitivity scaling (S_i in [1,5]) ===")
    rng = np.random.default_rng(42)
    scores = [1.0] * 15 + [5.0] * 5   # last 5 points are a home/work dwell
    ts_out_scaled = obfuscate_temporal(ts_in, rng=rng, sensitivity_scores=scores)
    print(f"S_i=1 point jitter (index 0): {ts_out_scaled[0] - ts_in[0]:+.1f}s")
    print(f"S_i=5 point jitter (index 19, before pause injection shifts it further): "
          f"{ts_out_scaled[19] - ts_in[19]:+.1f}s")
