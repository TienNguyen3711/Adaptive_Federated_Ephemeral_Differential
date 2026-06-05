import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

METERS_PER_DEGREE = 111_000

# ── Sensitivity table ─────────────────────────────────────────────────────────
# Maps POI / inferred-location category → sensitivity level S ∈ {1..5}.
SENSITIVITY_TABLE: dict = {
    "medical":       5,   # hospitals, clinics, pharmacies
    "religious":     5,   # mosques, churches, temples, synagogues
    "home_cluster":  4,   # inferred home address
    "work_cluster":  4,   # inferred workplace
    "legal":         4,   # courts, lawyers, embassies
    "political":     4,   # party offices, government buildings
    "education":     3,   # schools, universities
    "commercial":    2,   # shops, restaurants, gyms
    "transit":       1,   # bus stops, metro stations, roads
    "other":         1,
}

S_MAX = 5
S_MIN = 1

# ── Score weights (α + β + γ = 1) ────────────────────────────────────────────
_ALPHA = 0.4   # category contribution
_BETA  = 0.4   # proximity contribution
_GAMMA = 0.2   # dwell-time contribution


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class StayPoint:
    """A location where a user dwelled for a significant time."""
    lat: float
    lng: float
    arrival: float           # Unix seconds (or point index if no timestamps)
    departure: float
    duration: float          # seconds (or index steps)
    point_indices: List[int] = field(default_factory=list)
    visit_count: int = 1     # incremented across multiple visits


@dataclass
class POIRecord:
    """A single point-of-interest entry."""
    lat: float
    lng: float
    category: str            # key in SENSITIVITY_TABLE
    name: str = ""
    radius_m: float = 100.0  # influence radius for proximity scoring


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_m(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lng) points."""
    R = 6_371_000
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


def _centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


# ── Laplace sampler ───────────────────────────────────────────────────────────

def _laplace(scale: float) -> float:
    """Sample from Laplace(0, scale) via inverse-CDF transform."""
    u = random.uniform(-0.5 + 1e-10, 0.5 - 1e-10)
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Stay-point detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_stay_points(
    trajectory: List[Tuple[float, float]],
    timestamps: Optional[List[float]] = None,
    dist_thresh_m: float = 200.0,
    time_thresh_s: float = 300.0,
    min_points: int = 3,
) -> List[StayPoint]:
    n = len(trajectory)
    if n == 0:
        return []

    ts = timestamps if timestamps is not None else list(range(n))
    stay_points: List[StayPoint] = []
    i = 0

    while i < n - 1:
        cluster_pts = [trajectory[i]]
        cluster_idx = [i]
        j = i + 1

        while j < n:
            if _haversine_m(trajectory[i], trajectory[j]) <= dist_thresh_m:
                cluster_pts.append(trajectory[j])
                cluster_idx.append(j)
                j += 1
            else:
                break

        dwell = ts[cluster_idx[-1]] - ts[cluster_idx[0]]
        qualifies = (
            (timestamps is not None and dwell >= time_thresh_s)
            or (timestamps is None and len(cluster_pts) >= min_points)
        )

        if qualifies:
            clat, clng = _centroid(cluster_pts)
            stay_points.append(StayPoint(
                lat=round(clat, 7),
                lng=round(clng, 7),
                arrival=ts[cluster_idx[0]],
                departure=ts[cluster_idx[-1]],
                duration=dwell,
                point_indices=cluster_idx,
            ))
            i = cluster_idx[-1] + 1
        else:
            i += 1

    # Merge stay points that are within dist_thresh_m of each other
    merged: List[StayPoint] = []
    for sp in stay_points:
        matched = False
        for existing in merged:
            if _haversine_m((sp.lat, sp.lng), (existing.lat, existing.lng)) <= dist_thresh_m:
                existing.visit_count += 1
                existing.duration += sp.duration
                matched = True
                break
        if not matched:
            merged.append(sp)

    return sorted(merged, key=lambda s: s.arrival)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Home / work inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_home_work(stay_points: List[StayPoint]) -> List[POIRecord]:
    if not stay_points:
        return []

    ranked = sorted(stay_points, key=lambda s: (s.visit_count, s.duration), reverse=True)
    records: List[POIRecord] = []

    for rank, sp in enumerate(ranked):
        if rank == 0:
            cat = "home_cluster"
        elif rank == 1:
            cat = "work_cluster"
        else:
            cat = "commercial"
        records.append(POIRecord(lat=sp.lat, lng=sp.lng, category=cat, radius_m=200.0))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-point sensitivity scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_point(
    lat: float,
    lng: float,
    poi_records: List[POIRecord],
    stay_point: Optional[StayPoint],
    max_dwell: float,
) -> float:
    cat_score = S_MIN
    prox_score = 0.0

    if poi_records:
        best_dist = math.inf
        best_cat = S_MIN
        for poi in poi_records:
            d = _haversine_m((lat, lng), (poi.lat, poi.lng))
            if d < best_dist:
                best_dist = d
                best_cat = SENSITIVITY_TABLE.get(poi.category, S_MIN)
            # also check if within influence radius
            prox = max(0.0, 1.0 - d / max(poi.radius_m, 1.0))
            if prox > prox_score:
                prox_score = min(prox, 1.0)
                cat_score = max(cat_score, SENSITIVITY_TABLE.get(poi.category, S_MIN))

    dwell_score = 0.0
    if stay_point is not None and max_dwell > 0:
        dwell_score = min(1.0, stay_point.duration / max_dwell)

    raw = _ALPHA * cat_score + _BETA * prox_score * S_MAX + _GAMMA * dwell_score * S_MAX
    return max(float(S_MIN), min(float(S_MAX), raw))


def score_trajectory(
    trajectory: List[Tuple[float, float]],
    timestamps: Optional[List[float]] = None,
    poi_records: Optional[List[POIRecord]] = None,
    dist_thresh_m: float = 200.0,
    time_thresh_s: float = 300.0,
) -> List[float]:
    if not trajectory:
        return []

    # Step 1: detect stay points
    stay_pts = detect_stay_points(
        trajectory, timestamps, dist_thresh_m, time_thresh_s
    )

    # Step 2: build POI map if not provided
    if poi_records is None:
        poi_records = infer_home_work(stay_pts)

    # Step 3: find which stay point (if any) each GPS point belongs to
    stay_point_for_idx: dict = {}
    for sp in stay_pts:
        for idx in sp.point_indices:
            stay_point_for_idx[idx] = sp

    max_dwell = max((sp.duration for sp in stay_pts), default=1.0)

    scores = []
    for i, (lat, lng) in enumerate(trajectory):
        sp = stay_point_for_idx.get(i)
        s = _score_point(lat, lng, poi_records, sp, max_dwell)
        scores.append(round(s, 4))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SA-DP Laplace mechanism
# ─────────────────────────────────────────────────────────────────────────────

def apply_adaptive_laplace(
    trajectory: List[Tuple[float, float]],
    scores: List[float],
    epsilon_base: float,
    sensitivity_m: float = 500.0,
) -> List[Tuple[float, float]]:
    if epsilon_base <= 0:
        raise ValueError("epsilon_base must be > 0")
    if len(trajectory) != len(scores):
        raise ValueError("trajectory and scores must have the same length")

    delta_f = sensitivity_m / METERS_PER_DEGREE
    result = []
    for (lat, lng), s in zip(trajectory, scores):
        eps_i = epsilon_base / max(s, 1e-9)
        scale = delta_f / eps_i
        result.append((
            round(lat + _laplace(scale), 7),
            round(lng + _laplace(scale), 7),
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Budget analysis (Theorem 1 verification)
# ─────────────────────────────────────────────────────────────────────────────

def compute_budget_analysis(
    scores: List[float],
    epsilon_base: float,
    sensitivity_m: float = 500.0,
) -> dict:
    n = len(scores)
    if n == 0:
        return {}

    eps_uniform = n * epsilon_base
    eps_adaptive = epsilon_base * sum(1.0 / max(s, 1e-9) for s in scores)

    per_level: dict = {}
    for s in scores:
        key = round(s)
        if key not in per_level:
            eps_i = epsilon_base / s
            per_level[key] = {
                "count": 0,
                "eps_i": round(eps_i, 6),
                "noise_scale_m": round((sensitivity_m / METERS_PER_DEGREE) / eps_i * METERS_PER_DEGREE, 2),
            }
        per_level[key]["count"] += 1

    return {
        "n": n,
        "epsilon_base": epsilon_base,
        "eps_uniform": round(eps_uniform, 6),
        "eps_adaptive": round(eps_adaptive, 6),
        "budget_savings": round(eps_uniform - eps_adaptive, 6),
        "savings_pct": round((1.0 - eps_adaptive / eps_uniform) * 100, 2),
        "per_level": per_level,
        "theorem1_holds": eps_adaptive <= eps_uniform + 1e-9,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simple smoke-test with synthetic trajectory
    random.seed(42)

    traj = [
        (-37.8136, 144.9631),   # Melbourne CBD (commercial)
        (-37.8102, 144.9628),   # near hospital → sensitive
        (-37.8090, 144.9620),
        (-37.8080, 144.9615),
        (-37.8070, 144.9610),
        (-37.8060, 144.9605),   # home cluster (frequent stay)
        (-37.8055, 144.9600),
        (-37.8050, 144.9595),
    ]

    ts = [0, 30, 60, 90, 120, 300, 330, 600]  # last cluster = 300s dwell

    poi = [
        POIRecord(-37.8102, 144.9628, "medical", "Royal Melbourne Hospital", 300.0),
        POIRecord(-37.8136, 144.9631, "commercial", "CBD", 150.0),
    ]

    scores = score_trajectory(traj, ts, poi)
    print("Sensitivity scores:", scores)

    noisy = apply_adaptive_laplace(traj, scores, epsilon_base=1.0)
    print("First noisy point:", noisy[0])

    budget = compute_budget_analysis(scores, epsilon_base=1.0)
    print(f"\nBudget analysis:")
    print(f"  Uniform (baseline): ε_total = {budget['eps_uniform']}")
    print(f"  SA-DP adaptive: ε_total = {budget['eps_adaptive']}")
    print(f"  Savings:        {budget['savings_pct']:.1f}%")
    print(f"  Theorem 1 holds: {budget['theorem1_holds']}")
    print(f"  Per level: {budget['per_level']}")
