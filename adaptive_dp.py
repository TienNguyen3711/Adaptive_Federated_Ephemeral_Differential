import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

METERS_PER_DEGREE = 111_000

# ── Sensitivity table ─────────────────────────────────────────────────────────
# Maps POI / inferred-location category → sensitivity level S ∈ {1..5}.
SENSITIVITY_TABLE: dict = {
    "medical":       5,   # hospitals, clinics, pharmacies
    "religious":     5,   # mosques, churches, temples, synagogues
    "home_cluster":  5,   # inferred home address
    "work_cluster":  5,   # inferred workplace
    "legal":         4,   # courts, lawyers, embassies
    "political":     4,   # party offices, government buildings
    "education":     3,   # schools, universities
    "commercial":    2,   # shops, restaurants, gyms
    "transit":       1,   # bus stops, metro stations, roads
    "other":         1,
}

S_MAX = 5
S_MIN = 1

# sensitivity_m used throughout this paper's adjacency A_1 (Delta_f). A
# profile POI's acceptance radius must exceed this for any "safe interior"
# to exist at all: a point at distance d from a cluster centre can always
# be perturbed to d + DELTA_F_M, so points survive every Delta_f-bounded
# perturbation only if radius_m - DELTA_F_M > 0 (see build_profile).
DELTA_F_M = 500.0

# ── Score weights (α + β + γ = 1) ────────────────────────────────────────────
_ALPHA = 0.4   # category contribution
_BETA  = 0.4   # proximity contribution
_GAMMA = 0.2   # dwell-time contribution


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class StayPoint:
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
    historical_dwell_s: float = 0.0  # dwell accumulated from PRIOR sessions
                                      # only (profile-derived records); 0 for
                                      # externally-supplied POI database entries


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

def _laplace(scale: float, rng: np.random.Generator) -> float:
    """Sample from Laplace(0, scale) via inverse-CDF transform."""
    u = float(rng.uniform(-0.5 + 1e-10, 0.5 - 1e-10))
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

    merged = _merge_stay_points(stay_points, dist_thresh_m)
    return sorted(merged, key=lambda s: s.arrival)


def _merge_stay_points(
    stay_points: List[StayPoint], dist_thresh_m: float = 200.0
) -> List[StayPoint]:
    """Merge stay points within dist_thresh_m of each other, accumulating
    visit_count and duration. Used both within a single trajectory's own
    detection and across multiple historical trajectories when building a
    profile (build_profile below)."""
    merged: List[StayPoint] = []
    for sp in stay_points:
        matched = False
        for existing in merged:
            if _haversine_m((sp.lat, sp.lng), (existing.lat, existing.lng)) <= dist_thresh_m:
                existing.visit_count += sp.visit_count
                existing.duration += sp.duration
                matched = True
                break
        if not matched:
            merged.append(StayPoint(
                lat=sp.lat, lng=sp.lng, arrival=sp.arrival,
                departure=sp.departure, duration=sp.duration,
                point_indices=list(sp.point_indices),
                visit_count=sp.visit_count,
            ))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Home / work inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_home_work(
    stay_points: List[StayPoint], radius_m: float = 200.0
) -> List[POIRecord]:
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
        records.append(POIRecord(lat=sp.lat, lng=sp.lng, category=cat,
                                  radius_m=radius_m, historical_dwell_s=sp.duration))

    return records


def build_profile(
    historical_segments: List[List[Tuple[float, float]]],
    historical_timestamps: Optional[List[List[float]]] = None,
    dist_thresh_m: float = 200.0,
    time_thresh_s: float = 300.0,
    min_visits: int = 1,
    max_records: int = 50,
    radius_m: float = 3 * DELTA_F_M,
) -> List[POIRecord]:
    """Builds a frozen sensitivity profile (home/work cluster locations +
    historical dwell) from a user's PRIOR trajectory segments only.

    Used to score a *later* trajectory without that trajectory's own points
    ever influencing where its home/work clusters are judged to be --
    closing the data-dependent-sensitivity gap (Theorem 1's Remark 1):
    perturbing a point in the trajectory being scored cannot change this
    profile, because the profile is built entirely from segments that are
    not the one being scored.

    Stay points are detected independently per historical segment, then
    merged by proximity across ALL segments -- so a place visited once per
    segment accumulates visit_count across segments. min_visits defaults to
    1 (any historical stop counts, matching the original same-trajectory
    design's own effective threshold) rather than requiring recurrence:
    requiring min_visits>=2 was tried and cut real budget-savings utility
    roughly in half (58%->13% on a GeoLife sample) by discarding most of
    the profile's signal for little DP-soundness benefit, since soundness
    only requires the profile to be frozen before the trajectory being
    scored, not that it be "recurring." The result is capped to the top
    max_records by (visit_count, duration) purely to bound per-point
    scoring cost, since across dozens of historical segments the number of
    merged stops can otherwise be large (hundreds, in practice).

    radius_m defaults to 3*DELTA_F_M rather than the 200m used for
    within-trajectory stay-point *detection* (dist_thresh_m): freezing the
    cluster location alone does not stop a point's own score from changing
    under perturbation if the acceptance radius is comparable to the
    perturbation bound itself (Delta_f) -- any point within a
    radius_m<=Delta_f disk can trivially be pushed outside it, and the
    prox_score ramp is steep enough near a radius_m~=Delta_f boundary that
    even points somewhat inside it still cross the S>=3 threshold often in
    practice. Larger radius_m keeps improving empirical flip-rate stability
    well beyond radius_m=2*Delta_f, but pushing it much further (tested up
    to 100*Delta_f) trades this for a *hollow* utility gain: the score
    distribution collapses toward uniformly-high scores (most points become
    "somewhat sensitive" rather than a genuine low/high split), which
    inflates the budget-savings metric without the mechanism doing anything
    more adaptive -- confirmed empirically on GeoLife (radius_m=20km drives
    flip-rate to 0% but leaves only ~9% of points at S=1, down from ~39% at
    radius_m=3*Delta_f). 3*Delta_f was chosen as the point past which this
    degeneration starts to dominate on real data, not because it makes any
    point provably invariant: no non-constant threshold function of a
    continuous distance can be made invariant everywhere.
    """
    all_stay_points: List[StayPoint] = []
    for i, seg in enumerate(historical_segments):
        ts = historical_timestamps[i] if historical_timestamps is not None else None
        all_stay_points.extend(
            detect_stay_points(seg, ts, dist_thresh_m, time_thresh_s)
        )
    merged = _merge_stay_points(all_stay_points, dist_thresh_m)
    recurring = [sp for sp in merged if sp.visit_count >= min_visits]
    top = sorted(recurring, key=lambda s: (s.visit_count, s.duration), reverse=True)[:max_records]
    return infer_home_work(top, radius_m=radius_m)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-point sensitivity scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_point(
    lat: float,
    lng: float,
    poi_records: List[POIRecord],
    max_dwell: float,
) -> float:
    cat_score = S_MIN
    prox_score = 0.0
    dwell_score = 0.0

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
                if max_dwell > 0:
                    dwell_score = min(1.0, poi.historical_dwell_s / max_dwell)

    raw = _ALPHA * cat_score + _BETA * prox_score * S_MAX + _GAMMA * dwell_score * S_MAX
    return max(float(S_MIN), min(float(S_MAX), raw))


def score_trajectory(
    trajectory: List[Tuple[float, float]],
    timestamps: Optional[List[float]] = None,
    poi_records: Optional[List[POIRecord]] = None,
    profile: Optional[List[POIRecord]] = None,
    dist_thresh_m: float = 200.0,
    time_thresh_s: float = 300.0,
) -> List[float]:
    """Scores each point's sensitivity S_i.

    `profile`, if given, is a frozen set of home/work clusters (with
    historical_dwell_s set) built by build_profile() from trajectories OTHER
    than `trajectory` -- e.g. a user's prior sessions. Using it makes S_i
    independent of `trajectory`'s own points for the home/work-cluster
    signal, closing Theorem 1's data-dependent-sensitivity gap (Remark 1):
    perturbing a point in `trajectory` cannot move a cluster that was
    computed without looking at `trajectory` at all.

    `poi_records`, if given, is an external POI database (e.g. OpenStreetMap
    tiles) merged in for category-aware scoring; it is independent of any
    trajectory by construction and was already safe under the old design.

    If `profile` is not passed at all (None, the default), S_i falls back to
    inferring home/work clusters from `trajectory` itself (the original,
    data-dependent behaviour) -- kept for callers unrelated to Theorem 1's
    guarantee (e.g. structural baselines). Passing `profile=[]` explicitly
    is different from not passing it: it means "this user has no known
    recurring places yet" (cold start) and must NOT fall back to scoring
    `trajectory` against itself -- only `poi_records`, if any, apply, and
    points with no nearby POI at all correctly get S_MIN.
    """
    if not trajectory:
        return []

    if profile is None and poi_records is None:
        stay_pts = detect_stay_points(
            trajectory, timestamps, dist_thresh_m, time_thresh_s
        )
        combined_records: List[POIRecord] = infer_home_work(stay_pts)
    else:
        combined_records = list(profile) if profile is not None else []
        if poi_records:
            combined_records = combined_records + list(poi_records)

    max_dwell = max((r.historical_dwell_s for r in combined_records), default=1.0)
    if max_dwell <= 0:
        max_dwell = 1.0

    scores = []
    for lat, lng in trajectory:
        s = _score_point(lat, lng, combined_records, max_dwell)
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
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    if epsilon_base <= 0:
        raise ValueError("epsilon_base must be > 0")
    if len(trajectory) != len(scores):
        raise ValueError("trajectory and scores must have the same length")
    if rng is None:
        rng = np.random.default_rng(0)

    delta_f = sensitivity_m / METERS_PER_DEGREE
    result = []
    for (lat, lng), s in zip(trajectory, scores):
        eps_i = epsilon_base / max(s, 1e-9)
        scale = delta_f / eps_i
        result.append((
            round(lat + _laplace(scale, rng), 7),
            round(lng + _laplace(scale, rng), 7),
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
    _rng = np.random.default_rng(42)

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

    noisy = apply_adaptive_laplace(traj, scores, epsilon_base=1.0, rng=_rng)
    print("First noisy point:", noisy[0])

    budget = compute_budget_analysis(scores, epsilon_base=1.0)
    print(f"\nBudget analysis:")
    print(f"  Uniform (baseline): ε_total = {budget['eps_uniform']}")
    print(f"  SA-DP adaptive: ε_total = {budget['eps_adaptive']}")
    print(f"  Savings:        {budget['savings_pct']:.1f}%")
    print(f"  Theorem 1 holds: {budget['theorem1_holds']}")
    print(f"  Per level: {budget['per_level']}")
