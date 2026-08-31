import math
from typing import List, Tuple

import numpy as np

from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import POIRecord

METERS_PER_DEGREE = 111_000

_CATEGORY_WEIGHTS = {
    "commercial":  0.45,
    "transit":     0.20,
    "education":   0.15,
    "medical":     0.08,
    "religious":   0.05,
    "legal":       0.04,
    "political":   0.03,
}
_CATEGORIES = list(_CATEGORY_WEIGHTS.keys())
_WEIGHTS = list(_CATEGORY_WEIGHTS.values())


def _place_pois(pool, n_poi, jitter_m, radius_range_m, rng, name_prefix=""):
    if not pool:
        return []
    n_poi = min(n_poi, len(pool))
    idx = rng.choice(len(pool), size=n_poi, replace=False)

    pois: List[POIRecord] = []
    for i in idx:
        lat, lng = pool[int(i)]
        bearing = rng.uniform(0, 2 * math.pi)
        dist_m = rng.uniform(0, jitter_m)
        dlat = (dist_m * math.cos(bearing)) / METERS_PER_DEGREE
        dlng = (dist_m * math.sin(bearing)) / (
            METERS_PER_DEGREE * math.cos(math.radians(lat)) + 1e-9
        )
        category = rng.choice(_CATEGORIES, p=_WEIGHTS)
        radius_m = float(rng.uniform(*radius_range_m))
        pois.append(POIRecord(
            lat=round(lat + dlat, 7),
            lng=round(lng + dlng, 7),
            category=str(category),
            name=f"synthetic_{name_prefix}{str(category)}_{len(pois)}",
            radius_m=radius_m,
        ))
    return pois


def generate_synthetic_pois(
    trajectories: List[List[Tuple[float, float]]],
    n_poi: int = 200,
    jitter_m: float = 50.0,
    radius_range_m: Tuple[float, float] = (80.0, 150.0),
    seed: int = 42,
) -> List[POIRecord]:
    """Generate `n_poi` synthetic POIRecords placed near the observed
    activity density of `trajectories`, with categories drawn from
    `_CATEGORY_WEIGHTS`. Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    pool: List[Tuple[float, float]] = [pt for traj in trajectories for pt in traj]
    return _place_pois(pool, n_poi, jitter_m, radius_range_m, rng)


def generate_synthetic_pois_per_trajectory(
    trajectory: List[Tuple[float, float]],
    n_poi: int = 6,
    jitter_m: float = 50.0,
    radius_range_m: Tuple[float, float] = (80.0, 150.0),
    seed: int = 42,
) -> List[POIRecord]:
    """Same generation method as `generate_synthetic_pois`, scoped to a
    single trajectory's own points -- guarantees each trajectory has
    nearby synthetic POIs regardless of dataset-wide trajectory count.
    """
    rng = np.random.default_rng(seed)
    return _place_pois(trajectory, n_poi, jitter_m, radius_range_m, rng, name_prefix="traj_")
