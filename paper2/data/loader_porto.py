"""
loader_porto.py — Porto Taxi Dataset Loader

Dataset: ECML/PKDD 2015 Porto Taxi (1.7M trips)
Source:  train.csv
Format:  TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID,
         TIMESTAMP, DAY_TYPE, MISSING_DATA, POLYLINE
POLYLINE: JSON list of [longitude, latitude] pairs, sampled every 15s.

Returns trajectory segments as List[Tuple[float,float]] (lat, lon).
"""

import csv
import json
import os
from typing import Iterator, List, Optional, Tuple


_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "../../../data/porto_taxi/train.csv"
)
_DATA_ROOT = os.path.normpath(_DATA_ROOT)

# Porto bounding box
_LAT_MIN, _LAT_MAX = 41.0, 41.3
_LON_MIN, _LON_MAX = -8.75, -8.5


def _parse_polyline(raw: str) -> List[Tuple[float, float]]:
    """Parse POLYLINE JSON string → list of (lat, lon)."""
    try:
        pts = json.loads(raw)
        result = []
        for p in pts:
            if len(p) < 2:
                continue
            lon, lat = float(p[0]), float(p[1])
            if _LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX:
                result.append((lat, lon))
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _window(
    segment: List[Tuple[float, float]],
    window_size: int,
    stride: int,
) -> List[List[Tuple[float, float]]]:
    return [
        segment[i:i + window_size]
        for i in range(0, len(segment) - window_size + 1, stride)
    ]


def iter_trips(
    csv_path: str = None,
    n_trips: Optional[int] = None,
    min_points: int = 10,
    skip_missing: bool = True,
) -> Iterator[Tuple[str, str, List[Tuple[float, float]]]]:
    """
    Yield (trip_id, taxi_id, trajectory) for each valid trip.

    Parameters
    ----------
    n_trips      : max trips to read (None = all ~1.7M)
    min_points   : skip trips with fewer than this many valid points
    skip_missing : skip rows where MISSING_DATA == True
    """
    path = csv_path or _DATA_ROOT
    count = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if n_trips is not None and count >= n_trips:
                break
            if skip_missing and row.get("MISSING_DATA", "").strip() == "True":
                continue
            pts = _parse_polyline(row.get("POLYLINE", "[]"))
            if len(pts) < min_points:
                continue
            yield row["TRIP_ID"], row["TAXI_ID"], pts
            count += 1


def load_as_client_datasets(
    csv_path: str = None,
    n_taxis: int = 50,
    trips_per_taxi: int = 200,
    window_size: int = 50,
    stride: int = 25,
    min_points: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[List[List[Tuple[float, float]]]]]:
    """
    Group trips by TAXI_ID, return as federated client datasets.

    Returns (taxi_ids, datasets) where datasets[i] is a list of
    fixed-length trajectory windows for taxi i.

    Note: Porto has 442 unique taxis with many trips each.
    """
    from collections import defaultdict
    import random
    rng = random.Random(seed)

    path = csv_path or _DATA_ROOT
    taxi_trips: dict = defaultdict(list)

    # Read enough trips to fill n_taxis × trips_per_taxi
    max_read = n_taxis * trips_per_taxi * 5  # oversample to account for filtering
    for _, taxi_id, pts in iter_trips(
        path, n_trips=max_read, min_points=min_points
    ):
        if len(taxi_trips[taxi_id]) < trips_per_taxi:
            taxi_trips[taxi_id].append(pts)

    # Select n_taxis with enough trips
    eligible = [t for t, trips in taxi_trips.items() if len(trips) >= 5]
    selected = rng.sample(eligible, min(n_taxis, len(eligible)))

    taxi_ids = []
    datasets = []
    for tid in selected:
        windows = []
        for seg in taxi_trips[tid]:
            windows.extend(_window(seg, window_size, stride))
        if windows:
            taxi_ids.append(tid)
            datasets.append(windows)

    return taxi_ids, datasets


def load_flat(
    csv_path: str = None,
    n_trips: int = 5000,
    min_points: int = 20,
    seed: int = 42,
) -> List[List[Tuple[float, float]]]:
    """Load up to n_trips trajectories as a flat list (variable length)."""
    import random
    rng = random.Random(seed)

    all_trips = []
    for _, _, pts in iter_trips(csv_path or _DATA_ROOT,
                                n_trips=n_trips * 3,
                                min_points=min_points):
        all_trips.append(pts)
        if len(all_trips) >= n_trips * 3:
            break

    return rng.sample(all_trips, min(n_trips, len(all_trips)))


if __name__ == "__main__":
    print("=== Porto loader self-test ===")
    path = _DATA_ROOT
    print(f"CSV path: {path}")
    print(f"Exists: {os.path.isfile(path)}")

    count, total_pts = 0, 0
    taxis = set()
    for trip_id, taxi_id, pts in iter_trips(path, n_trips=1000, min_points=10):
        total_pts += len(pts)
        count += 1
        taxis.add(taxi_id)

    print(f"Trips read:          {count}")
    print(f"Unique taxis:        {len(taxis)}")
    print(f"Total points:        {total_pts}")
    print(f"Mean trip length:    {total_pts/max(1,count):.0f} pts")
