"""
loader_geolife.py — GeoLife Trajectories 1.3 Dataset Loader

Dataset: Microsoft Research GeoLife (182 users, ~17,000 trajectories)
Source:  Data/{user_id}/Trajectory/*.plt
Format:  6 header lines, then rows of:
             lat, lng, 0, altitude_ft, days_since_1899, date, time

Returns trajectory segments as List[Tuple[float,float]] (lat, lon).
"""

import os
import glob
from typing import Iterator, List, Tuple, Optional
from datetime import datetime


_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "../../../data/Geolife Trajectories 1.3/Data"
)
_DATA_ROOT = os.path.normpath(_DATA_ROOT)

_PLT_HEADER_LINES = 6


def _parse_plt(path: str) -> List[Tuple[float, float, datetime]]:
    """Parse one .plt file → list of (lat, lon, timestamp)."""
    points = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < _PLT_HEADER_LINES:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                dt  = datetime.strptime(
                    parts[5].strip() + " " + parts[6].strip(), "%Y-%m-%d %H:%M:%S"
                )
                points.append((lat, lon, dt))
            except (ValueError, IndexError):
                continue
    return points


def _segment_by_gap(
    points: List[Tuple[float, float, datetime]],
    gap_seconds: int = 1800,
    min_points: int = 10,
) -> List[List[Tuple[float, float]]]:
    """
    Split a trajectory at time gaps > gap_seconds.
    Returns list of (lat, lon) segments each with ≥ min_points points.
    """
    if not points:
        return []

    segments = []
    current = [points[0]]
    for i in range(1, len(points)):
        dt = (points[i][2] - points[i-1][2]).total_seconds()
        if dt > gap_seconds:
            if len(current) >= min_points:
                segments.append([(p[0], p[1]) for p in current])
            current = [points[i]]
        else:
            current.append(points[i])
    if len(current) >= min_points:
        segments.append([(p[0], p[1]) for p in current])
    return segments


def _window(
    segment: List[Tuple[float, float]],
    window_size: int,
    stride: int,
) -> List[List[Tuple[float, float]]]:
    """Slide a fixed-length window over a segment."""
    return [
        segment[i:i + window_size]
        for i in range(0, len(segment) - window_size + 1, stride)
    ]


def iter_user_trajectories(
    data_root: str = None,
    user_ids: Optional[List[str]] = None,
    min_points: int = 10,
    gap_seconds: int = 1800,
) -> Iterator[Tuple[str, List[List[Tuple[float, float]]]]]:
    """
    Yield (user_id, [trajectory_segments]) for each user.

    Parameters
    ----------
    data_root   : path to GeoLife Data/ directory
    user_ids    : subset of user IDs (e.g. ['000','001']); None = all 182
    min_points  : minimum points per segment (shorter segments dropped)
    gap_seconds : time gap threshold for splitting trajectories
    """
    root = data_root or _DATA_ROOT
    all_users = sorted(os.listdir(root)) if user_ids is None else user_ids

    for uid in all_users:
        traj_dir = os.path.join(root, uid, "Trajectory")
        if not os.path.isdir(traj_dir):
            continue
        plt_files = sorted(glob.glob(os.path.join(traj_dir, "*.plt")))
        user_segs = []
        for plt in plt_files:
            pts  = _parse_plt(plt)
            segs = _segment_by_gap(pts, gap_seconds=gap_seconds,
                                   min_points=min_points)
            user_segs.extend(segs)
        if user_segs:
            yield uid, user_segs


def load_as_client_datasets(
    data_root: str = None,
    n_users: int = 50,
    window_size: int = 50,
    stride: int = 25,
    min_points: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[List[List[Tuple[float, float]]]]]:
    """
    Load GeoLife data as federated client datasets.

    Returns (user_ids, datasets) where datasets[i] is a list of
    fixed-length trajectory windows for user i.

    Parameters
    ----------
    n_users     : number of users to load (sampled deterministically by seed)
    window_size : fixed trajectory length (points)
    stride      : window stride (overlap = window_size - stride)
    """
    import random
    rng = random.Random(seed)

    root = data_root or _DATA_ROOT
    all_users = sorted(os.listdir(root))
    selected  = rng.sample(all_users, min(n_users, len(all_users)))

    user_ids = []
    datasets = []

    for uid, segs in iter_user_trajectories(
        data_root=root, user_ids=selected, min_points=min_points
    ):
        windows = []
        for seg in segs:
            windows.extend(_window(seg, window_size, stride))
        if windows:
            user_ids.append(uid)
            datasets.append(windows)

    return user_ids, datasets


def load_flat(
    data_root: str = None,
    n_users: int = 50,
    min_points: int = 20,
    gap_seconds: int = 1800,
    seed: int = 42,
) -> List[List[Tuple[float, float]]]:
    """
    Load all trajectory segments (variable length) as a flat list.
    Suitable for SA-DP and E2E experiments.
    """
    import random
    rng = random.Random(seed)

    root = data_root or _DATA_ROOT
    all_users = sorted(os.listdir(root))
    selected  = rng.sample(all_users, min(n_users, len(all_users)))

    all_segs = []
    for _, segs in iter_user_trajectories(
        data_root=root, user_ids=selected,
        min_points=min_points, gap_seconds=gap_seconds,
    ):
        all_segs.extend(segs)
    return all_segs


if __name__ == "__main__":
    print("=== GeoLife loader self-test ===")
    root = _DATA_ROOT
    print(f"Data root: {root}")
    print(f"Exists: {os.path.isdir(root)}")

    count = 0
    total_pts = 0
    for uid, segs in iter_user_trajectories(
        data_root=root, user_ids=["000", "001", "002"],
        min_points=10,
    ):
        for s in segs:
            total_pts += len(s)
            count += 1
        print(f"  User {uid}: {len(segs)} segments")

    print(f"Total segments (3 users): {count}")
    print(f"Total points:             {total_pts}")
    print(f"Mean segment length:      {total_pts/max(1,count):.0f} pts")
