"""
loader_tdrive.py — T-Drive Taxi Trajectories Dataset Loader

Dataset: Microsoft Research T-Drive (10,357 taxis, 1 week, Feb 2008)
Source:  taxi_log_2008_by_id/{taxi_id}.txt
Format:  taxi_id, timestamp, longitude, latitude
         NOTE: longitude comes BEFORE latitude in the raw file.

Returns trajectory segments as List[Tuple[float,float]] (lat, lon).
"""

import os
import glob
from datetime import datetime
from typing import Iterator, List, Optional, Tuple


_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "../../../data/T-drive Taxi Trajectories/taxi_log_2008_by_id"
)
_DATA_ROOT = os.path.normpath(_DATA_ROOT)


def _parse_taxi_file(path: str) -> List[Tuple[float, float, datetime]]:
    """Parse one taxi .txt file → list of (lat, lon, timestamp)."""
    points = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                ts  = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M:%S")
                lon = float(parts[2])   # raw format: lon before lat
                lat = float(parts[3])
                # Basic sanity: Beijing bounding box
                if not (39.0 <= lat <= 41.0 and 115.0 <= lon <= 117.5):
                    continue
                points.append((lat, lon, ts))
            except (ValueError, IndexError):
                continue
    return points


def _segment_by_gap(
    points: List[Tuple[float, float, datetime]],
    gap_seconds: int = 1800,
    min_points: int = 10,
    keep_timestamps: bool = False,
) -> List[List[Tuple]]:
    if not points:
        return []

    def _emit(pts):
        if not keep_timestamps:
            return [(p[0], p[1]) for p in pts]
        t0 = pts[0][2]
        return [(p[0], p[1], (p[2] - t0).total_seconds()) for p in pts]

    # Sort by timestamp first (some files are not sorted)
    points = sorted(points, key=lambda p: p[2])
    segments, current = [], [points[0]]
    for i in range(1, len(points)):
        dt = (points[i][2] - points[i-1][2]).total_seconds()
        if dt > gap_seconds or dt < 0:
            if len(current) >= min_points:
                segments.append(_emit(current))
            current = [points[i]]
        else:
            current.append(points[i])
    if len(current) >= min_points:
        segments.append(_emit(current))
    return segments


def _window(
    segment: List[Tuple[float, float]],
    window_size: int,
    stride: int,
) -> List[List[Tuple[float, float]]]:
    return [
        segment[i:i + window_size]
        for i in range(0, len(segment) - window_size + 1, stride)
    ]


def iter_taxi_trajectories(
    data_root: str = None,
    taxi_ids: Optional[List[str]] = None,
    min_points: int = 10,
    gap_seconds: int = 1800,
    keep_timestamps: bool = False,
) -> Iterator[Tuple[str, List[List[Tuple[float, float]]]]]:
    """
    Yield (taxi_id, [trajectory_segments]) for each taxi.

    Parameters
    ----------
    taxi_ids        : list of taxi ID strings (e.g. ['1','2']); None = all
    min_points      : minimum points per segment
    gap_seconds     : time gap threshold for splitting
    keep_timestamps : if True, segments are (lat, lon, rel_seconds) triples
    """
    root = data_root or _DATA_ROOT
    if taxi_ids is not None:
        txt_files = [os.path.join(root, f"{tid}.txt") for tid in taxi_ids]
        txt_files = [f for f in txt_files if os.path.exists(f)]
    else:
        txt_files = sorted(glob.glob(os.path.join(root, "*.txt")))

    for fpath in txt_files:
        tid  = os.path.splitext(os.path.basename(fpath))[0]
        pts  = _parse_taxi_file(fpath)
        segs = _segment_by_gap(pts, gap_seconds=gap_seconds,
                               min_points=min_points,
                               keep_timestamps=keep_timestamps)
        if segs:
            yield tid, segs


def load_as_client_datasets(
    data_root: str = None,
    n_taxis: int = 50,
    window_size: int = 50,
    stride: int = 25,
    min_points: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[List[List[Tuple[float, float]]]]]:
    """
    Load T-Drive data as federated client datasets (one taxi = one client).

    Returns (taxi_ids, datasets) where datasets[i] is a list of
    fixed-length trajectory windows for taxi i.
    """
    import random
    rng = random.Random(seed)

    root = data_root or _DATA_ROOT
    all_files = sorted(glob.glob(os.path.join(root, "*.txt")))
    all_ids   = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    selected  = rng.sample(all_ids, min(n_taxis, len(all_ids)))

    taxi_ids = []
    datasets = []

    for tid, segs in iter_taxi_trajectories(
        data_root=root, taxi_ids=selected, min_points=min_points
    ):
        windows = []
        for seg in segs:
            windows.extend(_window(seg, window_size, stride))
        if windows:
            taxi_ids.append(tid)
            datasets.append(windows)

    return taxi_ids, datasets


def load_flat(
    data_root: str = None,
    n_taxis: int = 50,
    min_points: int = 20,
    gap_seconds: int = 1800,
    seed: int = 42,
) -> List[List[Tuple[float, float]]]:
    """Load all segments as a flat list (variable length)."""
    import random
    rng = random.Random(seed)

    root = data_root or _DATA_ROOT
    all_files = sorted(glob.glob(os.path.join(root, "*.txt")))
    all_ids   = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    selected  = rng.sample(all_ids, min(n_taxis, len(all_ids)))

    all_segs = []
    for _, segs in iter_taxi_trajectories(
        data_root=root, taxi_ids=selected,
        min_points=min_points, gap_seconds=gap_seconds,
    ):
        all_segs.extend(segs)
    return all_segs


def load_flat_with_timestamps(
    data_root: str = None,
    n_taxis: int = 50,
    min_points: int = 20,
    gap_seconds: int = 1800,
    seed: int = 42,
) -> List[List[Tuple[float, float, float]]]:
    """
    Same as load_flat(), but each segment's points are (lat, lon, rel_seconds)
    triples, where rel_seconds is real elapsed time since the segment's first
    GPS fix — for Temporal Obfuscation.
    """
    import random
    rng = random.Random(seed)

    root = data_root or _DATA_ROOT
    all_files = sorted(glob.glob(os.path.join(root, "*.txt")))
    all_ids   = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    selected  = rng.sample(all_ids, min(n_taxis, len(all_ids)))

    all_segs = []
    for _, segs in iter_taxi_trajectories(
        data_root=root, taxi_ids=selected,
        min_points=min_points, gap_seconds=gap_seconds,
        keep_timestamps=True,
    ):
        all_segs.extend(segs)
    return all_segs


def load_flat_with_profile_split(
    data_root: str = None,
    n_taxis: int = 50,
    min_points: int = 20,
    gap_seconds: int = 1800,
    seed: int = 42,
    max_eval_per_user: int = 10,
):
    """
    Per-taxi profile/eval split for the historical-profile SA-DP scorer
    (adaptive_dp.build_profile / score_trajectory(profile=...)). See
    loader_geolife.load_flat_with_profile_split() for the full contract
    (incl. why the result is shuffled and capped per taxi).
    """
    import random
    from Adaptive_Federated_Ephemeral_Differential.adaptive_dp import build_profile

    rng = random.Random(seed)
    root = data_root or _DATA_ROOT
    all_files = sorted(glob.glob(os.path.join(root, "*.txt")))
    all_ids   = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    selected  = rng.sample(all_ids, min(n_taxis, len(all_ids)))

    eval_segs, eval_ts, eval_profiles = [], [], []
    for _, segs in iter_taxi_trajectories(
        data_root=root, taxi_ids=selected,
        min_points=min_points, gap_seconds=gap_seconds,
        keep_timestamps=True,
    ):
        if len(segs) < 2:
            continue
        n_profile = max(1, len(segs) // 2)
        profile_segs, rest_segs = segs[:n_profile], segs[n_profile:]
        profile_trajs = [[(p[0], p[1]) for p in s] for s in profile_segs]
        profile_ts    = [[p[2] for p in s] for s in profile_segs]
        profile = build_profile(profile_trajs, profile_ts)
        for s in rest_segs[:max_eval_per_user]:
            eval_segs.append([(p[0], p[1]) for p in s])
            eval_ts.append([p[2] for p in s])
            eval_profiles.append(profile)

    order = list(range(len(eval_segs)))
    rng.shuffle(order)
    eval_segs     = [eval_segs[i] for i in order]
    eval_ts       = [eval_ts[i] for i in order]
    eval_profiles = [eval_profiles[i] for i in order]
    return eval_segs, eval_ts, eval_profiles


if __name__ == "__main__":
    print("=== T-Drive loader self-test ===")
    root = _DATA_ROOT
    print(f"Data root: {root}")
    print(f"Exists: {os.path.isdir(root)}")

    count, total_pts = 0, 0
    for tid, segs in iter_taxi_trajectories(
        data_root=root, taxi_ids=["1", "2", "3"], min_points=10
    ):
        for s in segs:
            total_pts += len(s)
            count += 1
        print(f"  Taxi {tid}: {len(segs)} segments")

    print(f"Total segments (3 taxis): {count}")
    print(f"Total points:             {total_pts}")
    print(f"Mean segment length:      {total_pts/max(1,count):.0f} pts")
