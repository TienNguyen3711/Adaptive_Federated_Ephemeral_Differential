"""
build_osm_poi_cache.py — converts raw Overpass API query results (real
OpenStreetMap POI data for Beijing and Porto) into cached POIRecord-shaped
JSON files, closing the "Semantic" framing gap flagged by peer review
Phase 7: prior experiments (exp_poi_semantic.py) placed synthetic POIs at
randomly jittered locations near each trajectory's own points, which never
exercised a real semantic map.

Raw query results (already fetched via the Overpass API, see
osm_queries/*.txt for the exact queries used) live in /tmp during the
fetch session; this script reads them once and writes a stable, offline
cache under data/osm_cache/ so re-running experiments never depends on
network access again.

Beijing's bounding box (39.6-40.3 N, 115.9-117.0 E) covers both GeoLife
and T-Drive, which are both Beijing-based datasets -- one real POI tile
serves both. Porto gets its own tile.
"""
import json
import os

_OUT_DIR = os.path.join(os.path.dirname(__file__), "osm_cache")
os.makedirs(_OUT_DIR, exist_ok=True)

# OSM tag -> this paper's SENSITIVITY_TABLE category (adaptive_dp.py)
_TAG_TO_CATEGORY = {
    ("amenity", "hospital"):        "medical",
    ("amenity", "clinic"):          "medical",
    ("amenity", "pharmacy"):        "medical",
    ("amenity", "doctors"):         "medical",
    ("amenity", "place_of_worship"): "religious",
    ("amenity", "courthouse"):      "legal",
    ("office", "government"):       "political",
    ("amenity", "townhall"):        "political",
    ("amenity", "school"):          "education",
    ("amenity", "university"):      "education",
    ("amenity", "college"):         "education",
    ("amenity", "kindergarten"):    "education",
    ("amenity", "restaurant"):      "commercial",
    ("amenity", "cafe"):            "commercial",
    ("amenity", "fast_food"):       "commercial",
    ("amenity", "bar"):             "commercial",
    ("amenity", "bus_station"):     "transit",
    ("railway", "station"):         "transit",
    ("railway", "subway_entrance"): "transit",
}

# Influence radius by category (metres) -- larger for institutional/campus-
# scale POIs (hospitals, universities), smaller for point amenities (a cafe
# counter, a bus stop), matching this codebase's existing synthetic_poi.py
# radius_range_m convention (80-150m) as the default and widening only
# where a real-world footprint plausibly justifies it.
_RADIUS_BY_CATEGORY = {
    "medical": 120.0, "religious": 100.0, "legal": 120.0, "political": 150.0,
    "education": 180.0, "commercial": 80.0, "transit": 100.0,
}


def _convert(raw_paths, out_name):
    records = []
    seen = set()
    for path in raw_paths:
        with open(path) as f:
            data = json.load(f)
        for el in data.get("elements", []):
            if el.get("type") != "node":
                continue
            tags = el.get("tags", {})
            category = None
            for (k, v), cat in _TAG_TO_CATEGORY.items():
                if tags.get(k) == v:
                    category = cat
                    break
            if category is None:
                continue
            key = (round(el["lat"], 6), round(el["lon"], 6), category)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "lat": el["lat"],
                "lng": el["lon"],
                "category": category,
                "name": tags.get("name", f"osm_{category}_{el['id']}"),
                "radius_m": _RADIUS_BY_CATEGORY[category],
            })

    out_path = os.path.join(_OUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    from collections import Counter
    cat_counts = Counter(r["category"] for r in records)
    print(f"{out_name}: {len(records)} real OSM POIs -> {out_path}")
    for cat, n in cat_counts.most_common():
        print(f"    {cat:<12} {n}")
    return records


if __name__ == "__main__":
    _convert(
        ["/tmp/osm_beijing_sensitive.json", "/tmp/osm_beijing_transit.json",
         "/tmp/osm_beijing_commercial.json"],
        "osm_pois_beijing.json",
    )
    _convert(["/tmp/osm_porto.json"], "osm_pois_porto.json")
