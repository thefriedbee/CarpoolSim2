"""Build a CarpoolSim trip file from the morning drive-alone trips.

The source CSV is one person-trip per row, tagged with 2020 model TAZ ids.
This draws one million of those trips, stratified by departure period, and
replaces each zone id with a point inside that zone.

Period 1 is 03:00–03:29, the start of the ARC model day, so period 7 is
06:00–06:29 and period 14 is 09:30–09:59. The written clock is one minute
inside that half hour.

Coordinates are EPSG:3857 meters, which is what ``TripCluster.from_csv``
reads. Longitude and latitude are kept beside them. ``orig_taz``,
``dest_taz``, and ``depart_period`` stay so a row can be traced back.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
TRIPS_PATH = ROOT / "trip_sov_AM.csv"
TAZ_PATH = ROOT / "TAZ2020" / "Model_Traffic_Analysis_Zones_2020.shp"
OUT_PATH = ROOT / "sov_am_1m_trips.csv"
NETWORK_NODES = ROOT.parent / "network" / "nodes.gpkg"

TARGET = 1_000_000
SEED = 20261021
# Period 1 begins at 03:00. Each period is 30 minutes.
DAY_START_MINUTE = 3 * 60
PERIOD_MINUTES = 30

TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    zones = load_zones()
    source = pd.read_csv(
        TRIPS_PATH,
        usecols=["orig_taz", "dest_taz", "depart_period"],
    )
    chosen = stratified_sample(source, TARGET, rng)
    print(f"source trips: {len(source)}")
    print(f"sampled trips: {len(chosen)}")

    print("sampling points inside zone polygons")
    origin_xy = points_in_zones(chosen["orig_taz"].to_numpy(), zones, rng)
    dest_xy = points_in_zones(chosen["dest_taz"].to_numpy(), zones, rng)
    assert_inside(chosen["orig_taz"].to_numpy(), origin_xy, zones)
    assert_inside(chosen["dest_taz"].to_numpy(), dest_xy, zones)

    departure = departure_clocks(chosen["depart_period"].to_numpy(), rng)
    origin_lon, origin_lat = TO_WGS84.transform(origin_xy[:, 0], origin_xy[:, 1])
    dest_lon, dest_lat = TO_WGS84.transform(dest_xy[:, 0], dest_xy[:, 1])

    out = pd.DataFrame(
        {
            "trip_id": np.arange(1, len(chosen) + 1),
            "origin_lon": np.char.mod("%.6f", origin_lon),
            "origin_lat": np.char.mod("%.6f", origin_lat),
            "dest_lon": np.char.mod("%.6f", dest_lon),
            "dest_lat": np.char.mod("%.6f", dest_lat),
            "origin_x": np.char.mod("%.1f", origin_xy[:, 0]),
            "origin_y": np.char.mod("%.1f", origin_xy[:, 1]),
            "dest_x": np.char.mod("%.1f", dest_xy[:, 0]),
            "dest_y": np.char.mod("%.1f", dest_xy[:, 1]),
            "departure_time": departure,
            "orig_taz": chosen["orig_taz"].to_numpy(),
            "dest_taz": chosen["dest_taz"].to_numpy(),
            "depart_period": chosen["depart_period"].to_numpy(),
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("sample by period:")
    print(chosen["depart_period"].value_counts().sort_index().to_string())
    print(f"earliest departure {min(departure)}, latest {max(departure)}")
    print(f"wrote {OUT_PATH}")
    report_network_overlap(origin_xy, dest_xy)


def load_zones() -> gpd.GeoDataFrame:
    """Return zone polygons in EPSG:3857, indexed by model TAZ id."""
    zones = gpd.read_file(TAZ_PATH)
    zones = zones.set_index(zones["MTAZ10"].astype(int))[["geometry"]]
    if not zones.index.is_unique:
        raise ValueError("MTAZ10 ids are not unique")
    zones = zones.to_crs(3857)
    if zones.geometry.is_empty.any() or not zones.geometry.is_valid.all():
        raise ValueError("TAZ polygons must be valid and non-empty")
    return zones


def stratified_sample(source: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``target`` rows without replacement, keeping each period's share.

    The draw uses the largest-remainder split, so the period counts sum to
    ``target``. Rows come back in source-file order.
    """
    periods = source["depart_period"].to_numpy()
    values, counts = np.unique(periods, return_counts=True)
    total = int(counts.sum())
    if target > total:
        raise ValueError(f"asked for {target} trips but the source has {total}")
    base = (target * counts) // total
    leftover = target - int(base.sum())
    remainder = target * counts - base * total
    base[np.argsort(-remainder, kind="stable")[:leftover]] += 1
    if np.any(base > counts):
        raise ValueError("a departure period does not have enough trips")

    picked: list[np.ndarray] = []
    for period, size in zip(values, base, strict=True):
        pool = np.flatnonzero(periods == period)
        picked.append(rng.choice(pool, size=int(size), replace=False))
    index = np.sort(np.concatenate(picked))
    return source.iloc[index].reset_index(drop=True)


def points_in_zones(
    taz_ids: np.ndarray,
    zones: gpd.GeoDataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one uniform point, in EPSG:3857, inside each trip's zone.

    Points for a zone are paired with that zone's trips in file order.
    """
    codes, counts = np.unique(taz_ids, return_counts=True)
    known = zones.index.to_numpy()
    missing = codes[~np.isin(codes, known)]
    if len(missing) > 0:
        listed = ", ".join(str(int(taz)) for taz in missing[:8])
        raise ValueError(f"{len(missing)} TAZ ids are missing from the zone file: {listed}")

    sampled = zones.geometry.loc[codes].sample_points(size=counts, rng=rng)
    blocks = [shapely.get_coordinates(geom) for geom in sampled]
    for taz, block, count in zip(codes, blocks, counts, strict=True):
        if len(block) != count:
            raise RuntimeError(f"zone {int(taz)} returned {len(block)} points, expected {count}")

    flat = np.concatenate(blocks, axis=0)
    order = np.argsort(taz_ids, kind="stable")
    xy = np.empty_like(flat)
    xy[order] = flat
    return xy


def assert_inside(taz_ids: np.ndarray, xy: np.ndarray, zones: gpd.GeoDataFrame) -> None:
    """Raise when a drawn point lies outside its zone polygon."""
    order = np.argsort(taz_ids, kind="stable")
    sorted_ids = taz_ids[order]
    sorted_xy = xy[order]
    breaks = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(sorted_ids)]))
    for taz, start, end in zip(sorted_ids[starts], starts, ends, strict=True):
        pts = shapely.points(sorted_xy[start:end, 0], sorted_xy[start:end, 1])
        inside = shapely.covers(zones.geometry.loc[int(taz)], pts)
        if not bool(np.all(inside)):
            raise RuntimeError(f"a point for TAZ {int(taz)} fell outside the zone")


def departure_clocks(periods: np.ndarray, rng: np.random.Generator) -> list[str]:
    """One ``HH:MM`` clock inside each trip's half-hour period."""
    if np.any(periods < 1):
        raise ValueError("depart_period starts at 1")
    offset = rng.integers(0, PERIOD_MINUTES, size=len(periods))
    minutes = DAY_START_MINUTE + (periods - 1) * PERIOD_MINUTES + offset
    return [f"{minute // 60:02d}:{minute % 60:02d}" for minute in minutes]


def report_network_overlap(origin_xy: np.ndarray, dest_xy: np.ndarray) -> None:
    """Print how many sampled trips sit inside the current drive-network box."""
    if not NETWORK_NODES.exists():
        print("network nodes not found; skipped the bounding-box check")
        return
    xmin, ymin, xmax, ymax = pyogrio.read_info(NETWORK_NODES)["total_bounds"]

    def inside(xy: np.ndarray) -> np.ndarray:
        return (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)

    both = int(np.count_nonzero(inside(origin_xy) & inside(dest_xy)))
    print(f"trips with both ends inside the network bounding box: {both} of {len(origin_xy)}")


if __name__ == "__main__":
    main()
