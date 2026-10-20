"""Keep large public parking lots from an OSM file and store them as points.

A lot is a place where two people can each drive, leave one car, and
continue together. The source is ``amenity=parking`` in the same OSM XML
file as the street network. OSM treats a parking lot with no ``access`` tag
as public, so those lots are kept. A lot tagged private, customers-only, or
as curb parking is dropped. Size is the footprint area, so only polygons are
kept.

The point table uses EPSG:3857, the same projection as the street nodes.
``lon`` and ``lat`` are the original WGS 84 coordinates. :func:`build_map`
reads streets and lots from one file.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx._errors import InsufficientResponseError

from carpoolsim.map_loader.network import WEB_MERCATOR
from carpoolsim.map_loader.network import _gdf_for_file
from carpoolsim.map_loader.network import build_drive_network

LOTS_FILENAME = "lots.gpkg"
MIN_AREA_M2 = 2000.0

# ``access`` values that mean a traveler cannot use the lot as a public
# meeting place. A missing tag is not in this set: OSM's default for
# ``amenity=parking`` is public.
EXCLUDED_ACCESS = frozenset(
    {
        "customers",
        "delivery",
        "destination",
        "employees",
        "no",
        "permit",
        "private",
        "residents",
    }
)

# Curb parking rather than a lot with room to meet.
EXCLUDED_PARKING = frozenset({"lane", "street_side"})

_LOT_COLUMNS = (
    "lon",
    "lat",
    "x",
    "y",
    "name",
    "access",
    "parking",
    "park_ride",
    "capacity",
    "area_m2",
)


def parking_lots(
    features: gpd.GeoDataFrame,
    min_area_m2: float = MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """Return one EPSG:3857 point for each large public parking polygon.

    ``features`` is an OSM feature table in a geographic CRS, as returned by
    OSMnx ``features_from_xml`` for ``amenity=parking``. Each row needs an
    element type and an OSM id, either as the ``(element, id)`` index or as
    columns. ``min_area_m2`` is the minimum footprint in square meters,
    measured in the local UTM zone.

    The result is indexed by ``(element, osmid)``. ``x`` and ``y`` are the
    representative point in EPSG:3857 meters. ``lon`` and ``lat`` are that
    point in WGS 84. ``area_m2`` is the footprint. An empty input, or a table
    with nothing left after the filter, returns an empty frame with those
    columns.
    """
    if min_area_m2 < 0:
        raise ValueError("min_area_m2 must be zero or positive")
    if features.crs is None or not features.crs.is_geographic:
        raise ValueError("parking features must use a geographic CRS")

    work = features.copy()
    work["geometry"] = work.geometry.make_valid()
    work = work.loc[~work.geometry.is_empty]
    polygons = work.loc[work.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if polygons.empty:
        return _empty_lots()

    area = polygons.to_crs(polygons.estimate_utm_crs()).area
    kept = polygons.loc[
        (area >= min_area_m2)
        & ~_column(polygons, "access").map(lambda value: _restricted(value, EXCLUDED_ACCESS))
        & ~_column(polygons, "parking").map(lambda value: _restricted(value, EXCLUDED_PARKING))
    ]
    if kept.empty:
        return _empty_lots()

    points_ll = kept.geometry.representative_point()
    points_m = points_ll.to_crs(epsg=WEB_MERCATOR)
    element, osm_id = _element_and_id(kept)
    lots = gpd.GeoDataFrame(
        {
            "lon": points_ll.x.to_numpy(),
            "lat": points_ll.y.to_numpy(),
            "x": points_m.x.to_numpy(),
            "y": points_m.y.to_numpy(),
            "name": _text(kept, "name").to_numpy(),
            "access": _text(kept, "access").to_numpy(),
            "parking": _text(kept, "parking").to_numpy(),
            "park_ride": _text(kept, "park_ride").to_numpy(),
            "capacity": _text(kept, "capacity").to_numpy(),
            "area_m2": area.loc[kept.index].to_numpy(),
        },
        geometry=list(points_m.geometry),
        crs=f"EPSG:{WEB_MERCATOR}",
        index=pd.MultiIndex.from_arrays(
            [element.astype(str).to_numpy(), pd.to_numeric(osm_id).astype("int64").to_numpy()],
            names=["element", "osmid"],
        ),
    )
    return lots.sort_index()


def lot_coordinates(lots: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``x`` and ``y`` in EPSG:3857 meters, in row order."""
    return (
        lots["x"].to_numpy(dtype=np.float64),
        lots["y"].to_numpy(dtype=np.float64),
    )


def save_lots(lots: gpd.GeoDataFrame, directory: str | Path) -> Path:
    """Write ``lots.gpkg`` under ``directory`` and return its path.

    The in-memory frame is not modified. Index names become columns.
    List-valued tags are written as comma-separated text.
    """
    lot_dir = Path(directory)
    lot_dir.mkdir(parents=True, exist_ok=True)
    path = lot_dir / LOTS_FILENAME
    _gdf_for_file(lots).to_file(path, driver="GPKG")
    return path


def load_lots(directory: str | Path) -> gpd.GeoDataFrame:
    """Reload the parking-lot points written by :func:`save_lots`.

    The frame is indexed by ``(element, osmid)``. List-valued tags come back
    as comma-separated text.
    """
    lots = gpd.read_file(Path(directory) / LOTS_FILENAME)
    if {"element", "osmid"} <= set(lots.columns):
        lots = lots.set_index(["element", "osmid"])
    return lots


def build_parking_lots(
    filepath: str | Path,
    directory: str | Path | None = None,
    *,
    min_area_m2: float = MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """Read large public parking lots from an OSM XML file.

    ``filepath`` is the same ``.osm`` or ``.osm.bz2`` extract used for the
    street network. When ``directory`` is set, the points are written there
    with :func:`save_lots`. A file with no parking features returns an empty
    frame.
    """
    lots = parking_lots(_parking_features(filepath), min_area_m2)
    if directory is not None:
        save_lots(lots, directory)
    return lots


def build_map(
    filepath: str | Path,
    directory: str | Path | None = None,
    *,
    min_area_m2: float = MIN_AREA_M2,
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load drivable streets and large public parking lots from one OSM file.

    The file is read once for the street graph and once for parking
    features. When ``directory`` is set, ``nodes.gpkg``, ``edges.gpkg``, and
    ``lots.gpkg`` are written there.
    """
    graph, nodes, edges = build_drive_network(filepath, directory)
    lots = build_parking_lots(filepath, directory, min_area_m2=min_area_m2)
    return graph, nodes, edges, lots


def _parking_features(filepath: str | Path) -> gpd.GeoDataFrame:
    """Return ``amenity=parking`` features, or an empty table when OSM has none."""
    try:
        return ox.features_from_xml(filepath, tags={"amenity": "parking"})
    except InsufficientResponseError:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            index=pd.MultiIndex.from_arrays([[], []], names=["element", "id"]),
        )


def _empty_lots() -> gpd.GeoDataFrame:
    """Return a lot table with the stored columns and no rows."""
    numeric = {"lon", "lat", "x", "y", "area_m2"}
    data = {
        column: pd.Series(dtype="float64" if column in numeric else "object")
        for column in _LOT_COLUMNS
    }
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.GeoSeries([], crs=f"EPSG:{WEB_MERCATOR}"),
        index=pd.MultiIndex.from_arrays([[], []], names=["element", "osmid"]),
    )


def _column(frame: gpd.GeoDataFrame, name: str) -> pd.Series:
    """Return a tag column, or missing values when OSM did not use that tag."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _text(frame: gpd.GeoDataFrame, name: str) -> pd.Series:
    """Return a tag column as text, with list values joined by commas."""
    return _column(frame, name).map(_plain_text)


def _plain_text(value: object) -> object:
    """Return one tag value as text, or missing when the tag is absent."""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(part) for part in value)
    if _missing(value):
        return pd.NA
    return str(value)


def _restricted(value: object, banned: frozenset[str]) -> bool:
    """Return whether a tag value names a banned class.

    A list or a semicolon-separated value is banned when any part is banned.
    A missing tag is not banned.
    """
    if isinstance(value, (list, tuple, set)):
        return any(_restricted(part, banned) for part in value)
    if _missing(value):
        return False
    return any(part.strip() in banned for part in str(value).split(";"))


def _missing(value: object) -> bool:
    """Return whether a tag value is absent."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _element_and_id(features: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the OSM element type and id aligned to ``features``."""
    index = features.index
    if isinstance(index, pd.MultiIndex):
        names = set(index.names)
        id_name = "osmid" if "osmid" in names else "id" if "id" in names else None
        if "element" in names and id_name is not None:
            return (
                pd.Series(index.get_level_values("element"), index=index),
                pd.Series(index.get_level_values(id_name), index=index),
            )
    if {"element", "osmid"} <= set(features.columns):
        return features["element"], features["osmid"]
    raise ValueError("parking features need an element type and an OSM id")
