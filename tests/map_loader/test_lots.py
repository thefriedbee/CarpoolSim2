"""Tests for large public parking lots."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry import Polygon

from carpoolsim.map_loader.lots import build_map
from carpoolsim.map_loader.lots import build_parking_lots
from carpoolsim.map_loader.lots import load_lots
from carpoolsim.map_loader.lots import lot_coordinates
from carpoolsim.map_loader.lots import parking_lots
from carpoolsim.map_loader.lots import save_lots

_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _square(lon: float, lat: float, size: float) -> Polygon:
    """Return a geographic square with its southwest corner at ``lon``, ``lat``."""
    return Polygon(
        [
            (lon, lat),
            (lon + size, lat),
            (lon + size, lat + size),
            (lon, lat + size),
        ]
    )


def _features(rows: list[tuple[str, int, object, dict[str, object]]]) -> gpd.GeoDataFrame:
    """Return an OSM-style feature table in WGS 84."""
    return gpd.GeoDataFrame(
        [tags for _element, _osmid, _geometry, tags in rows],
        geometry=[geometry for _element, _osmid, geometry, _tags in rows],
        crs="EPSG:4326",
        index=pd.MultiIndex.from_arrays(
            [
                [element for element, _osmid, _geometry, _tags in rows],
                [osmid for _element, osmid, _geometry, _tags in rows],
            ],
            names=["element", "id"],
        ),
    )


def test_large_public_lots_become_projected_points() -> None:
    footprint = _square(-84.396, 33.775, 0.002)
    features = _features(
        [
            (
                "way",
                1,
                footprint,
                {"name": "Open", "parking": "surface", "capacity": "40"},
            ),
            (
                "way",
                2,
                _square(-84.390, 33.775, 0.002),
                {"access": "permissive", "park_ride": "yes"},
            ),
            ("way", 3, _square(-84.380, 33.775, 0.0002), {}),
            ("way", 4, _square(-84.370, 33.775, 0.002), {"access": "private"}),
            ("way", 5, _square(-84.360, 33.775, 0.002), {"parking": "street_side"}),
            ("way", 6, _square(-84.350, 33.775, 0.002), {"parking": "lane"}),
            ("way", 7, _square(-84.340, 33.775, 0.002), {"access": "customers"}),
            ("node", 8, Point(-84.33, 33.78), {"amenity": "parking"}),
        ]
    )

    lots = parking_lots(features)

    assert lots.index.tolist() == [("way", 1), ("way", 2)]
    assert lots.crs.to_epsg() == 3857
    assert list(lots.columns[:4]) == ["lon", "lat", "x", "y"]
    assert lots.loc[("way", 1), "name"] == "Open"
    assert lots.loc[("way", 1), "capacity"] == "40"
    assert pd.isna(lots.loc[("way", 1), "access"])
    assert lots.loc[("way", 2), "park_ride"] == "yes"
    assert footprint.contains(Point(lots.loc[("way", 1), "lon"], lots.loc[("way", 1), "lat"]))
    x, y = _TO_MERCATOR.transform(lots.loc[("way", 1), "lon"], lots.loc[("way", 1), "lat"])
    assert lots.loc[("way", 1), "x"] == pytest.approx(x)
    assert lots.loc[("way", 1), "y"] == pytest.approx(y)
    assert lots["area_m2"].min() > 2000
    assert features.index.names == ["element", "id"]


def test_semicolon_or_list_access_drops_a_restricted_lot() -> None:
    features = _features(
        [
            ("way", 1, _square(-84.396, 33.775, 0.002), {"access": "yes;private"}),
            ("way", 2, _square(-84.390, 33.775, 0.002), {"access": ["permit"]}),
            ("way", 3, _square(-84.380, 33.775, 0.002), {"access": "yes"}),
        ]
    )

    lots = parking_lots(features)

    assert lots.index.tolist() == [("way", 3)]


def test_negative_min_area_is_rejected() -> None:
    features = _features([("way", 1, _square(-84.396, 33.775, 0.002), {})])

    with pytest.raises(ValueError, match="min_area_m2"):
        parking_lots(features, -1)


def test_projected_features_are_rejected() -> None:
    features = _features([("way", 1, _square(-84.396, 33.775, 0.002), {})]).to_crs(3857)

    with pytest.raises(ValueError, match="geographic"):
        parking_lots(features)


def test_features_without_an_id_are_rejected() -> None:
    features = gpd.GeoDataFrame(
        geometry=[_square(-84.396, 33.775, 0.002)],
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="OSM id"):
        parking_lots(features)


def test_no_parking_polygons_returns_an_empty_table() -> None:
    lots = parking_lots(
        gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            index=pd.MultiIndex.from_arrays([[], []], names=["element", "id"]),
        )
    )

    assert lots.empty
    assert lots.crs.to_epsg() == 3857
    assert lots.index.names == ["element", "osmid"]
    px, py = lot_coordinates(lots)
    assert px.shape == (0,)
    assert py.shape == (0,)


def test_save_load_round_trip(tmp_path: Path) -> None:
    features = _features(
        [("way", 10, _square(-84.396, 33.775, 0.002), {"name": ["Deck", "A"], "access": "yes"})]
    )
    lots = parking_lots(features)

    path = save_lots(lots, tmp_path / "nested" / "net")
    loaded = load_lots(path.parent)
    px, py = lot_coordinates(loaded)

    assert path.name == "lots.gpkg"
    assert loaded.index.tolist() == [("way", 10)]
    assert loaded.loc[("way", 10), "name"] == "Deck,A"
    assert loaded.loc[("way", 10), "x"] == pytest.approx(lots.loc[("way", 10), "x"])
    assert loaded.loc[("way", 10), "y"] == pytest.approx(lots.loc[("way", 10), "y"])
    assert px.tolist() == pytest.approx([lots.loc[("way", 10), "x"]])
    assert py.tolist() == pytest.approx([lots.loc[("way", 10), "y"]])
    assert lots.loc[("way", 10), "name"] == "Deck,A"


def test_build_map_reads_streets_and_lots_from_one_file(tmp_path: Path) -> None:
    filepath = tmp_path / "city.osm"
    filepath.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="33.770" lon="-84.390" version="1"/>
  <node id="2" lat="33.771" lon="-84.390" version="1"/>
  <node id="11" lat="33.772" lon="-84.390" version="1"/>
  <node id="12" lat="33.773" lon="-84.390" version="1"/>
  <node id="13" lat="33.774" lon="-84.390" version="1"/>
  <node id="14" lat="33.774" lon="-84.389" version="1"/>
  <node id="15" lat="33.774" lon="-84.388" version="1"/>
  <node id="16" lat="33.774" lon="-84.387" version="1"/>
  <node id="3" lat="33.775" lon="-84.396" version="1"/>
  <node id="4" lat="33.775" lon="-84.394" version="1"/>
  <node id="5" lat="33.777" lon="-84.394" version="1"/>
  <node id="6" lat="33.777" lon="-84.396" version="1"/>
  <node id="7" lat="33.780" lon="-84.390" version="1"/>
  <node id="8" lat="33.780" lon="-84.388" version="1"/>
  <node id="9" lat="33.782" lon="-84.388" version="1"/>
  <node id="10" lat="33.782" lon="-84.390" version="1"/>
  <way id="20" version="1">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="11"/>
    <nd ref="12"/>
    <nd ref="13"/>
    <nd ref="14"/>
    <nd ref="15"/>
    <nd ref="16"/>
    <tag k="highway" v="residential"/>
    <tag k="maxspeed" v="50"/>
  </way>
  <way id="30" version="1">
    <nd ref="3"/>
    <nd ref="4"/>
    <nd ref="5"/>
    <nd ref="6"/>
    <nd ref="3"/>
    <tag k="amenity" v="parking"/>
    <tag k="name" v="Public Lot"/>
    <tag k="parking" v="surface"/>
  </way>
  <way id="31" version="1">
    <nd ref="7"/>
    <nd ref="8"/>
    <nd ref="9"/>
    <nd ref="10"/>
    <nd ref="7"/>
    <tag k="amenity" v="parking"/>
    <tag k="access" v="private"/>
  </way>
</osm>
""",
        encoding="utf-8",
    )

    out = tmp_path / "network"
    graph, nodes, edges, lots = build_map(filepath, out)

    assert graph.number_of_edges() > 0
    assert not nodes.empty
    assert not edges.empty
    assert lots.index.tolist() == [("way", 30)]
    assert lots.loc[("way", 30), "name"] == "Public Lot"
    assert (out / "nodes.gpkg").exists()
    assert (out / "edges.gpkg").exists()
    assert (out / "lots.gpkg").exists()
    assert load_lots(out).index.tolist() == [("way", 30)]


def test_osm_file_without_parking_returns_no_lots(tmp_path: Path) -> None:
    filepath = tmp_path / "streets.osm"
    filepath.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="33.770" lon="-84.390" version="1"/>
  <node id="2" lat="33.771" lon="-84.390" version="1"/>
  <way id="20" version="1">
    <nd ref="1"/>
    <nd ref="2"/>
    <tag k="highway" v="residential"/>
  </way>
</osm>
""",
        encoding="utf-8",
    )

    lots = build_parking_lots(filepath)

    assert lots.empty
    assert lots.crs.to_epsg() == 3857
