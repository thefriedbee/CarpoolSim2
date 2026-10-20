"""Maps of individual drive-alone, direct, and park-and-ride trips."""

from __future__ import annotations

import math

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from evaluation.plots import ROLE_COLORS

# Same encoding as the four travel-plan diagrams: the driver is orange,
# the rider is blue, and a park-and-ride lot is a green square.
DRIVER_COLOR = "#AD4D28"
RIDER_COLOR = "#1E749A"
PNR_COLOR = "#1B8F78"
_PERSON_COLOR = {"driver": DRIVER_COLOR, "rider": RIDER_COLOR}

# Pickup and dropoff are the driver alone. Access is the rider's own drive
# to a lot. The shared ride is dashed, in the driver's color.
_CARPOOL_LEGS = (
    ("pickup", DRIVER_COLOR, 2.2, "-", "driver"),
    ("dropoff", DRIVER_COLOR, 2.2, "-", "driver"),
    ("access", RIDER_COLOR, 2.2, "-", "rider"),
    ("shared", DRIVER_COLOR, 2.5, "--", "shared"),
)


def plot_trip(
    trip_id: int,
    travelers: pd.DataFrame,
    matches: pd.DataFrame,
    legs: gpd.GeoDataFrame,
    *,
    basemap: bool = False,
) -> Figure:
    """Draw a saved traveler by trip ID, whether solo, driver, or rider.

    Drive-alone trips have one panel. Matched trips compare both travelers'
    solo routes with the pickup, shared ride, and dropoff; a PNR carpool
    also shows the rider's drive to the lot. Orange is the driver and blue
    is the rider. A triangle is an origin, a circle is a destination, and a
    green square is a park-and-ride lot. Solid lines are one person driving;
    the shared ride is dashed. The selected traveler's origin is outlined,
    and the title reports door-to-door time, waiting, and access.

    Coordinates and leg geometry use EPSG:3857, as in ``load_results``.
    New outputs store drive-alone routes with ``trip_id``. For older outputs
    without that geometry, the solo panel explicitly labels a dashed
    straight-line fallback; it does not infer a road route. No network or
    online map is needed unless ``basemap=True``.

    Example::

        tables = load_results("data_outputs/gt_2022")
        fig = plot_trip(42, tables.travelers, tables.matches, tables.legs)
    """
    person = _person(travelers, trip_id)
    role = person["role"]
    if role == "solo":
        return _drive_alone(person, legs, basemap=basemap)
    if role not in ("driver", "rider"):
        raise ValueError(f"trip {trip_id} has unknown role {role!r}")
    chosen = matches[
        (matches["driver_id"] == int(trip_id)) | (matches["rider_id"] == int(trip_id))
    ]
    if len(chosen) != 1:
        raise ValueError(f"trip {trip_id} must belong to exactly one match")
    match = chosen.iloc[0]
    if int(match[f"{role}_id"]) != int(trip_id):
        raise ValueError(f"trip {trip_id} has a role inconsistent with its match")
    fig = plot_match(int(match["match_id"]), travelers, matches, legs, basemap=basemap)
    mode = "PNR carpool" if _has_lot(match) else "Direct carpool"
    fig.axes[0].set_title("Drive alone baseline")
    fig.axes[1].set_title(mode)
    for ax in fig.axes:
        ax.scatter(
            person["origin_x"], person["origin_y"],
            facecolors="none", edgecolors="black", linewidths=1.5, marker="^", s=150, zorder=6,
        )
        ax.annotate(
            f"Trip {int(trip_id)}", (person["origin_x"], person["origin_y"]),
            xytext=(7, 7), textcoords="offset points", fontsize=9, zorder=6,
        )
    fig.suptitle(
        f"Trip {int(trip_id)} | {role.capitalize()} | {mode}\n"
        f"Solo {person['solo_min']:.1f} min → carpool {person['carpool_min']:.1f} min "
        f"({person['delta_min']:+.1f}); wait {person['wait_min']:.1f}, "
        f"access {person['access_min']:.1f} min | "
        f"Vehicle {match['vehicle_min_saved']:.1f} min saved",
        fontsize=11,
    )
    return fig


def _drive_alone(person: pd.Series, legs: gpd.GeoDataFrame, *, basemap: bool) -> Figure:
    """Use this solo traveler's saved road route, or label a schematic."""
    trip_id = int(person["trip_id"])
    selected = legs.iloc[0:0]
    if "trip_id" in legs.columns:
        selected = legs[(legs["trip_id"] == trip_id) & (legs["leg"] == "solo")]
    drawn = _with_geometry(selected)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if drawn.empty:
        ax.plot(
            [person["origin_x"], person["dest_x"]],
            [person["origin_y"], person["dest_y"]],
            color=ROLE_COLORS["solo"], linewidth=2, linestyle="--",
            label="Straight-line fallback", zorder=2,
        )
        _arrow(
            ax,
            LineString([(person["origin_x"], person["origin_y"]), (person["dest_x"], person["dest_y"])]),
            ROLE_COLORS["solo"],
            2.0,
            "--",
        )
        ax.text(
            0.02, 0.02, "Straight-line fallback: road route not stored; dashed line joins endpoints.",
            transform=ax.transAxes, fontsize=9, va="bottom", wrap=True,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    else:
        _draw_lines(ax, drawn, ROLE_COLORS["solo"], 2.5, "-", 2)
    _mark_stop(ax, person["origin_x"], person["origin_y"], ROLE_COLORS["solo"], "^")
    _mark_stop(ax, person["dest_x"], person["dest_y"], ROLE_COLORS["solo"], "o")
    ax.set_title("Drive alone")
    ax.set_axis_off()
    _zoom([ax], drawn, person, person, pd.Series({"lot_x": None, "lot_y": None}))
    _legend(ax, drive_alone=not drawn.empty, fallback=drawn.empty)
    if basemap:
        _add_basemap([ax])
    fig.suptitle(f"Trip {trip_id} | Drive alone | {person['solo_min']:.1f} min")
    return fig


def plot_match(
    match_id: int,
    travelers: pd.DataFrame,
    matches: pd.DataFrame,
    legs: gpd.GeoDataFrame,
    *,
    basemap: bool = False,
) -> Figure:
    """Draw one carpool before and after the match.

    The before panel is the two solo drives. The after panel is the driver's
    pickup and dropoff, the dashed shared ride, and, at a park-and-ride lot,
    the rider's own drive there. Orange is the driver and blue is the rider.
    Triangles are origins, circles are destinations, and a green square is
    the lot. Set ``basemap`` to lay a map under a real city's coordinates.
    """
    chosen = matches[matches["match_id"] == int(match_id)]
    if chosen.empty:
        raise ValueError(f"no match {match_id}")
    match = chosen.iloc[0]
    trip_legs = legs[legs["match_id"] == int(match_id)]
    driver = _person(travelers, match["driver_id"])
    rider = _person(travelers, match["rider_id"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    _solo_panel(axes[0], trip_legs, driver, rider)
    _carpool_panel(axes[1], trip_legs, match, driver, rider)
    _zoom(list(axes), trip_legs, driver, rider, match)
    if basemap:
        _add_basemap(list(axes))
    fig.suptitle(
        f"Driver {int(driver['trip_id'])}: {driver['delta_min']:+.1f} min, "
        f"rider {int(rider['trip_id'])}: {rider['delta_min']:+.1f} min, "
        f"vehicle {match['vehicle_min_saved']:.1f} min saved"
    )
    return fig


def plot_day(
    travelers: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    basemap: bool = False,
) -> Figure:
    """Draw each trip as a straight line from origin to destination.

    Line color is the role. Park-and-ride lots are green squares when a match
    recorded one. Set ``basemap`` to lay a map under a real city's coordinates.
    """
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    if travelers.empty:
        ax.set_axis_off()
        return fig
    for role, color in ROLE_COLORS.items():
        group = travelers[travelers["role"] == role]
        if group.empty:
            continue
        lines = gpd.GeoDataFrame(
            geometry=[
                LineString([(x0, y0), (x1, y1)])
                for x0, y0, x1, y1 in zip(
                    group["origin_x"],
                    group["origin_y"],
                    group["dest_x"],
                    group["dest_y"],
                    strict=True,
                )
            ],
            crs="EPSG:3857",
        )
        lines.plot(ax=ax, color=color, linewidth=1.2, alpha=0.8, label=role.capitalize())
        ax.scatter(group["origin_x"], group["origin_y"], c=color, s=18, zorder=3)
    lots = matches.dropna(subset=["lot_x", "lot_y"])
    if not lots.empty:
        ax.scatter(
            lots["lot_x"],
            lots["lot_y"],
            c=PNR_COLOR,
            marker="s",
            s=42,
            zorder=4,
            label="PNR facility",
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    _role_legend(ax)
    if basemap:
        _add_basemap([ax])
    return fig


def _solo_panel(ax: Axes, legs: gpd.GeoDataFrame, driver: pd.Series, rider: pd.Series) -> None:
    """Draw the two solo drives and the places they start and end."""
    drawn = {}
    for who in ("driver", "rider"):
        group = legs[(legs["leg"] == "solo") & (legs["who"] == who)]
        drawn[who] = _draw_lines(ax, group, _PERSON_COLOR[who], 2.2, "-", 2)
    _mark_ends(ax, driver, rider)
    ax.set_title("Before")
    ax.set_axis_off()
    _legend(ax, driver=drawn["driver"], rider=drawn["rider"])


def _carpool_panel(
    ax: Axes,
    legs: gpd.GeoDataFrame,
    match: pd.Series,
    driver: pd.Series,
    rider: pd.Series,
) -> None:
    """Draw the carpool legs and the places people start, stop, and meet."""
    drawn = {"driver": False, "rider": False, "shared": False}
    for name, color, width, style, key in _CARPOOL_LEGS:
        if _draw_lines(ax, legs[legs["leg"] == name], color, width, style, 3 if name == "shared" else 2):
            drawn[key] = True
    _mark_ends(ax, driver, rider)
    pnr = _has_lot(match)
    if pnr:
        _mark_lot(ax, match["lot_x"], match["lot_y"])
    ax.set_title("After")
    ax.set_axis_off()
    _legend(ax, driver=drawn["driver"], rider=drawn["rider"], shared=drawn["shared"], pnr=pnr)


def _role_legend(ax: Axes) -> None:
    """Place the day-overview legend under the map."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9)


class _ArrowLegend(HandlerPatch):
    """Draw a legend swatch as a short arrow, matching the plan diagrams."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        arrow = FancyArrowPatch(
            (xdescent, ydescent + height / 2),
            (xdescent + width, ydescent + height / 2),
            arrowstyle="-|>",
            mutation_scale=fontsize * 0.75,
            linewidth=orig_handle.get_linewidth() or 2,
            linestyle=orig_handle.get_linestyle(),
            color=orig_handle.get_edgecolor(),
        )
        arrow.set_transform(trans)
        return [arrow]


def _legend(
    ax: Axes,
    *,
    driver: bool = False,
    rider: bool = False,
    shared: bool = False,
    pnr: bool = False,
    drive_alone: bool = False,
    fallback: bool = False,
) -> None:
    """Place the plan legend under the map so it does not cover the trips."""
    handles: list = [
        Line2D(
            [], [], linestyle="None", marker="^", markersize=8,
            markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=1.4,
            label="Origin",
        ),
        Line2D(
            [], [], linestyle="None", marker="o", markersize=7.5,
            markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=1.4,
            label="Destination",
        ),
    ]
    if pnr:
        handles.append(
            Line2D(
                [], [], linestyle="None", marker="s", markersize=8,
                markerfacecolor=PNR_COLOR, markeredgecolor=PNR_COLOR,
                label="PNR facility",
            )
        )
    if driver:
        handles.append(_arrow_swatch(DRIVER_COLOR, "-", "Driver drives"))
    if rider:
        handles.append(_arrow_swatch(RIDER_COLOR, "-", "Rider drives"))
    if shared:
        handles.append(_arrow_swatch(DRIVER_COLOR, "--", "Shared"))
    if drive_alone:
        handles.append(_arrow_swatch(ROLE_COLORS["solo"], "-", "Drive alone route"))
    if fallback:
        handles.append(_arrow_swatch(ROLE_COLORS["solo"], "--", "Straight-line fallback"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        handler_map={FancyArrowPatch: _ArrowLegend()},
    )


def _arrow_swatch(color: str, linestyle: str, label: str) -> FancyArrowPatch:
    """A legend arrow in ``color``. The handler redraws it to the swatch size."""
    arrow = FancyArrowPatch(
        (0, 0), (1, 0),
        arrowstyle="-|>", color=color, linestyle=linestyle, linewidth=2.0, mutation_scale=10,
    )
    arrow.set_label(label)
    return arrow


def _draw_lines(
    ax: Axes,
    legs: gpd.GeoDataFrame,
    color: str,
    width: float,
    linestyle: str,
    zorder: int,
) -> bool:
    """Plot the lines in ``legs`` that have a geometry. Return whether any were drawn."""
    drawn = _with_geometry(legs)
    if drawn.empty:
        return False
    drawn.plot(ax=ax, color=color, linewidth=width, linestyle=linestyle, zorder=zorder)
    for geom in drawn.geometry:
        _arrow(ax, geom, color, width, linestyle)
    return True


def _arrow(ax: Axes, geom: BaseGeometry, color: str, width: float, linestyle: str) -> None:
    """Point a short arrow along the end of a route."""
    if geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            _arrow(ax, part, color, width, linestyle)
        return
    if geom.geom_type != "LineString" or geom.length <= 0:
        return
    coords = list(geom.coords)
    x1, y1 = coords[-1][0], coords[-1][1]
    for x0, y0 in reversed(coords[:-1]):
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist > 0:
            break
    else:
        return
    arrow_len = min(dist, max(geom.length * 0.14, dist * 0.35))
    sx = x1 - (x1 - x0) / dist * arrow_len
    sy = y1 - (y1 - y0) / dist * arrow_len
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(sx, sy),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": width,
            "linestyle": linestyle,
            "mutation_scale": 12,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def _mark_ends(ax: Axes, driver: pd.Series, rider: pd.Series) -> None:
    """Mark both origins and both destinations."""
    _mark_stop(ax, driver["origin_x"], driver["origin_y"], DRIVER_COLOR, "^")
    _mark_stop(ax, rider["origin_x"], rider["origin_y"], RIDER_COLOR, "^")
    _mark_stop(ax, driver["dest_x"], driver["dest_y"], DRIVER_COLOR, "o")
    _mark_stop(ax, rider["dest_x"], rider["dest_y"], RIDER_COLOR, "o")


def _mark_stop(ax: Axes, x: float, y: float, color: str, marker: str) -> None:
    """Draw one hollow origin triangle or destination circle."""
    ax.scatter(
        x, y,
        s=86 if marker == "^" else 74,
        marker=marker,
        facecolors="white",
        edgecolors=color,
        linewidths=1.7,
        zorder=5,
    )


def _mark_lot(ax: Axes, x: float, y: float) -> None:
    """Draw the park-and-ride lot as a green square with a P."""
    ax.scatter(
        x, y,
        c=PNR_COLOR,
        marker="s",
        s=150,
        zorder=5,
        label="PNR facility",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.annotate(
        "P",
        (x, y),
        ha="center",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
        zorder=6,
    )


def _person(travelers: pd.DataFrame, trip_id: object) -> pd.Series:
    """Return the traveler row for ``trip_id``."""
    rows = travelers[travelers["trip_id"] == int(trip_id)]
    if rows.empty:
        raise ValueError(f"trip {trip_id} is not in the traveler table")
    if len(rows) != 1:
        raise ValueError(f"trip {trip_id} occurs more than once in the traveler table")
    return rows.iloc[0]


def _with_geometry(legs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select usable geometry, including stationary zero-length routes."""
    present = legs[~legs.geometry.is_empty]
    return present[present.geometry.notna()]


def _has_lot(match: pd.Series) -> bool:
    """Lot coordinates identify PNR even when the mode has a custom name."""
    return bool(pd.notna(match["lot_x"]) and pd.notna(match["lot_y"]))


def _zoom(
    axes: list[Axes],
    legs: gpd.GeoDataFrame,
    driver: pd.Series,
    rider: pd.Series,
    match: pd.Series,
) -> None:
    """Frame both panels on the trips, with a margin."""
    xs = [float(driver["origin_x"]), float(driver["dest_x"]), float(rider["origin_x"]), float(rider["dest_x"])]
    ys = [float(driver["origin_y"]), float(driver["dest_y"]), float(rider["origin_y"]), float(rider["dest_y"])]
    if _has_lot(match):
        xs.append(float(match["lot_x"]))
        ys.append(float(match["lot_y"]))
    drawn = _with_geometry(legs)
    if not drawn.empty:
        minx, miny, maxx, maxy = drawn.total_bounds
        xs.extend([float(minx), float(maxx)])
        ys.extend([float(miny), float(maxy)])
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.08 * span
    for ax in axes:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal", adjustable="box")


def _add_basemap(axes: list[Axes]) -> None:
    """Lay a light basemap under axes that are already limited."""
    import contextily as cx

    for ax in axes:
        cx.add_basemap(ax, crs="EPSG:3857")
