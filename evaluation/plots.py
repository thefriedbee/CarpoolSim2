"""Charts of travel time before and after carpooling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROLE_COLORS = {
    "driver": "#1f77b4",
    "rider": "#ff7f0e",
    "solo": "#9aa0a6",
}


def plot_before_after(travelers: pd.DataFrame) -> Figure:
    """Scatter solo time against door-to-door carpool time.

    One panel when the run has a single mode, including people who drove
    alone. One panel per mode when several modes committed pairs. The dashed
    line is an unchanged trip.
    """
    frame = _with_mode(travelers)
    modes = _matched_modes(frame)
    low, high = _time_limits(frame)
    with sns.axes_style("whitegrid"):
        if len(modes) <= 1:
            fig, axes = plt.subplots(figsize=(6, 6), constrained_layout=True)
            panels = [axes]
            groups = [frame]
            titles = [_mode_title(modes[0]) if modes else "Solo"]
        else:
            fig, axes = plt.subplots(
                1,
                len(modes),
                figsize=(6 * len(modes), 6),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            panels = list(axes)
            groups = [frame[frame["mode"] == mode] for mode in modes]
            titles = [_mode_title(mode) for mode in modes]
        for ax, group, title in zip(panels, groups, titles, strict=True):
            _scatter(ax, group, low, high, title)
    return fig


def plot_delta(travelers: pd.DataFrame) -> Figure:
    """Histogram of door-to-door change, for drivers and for riders.

    The change is carpool time minus solo time, in minutes. A mark at zero
    is an unchanged trip.
    """
    frame = _with_mode(travelers)
    modes = _matched_modes(frame)
    matched = frame[frame["role"] != "solo"]
    span = 1.0 if matched.empty else max(float(matched["delta_min"].abs().max()), 1.0)
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, constrained_layout=True)
        for ax, role in zip(axes, ("driver", "rider"), strict=True):
            subset = frame[frame["role"] == role]
            _histogram(ax, subset, "delta_min", modes, ROLE_COLORS[role])
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlim(-span, span)
            ax.set_title(role.capitalize())
            ax.set_xlabel("Change in travel time (min)")
        axes[0].set_ylabel("Travelers")
        axes[1].set_ylabel("")
    return fig


def plot_vehicle_saved(matches: pd.DataFrame) -> Figure:
    """Histogram of driving minutes saved by each carpool.

    The title carries the total over the run. Waiting is not included.
    """
    total = 0.0 if matches.empty else float(matches["vehicle_min_saved"].sum())
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_title(f"Total {total:.0f} min saved")
        if not matches.empty:
            _histogram(ax, matches, "vehicle_min_saved", _matched_modes(matches), "#2ca02c")
            ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Vehicle minutes saved")
        ax.set_ylabel("Carpools")
    return fig


def plot_by_hour(travelers: pd.DataFrame) -> Figure:
    """Stacked departure counts and the share who found a partner.

    Hours are clock hours of ``departure_min``. Matched people are drivers
    and riders. The line is matched people divided by everyone departing
    in that hour.
    """
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.set_xlabel("Departure hour")
        ax.set_ylabel("Travelers")
        if travelers.empty:
            return fig
        frame = travelers.copy()
        frame["hour"] = (frame["departure_min"] // 60).astype(int)
        start = int(frame["hour"].min())
        end = int(frame["hour"].max())
        hours = list(range(start, end + 1))
        matched_counts = []
        solo_counts = []
        for hour in hours:
            in_hour = frame[frame["hour"] == hour]
            solo_counts.append(int((in_hour["role"] == "solo").sum()))
            matched_counts.append(int((in_hour["role"] != "solo").sum()))
        labels = [f"{hour:02d}:00" for hour in hours]
        positions = np.arange(len(hours))
        ax.bar(positions, solo_counts, color=ROLE_COLORS["solo"], label="Solo", width=0.8)
        ax.bar(
            positions,
            matched_counts,
            bottom=solo_counts,
            color=ROLE_COLORS["driver"],
            label="Matched",
            width=0.8,
        )
        ax.set_xticks(positions, labels)
        if len(hours) > 8:
            ax.tick_params(axis="x", labelrotation=45)
        rate_axis = ax.twinx()
        rates = [
            matched / (matched + solo) if matched + solo else 0.0
            for matched, solo in zip(matched_counts, solo_counts, strict=True)
        ]
        sns.lineplot(x=positions, y=rates, color="black", marker="o", ax=rate_axis, label="Match rate")
        rate_axis.set_ylabel("Match rate")
        rate_axis.set_ylim(0, 1)
        handles, labels_for_legend = ax.get_legend_handles_labels()
        rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
        ax.legend(handles + rate_handles, labels_for_legend + rate_labels, loc="upper right")
    return fig


def _with_mode(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy ``frame`` with a blank mode where the cell was empty."""
    out = frame.copy()
    if "mode" in out.columns:
        out["mode"] = out["mode"].fillna("")
    return out


def _matched_modes(frame: pd.DataFrame) -> list[str]:
    """Mode names on matched rows, in first-seen order."""
    if "role" in frame.columns:
        matched = frame[frame["role"] != "solo"]
    else:
        matched = frame
    seen: list[str] = []
    for mode in matched["mode"].fillna(""):
        if mode not in seen:
            seen.append(str(mode))
    return seen


def _mode_title(mode: str) -> str:
    """Panel title for a mode. An unnamed pipeline is labeled Carpool."""
    return mode if mode else "Carpool"


def _time_limits(travelers: pd.DataFrame) -> tuple[float, float]:
    """Axis limits covering solo and carpool times, with a margin."""
    if travelers.empty:
        return 0.0, 1.0
    low = float(min(travelers["solo_min"].min(), travelers["carpool_min"].min()))
    high = float(max(travelers["solo_min"].max(), travelers["carpool_min"].max()))
    if high == low:
        return low - 1.0, high + 1.0
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def _scatter(ax: Axes, travelers: pd.DataFrame, low: float, high: float, title: str) -> None:
    """Draw one before/after panel."""
    identity = ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1, zorder=1)[0]
    present = [role for role in ("solo", "driver", "rider") if (travelers["role"] == role).any()]
    if present:
        shown = travelers[travelers["role"].isin(present)].copy()
        shown["Role"] = shown["role"].str.capitalize()
        order = [role.capitalize() for role in present]
        palette = {role.capitalize(): ROLE_COLORS[role] for role in present}
        sns.scatterplot(
            data=shown,
            x="solo_min",
            y="carpool_min",
            hue="Role",
            hue_order=order,
            palette=palette,
            ax=ax,
            zorder=2,
            alpha=0.85,
            linewidth=0,
            legend=True,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[identity, *handles], labels=["Same time", *labels])
    else:
        ax.legend(handles=[identity], labels=["Same time"])
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Solo travel time (min)")
    ax.set_ylabel("Carpool travel time (min)")
    ax.set_title(title)


def _histogram(ax: Axes, frame: pd.DataFrame, column: str, modes: list[str], color: str) -> None:
    """Draw one series, or one series per mode when the frame has several."""
    if frame.empty:
        return
    bins = _bin_edges(frame[column])
    if len(modes) <= 1:
        sns.histplot(data=frame, x=column, bins=bins, color=color, edgecolor="white", ax=ax)
        return
    shown = frame.copy()
    shown["mode"] = shown["mode"].fillna("").map(_mode_title)
    order = []
    for mode in modes:
        title = _mode_title(mode)
        if title not in order and (shown["mode"] == title).any():
            order.append(title)
    sns.histplot(
        data=shown,
        x=column,
        hue="mode",
        hue_order=order,
        bins=bins,
        multiple="layer",
        alpha=0.65,
        edgecolor="white",
        ax=ax,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(None)


def _bin_edges(values: pd.Series) -> np.ndarray:
    """Bin edges that still span a sample whose values are all equal."""
    count = max(int(values.shape[0]), 1)
    pieces = min(20, max(5, int(count**0.5)))
    low = float(values.min())
    high = float(values.max())
    if low == high:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, pieces + 1)
