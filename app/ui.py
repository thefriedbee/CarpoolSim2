"""Small display helpers shared by the demo pages."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from carpoolsim.config import SimulationConfig

FILTER_LABELS = {
    "FilterDepartureTime": "Departure gap",
    "FilterOriginEuclideanDistance": "Origin distance",
    "FilterDestinationEuclideanDistance": "Destination distance",
    "FilterRerouteEuclideanDistance": "Euclidean detour",
    "FilterDropoffEuclideanDistance": "Dropoff distance",
    "FilterRerouteTravelTime": "Driving-time ratio",
    "FilterSharedTravelTime": "Shared-time ratio",
    "FilterPassengerWaitTime": "Passenger wait",
    "FilterPnrAccess": "Lot access",
    "FilterPnrRerouteEuclidean": "Euclidean detour",
    "FilterPnrPassengerAccess": "Passenger access",
    "FilterPnrRerouteTravelTime": "Driving-time ratio",
    "FilterPnrSharedTravelTime": "Shared-time ratio",
    "FilterPnrWait": "Wait at the lot",
}

_UNITS = {
    "delta1": "min",
    "delta2": "min",
    "r1": "km",
    "r2": "km",
    "mu_1": "ratio",
    "mu_2": "ratio",
    "mu_3": "ratio",
    "gamma_1": "ratio",
    "gamma_2": "ratio",
}


def show_metrics(summary: dict[str, object]) -> None:
    """Draw the four headline numbers for one run."""
    travelers, carpools, matched, saved = st.columns(4)
    travelers.metric("Travelers", f"{summary['travelers']:,}")
    carpools.metric("Carpools", f"{summary['carpools']:,}")
    matched.metric("Matched", f"{summary['match_rate']:.0%}")
    saved.metric("Minutes saved", f"{summary['minutes_saved']:,.0f}")
    seconds = summary.get("seconds")
    if seconds is not None:
        st.caption(f"Simulation time: {float(seconds):.1f}s. Scoring the routes is extra.")


def show_figure(figure: Figure) -> None:
    """Draw a Matplotlib figure and release it."""
    st.pyplot(figure)
    plt.close(figure)


def filter_table(config: SimulationConfig) -> pd.DataFrame:
    """One row per filter, in the order the simulator will apply them."""
    rows: list[dict[str, object]] = []
    for mode in config.modes:
        for spec in mode.filters:
            settings = ", ".join(
                f"{name} = {value:g} {_UNITS.get(name, '')}".rstrip()
                for name, value in spec.params.items()
            )
            rows.append(
                {
                    "Mode": mode.name,
                    "Lot reach (km)": "" if mode.r_p is None else mode.r_p,
                    "Filter": FILTER_LABELS.get(spec.name, spec.name),
                    "Setting": settings,
                }
            )
    return pd.DataFrame(rows)
