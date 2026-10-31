"""Small display helpers shared by the demo pages."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.figure import Figure


def show_metrics(summary: dict[str, object]) -> None:
    """Draw the four headline numbers for one run."""
    travelers, carpools, matched, saved = st.columns(4)
    travelers.metric("Travelers", f"{summary['travelers']:,}")
    carpools.metric("Carpools", f"{summary['carpools']:,}")
    matched.metric("Matched", f"{summary['match_rate']:.0%}")
    saved.metric("Minutes saved", f"{summary['minutes_saved']:,.0f}")


def show_figure(figure: Figure) -> None:
    """Draw a Matplotlib figure and release it."""
    st.pyplot(figure)
    plt.close(figure)
