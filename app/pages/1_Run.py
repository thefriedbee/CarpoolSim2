"""Choose a configuration and match the Georgia Tech day."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from carpoolsim.config import load_config
from app.cache import inputs
from app.jobs import run_simulation
from app.jobs import with_clock
from app.paths import demo_output
from app.paths import preset_note
from app.paths import presets
from app.paths import root
from app.ui import filter_table
from app.ui import show_metrics

st.set_page_config(page_title="Run", layout="wide")
st.title("Run")
st.write(
    "Pick a saved configuration, then change the horizon, the step, or the assigner. "
    "Filters stay as they are in the file. Matching starts when you press the button. "
    "The full day takes about a minute, and the street network loads once."
)

choices = presets()
if not choices:
    st.error("No config files were found under config/.")
    st.stop()

labels = [str(path.relative_to(root() / "config")) for path in choices]
preset_label = st.selectbox("Configuration", labels)
preset = choices[labels.index(preset_label)]
loaded = load_config(preset)
note = preset_note(preset)
if note:
    st.caption(note)

if st.session_state.get("chosen_preset") != preset_label:
    st.session_state["chosen_preset"] = preset_label
    st.session_state["horizon"] = float(loaded.horizon)
    st.session_state["dt"] = float(loaded.dt)
    st.session_state["assigner"] = loaded.assigner

horizon, step, assigner = st.columns(3)
horizon.number_input("Horizon (minutes)", min_value=1.0, step=1.0, key="horizon")
step.number_input("Step (minutes)", min_value=1.0, step=1.0, key="dt")
assigner.selectbox("Assigner", ["matching", "greedy", "bipartite"], key="assigner")

try:
    edited = with_clock(
        loaded,
        st.session_state["horizon"],
        st.session_state["dt"],
        st.session_state["assigner"],
    )
except ValueError as error:
    edited = None
    st.error(str(error))

if edited is not None:
    st.dataframe(filter_table(edited), hide_index=True, width="stretch")
    st.caption(
        "A run is written to data_outputs/demo/"
        f"{preset.stem} and replaces the previous demo of this file. "
        "Notebook outputs are left in place."
    )

if edited is not None and st.button("Run simulation", type="primary"):
    try:
        cluster, router, edges, lots = inputs()
        directory = demo_output(preset.stem)
        summary = run_simulation(edited, cluster, router, edges, lots, directory)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
    except Exception as error:
        st.exception(error)
    else:
        st.session_state["summary"] = summary
        st.session_state["result_dir"] = summary["directory"]

summary = st.session_state.get("summary")
if summary:
    st.subheader("Last run")
    show_metrics(summary)
    st.caption(str(Path(str(summary["directory"])).relative_to(root())))
    st.page_link("pages/2_Results.py", label="Open the charts")
