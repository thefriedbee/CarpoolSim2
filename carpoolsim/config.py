"""Load the clock, the matcher, and the filters for one simulation run."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from carpoolsim.optimizer import BipartiteAssigner
from carpoolsim.optimizer import GreedyAssigner
from carpoolsim.optimizer import MatchingAssigner
from carpoolsim.simulator import Simulator
from carpoolsim.simulator.simulator import DEFAULT_MAX_SIZE
from carpoolsim.trip_cluster import FilterDepartureTime
from carpoolsim.trip_cluster import FilterDestinationEuclideanDistance
from carpoolsim.trip_cluster import FilterDropoffEuclideanDistance
from carpoolsim.trip_cluster import FilterOriginEuclideanDistance
from carpoolsim.trip_cluster import FilterPassengerWaitTime
from carpoolsim.trip_cluster import FilterPnrAccess
from carpoolsim.trip_cluster import FilterPnrPassengerAccess
from carpoolsim.trip_cluster import FilterPnrRerouteEuclidean
from carpoolsim.trip_cluster import FilterPnrRerouteTravelTime
from carpoolsim.trip_cluster import FilterPnrSharedTravelTime
from carpoolsim.trip_cluster import FilterPnrWait
from carpoolsim.trip_cluster import FilterRerouteEuclideanDistance
from carpoolsim.trip_cluster import FilterRerouteTravelTime
from carpoolsim.trip_cluster import FilterSharedTravelTime
from carpoolsim.trip_cluster import PnrPlan
from carpoolsim.trip_cluster.cluster import TripCluster
from carpoolsim.trip_cluster.share import Filter

_ASSIGNERS = {
    "matching": MatchingAssigner,
    "greedy": GreedyAssigner,
    "bipartite": BipartiteAssigner,
}
_ASSIGNER_NAMES = {factory: name for name, factory in _ASSIGNERS.items()}
_CONFIG_KEYS = {"horizon", "dt", "et", "commit", "assigner", "max_size", "workers", "modes"}
_MODE_KEYS = {"name", "r_p", "filters"}


class _FilterKind:
    """How to build one filter from its numeric settings."""

    def __init__(
        self,
        factory: type,
        params: tuple[str, ...],
        needs_router: bool,
        needs_plan: bool,
    ) -> None:
        self.factory = factory
        self.params = params
        self.needs_router = needs_router
        self.needs_plan = needs_plan


def _kind(
    factory: type,
    params: tuple[str, ...] = (),
    needs_router: bool = False,
    needs_plan: bool = False,
) -> _FilterKind:
    return _FilterKind(factory, params, needs_router, needs_plan)


_FILTERS = {
    "FilterDepartureTime": _kind(FilterDepartureTime, ("delta1",)),
    "FilterOriginEuclideanDistance": _kind(FilterOriginEuclideanDistance, ("r1",)),
    "FilterDestinationEuclideanDistance": _kind(FilterDestinationEuclideanDistance, ("r2",)),
    "FilterRerouteEuclideanDistance": _kind(FilterRerouteEuclideanDistance, ("mu_1",)),
    "FilterDropoffEuclideanDistance": _kind(FilterDropoffEuclideanDistance, ("mu_2",)),
    "FilterRerouteTravelTime": _kind(FilterRerouteTravelTime, ("gamma_1",), needs_router=True),
    "FilterSharedTravelTime": _kind(FilterSharedTravelTime, ("gamma_2",), needs_router=True),
    "FilterPassengerWaitTime": _kind(FilterPassengerWaitTime, ("delta2",), needs_router=True),
    "FilterPnrAccess": _kind(FilterPnrAccess, needs_plan=True),
    "FilterPnrRerouteEuclidean": _kind(FilterPnrRerouteEuclidean, ("mu_1",), needs_plan=True),
    "FilterPnrPassengerAccess": _kind(FilterPnrPassengerAccess, ("mu_3",), needs_plan=True),
    "FilterPnrRerouteTravelTime": _kind(
        FilterPnrRerouteTravelTime, ("gamma_1",), needs_router=True, needs_plan=True
    ),
    "FilterPnrSharedTravelTime": _kind(
        FilterPnrSharedTravelTime, ("gamma_2",), needs_router=True, needs_plan=True
    ),
    "FilterPnrWait": _kind(FilterPnrWait, ("delta2",), needs_router=True, needs_plan=True),
}


class FilterSpec:
    """One filter name and its numeric settings.

    ``params`` holds the constructor numbers, such as ``delta1`` or ``gamma_2``.
    The router and the park-and-ride plan are supplied later, when the
    simulator is built.
    """

    def __init__(self, name: str, params: dict[str, float]) -> None:
        self.name = name
        self.params = dict(params)


class ModeConfig:
    """One named pipeline and, for park-and-ride, the lot reach ``r_p``.

    ``r_p`` is in kilometers. It is set when a filter in ``filters`` reads a
    ``PnrPlan``, and omitted for a door-to-door mode. Every park-and-ride
    filter in the mode shares that one plan.
    """

    def __init__(
        self,
        name: str,
        filters: Sequence[FilterSpec],
        r_p: float | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise ValueError("mode name must be a non-empty string")
        self.name = name
        self.filters = tuple(filters)
        self.r_p = None if r_p is None else float(r_p)
        needs_plan = any(_FILTERS[spec.name].needs_plan for spec in self.filters)
        if needs_plan and self.r_p is None:
            raise ValueError(f"park-and-ride mode {name!r} requires r_p")
        if self.r_p is not None and not needs_plan:
            raise ValueError(f"mode {name!r} has r_p but no park-and-ride filter")


class SimulationConfig:
    """Clock, matcher, and ordered modes for one run.

    ``horizon`` and ``dt`` are minutes. ``et`` limits who may drive; omitted,
    anyone in the pool may drive. ``commit`` is ``"window"`` or ``"step"``.
    ``assigner`` is ``"matching"``, ``"greedy"``, or ``"bipartite"``.
    ``max_size`` is the most travelers one matching group may hold. The pool
    is cut by trip similarity until every group is that small. The default is
    100. ``null`` leaves each pool as one group. ``workers`` is how many
    processes match those groups. The default is 1, which matches them in
    the calling process. A file that omits it keeps that default.

    The default is a 15-minute horizon, a 5-minute step, no ``et``, window
    commit, the maximum-cardinality matcher, groups of 100, one process, and
    one empty mode named ``"direct"``. A file replaces whichever of these it
    sets. A single mode with an empty name is the unnamed filter list
    ``Simulator`` uses when it is constructed without modes.

    ``build`` turns this into a ``Simulator``. Travel-time filters receive
    ``router``. A park-and-ride mode receives one ``PnrPlan`` built from
    ``lots``, which is ``(px, py)`` in EPSG:3857 meters, and from ``r_p``.
    """

    def __init__(
        self,
        horizon: float = 15,
        dt: float = 5,
        et: float | None = None,
        commit: str = "window",
        assigner: str = "matching",
        modes: Sequence[ModeConfig] | None = None,
        max_size: int | None = DEFAULT_MAX_SIZE,
        workers: int = 1,
    ) -> None:
        self.horizon = _real(horizon, "horizon")
        self.dt = _real(dt, "dt")
        self.et = None if et is None else _real(et, "et")
        if not isinstance(commit, str) or commit not in {"window", "step"}:
            raise ValueError("commit must be 'window' or 'step'")
        self.commit = commit
        if not isinstance(assigner, str) or assigner not in _ASSIGNERS:
            raise ValueError("assigner must be 'matching', 'greedy', or 'bipartite'")
        self.assigner = assigner
        self.max_size = None if max_size is None else _count(max_size, "max_size")
        self.workers = _count(workers, "workers")
        self.modes = (ModeConfig("direct", ()),) if modes is None else tuple(modes)
        _check_modes(self.modes)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.horizon < self.dt:
            raise ValueError("horizon must be at least dt")
        if self.et is not None and self.et < self.dt:
            raise ValueError("et must be at least dt")
        if self.et is not None and self.et > self.horizon:
            raise ValueError("et must be at most horizon")
        if self.max_size is not None and self.max_size < 2:
            raise ValueError("max_size must be at least 2")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SimulationConfig:
        """Build a config from the mapping a JSON or YAML file parses to."""
        mapping = _mapping(data, "config")
        _reject_unknown(mapping, _CONFIG_KEYS, "config setting")
        horizon = 15.0 if "horizon" not in mapping else _real(mapping["horizon"], "horizon")
        step = 5.0 if "dt" not in mapping else _real(mapping["dt"], "dt")
        et = None
        if "et" in mapping and mapping["et"] is not None:
            et = _real(mapping["et"], "et")
        commit = mapping.get("commit", "window")
        assigner = mapping.get("assigner", "matching")
        if not isinstance(commit, str):
            raise ValueError("commit must be 'window' or 'step'")
        if not isinstance(assigner, str):
            raise ValueError("assigner must be 'matching', 'greedy', or 'bipartite'")
        modes: tuple[ModeConfig, ...] | None = None
        if "modes" in mapping:
            listed = mapping["modes"]
            if not isinstance(listed, list):
                raise ValueError("modes must be a list")
            modes = tuple(_mode_from_dict(item) for item in listed)
        if "max_size" not in mapping:
            max_size: int | None = DEFAULT_MAX_SIZE
        elif mapping["max_size"] is None:
            max_size = None
        else:
            max_size = _count(mapping["max_size"], "max_size")
        workers = 1 if "workers" not in mapping else _count(mapping["workers"], "workers")
        return cls(horizon, step, et, commit, assigner, modes, max_size, workers)

    @classmethod
    def from_simulator(cls, simulator: Simulator) -> SimulationConfig:
        """Read the config back from a simulator that was already built."""
        try:
            assigner = _ASSIGNER_NAMES[type(simulator.assigner)]
        except KeyError as error:
            raise ValueError("assigner must be 'matching', 'greedy', or 'bipartite'") from error
        modes = tuple(_mode_from_filters(name, filters) for name, filters in simulator.modes)
        return cls(
            horizon=simulator.horizon,
            dt=simulator.dt,
            et=simulator.et,
            commit=simulator.commit,
            assigner=assigner,
            modes=modes,
            max_size=simulator.max_size,
            workers=simulator.workers,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the mapping ``load_config`` would read back."""
        modes: list[dict[str, object]] = []
        for mode in self.modes:
            record: dict[str, object] = {
                "name": mode.name,
                "filters": [_filter_dict(spec) for spec in mode.filters],
            }
            if mode.r_p is not None:
                record = {
                    "name": mode.name,
                    "r_p": mode.r_p,
                    "filters": record["filters"],
                }
            modes.append(record)
        record: dict[str, object] = {
            "horizon": self.horizon,
            "dt": self.dt,
            "et": self.et,
            "commit": self.commit,
            "assigner": self.assigner,
        }
        if self.max_size is not None:
            record["max_size"] = self.max_size
        if self.workers != 1:
            record["workers"] = self.workers
        record["modes"] = modes
        return record

    def with_horizon(self, horizon: float) -> SimulationConfig:
        """Return a copy that uses ``horizon`` and keeps the other settings."""
        return SimulationConfig(
            horizon=horizon,
            dt=self.dt,
            et=self.et,
            commit=self.commit,
            assigner=self.assigner,
            modes=self.modes,
            max_size=self.max_size,
            workers=self.workers,
        )

    def build(
        self,
        cluster: TripCluster,
        router: object | None = None,
        lots: tuple[object, object] | None = None,
    ) -> Simulator:
        """Build a simulator for ``cluster``.

        ``router`` is required by any travel-time filter. ``lots`` is
        ``(px, py)``, the lot coordinates in EPSG:3857 meters, and is required
        by any park-and-ride mode.
        """
        px: object | None = None
        py: object | None = None
        if lots is not None:
            px, py = _lot_coordinates(lots)
        built: list[tuple[str, tuple[Filter, ...]]] = []
        for mode in self.modes:
            plan = None
            if any(_FILTERS[spec.name].needs_plan for spec in mode.filters):
                if px is None or py is None or mode.r_p is None:
                    raise ValueError(f"park-and-ride mode {mode.name!r} requires lot coordinates")
                plan = PnrPlan(px, py, mode.r_p)
            filters = tuple(_build_filter(spec, router, plan) for spec in mode.filters)
            built.append((mode.name, filters))
        assigner = _ASSIGNERS[self.assigner]()
        if len(built) == 1 and built[0][0] == "":
            return Simulator(
                cluster,
                assigner,
                filters=built[0][1],
                horizon=self.horizon,
                dt=self.dt,
                et=self.et,
                commit=self.commit,
                max_size=self.max_size,
                workers=self.workers,
            )
        return Simulator(
            cluster,
            assigner,
            modes=built,
            horizon=self.horizon,
            dt=self.dt,
            et=self.et,
            commit=self.commit,
            max_size=self.max_size,
            workers=self.workers,
        )


def load_config(path: str | Path) -> SimulationConfig:
    """Read a simulation config from a ``.json``, ``.yaml``, or ``.yml`` file."""
    file = Path(path)
    suffix = file.suffix.lower()
    text = file.read_text()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError("config must be .json, .yaml, or .yml")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return SimulationConfig.from_dict(data)


def _mode_from_dict(data: object) -> ModeConfig:
    """Build one mode from its mapping."""
    mapping = _mapping(data, "mode")
    _reject_unknown(mapping, _MODE_KEYS, "mode setting")
    if "name" not in mapping or not isinstance(mapping["name"], str):
        raise ValueError("mode name must be a non-empty string")
    filters = mapping.get("filters", [])
    if not isinstance(filters, list):
        raise ValueError("filters must be a list")
    r_p = mapping.get("r_p")
    return ModeConfig(
        mapping["name"],
        tuple(_filter_from_dict(item) for item in filters),
        None if r_p is None else _real(r_p, "r_p"),
    )


def _filter_from_dict(data: object) -> FilterSpec:
    """Build one filter spec from its mapping."""
    mapping = _mapping(data, "filter")
    name = mapping.get("filter")
    if not isinstance(name, str) or name == "":
        raise ValueError("filter requires a name")
    try:
        kind = _FILTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown filter {name!r}") from error
    supplied = {key: value for key, value in mapping.items() if key != "filter"}
    _reject_unknown(supplied, set(kind.params), f"{name} setting")
    params: dict[str, float] = {}
    for param in kind.params:
        if param not in supplied:
            raise ValueError(f"{name} requires {param}")
        params[param] = _real(supplied[param], param)
    return FilterSpec(name, params)


def _mode_from_filters(name: str, filters: Sequence[object]) -> ModeConfig:
    """Read one mode from the filters stored on a simulator."""
    specs: list[FilterSpec] = []
    r_p: float | None = None
    for step in filters:
        kind_name = type(step).__name__
        try:
            kind = _FILTERS[kind_name]
        except KeyError as error:
            raise ValueError(f"unknown filter {kind_name!r}") from error
        if kind.needs_plan:
            plan = getattr(step, "plan", None)
            if not isinstance(plan, PnrPlan):
                raise ValueError(f"{kind_name} requires a park-and-ride plan")
            if r_p is None:
                r_p = float(plan.r_p)
            elif r_p != float(plan.r_p):
                raise ValueError("park-and-ride filters in one mode must share r_p")
        params = {param: float(getattr(step, param)) for param in kind.params}
        specs.append(FilterSpec(kind_name, params))
    return ModeConfig(name, specs, r_p)


def _filter_dict(spec: FilterSpec) -> dict[str, object]:
    """Return one filter in the order its settings are defined."""
    kind = _FILTERS[spec.name]
    record: dict[str, object] = {"filter": spec.name}
    for param in kind.params:
        record[param] = spec.params[param]
    return record


def _build_filter(spec: FilterSpec, router: object | None, plan: PnrPlan | None) -> Filter:
    """Construct one filter, injecting the router and plan it needs."""
    kind = _FILTERS[spec.name]
    if kind.needs_router and router is None:
        raise ValueError(f"{spec.name} requires a router")
    if kind.needs_plan and plan is None:
        raise ValueError(f"{spec.name} requires a park-and-ride plan")
    args: list[object] = []
    if kind.needs_plan:
        args.append(plan)
    if kind.needs_router:
        args.append(router)
    args.extend(spec.params[param] for param in kind.params)
    return kind.factory(*args)


def _check_modes(modes: tuple[ModeConfig, ...]) -> None:
    """Require at least one mode, and unique names."""
    if not modes:
        raise ValueError("at least one mode is required")
    seen: set[str] = set()
    for mode in modes:
        if not isinstance(mode, ModeConfig):
            raise ValueError("modes must be a list")
        if mode.name == "" and len(modes) != 1:
            raise ValueError("mode name must be a non-empty string")
        if mode.name in seen:
            raise ValueError("mode names must be unique")
        seen.add(mode.name)


def _lot_coordinates(lots: tuple[object, object]) -> tuple[object, object]:
    """Return ``(px, py)`` from the pair passed to ``build``."""
    if not isinstance(lots, tuple) or len(lots) != 2:
        raise ValueError("lots must be lot coordinates (px, py)")
    return lots


def _mapping(value: object, label: str) -> dict[str, object]:
    """Return ``value`` when it is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _reject_unknown(mapping: dict[str, object], allowed: set[str], label: str) -> None:
    """Raise when ``mapping`` contains a key outside ``allowed``."""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} {unknown[0]!r}")


def _real(value: object, label: str) -> float:
    """Return ``value`` as a float, rejecting booleans and other types."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _count(value: object, label: str) -> int:
    """Return ``value`` as an integer, rejecting booleans and fractional numbers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value
