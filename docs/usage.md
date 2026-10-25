# Usage

Install the package from a checkout of this repository. The documentation extra adds Sphinx, which is only needed to build these pages.

```bash
uv sync
uv sync --extra docs
```

## Run one configuration

A YAML or JSON file sets the clock, the assigner, and the filters. `load_config` reads it. `SimulationConfig.build` turns it into a `Simulator`.

Trip rows are EPSG:3857 meters. `departure_time` is `HH:MM`. `trip_id` is optional; without it, rows are numbered from zero. The network tables are the GeoPackages written by `save_network`.

```python
from carpoolsim.config import load_config
from carpoolsim.map_loader import load_network
from carpoolsim.network_prep import RoadRouter
from carpoolsim.trip_cluster import TripCluster

config = load_config("config/gt_2022.yaml")
trips = TripCluster.from_csv("trips.csv")
nodes, edges = load_network("data_inputs/network")
router = RoadRouter(nodes, edges)

simulator = config.build(trips, router=router)
simulator.run()
```

`simulator.matches` holds each committed pair. `driver` and `rider` are traveler ids, `t` is minutes after midnight, and `mode` is the pipeline that produced the pair. `simulator.solos` holds travelers who left without a partner.

Travel-time filters need `router`. A door-to-door file such as `config/gt_2022.yaml` uses those filters, so pass the router. A park-and-ride file also needs lot coordinates in EPSG:3857 meters:

```python
from carpoolsim.map_loader import load_lots
from carpoolsim.map_loader import lot_coordinates

config = load_config("config/gt_2022_pnr.yaml")
lots = lot_coordinates(load_lots("data_inputs/network"))
simulator = config.build(trips, router=router, lots=lots)
simulator.run()
```

## What the configuration controls

Times are minutes. Distances such as `r1` and `r_p` are kilometers. A ratio such as `gamma_1` compares the carpool tour with the driver's own trip.

`horizon` is how far ahead the pool looks. `dt` is the clock step. `et`, when set, limits who may drive at the current step. `commit` is `window` or `step`. `assigner` is `matching`, `greedy`, or `bipartite`. `max_size` is the most travelers one matching group may hold. The current pool is cut by trip similarity until every group is that small. The default is 100. Set it to `null` and the pool stays one group.

`modes` is an ordered list of pipelines. Each mode has a `name` and a `filters` list. A park-and-ride mode also has `r_p`, the farthest an origin may be from a lot. The filter class names and their settings are the public filter classes in the API reference.

## Build these pages

From the repository root, after `uv sync --extra docs`:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Read the Docs rebuilds the same site from `.readthedocs.yaml` when the repository is connected there.
