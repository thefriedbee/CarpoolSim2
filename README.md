# CarpoolSim2

*CarpoolSim2* matches drive-alone trips into carpools. A clock walks through the day. At each step the travelers still waiting, and departing soon enough to share a ride, are matched. Filters turn off pairs that are too far apart, too late, or too costly. An assigner then keeps a conflict-free set of driver and rider pairs.

The packaged example is the Georgia Tech 2022 off-campus commute on an Atlanta street network.

One door-to-door carpool and one park-and-ride carpool from that morning. On the left the driver picks the rider up at home, then they share the ride to campus. On the right each person drives to a public lot and they share the ride from there. Orange is the driver, blue is the rider, and the dashed line is the shared ride. Both end on campus, so the destination marks sit on the same point. Each bar is 2 km. The street map is Esri, HERE, Garmin, and OpenStreetMap contributors.

![A door-to-door carpool beside a park-and-ride carpool](static/modes.png)

## How a day is matched

The clock starts at the earliest departure minus a look-ahead, `horizon`, and advances by a step, `dt`. Both are minutes. At clock `t` the pool holds every still-unmatched traveler whose departure lies in `[t, t + horizon]`.

Each pipeline, called a mode, then runs in the order given in the configuration. Filters mark which pairs may share a ride. An assigner keeps a set in which each traveler has one seat. Both members of a committed pair leave the pool before the next mode runs. A traveler who is still unmatched after every mode, and whose departure the clock has passed, leaves alone.

A direct mode picks the rider up at the rider's origin, then drives both travelers to their destinations. A park-and-ride mode sends both travelers to a public parking lot and shares the ride from there. A run can use one mode or both. `config/gt_2022.yaml` is door-to-door. `config/gt_2022_pnr.yaml` is park-and-ride.

Times in a configuration file are minutes. Distances such as `r1` and `r_p` are kilometers. Filter names, assigners, and the rest of the settings are described in [docs/usage.md](docs/usage.md).

## Install

Python 3.11 or newer. From a checkout of this repository:

```bash
uv sync
```

## Run one configuration

Trip rows are EPSG:3857 meters. `departure_time` is `HH:MM`. Columns are `origin_x`, `origin_y`, `dest_x`, `dest_y`, and `departure_time`. `trip_id` is optional. The network tables are the GeoPackages written by `save_network`.

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

A park-and-ride file also needs lot coordinates. Build the Atlanta extract with the steps in [data_inputs/osm/README.md](data_inputs/osm/README.md), then follow the park-and-ride example in [docs/usage.md](docs/usage.md).

## Demo

The demo reviews each traveler and the summary plots from a saved Georgia Tech run. It does not match a new day. Those saved results are not part of the source tree. `data_outputs/gt_2022` needs `travelers.csv`, `matches.csv`, and `legs.gpkg`.

```bash
uv sync --extra app
uv run streamlit run app/Home.py
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
