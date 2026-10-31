# CarpoolSim2

CarpoolSim2 matches drive-alone trips into carpools. A clock walks through the day. At each step the travelers still waiting, and departing soon enough to share a ride, are matched. Filters turn off pairs that are too far apart, too late, or too costly. An assigner then keeps a conflict-free set of driver and rider pairs.

Two pipelines are built in. A direct mode picks the rider up at the rider's origin. A park-and-ride mode sends both travelers to a parking lot and shares the ride from there. A run can use one pipeline or both, in a chosen order.

```{toctree}
:hidden:
:maxdepth: 2

usage
api
```
