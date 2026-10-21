API reference
=============

These are the names each subpackage exports, plus the configuration objects in
``carpoolsim.config``.

Configuration
-------------

.. automodule:: carpoolsim.config
   :no-members:

.. autosummary::
   :toctree: generated

   SimulationConfig
   ModeConfig
   FilterSpec
   load_config

Simulation
----------

.. automodule:: carpoolsim.simulator
   :no-members:

.. autosummary::
   :toctree: generated

   Simulator
   Match
   Solo

Trips and filters
-----------------

.. automodule:: carpoolsim.trip_cluster
   :no-members:

.. autosummary::
   :toctree: generated

   TripCluster
   minutes_of_day
   Shareability
   FilterDepartureTime
   FilterOriginEuclideanDistance
   FilterDestinationEuclideanDistance
   FilterDropoffEuclideanDistance
   FilterRerouteEuclideanDistance
   FilterRerouteTravelTime
   FilterSharedTravelTime
   FilterPassengerWaitTime
   PnrPlan
   FilterPnrAccess
   FilterPnrPassengerAccess
   FilterPnrRerouteEuclidean
   FilterPnrRerouteTravelTime
   FilterPnrSharedTravelTime
   FilterPnrWait

Assigners
---------

.. automodule:: carpoolsim.optimizer
   :no-members:

.. autosummary::
   :toctree: generated

   Assigner
   MatchingAssigner
   GreedyAssigner
   BipartiteAssigner

Network
-------

.. automodule:: carpoolsim.map_loader
   :no-members:

.. autosummary::
   :toctree: generated

   build_map
   build_drive_network
   build_parking_lots
   save_network
   load_network
   save_lots
   load_lots
   lot_coordinates
   parking_lots
   load_osm_graph
   keep_drivable_edges
   is_drivable
   add_edge_travel
   graph_to_network_gdfs

Street and lot filters used while reading OSM XML.

.. py:data:: EXCLUDED_HIGHWAYS

   Highway values excluded while keeping drivable streets. These match OSMnx
   ``network_type="drive"``.

.. py:data:: EXCLUDED_SERVICE

   ``service`` values excluded by that same drive filter.

.. py:data:: EXCLUDED_ACCESS

   ``access`` values that mean a traveler cannot use the lot as a public
   meeting place. A missing tag is kept, because OSM's default for
   ``amenity=parking`` is public.

.. py:data:: EXCLUDED_PARKING

   Parking tags that mean curb parking rather than a lot with room to meet.

.. py:data:: MIN_AREA_M2

   Smallest parking polygon that is kept, in square meters. The value is
   ``2000.0``.

Routing
-------

.. automodule:: carpoolsim.network_prep
   :no-members:

.. autosummary::
   :toctree: generated

   RoadRouter
   Route
   NetworkPartition
   partition_network
