# OpenStreetMap extract for Atlanta

The map loader reads OSM XML (`.osm`), not a `.pbf` file. Geofabrik publishes Georgia as a `.pbf`. Download that file, cut a rectangle around Atlanta with osmium, keep streets and parking lots, then convert that smaller extract to XML.

Run the commands from the project root.

## Install osmium

```bash
brew install osmium-tool
```

## Download Georgia

Geofabrik’s Georgia extract is the source file. It is larger than the Atlanta cut and is reused when the rectangle changes.

```bash
curl -L -o data_inputs/osm/georgia-latest.osm.pbf \
  https://download.geofabrik.de/north-america/us/georgia-latest.osm.pbf
```

## Cut a rectangle around Atlanta

The box is `left,bottom,right,top` in degrees: west, south, east, north. This square keeps the east–west span of the earlier Atlanta extract and extends it north and south so the region reaches past Cumming and past Griffin. `complete_ways` keeps every node of a street that crosses the edge.

```bash
osmium extract \
  --bbox -84.7923611,33.4054444,-83.8108889,34.1051944 \
  --strategy complete_ways \
  --set-bounds \
  --overwrite \
  --output data_inputs/osm/atlanta.osm.pbf \
  data_inputs/osm/georgia-latest.osm.pbf
```

## Keep streets and parking lots

`graph_from_xml` reads every node, way, and relation in the XML file before the drive filter runs. Converting the full rectangle produces a multi-gigabyte `atlanta.osm` and makes `build_drive_network` very slow. Filter the PBF first. `w/highway` keeps every street way. `w/amenity=parking` and `r/amenity=parking` keep the lot polygons `build_map` reads. Referenced nodes are kept with those objects.

```bash
osmium tags-filter \
  --overwrite \
  --output data_inputs/osm/atlanta-roads.osm.pbf \
  data_inputs/osm/atlanta.osm.pbf \
  w/highway w/amenity=parking r/amenity=parking
```

## Convert the filtered extract to OSM XML

`load_osm_graph` and the parking-lot reader both call OSMnx on an XML file. Write `atlanta.osm` from the filtered PBF, not from `atlanta.osm.pbf`.

```bash
osmium cat \
  --overwrite \
  --output data_inputs/osm/atlanta.osm \
  data_inputs/osm/atlanta-roads.osm.pbf
```

## Check the bounds

```bash
osmium fileinfo -e data_inputs/osm/atlanta.osm.pbf
```

The bounding box in that report should match the `--bbox` values above.
