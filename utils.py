import folium
import gpxpy
import numpy as np
import osmnx as ox
from geoalchemy2.shape import to_shape
from shapely.geometry import (
    LineString,
    MultiLineString,
    Point,
)


def ensure_string(val):
    if isinstance(val, list):
        return ", ".join([str(v) for v in val])
    return val


def parse_gpx(gpx_filepath):
    with open(gpx_filepath, "r") as gpx_contents:
        gpx = gpxpy.parse(gpx_contents)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(Point(point.longitude, point.latitude))

    return points


def display_network(G):
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    center_lat = gdf_nodes.geometry.y.mean()
    center_lon = gdf_nodes.geometry.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=12, tiles="Cartodb Positron"
    )

    edge_color = "#2E86AB"
    node_color = "#F24236"

    for idx, row in gdf_edges.iterrows():
        if row.geometry.geom_type == "LineString":
            coords = [[coord[1], coord[0]] for coord in row.geometry.coords]

            folium.PolyLine(coords, color=edge_color, weight=1, opacity=0.7).add_to(m)

    for idx, row in gdf_nodes.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=node_color,
            fill=True,
            fillColor=node_color,
            fillOpacity=0.8,
            weight=0,
        ).add_to(m)

    return m


def display_points(run_points):
    center_lat = np.mean([point.y for point in run_points])
    center_lon = np.mean([point.x for point in run_points])

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=17, tiles="Cartodb Positron"
    )

    for i, point in enumerate(run_points):
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=3,
            color="blue",
            fill=True,
            popup=f"Point {i}",
        ).add_to(m)

    return m


def display_covered_streets(covered_streets):
    """Display matched edges with coverage information"""

    first_edge = covered_streets[0] if covered_streets else None
    if first_edge and hasattr(first_edge, "geometry"):
        geom = to_shape(first_edge.geometry)
        bounds = geom.bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    else:
        center_lat, center_lon = 48.8566, 2.3522  # Default to Paris

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=17, tiles="Cartodb Positron"
    )

    # Add matched edges with color coding based on coverage
    for street in covered_streets:
        coverage = getattr(street, "progress_percentage", 0)
        geom = to_shape(street.geometry)

        # Handle different geometry types
        if geom.geom_type == "LineString":
            coords = [[lat, lon] for lon, lat in geom.coords]
            folium.PolyLine(
                locations=coords,
                color="blue",
                weight=4,
                opacity=0.8,
                popup=f"""
                <b>{getattr(street, "name", "Unknown")}</b><br>
                Coverage: {coverage:.1f}%<br>
                Length: {geom.length:.6f}°<br>
                Type: {geom.geom_type}
                """,
            ).add_to(m)

        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = [[lat, lon] for lon, lat in line.coords]
                folium.PolyLine(
                    locations=coords,
                    color="blue",
                    weight=4,
                    opacity=0.8,
                    popup=f"""
                    <b>{getattr(street, "name", "Unknown")}</b><br>
                    Coverage: {coverage:.1f}%<br>
                    Length: {geom.length:.6f}°<br>
                    Type: {geom.geom_type}
                    """,
                ).add_to(m)

        elif geom.geom_type == "Point":
            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=6,
                color="blue",
                fill=True,
                popup=f"""
                <b>{getattr(street, "name", "Unknown")}</b><br>
                Coverage: {coverage:.1f}%<br>
                Type: {geom.geom_type}
                """,
            ).add_to(m)

    return m


def combine_linestrings(existing_geom, new_geom):
    """Combine existing and new geometries to show cumulative coverage"""
    # Convert both to lists of LineStrings
    existing_lines = []
    new_lines = []

    # Handle existing geometry
    if hasattr(existing_geom, "geoms"):  # MultiLineString
        existing_lines.extend(list(existing_geom.geoms))
    else:  # LineString
        existing_lines.append(existing_geom)

    # Handle new geometry
    if hasattr(new_geom, "geoms"):  # MultiLineString
        new_lines.extend(list(new_geom.geoms))
    else:  # LineString
        new_lines.append(new_geom)

    # Combine all lines
    all_lines = existing_lines + new_lines

    if len(all_lines) == 1:
        return all_lines[0]
    else:
        return MultiLineString(all_lines)


def convert_points_to_linestring(snapped_points_for_edge):
    """Create covered geometry by connecting the actual snapped points for this edge"""
    try:
        if len(snapped_points_for_edge) < 2:
            if len(snapped_points_for_edge) == 1:
                point = snapped_points_for_edge[0]
                return LineString([point, point])
            return None

        # Group points into continuous segments based on distance
        segments = []
        current_segment = [snapped_points_for_edge[0]]

        for i in range(1, len(snapped_points_for_edge)):
            prev_point = snapped_points_for_edge[i - 1]
            curr_point = snapped_points_for_edge[i]

            # If points are far apart, start a new segment
            distance = prev_point.distance(curr_point)
            if (
                distance > 0.0005
            ):  # Threshold for considering points disconnected (adjust as needed)
                # Finish current segment if it has enough points
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = [curr_point]
            else:
                current_segment.append(curr_point)

        # Add the last segment
        if len(current_segment) >= 2:
            segments.append(current_segment)

        if not segments:
            # Fallback: create one segment from all points
            coords = [point.coords[0] for point in snapped_points_for_edge]
            return LineString(coords)

        # Create geometry from segments
        if len(segments) == 1:
            coords = [point.coords[0] for point in segments[0]]
            return LineString(coords)
        else:
            # Multiple segments - create MultiLineString
            lines = []
            for segment in segments:
                coords = [point.coords[0] for point in segment]
                lines.append(LineString(coords))
            return MultiLineString(lines)

    except Exception:
        # Fallback to straight line between first and last points
        if len(snapped_points_for_edge) >= 2:
            return LineString([snapped_points_for_edge[0], snapped_points_for_edge[-1]])
        return None


def clip_linestring_to_street(covered_linestring, street_linestring):
    if covered_linestring.length > street_linestring.length:
        clipped_linestring = street_linestring.intersection(
            covered_linestring.buffer(0.0001)
        )
        if clipped_linestring.length > 0:
            covered_linestring = clipped_linestring

    return covered_linestring
