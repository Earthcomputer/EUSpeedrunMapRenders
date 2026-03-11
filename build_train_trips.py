from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests

MAX_MERCATOR_LAT = 85.0511287798066
DEFAULT_BROUTER_URL = "https://brouter.de/brouter"


def slugify(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "trip"


def class_part(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    parts = re.findall(r"[A-Za-z0-9]+", value)
    out = "".join(part.capitalize() for part in parts)
    return out or "Trip"


def clamp_lat(lat: float) -> float:
    return max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)


def latlon_to_global_pixel(lat: float, lon: float, zoom: float, tile_size: int = 256) -> tuple[float, float]:
    lat = clamp_lat(lat)
    sin_lat = math.sin(math.radians(lat))
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n * tile_size
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * tile_size
    return x, y


def compute_center_zoom(
    coordinates: list[list[float]],
    width_px: int = 1920,
    height_px: int = 1080,
    padding: float = 1.2,
) -> tuple[float, float, float]:
    if not coordinates:
        return 50.0, 10.0, 4.0
    lons = [float(coord[0]) for coord in coordinates]
    lats = [float(coord[1]) for coord in coordinates]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    if math.isclose(min_lon, max_lon) and math.isclose(min_lat, max_lat):
        return center_lat, center_lon, 14.0
    for step in range(64, 7, -1):
        zoom = step / 4.0
        left, top = latlon_to_global_pixel(max_lat, min_lon, zoom)
        right, bottom = latlon_to_global_pixel(min_lat, max_lon, zoom)
        span_x = abs(right - left) * padding
        span_y = abs(bottom - top) * padding
        if span_x <= width_px and span_y <= height_px:
            return center_lat, center_lon, zoom
    return center_lat, center_lon, 2.0


def parse_info_json(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Leg infoJson is not an object")
    return parsed


def collect_trip_points(travel: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for leg in travel.get("legs", []):
        info = parse_info_json(leg["infoJson"])
        stations = info.get("trainStopStations") or []
        for station in stations:
            coords = station.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is None or lon is None:
                continue
            points.append(
                {
                    "name": station.get("name", ""),
                    "station_id": station.get("stationId"),
                    "lat": float(lat),
                    "lon": float(lon),
                }
            )
    if not points:
        raise ValueError(f'No trainStopStations found for travel "{travel.get("from")} -> {travel.get("to")}"')
    deduped: list[dict[str, Any]] = []
    for point in points:
        if deduped and math.isclose(deduped[-1]["lat"], point["lat"]) and math.isclose(deduped[-1]["lon"], point["lon"]):
            continue
        deduped.append(point)
    if len(deduped) < 2:
        raise ValueError(f'Not enough distinct points for travel "{travel.get("from")} -> {travel.get("to")}"')
    return deduped


def request_brouter_route(
    points: list[dict[str, Any]],
    profile: str,
    timeout: int,
    endpoint: str,
    session: requests.Session,
) -> dict[str, Any]:
    lonlats = "|".join(f'{point["lon"]},{point["lat"]}' for point in points)
    response = session.get(
        endpoint,
        params={
            "lonlats": lonlats,
            "profile": profile,
            "alternativeidx": 0,
            "format": "geojson",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        snippet = response.text[:400].replace("\n", " ")
        raise RuntimeError(f"BRouter request failed ({response.status_code}): {snippet}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("BRouter response is not a JSON object")
    return payload


def extract_route_coordinates(route_payload: dict[str, Any]) -> list[list[float]]:
    features = route_payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("BRouter response has no features")
    geometry = features[0].get("geometry") or {}
    gtype = geometry.get("type")
    raw_coords = geometry.get("coordinates")
    if gtype == "LineString":
        coordinates = raw_coords
    elif gtype == "MultiLineString":
        coordinates = []
        for segment in raw_coords or []:
            if not coordinates:
                coordinates.extend(segment)
                continue
            if segment and coordinates[-1][:2] == segment[0][:2]:
                coordinates.extend(segment[1:])
            else:
                coordinates.extend(segment)
    else:
        raise ValueError(f"Unsupported route geometry type: {gtype}")
    clean: list[list[float]] = []
    for coordinate in coordinates or []:
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            continue
        lon = float(coordinate[0])
        lat = float(coordinate[1])
        if clean and math.isclose(clean[-1][0], lon) and math.isclose(clean[-1][1], lat):
            continue
        clean.append([lon, lat])
    if len(clean) < 2:
        raise ValueError("Route has fewer than two coordinates")
    return clean


def concat_coordinates(parts: list[list[list[float]]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for part in parts:
        for coord in part:
            if merged and math.isclose(merged[-1][0], coord[0]) and math.isclose(merged[-1][1], coord[1]):
                continue
            merged.append(coord)
    if len(merged) < 2:
        raise ValueError("Merged route has fewer than two coordinates")
    return merged


def route_trip(
    points: list[dict[str, Any]],
    profile: str,
    timeout: int,
    endpoint: str,
    session: requests.Session,
) -> tuple[list[list[float]], int]:
    try:
        full_payload = request_brouter_route(
            points=points,
            profile=profile,
            timeout=timeout,
            endpoint=endpoint,
            session=session,
        )
        return extract_route_coordinates(full_payload), 0
    except Exception:
        pass
    segments: list[list[list[float]]] = []
    fallback_segments = 0
    for idx in range(len(points) - 1):
        pair = [points[idx], points[idx + 1]]
        try:
            payload = request_brouter_route(
                points=pair,
                profile=profile,
                timeout=timeout,
                endpoint=endpoint,
                session=session,
            )
            coords = extract_route_coordinates(payload)
        except Exception:
            fallback_segments += 1
            coords = [
                [pair[0]["lon"], pair[0]["lat"]],
                [pair[1]["lon"], pair[1]["lat"]],
            ]
        segments.append(coords)
    return concat_coordinates(segments), fallback_segments


def ensure_unique_scene_name(base_name: str, seen: set[str], trip_index: int) -> str:
    candidate = base_name
    if candidate not in seen:
        seen.add(candidate)
        return candidate
    candidate = f"{base_name}{trip_index:02d}"
    seen.add(candidate)
    return candidate


def build_geojson_and_manifest(
    input_path: Path,
    geojson_dir: Path,
    manifest_path: Path,
    profile: str,
    timeout: int,
    endpoint: str,
    limit: int | None,
) -> dict[str, Any]:
    with input_path.open(encoding="utf-8") as handle:
        root = json.load(handle)
    travels = root.get("travels") or []
    if not isinstance(travels, list):
        raise ValueError("Input JSON field 'travels' is not a list")
    if limit is not None:
        travels = travels[:limit]
    geojson_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    seen_scene_names: set[str] = set()
    session = requests.Session()
    for index, travel in enumerate(travels, start=1):
        from_label = str(travel.get("from", "")).strip() or f"Trip {index} Start"
        to_label = str(travel.get("to", "")).strip() or f"Trip {index} End"
        file_stem = f"{index:02d}_{slugify(from_label)}_to_{slugify(to_label)}"
        scene_base = f"Trip{index:02d}{class_part(from_label)}To{class_part(to_label)}Scene"
        scene_name = ensure_unique_scene_name(scene_base, seen_scene_names, index)
        points = collect_trip_points(travel)
        coordinates, fallback_segments = route_trip(
            points=points,
            profile=profile,
            timeout=timeout,
            endpoint=endpoint,
            session=session,
        )
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "index": index,
                        "tripId": travel.get("tripId"),
                        "from": from_label,
                        "to": to_label,
                        "profile": profile,
                        "stationCount": len(points),
                        "fallbackSegments": fallback_segments,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates,
                    },
                }
            ],
        }
        geojson_path = geojson_dir / f"{file_stem}.geojson"
        with geojson_path.open("w", encoding="utf-8") as handle:
            json.dump(geojson, handle, ensure_ascii=False)
        start = points[0]
        end = points[-1]
        center_lat, center_lon, zoom = compute_center_zoom(coordinates)
        entries.append(
            {
                "index": index,
                "scene_name": scene_name,
                "output_name": file_stem,
                "from_label": from_label,
                "to_label": to_label,
                "trip_id": travel.get("tripId"),
                "geojson_path": geojson_path.as_posix(),
                "fallback_segments": fallback_segments,
                "start": {
                    "lat": start["lat"],
                    "lon": start["lon"],
                },
                "end": {
                    "lat": end["lat"],
                    "lon": end["lon"],
                },
                "map": {
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "zoom": zoom,
                },
            }
        )
        fallback_note = f" | fallback_segments={fallback_segments}" if fallback_segments else ""
        print(f'[{index:02d}/{len(travels):02d}] {from_label} -> {to_label} | {geojson_path}{fallback_note}')
    manifest = {
        "source": input_path.as_posix(),
        "profile": profile,
        "trip_count": len(entries),
        "trips": entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/euspeedrun_trains.json"))
    parser.add_argument("--geojson-dir", type=Path, default=Path("geojson/trains"))
    parser.add_argument("--manifest", type=Path, default=Path("data/train_trip_manifest.json"))
    parser.add_argument("--profile", default="rail")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--endpoint", default=DEFAULT_BROUTER_URL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_geojson_and_manifest(
        input_path=args.input,
        geojson_dir=args.geojson_dir,
        manifest_path=args.manifest,
        profile=args.profile,
        timeout=args.timeout,
        endpoint=args.endpoint,
        limit=args.limit,
    )
    print(
        f'Generated {manifest["trip_count"]} trips with profile "{manifest["profile"]}" into '
        f'{args.geojson_dir.as_posix()} and {args.manifest.as_posix()}'
    )


if __name__ == "__main__":
    main()
