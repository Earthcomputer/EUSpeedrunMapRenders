from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from manim import FadeIn, config
from tqdm import tqdm

from rendering.core.map_markers import MapMarker
from rendering.core.map_scene import TileMapScene
from rendering.core.route_visuals import TripRoute
from rendering.core.tile_map import TileMap

from .brouter import ensure_geojson_for_spec_with_status, find_cached_geojson
from .geojson import load_route_geo_points
from .geometry import cumulative_distances, station_progresses_on_route
from .manual import GeoJSONSpec, load_geojson_specs
from .naming import scene_name_for_identifier
from .paths import PathPoint, PathSpec, load_path_specs

config.renderer = "cairo"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 60
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"


@dataclass(frozen=True, slots=True)
class RenderPath:
    spec: PathSpec
    scene_name: str


@dataclass(frozen=True, slots=True)
class RenderGeoJSON:
    spec: GeoJSONSpec
    scene_name: str


class BaseRouteScene(TileMapScene):
    def route_map_view(self) -> tuple[float, float, float]:
        raise NotImplementedError

    def route_geojson_path(self) -> Path:
        raise NotImplementedError

    def route_labels(self) -> tuple[list[str], list[str]]:
        raise NotImplementedError

    def route_animation_options(self, geojson_path: Path) -> dict[str, object]:
        route_points = load_route_geo_points(geojson_path)
        total_distance = cumulative_distances(route_points)[-1]
        return {"total_distance": total_distance}

    def create_tile_map(self, **kwargs: Any) -> TileMap:
        center_lat, center_lon, zoom = self.route_map_view()
        return TileMap(center_lat, center_lon, max(0.0, zoom - 0.35), **kwargs)

    def _frame_size(self) -> tuple[float, float]:
        frame_width = getattr(self.camera, "frame_width", float(config["frame_width"]))
        frame_height = getattr(self.camera, "frame_height", float(config["frame_height"]))
        return float(frame_width), float(frame_height)

    @staticmethod
    def _downsample_points(points: Sequence[np.ndarray], max_points: int = 700) -> list[np.ndarray]:
        total = len(points)
        if total <= max_points:
            return [np.array(point, dtype=float) for point in points]
        step = max(1, total // max_points)
        sampled = [np.array(point, dtype=float) for point in points[::step]]
        if not np.array_equal(sampled[-1], np.array(points[-1], dtype=float)):
            sampled.append(np.array(points[-1], dtype=float))
        return sampled

    def _make_markers(self, route: TripRoute) -> tuple[list[MapMarker], list[MapMarker]]:
        start_labels, end_labels = self.route_labels()
        # Put English last
        start_labels = start_labels[1:] + [start_labels[0]]
        end_labels = end_labels[1:] + [end_labels[0]]
        
        start_markers = [MapMarker(start_label, route.start, label_direction=route.start_label_direction()) for start_label in start_labels]
        end_markers = [MapMarker(end_label, route.end, label_direction=route.end_label_direction()) for end_label in end_labels]
        frame_width, frame_height = self._frame_size()
        marker_route_points = self._downsample_points(route.points)
        longest_start_marker = max(start_markers, key=lambda marker: marker.label.width)
        longest_start_marker.choose_label_direction_with_route(frame_width, frame_height, marker_route_points, 0)
        longest_end_marker = max(end_markers, key=lambda marker: marker.label.width)
        longest_end_marker.choose_label_direction_with_route(frame_width, frame_height, marker_route_points, -1)
        for start_marker in start_markers:
            if start_marker is not longest_start_marker:
                start_marker.set_label_direction(longest_start_marker.label_direction)
            start_marker.clamp_label_within_frame(frame_width, frame_height)
        for end_marker in end_markers:
            if end_marker is not longest_end_marker:
                end_marker.set_label_direction(longest_end_marker.label_direction)
            end_marker.clamp_label_within_frame(frame_width, frame_height)
        
        return start_markers, end_markers

    def construct(self) -> None:
        geojson_path = self.route_geojson_path()
        route = self.load_geojson(geojson_path)
        start_markers, end_markers = self._make_markers(route)
        atmosphere = self.create_map_atmosphere(start_markers[0].dot.get_center(), end_markers[0].dot.get_center())
        self.play(FadeIn(atmosphere, run_time=0.8))
        self.play(*start_markers[0].animate_creation())
        start_markers[0].show_final_state()
        if len(start_markers) > 1:
            for i, animation in enumerate(MapMarker.animate_translation(start_markers)):
                if i != 0:
                    self.wait(0.2)
                self.play(animation)
        start_markers[-1].show_final_state()
        self.play(*end_markers[0].animate_creation())
        end_markers[0].show_final_state()
        if len(end_markers) > 1:
            for i, animation in enumerate(MapMarker.animate_translation(end_markers)):
                if i != 0:
                    self.wait(0.2)
                self.play(animation)
        end_markers[-1].show_final_state()
        marker_front = [*start_markers[-1].foreground_mobjects(), *end_markers[-1].foreground_mobjects(), start_markers[0].dot, start_markers[0].halo, end_markers[0].dot, end_markers[0].halo]
        route.create_and_animate(self, keep_on_top=marker_front, **self.route_animation_options(geojson_path))
        self.wait(0.4)


class PathScene(BaseRouteScene):
    path_spec: PathSpec

    def _speed_label_enabled(self) -> bool:
        start = self.path_spec.start
        end = self.path_spec.end
        return start.offset_minutes is not None and end.offset_minutes is not None

    def route_map_view(self) -> tuple[float, float, float]:
        return self.path_spec.map_view()

    def route_labels(self) -> tuple[str, str]:
        return self.path_spec.start.names, self.path_spec.end.names

    def route_animation_options(self, geojson_path: Path) -> dict[str, object]:
        options = super().route_animation_options(geojson_path)
        station_coords = [(float(point.lat), float(point.lon)) for point in self.path_spec.points()]
        station_progresses = station_progresses_on_route(station_coords, load_route_geo_points(geojson_path))
        options["station_progresses"] = station_progresses
        options["total_time"] = self.path_spec.end.offset_minutes
        options["top_speed"] = self.path_spec.top_speed
        return options

    def route_geojson_path(self) -> Path:
        cached = find_cached_geojson(self.path_spec)
        if cached is not None:
            with tqdm(
                total=1,
                desc="Using geojson cache",
                unit="seg",
                bar_format=PROGRESS_BAR_FORMAT,
            ) as progress:
                progress.update(1)
            return cached
        fallback_total = max(1, len(self.path_spec.route_points()) - 1)
        with tqdm(
            total=1,
            desc="Downloading geojson route",
            unit="seg",
            bar_format=PROGRESS_BAR_FORMAT,
        ) as progress:
            def update(event: dict[str, int | str]) -> None:
                total = max(1, int(event.get("total_segments", fallback_total)))
                done = max(0, min(total, int(event.get("done_segments", 0))))
                progress.total = total
                progress.n = done
                progress.refresh()

            path, _ = ensure_geojson_for_spec_with_status(self.path_spec, progress_callback=update)
        return path


class GeoJSONScene(BaseRouteScene):
    geojson_spec: GeoJSONSpec

    def route_map_view(self) -> tuple[float, float, float]:
        return self.geojson_spec.map_view()

    def route_geojson_path(self) -> Path:
        return self.geojson_spec.path

    def route_labels(self) -> tuple[list[str], list[str]]:
        return self.geojson_spec.start_names, self.geojson_spec.end_names


PATH_SPECS = load_path_specs()
PATH_SCENE_NAMES: list[str] = []
SCENE_NAME_BY_PATH: dict[str, str] = {}
PATH_RENDER_MANIFEST: list[RenderPath] = []

GEOJSON_SPECS = load_geojson_specs()
GEOJSON_SCENE_NAMES: list[str] = []
SCENE_NAME_BY_GEOJSON: dict[str, str] = {}
GEOJSON_RENDER_MANIFEST: list[RenderGeoJSON] = []


def _register_path_scenes() -> None:
    for spec in PATH_SPECS:
        scene_name = scene_name_for_identifier(spec.identifier)
        scene_class = type(scene_name, (PathScene,), {"path_spec": spec})
        globals()[scene_name] = scene_class
        PATH_SCENE_NAMES.append(scene_name)
        SCENE_NAME_BY_PATH[spec.identifier] = scene_name
        PATH_RENDER_MANIFEST.append(RenderPath(spec=spec, scene_name=scene_name))


def _register_geojson_scenes() -> None:
    for spec in GEOJSON_SPECS:
        scene_name = scene_name_for_identifier(spec.identifier)
        scene_class = type(scene_name, (GeoJSONScene,), {"geojson_spec": spec})
        globals()[scene_name] = scene_class
        GEOJSON_SCENE_NAMES.append(scene_name)
        SCENE_NAME_BY_GEOJSON[spec.identifier] = scene_name
        GEOJSON_RENDER_MANIFEST.append(RenderGeoJSON(spec=spec, scene_name=scene_name))


def get_scene_name_for_path(identifier: str) -> str:
    try:
        return SCENE_NAME_BY_PATH[identifier]
    except KeyError as exc:
        raise KeyError(f'Unknown path "{identifier}"') from exc


def get_scene_name_for_geojson(identifier: str) -> str:
    try:
        return SCENE_NAME_BY_GEOJSON[identifier]
    except KeyError as exc:
        raise KeyError(f'Unknown GeoJSON "{identifier}"') from exc


_register_path_scenes()
_register_geojson_scenes()
