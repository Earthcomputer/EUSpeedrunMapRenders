from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import (
    Arc,
    Animation,
    Arrow,
    Create,
    DARK_BLUE,
    DOWN,
    DrawBorderThenFill,
    FadeOut,
    FullScreenRectangle,
    LaggedStart,
    LEFT,
    MoveAlongPath,
    Polygon,
    RIGHT,
    Scene,
    Star,
    TAU,
    Text,
    UP,
    WHITE,
    Write,
    double_smooth,
)

from rendering.core.geometry import route_rate
from rendering.core.map_scene import TileMapScene
from rendering.core.tile_map import TileMap

EUROPE_GEOJSON_PATH = Path("data/geojson/europe_clean.geojson")
EU_COUNTRIES = (
    "Malta",
    "Italy",
    "France",
    "Spain",
    "Portugal",
    "Ireland",
    "Belgium",
    "Luxembourg",
    "Netherlands",
    "Germany",
    "Denmark",
    "Sweden",
    "Finland",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Poland",
    "Czech Republic",
    "Austria",
    "Slovenia",
    "Croatia",
    "Hungary",
    "Slovakia",
    "Romania",
    "Bulgaria",
    "Greece",
    "Cyprus",
)
INTRO_COUNTRY_ORDER = (
    "Sweden",
    "Finland",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Denmark",
    "Ireland",
    "Belgium",
    "Netherlands",
    "Luxembourg",
    "Germany",
    "Poland",
    "Slovakia",
    "Czech Republic",
    "France",
    "Austria",
    "Hungary",
    "Romania",
    "Bulgaria",
    "Croatia",
    "Slovenia",
    "Italy",
    "Spain",
    "Portugal",
    "Malta",
    "Greece",
    "Cyprus",
)


@dataclass(frozen=True, slots=True)
class CountryGeometry:
    name: str
    polygons: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(slots=True)
class CountryShape:
    name: str
    outlines: tuple[Polygon, ...]

    def animate_creation(self, run_time: float = 2.0) -> Animation:
        animations = tuple(
            DrawBorderThenFill(
                outline,
                stroke_width=1.3,
                run_time=run_time if len(self.outlines) == 1 else run_time / 1.2,
                rate_func=_country_rate(outline),
            )
            for outline in self.outlines
        )
        if len(animations) == 1:
            return animations[0]
        return LaggedStart(*animations, lag_ratio=0.2 / max(1, len(animations) - 1))

    def title_on_top(self) -> bool:
        return any(outline.get_bottom()[1] < -0.75 for outline in self.outlines)


def _country_rate(outline: Polygon):
    mapped_rate = route_rate(outline)

    def rate_func(t: float) -> float:
        if t < 0.5:
            return mapped_rate(t * 2.0) * 0.5
        return float(double_smooth(t))

    return rate_func


@lru_cache(maxsize=1)
def _country_geometries() -> dict[str, CountryGeometry]:
    with EUROPE_GEOJSON_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    geometries: dict[str, CountryGeometry] = {}
    for feature in payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            continue
        name = properties.get("name")
        if not isinstance(name, str) or not name:
            continue
        polygons = _extract_polygons(geometry)
        if polygons:
            geometries[name] = CountryGeometry(name=name, polygons=polygons)
    return geometries


def _extract_polygons(geometry: dict) -> tuple[tuple[tuple[float, float], ...], ...]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        return _polygon_from_rings(coordinates)
    if geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        polygons = tuple(
            polygon
            for item in coordinates
            for polygon in (_polygon_from_rings(item) if isinstance(item, list) else ())
        )
        return polygons
    return ()


def _polygon_from_rings(rings: list) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not rings:
        return ()
    outer_ring = rings[0]
    if not isinstance(outer_ring, list):
        return ()
    points = tuple(
        (float(lon), float(lat))
        for coordinate in outer_ring
        if isinstance(coordinate, list) and len(coordinate) >= 2
        for lon, lat in [coordinate[:2]]
    )
    return (points,) if len(points) >= 3 else ()


def display_country_name(name: str) -> str:
    return "Czechia" if name == "Czech Republic" else name


class EuropeMapScene(TileMapScene):
    def create_tile_map(self, **kwargs):
        return TileMap(57.51465184611381, 13.463387421839668, 3.45, **kwargs)

    def load_country_shape(self, name: str) -> CountryShape:
        geometry = _country_geometries().get(name)
        if geometry is None:
            raise ValueError(f'Unknown country "{name}"')
        outlines = tuple(self._build_outline(polygon) for polygon in geometry.polygons)
        return CountryShape(name=name, outlines=outlines)

    def _build_outline(self, polygon: tuple[tuple[float, float], ...]) -> Polygon:
        points = [
            np.array([*self.tm.latlon_to_scene_coords(lat, lon, self), 0.0], dtype=float)
            for lon, lat in polygon
        ]
        outline = Polygon(*points)
        outline.set_stroke(color=DARK_BLUE, width=1.3, opacity=0.65)
        outline.set_fill(opacity=0.0)
        return outline


class WholeEUIntroScene(EuropeMapScene):
    def construct(self):
        self.play(
            LaggedStart(
                *(self.load_country_shape(name).animate_creation(run_time=0.3) for name in INTRO_COUNTRY_ORDER),
                lag_ratio=0.2,
            )
        )
        self.wait(2)


class VisitedScene(EuropeMapScene):
    index: int

    def country_name(self) -> str:
        return EU_COUNTRIES[self.index]

    def visited_title(self, country: CountryShape) -> Text:
        title = Text(f"{display_country_name(self.country_name())} Visited!", font="Open Sans").set_color(DARK_BLUE)
        if country.title_on_top():
            title.to_edge(UP)
        else:
            title.to_edge(DOWN)
        return title

    def build_intro_animations(self, country: CountryShape) -> list[Animation]:
        return [country.animate_creation(), Write(self.visited_title(country))]

    def construct(self):
        for visited_name in EU_COUNTRIES[: self.index]:
            for outline in self.load_country_shape(visited_name).outlines:
                self.add(outline)
        self.play(*self.build_intro_animations(self.load_country_shape(self.country_name())))
        self.wait(2)


class VisitedMaltaScene(VisitedScene):
    index = 0

    def build_intro_animations(self, country: CountryShape) -> list[Animation]:
        malta_outline = country.outlines[0]
        arrow = Arrow(malta_outline.get_right() + RIGHT, malta_outline.get_right() + 0.1 * LEFT).set_color(DARK_BLUE)
        label = Text("Malta", font="Open Sans").scale(0.4).next_to(arrow, buff=0.1).set_color(DARK_BLUE)
        return super().build_intro_animations(country) + [Create(arrow), Write(label)]


class IntroTitle(Scene):
    def construct(self):
        title = Text("EU Speedrun", font="Open Sans").set_color(WHITE)
        stars = self._build_stars(title)
        background = FullScreenRectangle(fill_color=0x003399, fill_opacity=0, stroke_width=0)
        background.set_z_index(-10)
        self.add(background)
        self.play(
            LaggedStart(
                Write(title, run_time=1.5),
                DrawBorderThenFill(stars[0], run_time=1),
                lag_ratio=0.75,
            )
        )
        for star in stars[1:]:
            self.add(star)
        self.play(
            *(MoveAlongPath(star, path) for star, path in self._star_paths(title, stars)),
            background.animate.set_fill(opacity=1),
            run_time=2,
        )
        self.wait(0.5)
        self.play(FadeOut(title))
        self.wait(1)

    @staticmethod
    def _build_stars(title: Text) -> list[Star]:
        base_star = Star().set_color(0xFFCC00).set_opacity(1.0).scale(0.3)
        base_star.reverse_points()
        base_star.next_to(title, RIGHT)
        radius = np.linalg.norm(base_star.get_center() - title.get_center())
        base_star.move_to(title.get_center() + radius * UP)
        stars = [base_star]
        for _ in range(11):
            star = base_star.copy()
            star.move_to(base_star.get_center())
            stars.append(star)
        return stars

    @staticmethod
    def _star_paths(title: Text, stars: list[Star]):
        center = title.get_center()
        radius = np.linalg.norm(stars[0].get_center() - center)
        step = TAU / len(stars)
        start_angles = [
            np.arctan2(star.get_center()[1] - center[1], star.get_center()[0] - center[0])
            for star in stars
        ]
        base_angle = start_angles[0]
        target_angles = [base_angle + index * step for index in range(len(stars))]
        return [
            (
                star,
                Arc(start_angle=start_angle, angle=_clockwise_delta(start_angle, target_angle), radius=radius).shift(
                    center
                ),
            )
            for star, start_angle, target_angle in zip(stars, start_angles, target_angles)
        ]


def _clockwise_delta(start_angle: float, target_angle: float) -> float:
    delta = target_angle - start_angle
    return delta - TAU if delta > 0 else delta


VISITED_SCENE_NAMES: list[str] = []


def _register_visited_scenes() -> None:
    for index, country_name in enumerate(EU_COUNTRIES[1:], start=1):
        scene_name = f"Visited{country_name.replace(' ', '')}Scene"
        globals()[scene_name] = type(scene_name, (VisitedScene,), {"index": index})
        VISITED_SCENE_NAMES.append(scene_name)


_register_visited_scenes()

__all__ = [
    "EU_COUNTRIES",
    "EUROPE_GEOJSON_PATH",
    "EuropeMapScene",
    "IntroTitle",
    "INTRO_COUNTRY_ORDER",
    "VisitedMaltaScene",
    "VisitedScene",
    "VISITED_SCENE_NAMES",
    "WholeEUIntroScene",
    *VISITED_SCENE_NAMES,
]
