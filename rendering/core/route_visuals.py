from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from manim import (
    Animation,
    AnimationGroup,
    CapStyleType,
    Circle,
    Create,
    DARK_BLUE,
    DR,
    FadeIn,
    LineJointType,
    Scene,
    Text,
    ValueTracker,
    VMobject,
    VGroup,
    interpolate,
    linear,
)
from manim.typing import Point3DLike

from .geometry import as_point3, route_distance_lookup, route_proportion_for_distance, route_rate


@dataclass(frozen=True, slots=True)
class RouteLayerStyle:
    color: str
    width: float
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class RouteVisualStyle:
    layers: tuple[RouteLayerStyle, ...]
    draw_time: float = 3.1
    base_draw_time: float = 0.9


DEFAULT_ROUTE_STYLE = RouteVisualStyle(
    layers=(
        RouteLayerStyle("#1E3346", 9.4, 0.96),
        RouteLayerStyle("#6F89A6", 6.9, 1.0),
        RouteLayerStyle("#F7FBFF", 3.55, 1.0),
        RouteLayerStyle("#FFD76A", 1.15, 0.95),
    ),
)


class TripRoute:
    def __init__(
            self,
            points: Sequence[Point3DLike],
            style: RouteVisualStyle = DEFAULT_ROUTE_STYLE,
            distance_lookup: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        source = [as_point3(point) for point in points]
        if len(source) < 2:
            raise ValueError("Route requires at least two points")
        self.points = source
        self.style = style
        self.render_points = self._inset_endpoints(source)
        self.route_line = VMobject(joint_type=LineJointType.ROUND, cap_style=CapStyleType.ROUND)
        self.route_line.set_points_as_corners(self.render_points)
        self.route_line.set_fill(opacity=0)
        self._distance_fractions, self._distance_proportions = route_distance_lookup(
            self.route_line,
            lookup=distance_lookup,
        )
        self._length_rate = route_rate(
            self.route_line,
            linear,
            lookup=(self._distance_fractions, self._distance_proportions),
        )
        self.path_length = float(
            sum(
                np.linalg.norm((end - start)[:2])
                for start, end in zip(self.render_points, self.render_points[1:])
            )
        )

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def _distance_to_proportion(self, distance_fraction: float) -> float:
        return route_proportion_for_distance(
            self.route_line,
            distance_fraction,
            lookup=(self._distance_fractions, self._distance_proportions),
        )

    @staticmethod
    def _inset_endpoints(points: Sequence[np.ndarray], amount: float = 0.15) -> list[np.ndarray]:
        inset = [point.copy() for point in points]
        if len(inset) < 2:
            return inset
        start_length = float(np.linalg.norm((inset[1] - inset[0])[:2]))
        if start_length > 1e-6:
            start_ratio = min(1.0, amount / start_length)
            inset[0] = interpolate(inset[0], inset[1], start_ratio)
        end_length = float(np.linalg.norm((inset[-1] - inset[-2])[:2]))
        if end_length > 1e-6:
            end_ratio = min(1.0, amount / end_length)
            inset[-1] = interpolate(inset[-1], inset[-2], end_ratio)
        return inset

    def _endpoint_label_vector(self, anchor: np.ndarray, neighbor: np.ndarray | None) -> np.ndarray:
        if neighbor is None:
            vector = np.array([1.0 if anchor[0] <= 0 else -1.0, 0.35 if anchor[1] <= 0 else -0.35, 0.0], dtype=float)
        else:
            vector = as_point3(anchor - neighbor)
        if np.linalg.norm(vector[:2]) < 1e-6:
            vector = np.array([1.0 if anchor[0] <= 0 else -1.0, 0.35 if anchor[1] <= 0 else -0.35, 0.0], dtype=float)
        if abs(vector[0]) < 0.22:
            vector[0] += 0.35 if anchor[0] <= 0 else -0.35
        if abs(vector[1]) < 0.18:
            vector[1] += 0.28 if anchor[1] <= 0 else -0.28
        return vector

    def start_label_direction(self) -> np.ndarray:
        if len(self.points) <= 1:
            return self._endpoint_label_vector(self.start, None)
        window = self.points[1: min(len(self.points), 8)]
        weights = np.linspace(1.0, 0.45, len(window))
        center = sum(weight * point for weight, point in zip(weights, window)) / np.sum(weights)
        return self._endpoint_label_vector(self.start, center)

    def end_label_direction(self) -> np.ndarray:
        if len(self.points) <= 1:
            return self._endpoint_label_vector(self.end, None)
        window = self.points[max(0, len(self.points) - 8): -1]
        weights = np.linspace(0.45, 1.0, len(window))
        center = sum(weight * point for weight, point in zip(weights, window)) / np.sum(weights)
        return self._endpoint_label_vector(self.end, center)

    def _build_layer(self, layer_style: RouteLayerStyle) -> VMobject:
        layer = self.route_line.copy()
        layer.set_fill(opacity=0)
        layer.set_stroke(color=layer_style.color, width=layer_style.width, opacity=layer_style.opacity)
        return layer

    def _build_base_track(self) -> VGroup:
        base = VGroup(
            self.route_line.copy().set_fill(opacity=0).set_stroke(color="#1E3346", width=8.2, opacity=0.62),
            self.route_line.copy().set_fill(opacity=0).set_stroke(color="#F7FBFF", width=2.5, opacity=0.28),
        )
        return base

    def _route_normal(self, proportion: float) -> np.ndarray:
        p0 = self.route_line.point_from_proportion(max(0.0, proportion - 0.005))
        p1 = self.route_line.point_from_proportion(min(1.0, proportion + 0.005))
        tangent = p1 - p0
        tangent_norm = float(np.linalg.norm(tangent[:2]))
        if tangent_norm < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        tangent /= tangent_norm
        return np.array([-tangent[1], tangent[0], 0.0], dtype=float)

    def _build_station_marks(self, station_progresses: Sequence[float]) -> VGroup:
        marks = VGroup()
        if len(station_progresses) < 3:
            return marks
        for progress in station_progresses[1:-1]:
            clamped = float(np.clip(progress, 0.0, 1.0))
            point = self.route_line.point_from_proportion(self._distance_to_proportion(clamped))
            outer = Circle(radius=0.038).move_to(point).set_fill("#DFE9F3", opacity=0.74).set_stroke("#1A2A3A",
                                                                                                     width=0.75,
                                                                                                     opacity=0.52)
            inner = Circle(radius=0.015).move_to(point).set_fill("#4B6784", opacity=0.86).set_stroke(width=0)
            marks.add(VGroup(outer, inner))
        return marks

    @staticmethod
    def force_foreground(scene: Scene, mobjects: Sequence | None) -> None:
        if mobjects:
            scene.add_foreground_mobjects(*mobjects)

    def create_and_animate(
            self,
            scene: Scene,
            draw_time: float | None = None,
            keep_on_top: Sequence | None = None,
            total_distance: float | None = None,
            total_time: float | None = None,
            top_speed: float | None = None,
            station_progresses: Sequence[float] | None = None,
    ) -> TripRoute:
        recommended = float(np.clip(2.5 + self.path_length * 0.32, 2.8, 7.2))
        duration = draw_time if draw_time is not None else max(self.style.draw_time, recommended)
        layer_templates = [self._build_layer(layer_style) for layer_style in self.style.layers]
        layers = [template.copy() for template in layer_templates]
        base_track = self._build_base_track()
        progress = ValueTracker(0.0)
        speed_label = None
        station_marks = self._build_station_marks(station_progresses or [])

        if total_distance is not None or total_time is not None or top_speed is not None:
            stats = []
            if total_distance is not None:
                stats.append(("Distance:", f"{round(total_distance)}km ({round(total_distance * 0.621371)}mi)"))
            if total_time is not None:
                if total_time >= 60:
                    if total_time % 60 == 0:
                        stats.append(("Time:", f"{int(total_time // 60)}h"))
                    else:
                        stats.append(("Time:", f"{int(total_time // 60)}h {int(total_time % 60)}m"))
                else:
                    stats.append(("Time:", f"{int(total_time)}m"))
            if total_distance is not None and total_time is not None:
                average_speed = total_distance / total_time * 60
                stats.append(("Average speed:", f"{round(average_speed)}km/h ({round(average_speed * 0.621371)}mph)"))
            if top_speed is not None:
                stats.append(("Top speed:", f"{round(top_speed)}km/h ({round(top_speed * 0.621371)}mph)"))
            
            stats_obj = VGroup()
            for left, right in stats:
                stats_obj.add(Text(left, font="Open Sans"))
                stats_obj.add(Text(right, font="Open Sans"))
            
            stats_obj.scale(0.4)
            max_height = max((subobj.height for subobj in stats_obj.submobjects))
            stats_obj.arrange_in_grid(cols=2, col_alignments="rl", buff=0.1, row_heights=[max_height] * len(stats)).to_corner(DR).set_color(DARK_BLUE)
            scene.add_foreground_mobject(stats_obj)

        intro_animations: list[Animation] = [
            Create(base_track[0], run_time=self.style.base_draw_time, rate_func=self._length_rate),
            Create(base_track[1], run_time=self.style.base_draw_time * 1.05, rate_func=self._length_rate),
        ]
        if len(station_marks) > 0:
            intro_animations.append(FadeIn(station_marks, run_time=self.style.base_draw_time * 0.8))
        scene.play(AnimationGroup(*intro_animations, lag_ratio=0.08))
        trails = [0.0, 0.03, 0.06, 0.09]
        max_trail = max(trails)
        for index, (layer, template) in enumerate(zip(layers, layer_templates)):
            trail = trails[min(index, len(trails) - 1)]
            layer.pointwise_become_partial(template, 0.0, 0.0)
            layer.add_updater(
                lambda mob, dt, source=template, tail=trail: mob.pointwise_become_partial(
                    source,
                    0.0,
                    self._distance_to_proportion(
                        float(np.clip(progress.get_value() - tail, 0.0, 1.0))
                    ),
                )
            )
        scene.add(*layers)
        if speed_label is not None:
            scene.add_foreground_mobjects(speed_label)
        self.force_foreground(scene, keep_on_top)
        scene.play(
            progress.animate(rate_func=linear).set_value(1.0 + max_trail),
            run_time=duration,
        )
        for layer, template in zip(layers, layer_templates):
            layer.clear_updaters()
            layer.become(template)
        if speed_label is not None:
            speed_label.clear_updaters()
            scene.remove_foreground_mobjects(speed_label)
            scene.remove(speed_label)
        self.force_foreground(scene, keep_on_top)
        return self
