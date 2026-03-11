from __future__ import annotations
import json
from bisect import bisect_right
from pathlib import Path
from manim import *
from manim.typing import *
from manimpango import register_font
from tilemap_manim import *
from typing import Callable, List, Sequence, Tuple

try:
    from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
except Exception:
    OpenGLVMobject = None

register_font("OpenSans-Bold.ttf")

def load_tilemap_settings():
    try:
        with open("tilemap_settings.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

tilemap_settings = load_tilemap_settings()
TRAIN_ICON_PATH = Path("assets/train-front-fill.svg")

def route_rate(route: VMobject, rate_func: Callable[[float], float] = smooth) -> Callable[[float], float]:
    curve_lengths = [c[1] for c in route.get_curve_functions_with_lengths()]
    t_per_interval = 1 / len(curve_lengths)
    relative_time = [(l / sum(curve_lengths)) for l in curve_lengths]
    v = 0
    cumulative_t = [0] + [v := v+t for t in relative_time]

    def curve_rf(t):
        t = rate_func(t)
        if t <= 0: return 0
        if t >= cumulative_t[-1]: return 1

        # find index of current curve
        i = bisect_right(cumulative_t, t) - 1

        a, b = cumulative_t[i], cumulative_t[i+1]
        t1 = i * t_per_interval
        t2 = (i + 1) * t_per_interval

        return interpolate(t1, t2, inverse_interpolate(a, b, t))

    return curve_rf

class DashedLine(VMobject):
    def __init__(self, base_obj: VMobject, dash_len: float = 0.2, dash_offset: float = 0.0, colors: List[ParsableManimColor] = [BLUE, WHITE], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_obj = base_obj
        self._dash_len = dash_len
        self._dash_offset = dash_offset
        self.colors = colors
        self.refresh_dashes()
    
    def set_dash_len(self, dash_len: float) -> DashedLine:
        self._dash_len = dash_len
        self.refresh_dashes()
        return self

    def set_dash_offset(self, dash_offset: float) -> DashedLine:
        self._dash_offset = dash_offset
        self.refresh_dashes()
        return self

    def refresh_dashes(self):
        self.submobjects.clear()

        # calculate the entire length by adding up short line-pieces
        norms = np.array(0)
        for k in range(self.base_obj.get_num_curves()):
            norms = np.append(norms, self.base_obj.get_nth_curve_length_pieces(k))
        # add up length-pieces in array form
        length_vals = np.cumsum(norms)
        ref_points = np.linspace(0, 1, length_vals.size)
        curve_length = length_vals[-1]
        
        if curve_length > 0 and self._dash_len > 0:
            dash_len = min(self._dash_len, curve_length)
            dash_proportion = dash_len / curve_length
            dash_offset = self._dash_offset % 1
            color_offset = -int(self._dash_offset) % len(self.colors)
            offset = dash_offset * dash_proportion

            if offset > 0:
                self.add(self.base_obj.get_subcurve(
                    0,
                    np.interp(offset * curve_length, length_vals, ref_points)
                ).match_style(self).set_color(self.colors[(color_offset - 1) % len(self.colors)]))
            i = 0
            while offset < curve_length:
                self.add(self.base_obj.get_subcurve(
                    np.interp(offset * curve_length, length_vals, ref_points), 
                    min(1, np.interp((offset + dash_proportion) * curve_length, length_vals, ref_points))
                ).match_style(self).set_color(self.colors[(i + color_offset) % len(self.colors)]))
                offset += dash_proportion
                i += 1

class DashOffsetAnimation(Animation):
    def __init__(self, mobject: DashedLine, dash_move_speed: float = 2.0, fade_in_time: float = 1.0, rate_func: Callable[[float], float] = linear, *args, **kwargs):
        super().__init__(mobject, *args, rate_func=rate_func, **kwargs)
        self.initial_offset = mobject._dash_offset
        self.dash_move_speed = dash_move_speed
        self.fade_in_time = fade_in_time

    def interpolate_mobject(self, alpha: float):
        time = alpha * self.run_time
        opacity = 1 if time >= self.fade_in_time or self.fade_in_time <= 0 else time / self.fade_in_time
        offset = self.initial_offset + time * self.dash_move_speed
        self.mobject.become(self.mobject.copy().set_dash_offset(offset))
        self.mobject.set_opacity(opacity)

class Geojson:
    def __init__(self, points: List[Point3D], route_line: VMobject, dashed_route_line: DashedLine | None):
        self.points = points
        self.route_line = route_line
        self.dashed_route_line = dashed_route_line
    
    def animate_line_creation(self) -> Animation:
        return Create(self.route_line, rate_func=route_rate(self.route_line, linear))
    
    def animate_dashes(self, run_time: float = 10) -> Animation:
        if self.dashed_route_line is None:
            raise ValueError("Dashed route line is not available")
        return DashOffsetAnimation(self.dashed_route_line, run_time=run_time)

    def create_train_icon(self) -> VGroup:
        if TRAIN_ICON_PATH.exists():
            train_svg = SVGMobject(str(TRAIN_ICON_PATH))
            train_svg.set_fill(WHITE, opacity=1.0).set_stroke(DARK_BLUE, width=2.6, opacity=1.0)
            train_svg.scale_to_fit_height(0.42)
        else:
            train_svg = Triangle().set_fill(WHITE, opacity=1.0).set_stroke(DARK_BLUE, width=2.6, opacity=1.0).scale(0.16).rotate(-PI / 2)
        shadow = train_svg.copy().set_fill(BLACK, opacity=0.22).set_stroke(width=0).shift(0.045 * DOWN + 0.04 * RIGHT)
        return VGroup(shadow, train_svg).scale(1.18)

    def create_and_animate(
        self,
        scene: Scene,
        dash_animate_time: float = 10,
        keep_on_top: Sequence[Mobject] | None = None,
    ) -> Geojson:
        renderer = str(config["renderer"]).lower()
        if "opengl" in renderer:
            route_glow = self.route_line.copy().set_stroke(color=WHITE, width=10.0, opacity=0.2)
            scene.play(
                Create(route_glow, rate_func=route_rate(route_glow, linear)),
                self.animate_line_creation(),
            )
            dash_count = max(30, min(220, len(self.points) // 5))
            dotted_route = DashedVMobject(
                self.route_line.copy(),
                num_dashes=dash_count,
                dashed_ratio=0.5,
            )
            dotted_route.set_stroke(color=WHITE, width=3.2, opacity=0.9)
            if hasattr(dotted_route, "set_z_index"):
                dotted_route.set_z_index(2)
            highlight = self.create_train_icon().move_to(self.start)
            if hasattr(highlight, "set_z_index"):
                highlight.set_z_index(3)
            scene.play(FadeIn(dotted_route, run_time=0.3))
            if keep_on_top:
                scene.bring_to_front(*keep_on_top, highlight)
            scene.add(highlight)
            scene.play(MoveAlongPath(highlight, self.route_line), run_time=dash_animate_time, rate_func=linear)
            scene.play(FadeOut(highlight, run_time=0.15))
            return self
        scene.play(self.animate_line_creation())
        if self.dashed_route_line is not None:
            scene.play(self.animate_dashes(run_time=dash_animate_time))
        if keep_on_top:
            scene.bring_to_front(*keep_on_top)
        return self

    @property
    def start(self) -> Point3D:
        return self.points[0]
    
    @property
    def end(self) -> Point3D:
        return self.points[-1]

class MapMarker(VGroup):
    def __init__(self, label: str, location: Point3DLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dot = Dot().set_color(DARK_BLUE).move_to(location)
        self.add(self.dot)
        self.text = Text(label, font="Open Sans", font_size=24).set_color(DARK_BLUE).next_to(self.dot)
        self.add(self.text)
        if hasattr(self, "set_z_index"):
            self.set_z_index(1)

    def animate_creation(self) -> Sequence[Animation]:
        renderer = str(config["renderer"]).lower()
        if "opengl" in renderer:
            return [GrowFromCenter(self.dot), FadeIn(self.text, shift=0.1 * UP, run_time=1)]
        return [GrowFromCenter(self.dot), Write(self.text, run_time=1)]

class TileMapScene(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tm = self.create_tile_map(**tilemap_settings)
        self.tm_image: ImageMobject = None
    
    def create_tile_map(self, **kwargs) -> TileMap:
        raise NotImplementedError("Subclasses must implement create_tile_map")
    
    def setup(self):
        super().setup()
        self.tm_image = self.tm.make_image_mobject(self, set_width_to_frame=True)
        self.add(self.tm_image)
    
    def load_geojson(self, filename: str) -> Geojson:
        with open(filename) as f:
            data = json.load(f)
        features = data["features"]
        if len(features) != 1:
            raise Exception(f"Geojson {filename} should have exactly one feature, found {len(features)}")
        geometry = features[0]["geometry"]
        if geometry["type"] != "LineString":
            raise Exception(f"Geojson {filename} should have LineString geometry, found {geometry["type"]}")
        points = [self.tm.latlon_to_scene_coords(lat, lon, self) for [lon, lat, *_] in geometry["coordinates"]]
        points = [np.array([sx, sy, 0.0], dtype=float) for sx, sy in points]

        renderer = str(config["renderer"]).lower()
        if "opengl" in renderer and OpenGLVMobject is not None:
            route_line = OpenGLVMobject()
        else:
            route_line = VMobject()
        route_line.set_points_as_corners(points)
        route_line.set_stroke(color=DARK_BLUE, width=6.0)

        if "opengl" in renderer:
            dashed_route_line = None
        else:
            dashed_route_line = DashedLine(route_line)

        return Geojson(points, route_line, dashed_route_line)
    
    def create_marker(self, label: str, lat: float, lon: float) -> MapMarker:
        sx, sy = self.tm.latlon_to_scene_coords(lat, lon, self)
        return MapMarker(label, np.array([sx, sy, 0.0]))
