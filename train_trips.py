from __future__ import annotations

import json
from pathlib import Path

from maputils import *

MANIFEST_PATH = Path("data/train_trip_manifest.json")


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    trips = payload.get("trips")
    if not isinstance(trips, list):
        raise ValueError(f'Invalid manifest at "{MANIFEST_PATH.as_posix()}": missing "trips" list')
    return trips


class TrainTripScene(TileMapScene):
    trip: dict

    def create_tile_map(self, **kwargs):
        map_info = self.trip["map"]
        return TileMap(
            map_info["center_lat"],
            map_info["center_lon"],
            map_info["zoom"],
            **kwargs,
        )

    def construct(self):
        route = self.load_geojson(self.trip["geojson_path"])
        start_marker = self.create_marker(
            self.trip["from_label"],
            self.trip["start"]["lat"],
            self.trip["start"]["lon"],
        )
        end_marker = self.create_marker(
            self.trip["to_label"],
            self.trip["end"]["lat"],
            self.trip["end"]["lon"],
        )
        self.play(*start_marker.animate_creation())
        self.play(*end_marker.animate_creation())
        route.create_and_animate(self, dash_animate_time=6, keep_on_top=[start_marker, end_marker])
        self.wait(0.4)


TRIP_MANIFEST = load_manifest()
TRIP_SCENE_NAMES: list[str] = []

for trip in TRIP_MANIFEST:
    scene_name = trip["scene_name"]
    if not isinstance(scene_name, str) or not scene_name:
        continue
    scene_class = type(scene_name, (TrainTripScene,), {"trip": trip})
    globals()[scene_name] = scene_class
    TRIP_SCENE_NAMES.append(scene_name)
