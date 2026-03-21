from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .geojson import load_geojson_line_coordinates
from .geometry import compute_center_zoom
from .naming import display_name_pair_for_identifier, scene_name_for_identifier

DEFAULT_GEOJSONS_DIR = Path("data/geojson/manual")


@dataclass(frozen=True, slots=True)
class GeoJSONSpec:
    identifier: str
    path: Path
    start_names: list[str]
    end_names: list[str]

    @property
    def file_stem(self) -> str:
        return self.path.stem

    def map_view(self) -> tuple[float, float, float]:
        return compute_center_zoom([list(point) for point in load_geojson_line_coordinates(self.path)])


def load_geojson_spec(path: Path) -> GeoJSONSpec:
    load_geojson_line_coordinates(path)
    start_names, end_names = display_name_pair_for_identifier(path.stem)
    return GeoJSONSpec(
        identifier=path.stem,
        path=path,
        start_names=start_names,
        end_names=end_names,
    )


def load_geojson_specs(geojson_dir: Path = DEFAULT_GEOJSONS_DIR) -> list[GeoJSONSpec]:
    if not geojson_dir.exists():
        return []
    files = sorted(geojson_dir.rglob("*.geojson"), key=lambda path: path.as_posix())
    specs = [load_geojson_spec(path) for path in files]
    seen: set[str] = set()
    for spec in specs:
        if spec.identifier in seen:
            raise ValueError(f'Duplicate GeoJSON scene identifier "{spec.identifier}" in "{geojson_dir.as_posix()}"')
        seen.add(spec.identifier)
    return specs
