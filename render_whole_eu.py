from __future__ import annotations

from rendering import europe as _impl

EU_COUNTRIES = _impl.EU_COUNTRIES
EUROPE_GEOJSON_PATH = _impl.EUROPE_GEOJSON_PATH
INTRO_COUNTRY_ORDER = _impl.INTRO_COUNTRY_ORDER
VISITED_SCENE_NAMES = list(_impl.VISITED_SCENE_NAMES)

EuropeMapScene = type("EuropeMapScene", (_impl.EuropeMapScene,), {"__module__": __name__})
WholeEUIntroScene = type("WholeEUIntroScene", (_impl.WholeEUIntroScene,), {"__module__": __name__})
VisitedScene = type("VisitedScene", (_impl.VisitedScene,), {"__module__": __name__})
VisitedMaltaScene = type("VisitedMaltaScene", (_impl.VisitedMaltaScene,), {"__module__": __name__})
IntroTitle = type("IntroTitle", (_impl.IntroTitle,), {"__module__": __name__})

for _scene_name in VISITED_SCENE_NAMES:
    _base = getattr(_impl, _scene_name)
    globals()[_scene_name] = type(_scene_name, (_base,), {"__module__": __name__})

__all__ = [
    "EU_COUNTRIES",
    "EUROPE_GEOJSON_PATH",
    "EuropeMapScene",
    "INTRO_COUNTRY_ORDER",
    "IntroTitle",
    "VisitedMaltaScene",
    "VisitedScene",
    "VISITED_SCENE_NAMES",
    "WholeEUIntroScene",
    *VISITED_SCENE_NAMES,
]
