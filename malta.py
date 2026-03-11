from maputils import *

class VallettaToPozzalloScene(TileMapScene):
    def create_tile_map(self, **kwargs):
        return TileMap(36.32441076142739, 14.7138834390955, 9.25, **kwargs)

    def construct(self):
        route = self.load_geojson("geojson/malta/valletta_to_pozzallo.geojson")
        self.play(MapMarker("Valletta Port", route.start).animate_creation())
        self.play(MapMarker("Pozzallo Port", route.end).animate_creation())
        route.create_and_animate(self)
