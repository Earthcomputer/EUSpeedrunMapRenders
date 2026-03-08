"""
tilemap_manim.py

Dependencies:
  pip install requests pillow
  manim (community edition)

Usage:
  from manim import *
  from tilemap_manim import TileMap, TileMapExampleScene

  # then render TileMapExampleScene with manim
"""

import math
import os
import io
import requests
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# Manim import (Community edition)
from manim import ImageMobject, Scene


# ---- Web-Mercator helpers ----
MAX_MERCATOR_LAT = 85.0511287798066  # clamp latitude

DEFAULT_TILE_SIZE = 256


def clamp_lat(lat: float) -> float:
    return max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)


def latlon_to_global_pixel(lat: float, lon: float, zoom: int, tile_size: int = DEFAULT_TILE_SIZE) -> Tuple[float, float]:
    """
    Convert lat/lon -> global pixel coordinates at given zoom level (Web Mercator).
    Returns (pixel_x, pixel_y) where full world is tile_size * 2^zoom in size.
    """
    lat = clamp_lat(lat)
    sin_lat = math.sin(math.radians(lat))
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * tile_size
    # y using Mercator
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * tile_size
    return x, y


def global_pixel_to_tile(px: float, py: float, tile_size: int = DEFAULT_TILE_SIZE) -> Tuple[int, int]:
    return int(px // tile_size), int(py // tile_size)


# ---- Tile fetching + stitching ----
DEFAULT_SUBDOMAINS = ["a", "b", "c", "d"]


class TileFetcher:
    """
    Fetches and caches XYZ tiles.
    """

    def __init__(self, template_url: str, cache_dir: Optional[str] = None, subdomains=None, timeout=10):
        """
        template_url: e.g. "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
        """
        self.template = template_url
        self.subdomains = subdomains or DEFAULT_SUBDOMAINS
        self.timeout = timeout
        if cache_dir is None:
            cache_dir = "manim_tile_cache"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def _tile_cache_path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self.cache_dir, f"tile_z{z}_x{x}_y{y}.png")

    def fetch(self, z: int, x: int, y: int) -> Image.Image:
        """Return a PIL Image for the requested tile. If fetch fails, return a blank tile image."""
        path = self._tile_cache_path(z, x, y)
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGBA")
            except Exception:
                # fallback to re-download
                pass

        # pick a subdomain deterministically
        sub = self.subdomains[(x + y) % len(self.subdomains)]
        url = self.template.format(s=sub, z=z, x=x, y=y)
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            img.save(path)  # cache
            return img
        except Exception as e:
            print(f"Tile fetch error for z={z} x={x} y={y}: {e}")
            # Return a transparent tile on failure
            tile = Image.new("RGBA", (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), (0, 0, 0, 0))
            return tile


def stitch_tiles(fetcher: TileFetcher, z: int, min_tx: int, min_ty: int, max_tx: int, max_ty: int, tile_size: int = DEFAULT_TILE_SIZE) -> Image.Image:
    """
    Stitch tiles from tile coords [min_tx..max_tx] x [min_ty..max_ty] inclusive.
    Returns a PIL Image (RGBA).
    """
    cols = max_tx - min_tx + 1
    rows = max_ty - min_ty + 1
    out_w = cols * tile_size
    out_h = rows * tile_size
    out = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    for ix, tx in enumerate(range(min_tx, max_tx + 1)):
        for iy, ty in enumerate(range(min_ty, max_ty + 1)):
            tile_img = fetcher.fetch(z, tx % (2 ** z), ty % (2 ** z))  # wrap x; y wrap depends on server (we modulo to be safe)
            if tile_img.width != tile_size or tile_img.height != tile_size:
                raise Exception(f'Image size ({tile_img.width}, {tile_img.height} does not match tile size {tile_size})')
            out.paste(tile_img, (ix * tile_size, iy * tile_size))
    return out


# ---- High-level TileMap utility ----
class TileMap:
    """
    High-level helper to create a stitched basemap centered at (lat, lon) at integer zoom,
    with a width/height specified in number of tiles (odd numbers center nicely).
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        zoom: float,
        output_width_px: Optional[int] = None,
        output_height_px: Optional[int] = None,
        template_url: str = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
        tile_size: int = DEFAULT_TILE_SIZE,
        cache_dir: Optional[str] = None,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = float(zoom)
        self.output_width_px = output_width_px
        self.output_height_px = output_height_px
        self.tile_size = tile_size
        self.fetcher = TileFetcher(template_url, cache_dir=cache_dir)
        # filled after build()
        self.image: Optional[Image.Image] = None
        self.min_tx = self.min_ty = self.max_tx = self.max_ty = None
        self.global_top_left_px = None  # global pixel coords of top-left of stitched image (x, y)
        self.width_px = None
        self.height_px = None
        self._fetch_z = None

    def build(self, scene = None) -> Image.Image:
        """
        Download and stitch tiles. Sets self.image and geometry attributes and returns PIL.Image.
        """
        z_tile = int(math.ceil(self.zoom))
        self._fetch_z = z_tile

        if self.output_width_px is None or self.output_height_px is None:
            if scene is None:
                raise ValueError("Scene must be provided if output image size is not specified")
            self.output_width_px = scene.camera.pixel_width
            self.output_height_px = scene.camera.pixel_height
        
        scale_frac = 2 ** (self.zoom - z_tile)

        required_px_at_tile_z_x = self.output_width_px / (scale_frac if scale_frac > 0 else 1.0)
        required_px_at_tile_z_y = self.output_height_px / (scale_frac if scale_frac > 0 else 1.0)
        tiles_x = int(math.ceil(required_px_at_tile_z_x / self.tile_size))
        tiles_y = int(math.ceil(required_px_at_tile_z_y / self.tile_size))
        if tiles_x % 2 == 0:
            tiles_x += 1
        if tiles_y % 2 == 0:
            tiles_y += 1
        

        # center pixel in global pixel coordinates
        center_px_tile, center_py_tile = latlon_to_global_pixel(self.center_lat, self.center_lon, z_tile, self.tile_size)
        center_tile_x, center_tile_y = global_pixel_to_tile(center_px_tile, center_py_tile, self.tile_size)

        # compute tile ranges (center in the middle)
        half_x = tiles_x // 2
        half_y = tiles_y // 2
        min_tx = center_tile_x - half_x
        min_ty = center_tile_y - half_y
        max_tx = min_tx + tiles_x - 1
        max_ty = min_ty + tiles_y - 1

        # stitch
        stitched = stitch_tiles(self.fetcher, z_tile, min_tx, min_ty, max_tx, max_ty, self.tile_size)

        # compute the global-pixel coordinate of the top-left pixel of the stitched image
        top_left_px_tile = (min_tx * self.tile_size, min_ty * self.tile_size)

        if scale_frac != 1.0:
            new_w = max(1, int(round(stitched.width * scale_frac)))
            new_h = max(1, int(round(stitched.height * scale_frac)))
            resized = stitched.resize((new_w, new_h), resample=Image.LANCZOS)
        else:
            resized = stitched
        
        final_img = resized
        crop_left_frac = 0.0
        crop_top_frac = 0.0
        target_w = self.output_width_px
        target_h = self.output_height_px
        cur_w, cur_h = resized.size
        
        # compute where the centre pixel is in the resized image (fractional px units)
        # center_px_tile is in tile-zoom pixels; convert to fractional px
        rel_center_x_tile = center_px_tile - top_left_px_tile[0]
        rel_center_y_tile = center_py_tile - top_left_px_tile[1]
        rel_center_x_frac = rel_center_x_tile * scale_frac
        rel_center_y_frac = rel_center_y_tile * scale_frac

        # desired top-left within resized such that centre maps to target center:
        desired_left = rel_center_x_frac - (target_w / 2.0)
        desired_top = rel_center_y_frac - (target_h / 2.0)

        # Now decide: if the desired crop region is fully inside resized -> crop.
        # Otherwise we'll paste resized onto a transparent canvas of target size at (-desired_left, -desired_top)
        if (desired_left >= 0 and desired_left <= cur_w - target_w) and (desired_top >= 0 and desired_top <= cur_h - target_h):
            # direct crop
            left = int(round(desired_left))
            upper = int(round(desired_top))
            final_img = resized.crop((left, upper, left + target_w, upper + target_h))
            crop_left_frac = desired_left
            crop_top_frac = desired_top
        else:
            # need to pad (or partly pad). Create transparent canvas and paste resized at offset.
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            paste_x = int(round(-desired_left))
            paste_y = int(round(-desired_top))
            # paste with alpha (third arg). PIL allows negative paste coords; if negative, the pasted image is clipped.
            canvas.paste(resized, (paste_x, paste_y), resized)
            final_img = canvas
            # When we pasted, the pixel at resized coordinate desired_left maps to canvas x=0,
            # so the crop offset used relative to resized was desired_left (even if out of range). Keep that.
            crop_left_frac = desired_left
            crop_top_frac = desired_top
        
        top_left_frac_x = top_left_px_tile[0] * scale_frac + crop_left_frac
        top_left_frac_y = top_left_px_tile[1] * scale_frac + crop_top_frac

        # store
        self.image = final_img
        self.min_tx, self.min_ty, self.max_tx, self.max_ty = min_tx, min_ty, max_tx, max_ty
        self.global_top_left_px = (top_left_frac_x, top_left_frac_y)
        self.width_px, self.height_px = final_img.size
        return final_img

    
    def get_numpy_image(self, scene = None) -> np.ndarray:
        """
        Ensure the tilemap is built and return a HxWxC uint8 numpy array suitable for Manim's ImageMobject.
        Channels will be 3 (RGB) or 4 (RGBA) depending on the source tiles.
        """
        if self.image is None:
            self.build(scene)

        # PIL -> numpy (uint8). This yields shape (height, width, channels).
        arr = np.asarray(self.image)
        # Ensure dtype is uint8
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.dtype == np.float32 else arr.astype(np.uint8)
        return arr

    def make_image_mobject(self, scene: "Scene", set_width_to_frame: bool = True) -> "ImageMobject":
        """
        Build if necessary, then return a Manim ImageMobject using a NumPy array (the preferred input).
        By default we scale to scene.camera.frame_width so the image fills the width of the frame (keeps aspect).
        """
        # ensure image built
        arr = self.get_numpy_image(scene)

        # ImageMobject accepts a NumPy array (H, W, 3/4) uint8
        img_m = ImageMobject(arr)

        if set_width_to_frame:
            img_m.set_width(scene.camera.frame_width)
            img_m.move_to(scene.camera.frame_center)

        return img_m

    def latlon_to_scene_coords(self, lat: float, lon: float, scene: Scene) -> Tuple[float, float]:
        """
        Convert lat/lon to Manim scene coordinates (x, y) assuming the stitched
        image has been added at center and scaled to `scene.camera.frame_width`.
        This uses the same scaling performed in make_image_mobject(...).
        """
        if self.image is None or self.global_top_left_px is None:
            raise RuntimeError("TileMap.build() must be called before converting coordinates.")

        # compute global pixel coordinate for lat/lon
        px, py = latlon_to_global_pixel(lat, lon, self.zoom, self.tile_size)

        # compute pixel coordinate relative to top-left of stitched image
        top_left_x, top_left_y = self.global_top_left_px
        rel_x = px - top_left_x
        rel_y = py - top_left_y

        # map rel_x/rel_y in [0..width_px],[0..height_px] -> scene coordinates:
        frame_w = scene.camera.frame_width
        # frame height depends on aspect
        frame_h = scene.camera.frame_height

        # origin: scene center corresponds to the center of the image after set_width
        # x: left -> negative, right -> positive
        scene_x = (rel_x - self.width_px / 2.0) * (frame_w / self.width_px)
        # y: pixel down is positive, but scene y up is positive -> flip sign
        scene_y = -(rel_y - self.height_px / 2.0) * (frame_h / self.height_px)

        return scene_x, scene_y
