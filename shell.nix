{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python3
    uv
    pkg-config
    cairo
    meson
    ninja
    pango
    stdenv.cc.cc.lib
    zlib
    libGL
    libGLU
    glib
    xorg.libX11
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    zlib
    libGL
    libGLU
    glib
    xorg.libX11
  ]);
}
