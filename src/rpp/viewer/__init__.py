"""
RPV, the Rover Photogrammetry Viewer: one static web page for the station products of :mod:`rpp.ibr`.

``index.html`` (three.js, WebXR) and ``vendor/`` are the whole viewer.  It reads ``panos.json``, the list of
stations that ``rpp single-sites --publish DIR`` writes (:mod:`rpp.ibr.web`), and shows each station as a stereo
panorama or as a 3D terrain with the rover, in a browser or in a VR headset.  The public copy is
https://ctate7163.github.io/rpv/ .

``README.md`` is the front page of the viewer's own repository (github.com/ctate7163/rpv); it is kept here with
the page and copied there by hand when it changes.

The page is package data, so an installed RPP can make a local copy of the viewer::

    rpp single-sites STATION --publish my_viewer      # writes my_viewer/index.html, vendor/, panos.json, panos/
    python -m http.server --directory my_viewer       # then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

PAGE = "index.html"


def page_dir() -> Path:
    """The folder that holds the viewer page (``index.html``) and its ``vendor/`` folder."""
    return Path(__file__).resolve().parent
