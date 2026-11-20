# RPV: the Rover Photogrammetry Viewer

RPV shows the station products of RPP in a web browser or a VR headset. It is one static web page (three.js and
WebXR), so it needs no server-side code and no installation.

[Open RPV](https://ctate7163.github.io/rpv/){ .md-button .md-button--primary target="_blank" rel="noopener" }

!!! note "Experimental"
    RPV is experimental in 1.0. The stereo panoramas are the most tested view. See
    [What is validated and what is experimental](status.md).

## What it shows

Each entry is one **rover station**: the place where the rover stood when it took a 360° panorama. A station made
from Mastcam-Z frames has `(zcam)` after its name; the others are made from Navcam frames.

The **terrain** menu chooses how the station is drawn:

| Terrain | What it is |
|---|---|
| terrain stereo | a stereo 360° panorama, one image per eye. The viewpoint is fixed |
| terrain local 3D | the textured terrain mesh within 20 m, in front of the terrain-only panorama |
| terrain full 3D | one textured surface from under the rover to the horizon and the sky. On a screen the eye moves a little so that the depth can be seen |

The **rover** button shows or hides the rover. In the stereo terrain it switches to the panorama that includes the
rover as the cameras saw it. In the 3D terrains it adds a 3D model of the rover, textured from the same images.

The **view** menu chooses the colors:

| View | What it is |
|---|---|
| standard color | the pipeline's white-balanced color |
| raw color | the white balance taken out again |
| enhanced color, enhanced tone | automatic display transforms that bring out color and texture. They are not calibrated color |
| dcs color | a decorrelation stretch of the sunlit terrain |
| normal map | the direction each surface faces |
| distance map | the range from the viewpoint |

Every panorama keeps a record of how it was made, and a picture of which source image each pixel came from. Every
file says who made it and with which RPP version: XMP in the pictures, `asset` in the 3D models, `author` and
`software` in the JSON records.

Straight down, a light blue disc on the ground carries the station's name, the image credit and the author. It is a
separate picture laid on the ground; the panoramas themselves do not carry it.

## Controls

On a screen:

| Key or action | Effect |
|---|---|
| drag, or W A S D | look around |
| wheel | zoom |
| left and right arrows | previous and next station |
| space (Shift+space) | next (previous) view |
| V | next terrain |
| B | switch between terrain stereo and terrain full 3D, in standard color |
| R | rover on or off |
| L | next eye: left, right, and center where the station has it |
| N, E, S, W | face that way |
| O | face the front of the rover |
| I | information and a map |
| H | home (the welcome view) |
| P | the screen's eye movement on or off |

In a headset:

| Control | Effect |
|---|---|
| right trigger, left trigger | next and previous station |
| thumbstick left or right | turn |
| A | next view |
| B | switch between terrain stereo and terrain full 3D, in standard color |
| X | information and a map, on the left hand |
| Y | home |
| right grip | next terrain |
| left grip | rover on or off |

The welcome view lists the same controls.

## Links

A link can name one station, variant and view:

```text
https://ctate7163.github.io/rpv/#Sol0052_S003D2046/rover/enhanced
```

`?experimental=1` before the `#` shows features that are still being tried, such as the simulated rover drive.

## Where the files are

- The page is part of the RPP package (`rpp.viewer`: `index.html` and `vendor/`).
- The public site is the repository [ctate7163/rpv](https://github.com/ctate7163/rpv), which holds the page and the
  list of stations, `panos.json`.
- The pictures and 3D models are in the Hugging Face dataset
  [ctate7163/rpp-rpv](https://huggingface.co/datasets/ctate7163/rpp-rpv). Each station has a `viewer/` folder (what
  the page loads) and a `record/` folder (the 16-bit panoramas, the depth from the viewpoint, and the JSON records
  of how each was made).

## Your own copy

RPV can show your own stations from a folder on your computer. Install the station extra
(`pip install "rover-photogrammetry[pipeline,stations]"`), align a block, then:

```bash
rpp single-sites STATION --root BLOCKS_ROOT --publish my_viewer
python -m http.server --directory my_viewer
```

Open <http://localhost:8000>. `rpp single-sites --list` prints the stations RPP knows, and
[Station products](station-products.md) explains each step and its options. The notebook
[`van_zyl_ods_example.ipynb`](https://github.com/ctate7163/rpp/blob/main/notebooks/van_zyl_ods_example.ipynb) does
all of this for one station, starting from the PDS download.

The page can also read a list from another address: `?index=URL`, where the URL is on the same origin or on a host
that allows cross-origin requests.

## Credits

Made by Christian Tate at Malin Space Science Systems with RPP. Image credits: NASA/JPL-Caltech/MSSS (Navcam);
NASA/JPL-Caltech/MSSS/ASU (Mastcam-Z). The rover model is derived from NASA's "Mars Perseverance Rover, 3D Model"
(NASA/JPL-Caltech).
