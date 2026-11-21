# RPV: Rover Photogrammetry Viewer

**Stand where a Mars rover stood.** RPV shows 3D reconstructions of Mars 2020 Perseverance rover stations in a web
browser or a VR headset.

**Open it: <https://ctate7163.github.io/rpv/>**

Each station is a place where the rover took a 360° panorama. RPV shows it as a stereo panorama, or as a 3D terrain
that you can look around in, with or without the rover. The terrain is in true scale: it is reconstructed from the
rover's own images, taken at this stop and at nearby stops.

## Controls

On a screen: drag to look, wheel to zoom, the arrow keys for the next station, space for the next color view, `V`
for the next terrain, `R` for the rover, `I` for information and a map, `H` for home.

In a headset (open the link in the headset's browser and press *Enter VR*): triggers for the next and previous
station, thumbstick to turn, `A` for the next view, `B` to switch between the stereo panorama and the full 3D
terrain, `X` for information, `Y` for home.

The welcome view lists all controls. The user guide is at <https://ctate7163.github.io/rpp/viewer/>.

## What is in this repository

- `index.html` and `vendor/`: the viewer, one static page (three.js, WebXR).
- `panos.json`: the list of stations.

The pictures and 3D models are in the Hugging Face dataset
[ctate7163/rpp-rpv](https://huggingface.co/datasets/ctate7163/rpp-rpv).

This repository is written by the Rover Photogrammetry Pipeline (RPP), which makes the station products and
publishes them here. To change the viewer, or to make your own stations, see
[github.com/ctate7163/rpp](https://github.com/ctate7163/rpp). Questions and bug reports go to
[its issue tracker](https://github.com/ctate7163/rpp/issues).

## Status

RPV is experimental. The stereo panoramas are the most tested view; the 3D terrains are newer. The colors named
"enhanced" are display transforms, not calibrated color. See
[what is validated and what is experimental](https://ctate7163.github.io/rpp/status/).

## Credits

Made by Christian Tate at Malin Space Science Systems with RPP. Image credits: NASA/JPL-Caltech/MSSS (Navcam);
NASA/JPL-Caltech/MSSS/ASU (Mastcam-Z). The rover model is derived from NASA's "Mars Perseverance Rover, 3D Model"
(NASA/JPL-Caltech). The viewer includes three.js (MIT license).
