# Roadmap

What is being worked on and planned. Release notes are in the
[changelog](https://github.com/ctate7163/MPPP/blob/main/CHANGELOG.md).

## Now: the freeze run

The camera models and the alignment rules are frozen, and every block is aligned again from scratch with them:

- the Navcam tied joint and its stereo rig (drift with sol, a knee at sol 250);
- the Mastcam-Z joints per zoom: 34, 48, 63, 79 and 110 mm, with their two focus-backlash states;
- the Hazcam fit;
- the combined Navcam + Mastcam-Z 34 mm check, and the deck-fiducial adjustment that gives the rigs and the scale.

The results of that run are the basis of the papers, and its models replace the ones shipped with the package.

## Now: station panoramas

The station products are in place (see [Station products and the stereo panorama viewer](station-products.md)): a
cleaned mesh with the rover model, per-image depth and photometric maps, a 16K stereo panorama, and a prototype web
viewer. Open:

- the viewer checked in a headset; tiled 16K / 32K images and zoom;
- Mastcam-Z 34 mm frames in the panorama (they need the combined Navcam + Mastcam-Z block);
- the rover model following the arm, the antenna and the suspension from the image labels
  (see [Rover model](rover-model.md));
- rendering the two variants in parallel.

## Later

- A browser viewer for whole blocks: meshes, splats and station panoramas; layers for instrument, time of day and
  uncertainty; A to B measurement with its uncertainty.
- Independent checks of the camera models: the Mastcam-Z calibration target as a scale monitor, Sun images for
  heading, night-sky frames.
