# Roadmap

What is being worked on and planned. Release notes are in the
[changelog](https://github.com/ctate7163/rpp/blob/main/CHANGELOG.md).

## Now: the freeze run

The camera models and the alignment rules are frozen, and every block is aligned again from scratch with them:

- the Navcam tied joint and its stereo rig (drift with sol, a knee at sol 250);
- the Mastcam-Z joints per zoom: 34, 48, 63, 79 and 110 mm, with their two focus-backlash states;
- the Hazcam fit;
- the combined Navcam + Mastcam-Z 34 mm check, and the deck-fiducial adjustment that gives the rigs and the scale.

The results of that run are the basis of the papers, and its models replace the ones shipped with the package.

## Now: station panoramas

The station products are in place (see [Station products and the stereo panorama viewer](station-products.md)): a
cleaned mesh with the rover model, per-image depth and photometric maps, a stereo panorama (8K, or 16K on request), and a prototype web
viewer. Open:

- the viewer checked in a headset; tiled 16K / 32K images and zoom;
- Mastcam-Z 34 mm frames in the panorama (they need the combined Navcam + Mastcam-Z block);
- the rover model following the arm, the antenna and the suspension from the image labels
  (see [Rover model](rover-model.md));
- rendering the two variants in parallel.

## Next: beyond the stereo 360 - a station in six degrees of freedom

The goal is one rover station that a viewer can move around in. The terrain within about 5 m is a textured mesh under
the 3D rover, and the terrain from 5 to 75 m is a coarse mesh. The stereo 360 carries only the sky and the far field.
RPV runs it on a Quest 3 at 72 frames per second.

Why the mesh goes out to 75 m: a mesh that fades into the 360 at 5 m is out of step with it by about 40 display pixels
when the head moves 15 cm. At 75 m the step is about 3 pixels. The 5-75 m ring costs few triangles, because the
tolerated depth error grows with the square of the range.

1. **RPV groundwork.**
   - three.js r180 or later (Spark needs it), and the KTX2 loader.
   - A performance readout in the headset: frame rate, triangles, draw calls, texture memory.
   - KTX2 textures for the panoramas.
2. **The mesh, textured from the pivot panorama.**
   - The station mesh, cropped and simplified by screen-space error: about 0.3-0.5 M triangles within 5 m and about
     50 k from 5 to 75 m.
   - Textured in the shader from the pivot's 16K panorama and its range, as the 3D rover is.
   - A fourth viewing mode.
3. **A texture atlas baked from the frames.**
   - A UV atlas (xatlas). Each texel is colored by the panorama's blending, with provenance and no-data kept.
   - An 8K KTX2 atlas: about 85 MB on the GPU.
   - Mastcam-Z tiles where they cover the near field, for leaning in.
4. **Gaussian splats, as an experiment.**
   - Nerfstudio splatfacto from the block's export, with the tight masks and the ray-cast depth maps as supervision.
   - Cropped to 10 m, at most 0.5-1 M splats, stored as SPZ, and drawn with Spark in RPV.
   - Checked in the headset in the same way as the mesh.
5. **Whole blocks:** tiles and streaming (the browser viewer below).

## Later

- A browser viewer for whole blocks: meshes, splats and station panoramas; layers for instrument, time of day and
  uncertainty; A to B measurement with its uncertainty.
- Independent checks of the camera models: the Mastcam-Z calibration target as a scale monitor, Sun images for
  heading, night-sky frames.
