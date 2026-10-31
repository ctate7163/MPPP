# Roadmap

This page lists what is being worked on and planned after 0.95. Release notes are in the
[changelog](https://github.com/ctate7163/MPPP/blob/main/CHANGELOG.md).

## Now: the 0.95 freeze

The camera models and the alignment code are frozen, and every block is run again from scratch with them:

- the Navcam tied joint;
- the Mastcam-Z joints per zoom: 34, 48, 63, 79 and 110 mm;
- the Hazcam temperature law;
- the combined Navcam + Mastcam-Z 34 mm check.

The results of that run are the basis of the papers. The rover mask model trained on the v8 mask set
(`mppp_mask_v4`, see [Rover masks](masks.md)) is prepared for release on Hugging Face.

## Next: stereo panoramas from the block's own images

The goal is an 8K omnidirectional-stereo (ODS) panorama of one rover station. The panorama is projected only from
that station's images, keeps the alignment of the whole block, and is centred on the mast's centre of rotation,
which is fitted from the station's own poses. Navcam comes first, then Mastcam-Z 34 mm. The steps:

1. **Navcam-only dense subset** (`mppp dense --navcam-only --station`). Navcam patch-match stereo at half
   resolution, with references near the station and sources from the whole block.
2. **Mesh, cleanup and a far-field sphere** (`mppp mesh`):
   - a Delaunay mesh of the fused points;
   - floaters, rubber sheets across occlusion edges and unsupported surface removed;
   - terrain clipped at about 60 m;
   - a sphere at about 100 m, beyond which the stereo disparity of an 8K panorama is under one pixel.
3. **Per-image depth and normal maps** (`mppp depthmaps`). Each map is ray-cast from the mesh through the image's
   exact camera model (THIN_PRISM_FISHEYE, FULL_OPENCV), so the depth maps of all images agree on one surface. The
   Mastcam-Z frames, which are not used for the mesh, give an independent check.
4. **Rendering** (`mppp ods`). For each pixel, the best source image is chosen, seams are feathered, and holes are
   left black and flagged with their provenance.

The normal maps also feed the photometric angle maps: incidence, emission and phase per pixel.

## Later

- A browser viewer for the blocks: meshes, splats and station panoramas; layers for instrument, time of day and
  uncertainty; A→B measurement with its uncertainty; WebXR.
- More Mastcam-Z zooms in the calibrated scope.
