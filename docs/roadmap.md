# Roadmap

Where RPP and RPV are going. The [changelog](https://github.com/ctate7163/rpp/blob/main/CHANGELOG.md) says what
changed in each release, and [What is validated and what is experimental](status.md) says where each part stands
today.

The goal of the 1.x series is that someone outside the Mars 2020 team can install RPP, make a true-scale model of a
place on Mars, trust it, and show it to others.

## 1.0: the validated pipeline, an experimental viewer

- The pipeline from PDS products to an aligned, true-scale block, on PyPI as `rover-photogrammetry`.
- Camera, rig and mast models from the third freeze run, in which every Mars 2020 block is aligned again with one
  version of the code and the models. The model forms are close to final.
- The rover mask model `rpp_mask_v5` as the default.
- Station products and RPV as experimental parts.
- A citable release: a DOI for the software and a software paper.

## 1.x: make it easy to use

**Install and first run**

- Tests on Linux and Windows for every change.
- A first run that needs no settings: download, process, align, look.
- Clearer messages when a download, a GPU or a model is missing.

**Trust**

- Published validation against the mission's own terrain meshes and against orbital (HiRISE) terrain models.
- The independent checks of the camera models, with their numbers, in the documentation.
- An uncertainty for every measurement: the reconstruction-error maps as a product.

**Viewer (RPV)**

- **Faster loading of the 3D stations** (first). A station's full 3D terrain is a 41-48 MB model with an 8192² JPEG
  texture inside, and each color view adds a 10-27 MB JPEG; on the GPU one 8192² texture takes about 340 MB, and
  decoding it stalls a headset for seconds. In order:
  1. GPU-compressed textures: KTX2 with Basis Universal (UASTC for the terrain color, ETC1S tried for the color
     views), the mipmaps in the file, the JPEG kept as a fallback. three.js transcodes them to ASTC on a Quest and
     to BC7 on a desktop; they stay compressed on the GPU.
  2. The terrain texture out of the model file; the terrain, the far shell and the rover fetched in parallel; a
     loading indicator in VR as well as on a screen.
  3. Compressed geometry (meshopt with quantized positions and texture coordinates).
  4. A small texture first, the full one after it; the next station loaded in the background.
- Measurement: distance, height, slope, and the dip and strike of a plane, each with its uncertainty.
- Click to source: pick a pixel and open the PDS product it came from.
- Information per station: which cameras it is made from, and their local times.
- A map of the traverse to choose a station from.

**Data**

- Downloads for every station: the mesh, the terrain model, the panoramas and their records, as a versioned data
  release with a DOI.
- A catalog of the end-of-drive stations along the traverse.

**More cameras and rovers**

- Curiosity: the Navcam and Hazcam models from more of the mission, then the Mastcam (34 and 100 mm).
- Mastcam-Z at 26, 48, 63, 79 and 100 mm from more blocks.
- The mast as a versioned model file next to the camera models, so that a camera pose can be built from the mast's
  angles alone.

## Later

- Appearance: separate the material of the terrain from the lighting, so that images taken at different times of
  day agree, and a model can be shown under another sun.
- A viewer for whole blocks: meshes and station panoramas together, with layers for instrument, time of day and
  uncertainty.
- Overlays for science: named targets, sample sites, multispectral maps on the 3D terrain.
- Other missions with mast stereo cameras.

## How to help

Tell us what you tried and where it stopped: the [issue tracker](https://github.com/ctate7163/rpp/issues) is the
place. Requests from people who want to use RPP for their own science decide the order of the list above.
