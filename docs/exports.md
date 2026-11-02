# Downstream exports: dense, Metashape, Nerfstudio

An aligned block (`<WORK>\colmap\`) can go on to three downstream tools. Each export reads the block's
**native-pixel model**, `gui_native/sparse`: the COLMAP model in each image's own pixels, which MPPP writes at the end
of every run. None of the exports re-aligns anything: the poses, cameras and tie points are the block's.

| export | command | folder | for |
|---|---|---|---|
| Dense (COLMAP MVS) | `mppp dense WORK` | `<project>\dense\` | depth and normal maps, a fused point cloud, a mesh |
| Metashape project | `mppp metashape WORK` | `<project>\metashape\` | Metashape's depth maps, meshes, textures, orthomosaics |
| Nerfstudio (splatfacto) | `mppp nerfstudio WORK` | `<project>\nerfstudio\` | 3D Gaussian splatting |

On Windows, `scripts\windows\ecam_block.bat SITE` aligns a site's Navcam + Hazcam block and writes the Metashape and
Nerfstudio exports in one go (`mppp run SITE --metashape --nerfstudio`).

## Dense reconstruction (`mppp dense`)

The COLMAP workspace is preloaded from the alignment. It holds:

- the images undistorted to pinhole cameras;
- the rover masks, undistorted the same way and applied in the fusion;
- `patch-match.cfg` and `fusion.cfg`.

The steps:

```bat
python -m mppp dense D:\rover_blocks\<site>\WORK                       & rem the workspace only
python -m mppp dense D:\rover_blocks\<site>\WORK --stereo --fusion     & rem patch-match stereo (CUDA) and fusion
python -m mppp dense D:\rover_blocks\<site>\WORK --maps png            & rem depth and normal maps as images
```

`dense_reconstruct.bat`, next to `open_in_colmap.bat` in every project, runs the same steps with the COLMAP
executable. COLMAP's GUI (Reconstruction > Dense reconstruction) also accepts the `dense\` folder as its workspace.
The working resolution is 2000 px on the long side by default (`--max-image-size`). See [mppp dense](cli.md#mppp-dense).

## Metashape project (`mppp metashape`)

This builds `<block>.psx` with the following:

- the photos, linked from the COLMAP project's `images\` (not copied);
- one sensor per COLMAP camera, with its calibration converted and fixed;
- every camera's COLMAP pose as its position and reference;
- the tie points, re-triangulated by Metashape from COLMAP's tracks;
- the masks;
- a local metric frame: MPPP's world frame (x east, y north, z up), with the block offset in the chunk transform.

Nothing is aligned or optimised in Metashape. Double-click `make_metashape_project.bat` (Metashape Professional)
and continue with *Build Depth Maps* or *Build Mesh*. `metashape_check.json` compares Metashape's reprojection of
the tie points with COLMAP's residuals. The calibration mapping (THIN_PRISM_FISHEYE → Fisheye,
FULL_OPENCV → Frame) is in the table under [mppp metashape](cli.md#mppp-metashape).

## Nerfstudio (`mppp nerfstudio`)

Every frame is resampled through COLMAP's exact camera model into one pinhole size, 2048 × 1536. The export holds:

- `images\` and `masks\`, plus the half-size `images_2\` and `masks_2\`;
- `transforms.json`, with per-frame intrinsics and OpenGL camera-to-world matrices in MPPP's world frame;
- `sparse_pc.ply`, the tie points, which splatfacto starts from.

Mastcam-Z frames lose 22 columns on the left and 16 on the right, the dark edge columns.

`train-ns.bat` in the folder starts a small splatfacto run tuned for these scenes:

- half-size images (`--downscale-factor 2`);
- starting from the sparse points (no random initialisation);
- no camera optimisation, since the poses are the block's;
- regular pruning of floaters (opacity resets, alpha and scale culling, scale regularisation);
- 15,000 iterations.

Extra options go to `ns-train` (`train-ns.bat --max-num-iterations 30000`). See
[mppp nerfstudio](cli.md#mppp-nerfstudio).

## What comes next

Image-based rendering of stereo panoramas (ODS) from the block's own images is planned: a Navcam-only dense subset,
a cleaned mesh with a far-field sphere, and per-image depth and normal maps ray-cast through each image's exact
camera model. See the [roadmap](roadmap.md).
