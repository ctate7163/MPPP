# Station products and the stereo panorama viewer

From a finished block MPPP makes the products of one **station** (one stop of the rover): a cleaned mesh, per-image
depth and photometric maps, and an omnidirectional stereo panorama that can be looked at in a headset. The code is
`mppp.ibr` (the `[ibr]` extra: open3d). The commands and every option are in [Command line](cli.md).

**See the result:** the [stereo panorama viewer](ods/index.html) (a prototype, below).

## The steps

| step | command | what it makes |
|---|---|---|
| dense | `mppp dense WORK --station S` | COLMAP patch-match stereo and fusion for the station's frames (`fused.ply`); cached, so a second run costs nothing |
| mesh | `mppp mesh WORK --station S` | `mesh.ply`: the terrain out to 75 m, a ground plane beyond it to 200 m, the rover model |
| depthmaps | `mppp depthmaps WORK --station S --photometry` | per image: depth, range, normal, incidence, emission, phase, cast shadow (also from the rover's mast, booms and camera head) |
| ods | `mppp ods WORK --station S` | the stereo panorama and its companions |

`mppp single-sites` runs all four for the stations listed in `src/mppp/data/single_sites.json`, each in two variants:
**with the rover** (the rover model in the mesh, the images unmasked) and **terrain only** (the rover masked out and
the ground under it filled in). The second variant reuses the first one's terrain mesh. A step whose command, code
and inputs have not changed since its last good run is skipped (`--force` runs everything);
`single_sites_status.json` says which station and step is running; `single_sites_summary.md` lists the results and
the minutes per step. On Windows: `scripts\windows\run_single_site_ibr.bat` (set `MPPP_NO_PAUSE=1` to run it
unattended).

## The mesh

- The terrain is a Poisson surface over the fused points within 75 m of the mast axis (`--mesh-radius`), cleaned of
  small components, of surface no point supports, and of surface floating above the ground.
- Beyond it a ground plane fitted to the far points and tie points, out to 200 m (`--plane-radius`); it also lies
  under the mesh's holes, and the ground under the rover is filled from the terrain around it.
- With the rover shown, the [rover model](rover-model.md) is added at the station's rover pose.
- The sky is a sphere of 200 m radius (`mppp ods --sky-radius`).

## The per-image maps

`maps/<image>.npz` holds depth (float32, m), range (float16, m), the surface normal in the camera frame (int8 / 127)
and a label per pixel; `photometry/<image>.npz` holds incidence, emission and phase (float16, degrees) and the cast
shadow (0 unknown, 1 lit, 2 shadow). **These files are the numbers.**

The shadows are cast by the station mesh and by the rover hardware that is not in it: the mast and its two booms,
and the camera head turned to where it was when the image was taken (see [Rover model](rover-model.md)). With the
terrain-only mesh the whole rover model casts shadows. A surface the Sun meets at more than 89 degrees is shadow
too; gaps of a few pixels inside a shadow are closed, and the shadow is then grown by 3 pixels, so the stored map
errs on the side of shadow.

`--add-cameras F` adds the station's front Hazcam pair of the same sols (on in the station list); for its shadows
the camera head is where the Navcam frame nearest in time had it.

`photometry_jpg/` holds a picture of each map for looking at. Their colour scales are constants, the same for every
image and run:

| picture | quantity | scale |
|---|---|---|
| `_depth_linear.jpg` | range from the camera centre | 0 to 75 m, linear, colour map TURBO |
| `_depth_log.jpg` | the same | 0.5 to 200 m, log10, TURBO |
| `_normal.jpg` | surface normal, camera frame (x right, y down, z forward) | n = 2 RGB / 255 - 1 |
| `_incidence.jpg`, `_emission.jpg` | angle of the normal to the Sun, to the camera | 0 to 90 degrees, linear, VIRIDIS |
| `_phase.jpg` | angle Sun - surface - camera | 0 to 180 degrees, linear, VIRIDIS |
| `_shadow.jpg` | lit / shadow / unknown | three fixed colours |
| all but the shadow picture | the sky | white |
| `_rad.jpg` | the 8-bit picture of the frame | a view, not radiance |

Each picture carries its scale as JSON in its XMP metadata (`mppp:Scale`), `photometry_jpg/scales.json` lists them
all, and `mppp.ibr.scales.decode(path)` returns the values as floats with their units. Read back from a JPEG, a
value is good to about 0.2 degrees (incidence, emission), 0.4 degrees (phase), 0.15 m (linear range), 1 % (log
range) and 0.5 degrees (normal) on smooth surfaces, and worse at edges; for photometry use the `.npz` files.

## The panorama

Each eye is an equirectangular image, **16384 x 8192** by default (`--width`): the left edge looks north and the
azimuth grows eastward, so it reads N, E, S, W from left to right; the top row is the zenith. Every pixel's ray is
cast into the mesh and coloured from the station frames that see that point best, blended in two frequency bands.

Frames of other cameras can fill in what the panorama's own frames do not see: `--fill-cameras F` adds the front
Hazcam pair of the same sols, which colours the ground under the arm. A fill frame has weight only where no regular
frame sees the surface, and its colours are matched to the regular frames first, per channel, on the surface both
see (`ods.json`, `fill_colour_match`). The regular frames are not changed.

On surfaces nearer than 3 m - the rover, and the ground beside it - a frame is weighted by how little parallax it
has against the panorama's ray, much more steeply than on the terrain. There a wrong depth in the mesh moves the
picture by (the camera's distance from the ray) x (the depth error) / range squared: at 1.2 m, 20 cm and 5 cm
make 23 pixels of a 16K panorama. The frame nearest the ray shows what the panorama should show whatever the
model's shape.

Files (the prefix is the station, with `_rover` for the with-rover variant):

| file | content |
|---|---|
| `<prefix>_360_3DTD.png`, `.jpg` | both eyes in one square, **the left eye on top** (8-bit) |
| `<prefix>_360_3DTD_16.png` | the same in 16-bit linear radiance units: the record for colour work |
| `<prefix>_enh_360_3DTD.jpg`, `_enhlocal_360_3DTD.jpg` | contrast-enhanced views (terrain stretched; sky and deck may clip) |
| `<prefix>_ods_left`, `_right`, `_centre` `.png` / `_16.png` | the eyes apart, and the view from the pivot |
| `<prefix>_ods_centre_depth.npz`, `_depth_16.png`, `_depth_*.jpg` | the range from the pivot for every pixel of the centre view |
| `provenance.png`, `ods.json` | which frame coloured what; every setting and the coverage |

The PNG files are the data record; the JPEGs are for viewing. `_360_3DTD` in a file name is the convention headset
viewers read (360 degrees, 3D, top-down). The stacked files also carry XMP metadata (Google Photo Sphere
`GPano`, `GSpherical:StereoMode = top-bottom`, and `mppp:` fields); there is no single standard for stereo stills,
and a viewer that reads only `GPano` shows the square as one mono sphere.

### How the stereo is made comfortable

A strict omnidirectional stereo pair is right only at the horizon, so the eye separation (64 mm) is reduced where
it would hurt:

- **towards the poles:** full to 30 degrees above the horizon and to 60 degrees below it, then merged to one image
  at the zenith and the nadir;
- **on near surfaces:** a surface nearer than 2.5 m gets range / 2.5 of the separation, so no disparity exceeds
  1.5 degrees (the deck, 1 m below the cameras, would have 3.7). Terrain from 2.5 m on keeps its true depth;
- **far away:** the separation fades from 75 m to zero at 200 m, where the mesh, the plane and the sky meet.

Where the separation is reduced by a share s, a point at distance d is seen at d / s: the deck looks flatter and
farther than it is. These numbers are constants at the top of `mppp/ibr/ods.py` and are recorded in `ods.json`.

### The mark at the nadir

Within 10 degrees of straight down the panorama shows no terrain (the mast, or filled-in ground). A light blue
disc there carries, in black, one line that runs once around it near the rim: "Mars 2020 Perseverance Rover", the
sol and the site, "made by C. Tate" and the image credit (NASA/JPL for Navcam and Hazcam frames; NASA/JPL/MSSS/ASU
when Mastcam-Z frames are in). In the flat image it is a straight line along the bottom, starting at north; in a
headset it is a ring. The letters are made about 3 times too wide in the flat image, which leaves them about 3
times too narrow in the headset: the strip is stretched 7 to 9 times there, and no letter can have its own shape in
both. It is in every colour panorama, not in the depth products; `ods.json` records its rows (`nadir_mark`), and
`mppp ods --no-nadir-mark` leaves it out.

### Sky and brightness

The sky no frame saw is filled from the sky that was seen. Frames that clipped on the bright sky are left out of the
brightness matching between frames, and dark patches in the sky smaller than 30 % of the panorama's width are lifted
to the sky around them. The 16-bit product gets one exposure per station; the 8-bit views share their scaling
between a station's two variants.

## The viewer

`docs/ods/index.html` is one static page (three.js, WebXR): it shows the `_360_3DTD` squares on a sphere per eye,
with menus for station, variant and view, a link per view (`#<station>/<variant>/<view>`), drag and wheel on a
screen, and in a headset: trigger = next station, grip = variant, A or X = view. A line at the bottom reports what
the device can show (maximum texture size, compressed-texture formats).

- On the documentation site: <https://ctate7163.github.io/MPPP/ods/>.
- To add stations: `mppp single-sites S003D2046 --publish docs/ods --publish-only` copies the stereo JPEGs (reduced
  to 8192) into `docs/ods/panos/` and lists them in `docs/ods/panos.json`; commit and push `docs/ods`.
- Another image list: `.../ods/?index=URL` (same origin, or a host that allows cross-origin requests).

Status: a prototype. It has been checked in a desktop browser with a labelled test pair; the stereo in a headset,
tiled 16K / 32K images and zoom are not done.
