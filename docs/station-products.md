# Station products and the stereo panorama viewer

From a finished block RPP makes the products of one **station** (one stop of the rover): a cleaned mesh, per-image
depth and photometric maps, and an omnidirectional stereo panorama that can be looked at in a headset. The code is
`rpp.ibr` (the `[ibr]` extra: open3d). The commands and every option are in [Command line](cli.md).

**See the result:** the [stereo panorama viewer](ods/index.html){target="_blank" rel="noopener"} (a prototype, below).

## The steps

| step | command | what it makes |
|---|---|---|
| dense | `rpp dense WORK --station S` | COLMAP patch-match stereo and fusion for the station's frames (`fused.ply`); cached, so a second run costs nothing |
| mesh | `rpp mesh WORK --station S` | `mesh.ply`: the terrain out to 75 m, a ground plane beyond it to 200 m (300 m in `rpp ods`), the rover model |
| depthmaps | `rpp depthmaps WORK --station S --photometry` | per image: depth, range, normal, incidence, emission, phase, cast shadow (also from the rover's mast, booms and camera head) |
| ods | `rpp ods WORK --station S` | the stereo panorama and its companions |

`rpp single-sites` runs all four for the stations listed in `src/rpp/data/single_sites.json`, each in two variants:
**with the rover** (the rover model in the mesh, the images unmasked) and **terrain only** (the rover masked out and
the ground under it filled in). The second variant reuses the first one's terrain mesh. A step whose command, code
and inputs have not changed since its last good run is skipped (`--force` runs everything);
`single_sites_status.json` says which station and step is running; `single_sites_summary.md` lists the results and
the minutes per step. On Windows: `scripts\windows\run_single_site_ibr.bat` (set `RPP_NO_PAUSE=1` to run it
unattended).

## The mesh

- The terrain is a Poisson surface over the fused points within 75 m of the mast axis (`--mesh-radius`), cleaned of
  small components, of surface no point supports, and of surface floating above the ground.
- Beyond it a ground plane fitted to the far points and tie points, out to 200 m (`--plane-radius`). The plane is
  the regional slope, so it stands in for terrain only from 10 m of the mast axis on. Nearer, the mesh is made
  whole from itself: cells without surface (under the rover, behind rocks) are interpolated from the cells around
  them, and a second sheet of that interpolation lies 0.2 m under the terrain to catch a ray through a small gap.
- With the rover shown, the [rover model](rover-model.md) is added at the station's rover pose.
- The sky is a sphere of 300 m radius (`rpp ods --sky-radius`), and the ground plane reaches to it. `rpp ods`
  refines what lies beyond the plane per direction: see "The far field" below.

## The per-image maps

`maps/<image>.npz` holds depth (float32, m), range (float16, m), the surface normal in the camera frame (int8 / 127)
and a label per pixel; `photometry/<image>.npz` holds incidence, emission and phase (float16, degrees) and the cast
shadow (0 unknown, 1 lit, 2 shadow). **These files are the numbers.**

`rpp single-sites` makes the per-image maps for the terrain-only variant alone, and depth and normals only (since
0.97.14): they were a fifth to a half of a station's run. In the station list `"photometry": true` adds the
photometric maps and `"depthmaps_variants": ["rover", "masked"]` makes them with the rover shown as well. The
panoramas do not need the per-image maps.

The shadows are cast by the station mesh and by the rover hardware that is not in it: the mast and its two booms,
and the camera head turned to where it was when the image was taken (see [Rover model](rover-model.md)). With the
terrain-only mesh the whole rover model casts shadows. A surface the Sun meets at more than 86 degrees is shadow
too. The mask is sharp: it is what the rays say, pixel by pixel, with no closing and no growing. The shadow ray
starts 1 cm off the surface along the surface normal, so a surface the Sun only grazes does not shadow itself.

`--add-cameras F` adds the station's front Hazcam pair when it was taken within 2 hours of the Navcam frames (on
in the station list); for its shadows
the camera head is where the Navcam frame nearest in time had it.

`photometry_jpg/` holds a picture of each map for looking at. Their color scales are constants, the same for every
image and run:

| picture | quantity | scale |
|---|---|---|
| `_depth_linear.jpg` | range from the camera center | 0 to 75 m, linear, color map TURBO |
| `_depth_log.jpg` | the same | 0.5 to 200 m, log10, TURBO |
| `_normal.jpg` | surface normal, camera frame (x right, y down, z forward) | n = 2 RGB / 255 - 1 |
| `_incidence.jpg`, `_emission.jpg` | angle of the normal to the Sun, to the camera | 0 to 90 degrees, linear, VIRIDIS |
| `_phase.jpg` | angle Sun - surface - camera | 0 to 180 degrees, linear, VIRIDIS |
| `_shadow.jpg` | lit / shadow / unknown | three fixed colors |
| all but the shadow picture | the sky | white |
| `_rad.jpg` | the 8-bit picture of the frame | a view, not radiance |
| `_image.jpg` | the block's frame as it is, at its own size (0.97.20) | a copy for comparing with the maps |
| depth and normal pictures | masked pixels (the rover, with the rover masked), no surface, the frame's black padding | black |

The panorama step also writes the model alone through the panorama's own rays (0.97.20, for checking the model):
`<station>_range_log_360_3DTD.jpg`, `<station>_range_linear_360_3DTD.jpg` and `<station>_normal_360_3DTD.jpg`, the
left eye on top as in the color panoramas. Range is from each eye's ray origin in the colors of the table above;
the normal is in the station's frame, RGB = 127.5 (1 + [east, north, up]), so level ground is blue. No source frame
is used, so a fault seen there is in the mesh, the plane, the rover model or the far field and not in the blending.
`rpp ods --no-geometry` leaves them out. The viewer lists them as "distance map" (log), "distance map linear" and
"normal map" (0.98.30; "range log", "range linear" and "normal" before - those names still work in
`--publish-views`). The upload's default views (0.99.37) are standard, raw color, enhanced local, enhanced tone, dcs
color, normal map and distance map; the page shows standard as "standard color" and enhanced local as "enhanced color". The normal
map shows only the surface meshed from the cloud and the rover model: the fill under the rover, the apron to the
ground plane's reach, the deck floors and the plane are black (a mesh made before 0.98.30 needs the mesh step again
for this). The eyes are rendered at 16384 x 8192 or 8192 x 4096 (`rpp ods --width`); publishing reduces them to 8192.
Without `--width` (0.99.40) the width comes from the station's own frames: 16384 for a Mastcam-Z station and where
the median of their pixel angles (1 / focal length in pixels, at the scale they are read) is finer than 0.5 mrad -
full-resolution Navcam (0.33 mrad) - and 8192 otherwise: 2x-binned Navcam (0.65 mrad) is resolved by an
8192-wide panorama (0.77 mrad a pixel). `ods.json` records the rule under `width_rule`.

`run_single_site_ibr.bat <station>` publishes and uploads each station when it is done (0.97.24: by default;
`--no-upload` skips it; `--publish-only` does it without rendering; `--all` takes the experimental stations too).
Since 0.99.3 the pictures go to the public Hugging Face dataset `ctate7163/rpp-rpv` (`RPP_HF_DATASET`), one
commit per station (`rpp.ibr.hf`):

- `<station>/viewer/`: what the viewer loads - the 8K stereo JPEGs of every view and variant, the nadir mark, the
  rover model;
- `<station>/record/` (and `record/with_rover/`): the 16-bit stacked panoramas, the pivot's 16-bit depth, `ods.json`,
  `station.json`, the provenance picture and the tone levels (`--no-hf-record`: the viewer's pictures only).

`panos.json` stays in the viewer's GitHub checkout (`D:\code\RPV`, published at https://ctate7163.github.io/rpv/),
committed and pushed as before; its addresses point at the station's commit on the dataset
(`https://huggingface.co/datasets/ctate7163/rpp-rpv/resolve/<commit>/<station>/viewer/<file>`), so a link
never changes under a browser's cache. Pictures an earlier upload left in the checkout are removed once the station
is on the dataset. A station whose upload fails keeps its entry of before; its pictures stay in
`<output root>/_hf_stage/panos` and go up with the next run. `rpp hf-check [--create]` makes the dataset and its card
(licence `RPP_HF_LICENSE`, default `cc-by-4.0`) and checks that it serves the viewer's origin (CORS: the page fetches
the pictures). `--host github` puts the pictures into the checkout as before 0.99.3; a push that fails is tried three
times, the commit is kept and goes up with the next run, and `upload_pending.json` says what did not go through.

The mark at the nadir is a light blue disc 10 degrees from the nadir with two rows of text and a compass whose solid
point is the panorama's north (no letters: they would be sideways or upside down for most viewers). `<station>_nadir_mark.png` is the same mark seen from above; the viewer page
lays it on the ground as a real surface (`ground_mark` in `panos.json`: its depth below the eyes and its radius), so
in a headset it has the disparity of a thing on the ground.

Each picture carries its scale as JSON in its XMP metadata (`rpp:Scale`), `photometry_jpg/scales.json` lists them
all, and `rpp.ibr.scales.decode(path)` returns the values as floats with their units. Read back from a JPEG, a
value is good to about 0.2 degrees (incidence, emission), 0.4 degrees (phase), 0.15 m (linear range), 1 % (log
range) and 0.5 degrees (normal) on smooth surfaces, and worse at edges; for photometry use the `.npz` files.

## The panorama

Each eye is an equirectangular image, 8192 x 4096 or 16384 x 8192 (see above; `--width` to choose): the left edge looks north and the
azimuth grows eastward, so it reads N, E, S, W from left to right; the top row is the zenith. Every pixel's ray is
cast into the mesh and colored from the station frames that see that point best, blended in two frequency bands.

Frames of other cameras can fill in what the panorama's own frames do not see: `--fill-cameras F` adds the front
Hazcam pair when it was taken within 2 hours of them, which colors the ground under the arm. A fill frame has weight only where no regular
frame sees the surface, and its colors are matched to the regular frames first, per channel, on the surface both
see (`ods.json`, `fill_colour_match`). The regular frames are not changed.

**Context frames** (`"context"` in the station list, `rpp ods --context-images`) are frames of the block taken
from another place, or from this place on another sol. They add to the terrain in the dense step, and in the
panorama they are fill-only sources like the Hazcam pair, with three differences: their rover mask is applied in
both variants (the rover in them is somewhere else), they never color the rover model, and each is color-matched
to the panorama on its own, whenever it was taken.

On near surfaces - the rover, and the ground beside it - a frame is weighted more steeply by how little parallax it
has against the panorama's ray. There a wrong depth in the mesh moves the picture by (the camera's distance from
the ray) x (the depth error) / range squared: at 1.2 m, 20 cm and 5 cm make 23 pixels of a 16K panorama, and the
frame nearest the ray shows what the panorama should show whatever the model's shape. The weight of a frame is its
feather (wide for the color band, narrow for the detail band) x the parallax term to a power: 1 (color) and 3
(detail) on the terrain, rising gradually between 4 m and 2 m to 2 and 4. The color of a near surface is a blend of
the frames that see it; its detail comes mostly from the frame nearest the ray.

### The far field

Beyond the ground plane lie hills from a hundred meters to kilometers, and the sky. The frames' cameras are up to
0.4 m apart, so where two frames overlap, a surface drawn at the wrong range is drawn twice (0.4 m at 100 m against
infinity: 5 pixels of an 8K panorama). Before the eyes are rendered, every far direction is tried at 100 m, at the
sky sphere's 300 m and at infinity; the radius at which the overlapping frames agree best in their detail is the
sharpest. The result is one smooth field of 1 / radius per direction, not three shells: a soft minimum over the
three, filled from the decided directions around where the frames do not tell (one frame only, plain sky), smoothed
over 1.5 degrees, and equal to the sky sphere's radius at the ground plane's rim, so the plane and the far field
meet without a step. `ods.json` records it (`far_field`: the share decided, the shares nearest each radius),
`far_field.npz` holds the field, and the depth products show it. `rpp ods --no-far-field` keeps the one sphere.

Files (the prefix is the station's code, with `_rover` for the with-rover variant). The code is
`Sol<sol, 4 digits>_S<site, 3>D<drive, 4>`, for example `Sol0052_S003D2046`: the first sol of the frames the
panorama is made from, then the rover's site and drive counters. The station's dense workspace in the block keeps
the place alone (`dense_S003D2046`), and `--station` takes either form.

| file | content |
|---|---|
| `<prefix>_360_3DTD.png`, `.jpg` | both eyes in one square, **the left eye on top** (8-bit) |
| `<prefix>_360_3DTD_16.png` | the same in 16-bit linear radiance units: the record for color work |
| `<prefix>_enh_360_3DTD.jpg`, `_enhlocal_360_3DTD.jpg` | contrast-enhanced views, made from the 16-bit panoramas (terrain stretched; the sky smoothed; nothing clips): `_enhlocal` is the viewer's "enhanced color" |
| `<prefix>_raw_360_3DTD.jpg` | raw color (0.99.37; RPV calls it "original color" from 0.99.39): the standard view of the 16-bit values with the white balance divided out (`ods.json` linear16.white_balance) - the color of the RAD products |
| `<prefix>_enhtone_360_3DTD.jpg` | enhanced tone (0.99.37): base / detail tone mapping - the local brightness (a guided filter, 1 deg) compressed, the detail x 1.8, each pixel's color ratios kept |
| `<prefix>_dcs_360_3DTD.jpg` | dcs color (0.99.37; RPV calls it "decorrelation stretch" from 0.99.39): a decorrelation stretch fitted to the sunlit terrain; the sky white, the rover and its shadow black, the nadir disc as in the standard view |
| `<prefix>_color_products.json` | the parameters of the three (`rpp.ibr.color_products`; one set per station, from the left eye, in `tone_levels.json`) |
| `<prefix>_ods_sky_weight.png` | small map of the sky above the skyline: where the enhanced views smooth |
| `<prefix>_ods_left`, `_right` `.png` / `_16.png` | the eyes apart (`--eyes left right center` adds the view from the pivot, a third more time) |
| `<prefix>_ods_centre_depth.npz`, `_depth_16.png`, `_depth_*.jpg` | the range from the pivot for every pixel; from rays alone, seconds |
| `provenance.png`, `ods.json` | which frames colored what, their colors mixed by their weights; every setting and the coverage |

The PNG files are the data record; the JPEGs are for viewing. `_360_3DTD` in a file name is the convention headset
viewers read (360 degrees, 3D, top-down). The stacked files also carry XMP metadata (Google Photo Sphere
`GPano`, `GSpherical:StereoMode = top-bottom`, and `rpp:` fields); there is no single standard for stereo stills,
and a viewer that reads only `GPano` shows the square as one mono sphere.

### The eye separation

A strict omnidirectional stereo pair is right only at the horizon: toward a pole the columns' eye offsets turn
with the azimuth, and a viewer looking down sees them misaligned. Since 0.97.13 the eye separation is

    64 mm x cos(elevation)

above and below the horizon, and nothing else scales it: no limit on near surfaces, no fade with range (0.97.5
to 0.97.12 had both, and the cosine only near the nadir).

What the cosine does: with a share s of the separation a point at distance d is seen at d / s. Everything along
one line of sight is seen 1 / cos(elevation) farther - the rover and the terrain behind it alike, so their depths
stay in proportion. Flat ground h below the eyes is seen h / cos(elevation) below them: 1.15 h at -30 degrees,
1.41 h at -45, 2 h at -60. The ground appears to dip toward the rover. `rpp ods --pole-merge START END` and
`--pole-merge-down START END` set where the fall starts and ends on each side (`0 90` is the cosine; `90 90`
none); the constants are at the top of `rpp/ibr/ods.py`, and `ods.json` records what was used.

### The mark at the nadir

Within 20 degrees of straight down the panorama shows no terrain worth keeping (the mast, or filled-in ground). A
light blue disc there carries two rows of black text near its rim, each over at most half the circle and centered on
the rover's front: "Mars 2020 Perseverance Rover" with the sol and the site (with Mastcam-Z frames in, only the
first sol of those: "Sol 690 - Three Forks North", 0.99.36), and the image credit
(NASA/JPL-Caltech/MSSS for Navcam and Hazcam frames; NASA/JPL-Caltech/MSSS/ASU when Mastcam-Z frames are in)
followed by "made by C. Tate". In the flat image they are two straight lines along the bottom; in a headset, facing the rover's front and
looking down, they are two arcs ahead of you, upright. The letters are made about twice too wide in the flat image,
which leaves them about twice too narrow in the headset: the strip is stretched 3 to 4 times there, and no letter
can have its own shape in both. The first row is 1.6 times as tall as the second. The disc lies on the surface
under the mast - the deck with the rover shown, the ground without (the median depth below the pivot of what is
seen at the disc's rim, `depth_below_pivot_m`): each eye sees it with that surface's disparity, so it is at the
surface's distance and not at infinity. The disc is in every color panorama, not in the depth products; `ods.json`
records it (`nadir_mark`), and `rpp ods --no-nadir-mark` leaves it out.

### Sky and brightness

The sky no frame saw is filled from the sky that was seen. Frames that clipped on the bright sky are left out of the
brightness matching between frames, and dark patches in the sky smaller than 30 % of the panorama's width are lifted
to the sky around them. The 16-bit product gets one exposure per station; the 8-bit views share their scaling
between a station's two variants.

## The viewer

`docs/ods/index.html` is one static page (three.js, WebXR): it shows the `_360_3DTD` squares on a sphere per eye,
with menus for station, variant and view, a link per view (`#<station>/<variant>/<view>`), drag and wheel on a
screen and a choice of the eye shown there (left, right, or the center where there is one; the L key), the space bar for
the next view (0.99.37; Shift+space back), R for the rover on and off (as the rover button, 0.99.37), and in a headset: right / left trigger = next / previous station, grip = variant,
A = view, X = info and map, Y = home (the welcome view, facing west), B = terrain only, thumbstick = turn (0.98.28). On a screen: the house
button or H = home, B = terrain only, and the N E S W O cross at the upper right, level with the bar (O = level along
the rover's +x). On a screen the welcome view lists the keys (in a headset, the controller's buttons). A line at the bottom reports what
the device can show (maximum texture size, compressed-texture formats).

- On the documentation site: [https://ctate7163.github.io/rpv/](https://ctate7163.github.io/rpv/){target="_blank" rel="noopener"}
  (opens in a new tab from these pages; GitHub's own rendering of the README does not allow that).
- To add stations: `rpp single-sites Sol0052_S003D2046 --publish D:\code\RPV --upload --publish-only` puts the stereo JPEGs
  (reduced to 8192) on Hugging Face (`--host hf`, the default) and lists them in the viewer checkout's `panos.json`, then
  commits and pushes that checkout (`--host github`: the pictures into its `panos/`, as before 0.99.3). Since 0.99.10 the
  repository's own `docs/ods` holds the page and `vendor/` only: the panoramas and their list live in `ctate7163/rpv`.
- Another image list: `.../ods/?index=URL` (same origin, or a host that allows cross-origin requests).
- Since 0.99.36 the upload commits the page (`index.html`, `vendor/`) with the panoramas. Before it only `panos/` and
  `panos.json` were committed in `ctate7163/rpv`, so the live page stayed at the one of 17 Nov (0.99.13) while the
  stations went up. `rpp single-sites <station> --publish-only --upload` puts a new page up with one station.
- The menu and the info give a station's name from the station list (`single_sites.json`), not from its disc, so a
  station renamed since its panoramas were made shows the new name at the next publish of any station: the Navcam
  Three Forks North (sols 686, 688) had the disc's "Three Forks". A Mastcam-Z station has `(zcam)` after the name
  ("Three Forks North (zcam)", its first sol only); its disc has the name alone, as the Navcam one (0.99.13).
- The lower left shows two lines: the viewer's own version (`RPV viewer <version>`, the RPP release that wrote the
  page) above the station, its note, the variant, the size of the picture shown and the RPP version it was rendered
  with.
- **The center panorama on a screen (0.99.36).** Where a variant has the panorama from the pivot itself (the
  with-rover variant does when the rover surface is made: `rpp ods --eyes left right center`), the eye menu offers
  `center` after the left and right eyes (the L key goes left, right, center). The publish step writes it as
  `<prefix>_360_center.jpg` at 8192 x 4096, and as `<prefix>_360_center_16k.jpg` at 16384 x 8192 when the panorama is
  wider (`center` and `center_full` in `panos.json`). The page loads the 16K one where the GPU takes a 16384 texture,
  the device reports 8 GB or more and it is not a phone or a headset's browser; else, or when it fails, the 8K one.
  A headset shows the two eyes as before.

Status: a prototype. The fifth and sixth modes (0.99.32-0.99.33) have been loaded only in a headless Chrome, on a
synthetic station; nothing of 0.99.32-0.99.33 has been tried in a headset yet. Tiled 16K / 32K images are not done.

### The viewing modes

0.99.40: the page's menu chooses the terrain and the rover button (R; in a headset the left grip) shows or hides the
rover; `panos.json` keeps the variants below, and the two choices pick one of them:

| terrain (menu) | rover hidden | rover shown |
|---|---|---|
| terrain stereo | `masked` | `rover` (the with-rover pair; `rover3d` where a station has no `rover`) |
| terrain local 3D | `terrain3d`, its 3D rover hidden | `terrain3d` |
| terrain full 3D | `scape3d`, its 3D rover hidden | `scape3d` |

The full 3D terrain loads no panorama (its texture holds the whole view from the pivot): its views are its texture
(standard color) and the normal map. The local 3D terrain is the full 3D file's terrain (the same 20 m mesh and
texture, the shell to the horizon left out) in front of the terrain-only panorama, where the station has the full 3D
file (0.99.41); else the station mesh. Each part's texture is also kept in 16 bits (`<stem>_16.png`, the panoramas'
encoding), and `rpp ods --retone` makes the textures again from it with the panoramas' 8-bit conversion (0.99.41).
The full 3D texture also has the color skins of the panoramas - enhanced color, enhanced tone, original color,
decorrelation stretch (`<stem>_enhlocal.jpg`, `_enhtone`, `_raw`, `_dcs`), made from its 16-bit texture with the
station's parameters - and the page draws the normal and distance maps of the 3D modes from the surfaces themselves.
B switches terrain stereo and terrain full 3D (0.99.41). The drive (`drive3d`, the full 3D terrain and the rover) is the steering wheel
above the rover button, shown only with `?experimental=1`. A link to a 3D mode with the rover hidden ends in
`/no-rover`; the wheel carries a "Beta" tag, and in a headset a thumbstick click starts and ends the drive (the sticks'
forward / back: the tracks; left / right still turn the view). The directions in the sky show in the welcome view and
with the info page (I, X). V (headset: the right grip) steps through the terrain modes; a station change keeps the terrain and the
rover where the next station has them (else the full 3D terrain falls back to the local, and that to stereo).

| variant (before 0.99.40, the menu) | key | what is drawn | parallax |
|---|---|---|---|
| terrain only | `masked` | the rover-masked stereo pair on two spheres | binocular only (the ODS pair), none for head motion |
| terrain with rover (projected) | `rover` | the with-rover pair | as above |
| terrain with rover (3D model) | `rover3d` | the terrain-only pair + the textured Perseverance model | head motion on the rover |
| 3D terrain and rover | `terrain3d` | the terrain-only pair beyond 20 m + the station mesh (terrain within 20 m, rover; 10 m until 0.99.39) | head motion within 20 m |
| 3D terrain to the horizon and rover | `scape3d` | one surface from under the rover to the far field and a sky dome at 3 km, no panorama | head motion everywhere; true to the far field |
| Drive the rover (simulation) | `drive3d` | the `scape3d` file, the rover moved over the terrain | as `scape3d` |

The station mesh (`rpp.ibr.station_mesh`, made by the `ods` step with `--station-mesh`, the single-sites default) is
textured from a point 0.5 m above the pivot: one 8192 x 8192 texture whose center is straight down and whose radius
grows evenly with the angle from straight down, so every texel covers the same angle seen from there and the near
ground gets the most texels per square meter. Triangles that point cannot see get cells of their own in the corners,
colored from the frames that see them. The rover has its own 4096 x 4096 texture (the triangles seen from within 1 m
of the pivot; the rest the rover gray). In the fifth mode the same kind of texture runs on past the horizontal: the
terrain up to the skyline (at least 10 deg above the horizontal) fills 90 % of the disc's radius, the sky the rest. The
near terrain fades out between 18 and 20 m from the mast axis (in the fourth mode into the panorama, in the fifth into
the surface beyond; 0.99.39 - 8 to 10 m before, and the terrain simplified to 500,000 triangles, 300,000 before).
Stations made before 0.99.39 keep their 10 m mesh until they are rendered again.

On a screen the eye moves a little where a 3D surface is shown (0.99.32): it sits 0.15 m in front of the point the
view turns about (as eyes are in front of the neck) and follows the mouse by up to 5 cm (2 cm in 0.99.38; a slow sway
when left alone until 0.99.37). P turns it off; `?still=1` opens without it. In the 3D modes the normal-map view draws
the surfaces with their own normals, so the shading moves with the eye, and the rover is the projected rover surface
of the third mode (the pivot's panorama at its full resolution; the station mesh's own rover texture is 4096 texels a
side for the whole model) - its parts the pivot does not see are gray (0.99.38). In
the panorama-only modes (terrain only, terrain with the projected rover) the picture is a sphere around the eye, so
there is nothing for the eye to move against and the movement is off.

The sixth mode is a simulation: each thumbstick's forward / back axis is the track on that side (skid steering: speed =
their mean x 0.25 m/s, turn rate = their difference / 2.3 m; on a screen the arrow keys). The viewer rides at the mast
head: the ground is moved and turned under the rover, which takes the height, pitch and roll of the plane through the
terrain under its six wheels. It stops 15 m from where the rover stood (the textured terrain ends at 20 m; 6 m and
10 m until 0.99.39). In a
headset the rover is hidden while the head is more than 1 m from its pivot. The ground under the real rover is
interpolated (no frame saw it), and its shadow is in the texture.

### The mesh near the mast (0.99.32)

The 0-4 m zone is a height field seen from the mast (one median height per 1.5 cm cell of a horizontal grid; small
gaps closed), not a Poisson surface: it cannot fold over a hole's rim. The holes left (under and behind the rover) are
filled by a smooth sheet hung from the low side of the surface around them, 5 mm under it, so it meets a hole's rim
from below. The normal panorama shows that fill in its own normal's color at 45 % brightness.

0.99.38: a cell's height is the median of at least 3 points (a cell of two took the higher one: needles of 1-4 m at
2-4 m from the mast); a cell of fewer than 8 points more than 10 cm from the median of the 4x coarser grid is dropped;
empty cells take the heights of the 2x, 4x and 8x grids where those have points, so the thinner cloud beyond 2-3 m
leaves fewer holes to the fill; and where neighboring cells differ by more than 6 cm - the face of a rock, which a
height field can only draw as a wall - the cells about it are meshed by Poisson from the same points.

### Color for display (a study, 0.99.33)

`python -m rpp_studies.misc.color_enhance PANORAMA_16.png OUT NAME --depth CENTER_DEPTH.npz [--rover-depth ...]` makes
three fixed display transforms of the 16-bit linear panorama, with one parameter set per station written to a JSON:
A, base/detail (an edge-preserving split of log luminance, the large scale compressed, the detail multiplied, each
pixel's color ratios kept); B, Mantiuk et al. 2006 (contrast mapped through a model of visible contrast); C, a
decorrelation stretch of the terrain (the sky and the rover left out and shown as they are). None is in the panorama
step yet.

## The rover as a 3D model (0.98.14)

A station run also renders the with-rover panorama from the pivot and writes `with_rover/<prefix>_rover_surface.glb`:
the Perseverance model where `rpp mesh --rover` put it (all its triangles, in meters about the pivot), the rows of
that panorama that hold the rover, and their range. The viewer colors a point of the model with the panorama's pixel
in its direction when it is the surface the pivot sees there (its distance against the range image, with a bias of
12 mm + 1.2 %); what the pivot does not see was seen by no camera and is the rover gray. So the rover has no holes
from any viewpoint and its edges are the model's. `<prefix>_rover_surface.json` gives the share of the model's area
seen from the pivot. To build it again from a folder that already has the center panorama:

```
python -m rpp.ibr.rover_surface D:\rover_blocks\single_sites\Sol0052_S003D2046\with_rover
```

On a screen, `.../ods/?head=0.15,0,0` puts the eye 0.15 m east of the pivot (east, up, south, meters) to look at the
model's parallax. (0.98.2-0.98.13: a grid mesh cut from the panorama's depth, with a second layer behind its edges.)

## Experimental: a Mastcam-Z panorama on the Navcam terrain (0.98.2)

`python -m rpp single-sites Sol0690_S032D1184_Z34` renders the 34 mm frames of sol 690 at Three Forks onto the
terrain made from the station's Navcam frames. Stations marked `"experimental": true` in `single_sites.json` run only
when named.

### 0.98.4

- (The rover surface of 0.98.4-0.98.13 had two grid layers; replaced by the model in 0.98.14.)
- The sky's last step, `heal_sky`, replaces frame-shaped patches and bands by the sky's own smooth version.
- Viewer: right / left trigger = next / previous station, thumbstick = turn, A = site and sol, X = color view.
