# Streamer analysis framework

A Python replacement for the parts of **Streamertools** (S. Nijdam, TU/e —
manual of 31 March 2022) needed for ICCD streamer image analysis, plus a
streamer-velocity measurement the original suite does not have.

```
Streamer_Analysis_V3.ipynb  the control center — this is what you open
Streamer_Analysis_V2.ipynb  previous version (pop-up pickers), kept for reference
Streamer_Analysis.ipynb     V1 (ipywidgets), kept for reference
calibrate.py                pixel size; run once per optical setup
check_install.py            "is the package actually loaded?" diagnostic
streamertools/              all functions and classes
tests/test_streamertools.py 109 self-tests on synthetic data (all passing)
```

Nothing in the notebook defines a function; it only drives the package. Edit
the package, keep `%autoreload 2` on, and the notebook picks it up.

## Install

```bash
pip install numpy scipy matplotlib pillow pandas sif-parser
pip install ipywidgets ipympl        # for the click-driven tools
pip install imageio imageio-ffmpeg   # only for mp4 output
```

Put `streamertools/` next to your notebook (the notebook already adds its own
directory to `sys.path`).

## What maps to what

| Manual | Implemented in | Notes |
|---|---|---|
| 2.1 file types | `io.py` | reads every type of the manual (table below); writes the two Streamertools count formats (2.1.3) and all colour formats. Add a reader to `io.READERS` and it works everywhere. |
| 2.2 colour representation | `palette.py`, `display.py` | `*.pal` files, eq. 2.1 and the log branch of eq. 2.2, AutoMin/AutoMax/histogram limits, the deprecated *Drawing style* numbers |
| 2.3 zooming | `geometry.zoom_out` | integer zoom-out by block averaging; measurements always use the original data |
| 2.4 pixel size | `calibration.py` | ruler shot, one square, a periodic row fit, or a full checkerboard via OpenCV |
| 2.6 movies | `tilemaker.save_movie`, converter batch | GIF/MP4 instead of AVI — no Windows codec needed |
| 3 Streamer Width Analyzer | `width.py`, `fwhm.py`, `geometry.py` | oblique grid (fig. 3.3) with bilinear interpolation (fig. 3.4); the three profile methods (eq. 3.1, per line, eq. 3.2); width and length (3.2.2); Savitzky-Golay smoothing (3.2.1); angles, area statistics (3.4); RGB frame overlay (3.5); profile and histogram plots (3.3) |
| A FWHM code | `fwhm.calc_fwhm` | line-by-line port of the Delphi routine, including the doubled bracket, the least-squares baseline and the sub-pixel edge interpolation |
| 5 Image Converter | `converter.py` | the seven actions in the manual's order, single-file preview and batch mode; text/arrow/overlay refused for count-based output, as in the original |
| 6 Tile maker | `tilemaker.py` | matrix images, borders and extra borders, comments (`/b /n /max /file /frame`), four arrow sections, overlay, outside row/column comments, groups with `ParentGroup`, stack modes 1–7, movie output; `read_tlmkr` / `write_tlmkr` for the original ini scripts |
| **new** | `velocity.py` | propagation velocity: the literature front method over a delay series, two images, frame to frame, detected or clicked |
| **new** | `velocity_window.py`, `thickness_window.py` | the two permanent analysis windows of the V3 notebook |
| **new** | `workbench.py` | what the two windows share: series, saving, crop, zoom, the Qt window |
| — | `preprocess.py` | dark/bias subtraction, Mf brightness normalisation, hot-pixel and blob filters |
| — | `interactive.py` | `Viewer`, `Cropper`, `LinePicker`, `HeadPicker` |

Chapter 4 (Stereo Streamers) is deliberately not implemented.

## The velocity measurement

Implements exactly this procedure:

> Two images with a very short camera exposure time (5 ns), taken when the
> streamers crossed half of the gap, with a difference in exposure delay of
> 10 ns. The velocity is the distance between the streamer head positions in
> the two images divided by that delay.

```python
v = st.measure_velocity(image_a, image_b, delay_ns=10.0, pixel_size_um=17.0,
                        axis="down", origin=(x_tip, y_tip),
                        delay_jitter_ns=0.5, pixel_size_rel_error=0.02)
v.velocity_m_per_s, v.sigma_velocity_m_per_s, v.axial_distance_px
```

* **Sources** — two files, two arrays, or `(stack, frame_index)` pairs, so two
  frames of one kinetic series work without any extra step.
* **Head position** — `method="front"` (default) is the literature
  definition, see below. `"tip"` (the pixel reaching furthest, refined with
  an intensity-weighted centroid), `"centroid"` and `"max"` remain available.
  `manual_heads=((xa,ya),(xb,yb))` bypasses detection.
* **Axis** — `"down"/"up"/"left"/"right"`, a vector, two points, or `None` to
  fit it from the image (give `origin` so the sign is unambiguous).
* **Branches** — `measure_velocity_branches` detects every channel in both
  images, matches them one-to-one by minimum total displacement, and returns
  one velocity per branch plus a DataFrame.
* **Series** — `velocity_from_fronts` is the literature method over a whole
  delay series (below); `velocity_from_series` is the same fit without the
  end-of-exposure time and frame selection.
* **Error bars** — head-localisation spread, timing jitter and pixel-size
  uncertainty are propagated into `sigma_velocity_m_per_s`. The 5 ns exposure
  does not enter the arithmetic but does smear the head; if that smear is
  comparable to a pixel, set `position_uncertainty_px` so the error bar is
  honest.

## The literature head and velocity

The automatic head follows Briels *et al* (J. Phys. D 41 234004, 2008),
Winands *et al* (J. Phys. D 41 234001, 2008) and Nijdam (PhD thesis, TU/e
2011, §3.4.3):

1. **The streamer that protruded furthest** into the gap is the one evaluated
   — it propagates closest to the camera's focal plane.
2. **Its head is a half-maximum edge.** Lengths along the propagation
   direction are measured with the same FWHM criterion as diameters. A strip
   across the channel (`front_strip_px`) is averaged into an axial profile;
   from the head maximum, the front is where the profile falls to
   `front_level` (0.5) of the way down to the local background ahead of it,
   interpolated. During a short gate the head is a stripe of light; this edge
   is where the head was at the *end* of the gate. Its uncertainty is the
   profile noise over the edge slope.
3. **The velocity is a slope**: head position against end-of-exposure time
   (delay + gate), fitted with a straight line over the frames where it is
   constant (the middle of the gap, `fit_frames`).

```python
r = st.velocity_from_fronts(stack, delay_ns=10.0, gate_ns=5.0, axis="down",
                            origin=(x_tip, y_tip), fit_frames=(15, 30))
r.velocity_px_per_ns, r.sigma_px_per_ns, r.velocity_m_per_s
r.track     # per frame: end-of-exposure time, head, distance from the tip, in_fit
r.pairs     # frame to frame, the same table as velocity_frame_to_frame
```

On synthetic channels whose end is a blurred step the front lands within
0.15 px of the step, and a series at 3 px/ns comes back as 3.000 px/ns.

## The two windows (V3)

Two Qt windows that stay open for the whole session, each driven by its own
notebook cell and never merged: the **velocity window** (tools view, crop,
head) and the **thickness window** (view, crop, line). The notebook stays on
`%matplotlib inline`: the windows are not pyplot figures but Qt main windows
with an embedded matplotlib canvas, run by the kernel's Qt event loop (the
hook `%gui qt` installs). Nothing switches backend, so nothing closes.

```python
SERIES = ["a.sif", "b.sif", "c.sif"]          # one file per delay series
PREP = dict(rot90=2, dark_from_frames=2)       # how each is loaded
vwin = st.open_velocity_window(SERIES, prepare=PREP, out_dir=OUT_DIR,
                               frame=17, tool="head", delay_ns=10.0,
                               gate_ns=5.0, axis="down")
twin = st.open_thickness_window(SERIES, prepare=PREP, out_dir=OUT_DIR,
                                tool="line", box_width=75)
```

Re-run a window's cell to change it: the first run opens it, later runs
update it in place, and a closed window comes back. Settings are applied on
every run; series, frame, crop and tool, which the window changes as well,
only when their value in the cell changed since its last run. A misspelt
keyword raises rather than being ignored.

**Series.** Both windows work through `SERIES` with three buttons:

| button | does |
|---|---|
| *Save mean velocity* / *Save mean thickness* (Ctrl+S) | one row per series in `velocity_summary.csv` / `thickness_summary.csv`; saving again replaces it |
| *Next series >* | moves on, asking first if the series changed since its mean was saved |
| *Save and close* | saves the series shown and closes the window |

Clicks are written to `<series>_clicked_heads.csv` /
`<series>_thickness_lines.csv` after every change and restored when a series
is opened again; `vwin.summary` and `twin.summary` read the saved rows back,
from earlier sessions too. `st.prepare_series` loads a file (rotation, dark,
normalisation), and a series still held by the notebook or the other window
is not loaded twice.

**Velocity by hand.** Click the head in frame 17, 18, 19 …; after every click
the velocity from the previous head is recomputed and shown in the readout,
the table and a live v(t) plot. A skipped frame leaves its two pairs empty,
or with `bridge_gaps=True` pairs its neighbours over the longer interval.
*Detect* places literature fronts as a first pass to correct. The saved mean
holds the mean of the frame-to-frame velocities and the slope of position
against end-of-exposure time.

**Thickness.** Click both ends of a straight channel section: the diameter
is the FWHM of the averaged cross section (manual eq. 3.1; Briels 2008;
Nijdam 2011), measured at once. Lines narrower than `min_diameter_px`
(10 px, Nijdam) are flagged; lines without half-maximum edges are marked
invalid and left out of the mean. *Detect channel* uses
`auto_measure_width`.

**Coordinates** in a window are those of its cropped series, `work`. Results
are stored uncropped underneath, so a new crop never invalidates them.

Things to know: a window does not respond while a cell runs; any
`%matplotlib …` magic unhooks the Qt loop (re-run a window cell); and
`%autoreload` does not orphan the windows, because they are remembered
outside the reloaded modules.

## Measuring without widgets

Nothing in the analysis needs ipywidgets — the pickers only help you *find*
coordinates. Three ways round them, best first:

```python
# 1. don't pick at all: find the channel and measure it
r = st.auto_measure_width(frame, pixel_size_um=17.0)
r.fwhm_averaged, r.passes, r.auto_info["axis"]

# 2. click in a pop-out window (Qt/Tk talk to the OS, not to the notebook)
box = st.pick_line(frame, box_width=30)      # opens a window, click twice
heads = st.pick_heads(image_a, image_b)      # one click per image

# 3. read coordinates off a labelled grid, inline, in any backend
st.coordinate_grid(frame, points=[(410, 300)], box=box)
```

`auto_measure_width` finds the channel the same way the velocity code finds
a head — smooth, threshold, largest connected component, principal axis —
places the line along it, then iterates the box width until it settles.
Accurate to about 1 % for channels of sigma 3–12 px, and stable to ±0.03 px
at 60 counts of noise. It follows the *largest* channel, so for a branched
tree crop to the branch you want first.

`st.use_window()` switches matplotlib to Qt or Tk, whichever is installed,
running the `%matplotlib` magic for you. That path never touches ipywidgets,
ipympl or the notebook frontend.

## If the widgets do not display

`Error displaying widget: model not found` is a *frontend* message — the
browser is being asked to draw a widget whose model is not in the kernel. Run

```python
import streamertools as st
st.diagnose_widgets()        # versions, backend, and what to fix
```

In order of how often it is the cause:

1. **Stale output.** The notebook was reopened, or the kernel restarted, with
   old widget output still on screen. Re-run the cell. Widget output never
   survives being saved and reopened.
2. **`ipympl` missing from the kernel's environment** — which is not always
   the one you ran `pip install` in. `st.diagnose_widgets()` prints
   `sys.executable` so you can check.
3. **Version mismatch.** ipywidgets 8 needs `jupyterlab_widgets` 3.x /
   `widgetsnbextension` 4.x; ipywidgets 7 needs 1.x / 3.x. Mixing them gives
   exactly this message.

You are not blocked either way. `Viewer` and `Cropper` never needed ipympl.
`LinePicker` and `HeadPicker` take `mode="auto"` (the default) and fall back
to **coordinate sliders** when the backend cannot deliver mouse events —
same live preview, same readout, same printed coordinates to paste into a
script. To click instead, either fix the widget stack or use a pop-out
window, which bypasses it entirely:

```python
%matplotlib qt      # or tk — a real window, clicking works
picker = LinePicker(frame, pixel_size_um=17.0, box_width=30)
```

`mode="click"` and `mode="sliders"` force either one.

## Calibration with a black-and-white square sheet

Pixel size is a property of the optics, not of a shot, so it is not part of
the analysis run: measure it once with `calibrate.py` and keep the number as
a constant in the notebook's config cell.

```bash
python calibrate.py calibration.sif --square-mm 1.0 --pattern 9x7 --show
```

It tries the checkerboard detector, falls back to the periodic row fit and
then to a single square, prints every warning worth acting on, ends with the
line to paste (`PIXEL_SIZE_UM = 24.9997`), and appends a dated row to
`calibration_log.csv` — file, method, value, relative error, warnings, your
`--note` — so a number can be traced back to the shot it came from months
later.

Four methods, in increasing order of how much they get out of one shot:

| function | uses | scatter on a synthetic 40 px square |
|---|---|---|
| `calibrate_from_line` | two identified points | set by your baseline |
| `calibrate_from_grid` | one square, mid-level crossings | ±0.14 px |
| `calibrate_from_period` | every edge along one row | ±0.008 px (17× tighter) |
| `calibrate_from_checkerboard` | all inner corners, via OpenCV | ±0.001 px (140× tighter) |

```python
r = st.calibrate_from_checkerboard(calib[0], (9, 7),   # INNER corners, not squares
                                   square_size_mm=1.0, show=True)
r.report()          # pixel size + every warning worth acting on
stack.pixel_size_um = r.pixel_size_um
```

The checkerboard method is the one to use if you have a checkerboard: as well
as being the most precise, it is the only one that tells you whether the sheet
was **flat and square to the camera**. `anisotropy` is the fractional
difference between horizontal and vertical corner spacing — ~0 when the sheet
is square on, −6 % for a 30 px perspective shift across a 400 px board — and
`rms_px` grows the same way. Both appear automatically in `report()`.

### If only part of the sheet is in view

Normal at high magnification, and fine — you tell the detector the *sub-grid
you can actually see*, not the whole sheet.

| squares in view | use | accuracy |
|---|---|---|
| 5 × 5 or more | `calibrate_from_checkerboard`, pattern = (n−1) inner corners | ±0.01 % |
| 4 × 4 | checkerboard, pattern (3,3) — but check the warnings | unreliable, ±9 % seen |
| 2–3 across a row | `calibrate_from_period` (needs ≥ 4 edges) | ~0.1 % |
| 1 | `calibrate_from_grid` | ~0.3 % |

Two things to know:

* **3 × 3 inner corners is OpenCV's hard floor** (a 4 × 4 block of squares).
  Below that the call is refused with a message pointing at the fallbacks.
  In testing, 3 × 3 also *detected but lied* (+15 %), so treat 5 × 5 squares
  as the practical minimum and read `report()` — the fit's relative error
  flags a bad correspondence.
* **Never ask for a pattern smaller than what is visible.** On a full 14 × 12
  board, asking for (3,3) returned 58.8 px against a true 40, and (5,4)
  returned 84.9 — the detector locks onto a wrong correspondence and is
  confident about it. The `rel_error` warning catches both, which is why
  `report()` is worth reading rather than just taking `.pixel_size_um`.

A quiet margin around the pattern helps but is not required: a board flush to
the frame edge still detected, as long as the pattern asked for is the one
fully inside.

What actually limits a calibration is never the algorithm:

1. **The sheet must be in the object plane** — exactly where the discharge is,
   parallel to the sensor. 1 cm of depth error at 30 cm object distance is a
   3 % scale error on every width and velocity you quote afterwards.
2. **A printed "1 mm" square is not 1 mm.** Printers are off by a few tenths
   of a percent and paper moves with humidity. Measure ten squares with
   callipers and divide; a systematic print error does not average out.
3. **Do not saturate the white squares.** Clipping shifts the mid-level and
   biases the width — narrow for a dark square, wide for a bright one; a 50 %
   clip cost 6 % in testing. The periodic fit is far less sensitive and corner
   detection is immune (clipping does not move the point where four squares
   meet), but all three warn when they see it.
4. **Watch for an illumination gradient.** A 20 % gradient across the field
   biases a square by ~0.5 %. Use squares near the centre, or average across.

Defocus is *not* on that list: the 50 % crossing sits on the true edge for any
symmetric blur, so a slightly soft calibration shot costs nothing and should
not be sharpened first (verified exact to σ = 8 px of blur).

## File types (manual 2.1)

| type | reader | notes |
|---|---|---|
| `.img` Stanford Computer Optics | one frame, 16/32 bit | layout from StreamerTools' own Matlab toolbox (`T2DFileImage`) |
| `.img` Agema/FLIR thermal | refused, with the reason | same extension, undocumented layout; a sample file would make it possible |
| `.rtv` Stanford movie | 16 bit, multi-frame | 8192-byte header, 4096-byte page blocks (same toolbox) |
| `.spe` Princeton | versions 2.x and 3.0 | 3.0: frame stride from the XML footer, so per-frame metadata is skipped |
| `.sif` Andor | via `sif_parser` | checked on the toolbox's `test_09.sif` |
| `.st2k` SBIG | plain and compressed | per SBIG's own `CSBIGImg` code |
| `.tsv` CWI | tab-separated grid, or `x y value` columns | CWI's refined sub-grids are files of their own |
| `.fit` `.fits` | via `astropy` | |
| `.cih` + `.raw` Photron | one `.raw`, or one `.raw` per frame | a `.raw` opened directly finds its `.cih` |
| `.dat` Shimadzu | 312×260 only | as in the manual |
| `.im7` LaVision | uncompressed or zlib images | layout from LaVision's ReadIMX; old IMX packing, 12-bit packing, vector and sparse buffers are refused |
| `.txt` `.streamer` | read and write | the two Streamertools formats (2.1.3) |
| `.avi` | uncompressed decoded directly; other codecs through OpenCV | RGB averaged into counts |
| `.jpg` `.png` `.bmp` `.tif` | read and write | multi-page TIFF is a movie; `keep_colour=True` keeps RGB |

The multi-frame types of manual 2.1.4 (`.cih .raw .avi .dat .rtv .spe .tif`,
plus `.sif`) all load as movies. Every reader is tested on a file written to
the published layout; the `.sif`, `.jpg` and `.png` examples shipped with
StreamerTools are read as well when it is installed. No real `.img`, `.rtv`,
`.st2k`, `.im7` or `.tsv` file was available to check against — the first
real file of each is worth a look.

## Single-frame and multi-frame files

A delay series is stored one of two ways, and `FILE_MODE` in the notebook
switches between them:

* `"multi"` — one multi-frame file per series (a kinetic `.sif`, an `.rtv`);
* `"single"` — one file per delay (a folder of `.img` shots, as in
  Briels/Winands, where every photograph is a different discharge).

```python
st.prepare_series("run.sif", mode="multi", dark_from_frames=2)
st.prepare_series("shots/run3", mode="single", dark_path="shots/dark")      # a folder
st.prepare_series("shots/run3/img_*.img", mode="single", dark_path="dark.img")  # a pattern
st.prepare_series(["a.img", "b.img", "c.img"], mode="single",
                  subtract_dark=False)                                      # a list, in this order
st.frame_files("shots/run3")          # which files, in which order
st.load_stack("shots/run3")           # the same, without dark subtraction
```

Folders and patterns are read in name order with numbers compared as numbers
(`shot2` before `shot10`); a list keeps its own order. Both windows take the
same `SERIES` list in either mode (`prepare=dict(mode=...)`); in `"single"`
mode each entry is one series, and a plain list of files is one series.

**Dark / background.** For single-frame files the subtraction is a switch,
`SUBTRACT_DARK` in the notebook (`subtract_dark=` of `prepare_series`):

| | subtracted from every frame |
|---|---|
| `"single"`, on | the dark at `DARK_PATH` — one file (its frames averaged), or a folder, pattern or list of dark shots (all averaged). Required: the shots are separate discharges, so none of them stands in for the background |
| `"single"`, off | nothing; the counts stay as recorded |
| `"multi"` | `DARK_PATH` if given, otherwise the mean of the first `DARK_FROM_FRAMES` frames, otherwise the median of the image corners |

What was subtracted is kept in `stack.metadata["dark"]`, printed by the load
cell and shown in both windows' title bars. Flipping the switch and re-running
a window cell reloads the series and keeps its clicks. `subtract_dark=False`
switches multi-frame files off too, but the notebook passes the switch only in
`"single"` mode. Thresholds in counts (`min_peak`) include the camera offset
when nothing is subtracted.

## Multi-frame files

A kinetic series and a single image are the same code path: `load` always
returns `(frames, H, W)` and a single image is `frames == 1`. Nothing needs a
different call.

```python
stack = st.load("series.sif", pixel_size_um=17.0)     # (40, H, W)
stack.n_frames, stack.is_movie                        # 40, True
st.load("series.sif", frames=slice(3, 9))             # a subset, on load

dark  = st.make_dark(stack, from_frames=3)            # first 3 frames as dark
ana   = st.StreamerWidthAnalyzer(stack, frame=17)     # measure on one frame
v     = st.measure_velocity((stack, 7), (stack, 8), delay_ns=10, ...)
st.velocity_from_series(stack, times_ns=..., ...)     # fit over the whole series
st.rgb_overlay(stack, 6)                              # frames 6,7,8 as R,G,B
conv.convert_files([path], out, all_frames=True)      # export every frame
FileEntry("series.sif", all_frames=True)              # one tile per frame ('?*')
FileEntry("series.sif", frames=(3, 8))                # or a range ('?3:8')
```

Two things specific to a series:

* **Comparable brightness.** Per-frame limits make every frame look equally
  bright. Use `DisplaySettings(scope="global")` — the limits are then computed
  once from the whole stack — whenever frame 3 and frame 27 need to mean the
  same thing.
* **Frames with no discharge.** A relative threshold always finds *something*,
  so on a pre-discharge background frame the head detector would happily
  return a noise blob. It now requires the head to stand `min_snr` robust
  sigmas (default 5) above the frame's background, and `velocity_from_series`
  drops the frames that fail, reporting how many. Set `min_snr=0` to disable,
  or `min_peak` for an absolute floor in counts.

Frame timing is not read from the `.sif` header — supply `times_ns` yourself
from the gate delays you set on the camera.

## Two things worth knowing

**Gamma.** The old notebook stretched each frame between two percentiles and
then applied `disp**gamma`. Here the percentiles only *choose* Imin/Imax and
the exponent follows the manual's `1/gamma` convention, with the logarithmic
branch for `gamma >= 10`. So a number written down as "gamma 1.5" means the
same thing here as in Streamertools, and gamma between 1 and 2 makes the dark
regions more pronounced, as the manual describes. Brightness and contrast are
applied to the counts *before* the normalisation, so they behave like camera
settings rather than like a curve.

**Box width.** The FWHM routine fits its baseline on whatever falls outside
the doubled peak bracket. If the measurement box is barely wider than the
channel, that fit eats into the Gaussian tails and the width comes out low.
Measured across channels of sigma 4, 6 and 9 px, as a multiple of the true
width: **2x** costs 9–11 %, **3x** costs 0.3–2.7 %, **4x** costs 0.5–1.3 %,
**5x** costs 0.2–0.6 %, **8x** is unbiased. So give the box **four to five
times** the expected width — an earlier version of this note said three,
which was based on one test case and is optimistic. `auto_measure_width`
iterates to a 5x box for you. This is the behaviour of the original
algorithm, not a deviation from it.

## Verification

`python tests/test_streamertools.py` builds synthetic images with known
answers and checks them: a Gaussian profile of known FWHM (recovered to <1 %),
a tilted channel of known width and length (<3 %), a head displaced by exactly
30 px in 10 ns, two branches moving 25 px each, a six-frame series with a
constant velocity, clicked heads with and without a skipped frame, plus
round-trips for the `.txt`/`.streamer`/`.pal`/`.tlmkr` formats and the
geometry of the tile matrix, a file of every type in manual 2.1 written to
its published layout, single- versus multi-frame series, and the dark switch.
The windows are tested twice: their models without Qt, and the real windows
offscreen (`QT_QPA_PLATFORM=offscreen`), with mouse events sent through
matplotlib's own wiring. 109/109 pass.
