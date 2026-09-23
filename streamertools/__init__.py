"""
streamertools
=============

A Python re-implementation of the parts of the Streamertools suite
(S. Nijdam, TU/e, manual of 31 March 2022) that matter for ICCD streamer
image analysis, plus a streamer-velocity measurement that the original suite
does not have.

Covered
-------
chapter 2  common functionality  -- :mod:`.io`, :mod:`.palette`,
           :mod:`.geometry`, :mod:`.display`, :mod:`.calibration`
chapter 3  Streamer Width Analyzer -- :mod:`.fwhm`, :mod:`.width`
chapter 5  Image Converter        -- :mod:`.converter`
chapter 6  Tile maker             -- :mod:`.tilemaker`
new        propagation velocity   -- :mod:`.velocity`

Support modules: :mod:`.preprocess` (dark subtraction, Mf normalisation,
denoising), :mod:`.annotate` (text, arrows, overlays), :mod:`.plotting`,
:mod:`.velocity_window` and :mod:`.thickness_window` (the two permanent
analysis windows of the V3 notebook, built on :mod:`.workbench`),
:mod:`.pick` and :mod:`.interactive` (the older pop-up and widget pickers).

Quick start
-----------
>>> import streamertools as st
>>> stack = st.load("shot.sif", pixel_size_um=17.0, rot90=2)
>>> stack = st.subtract_dark(stack, st.make_dark(stack, from_frames=3))
>>> ana = st.StreamerWidthAnalyzer(stack, frame=10)
>>> res = ana.measure((410, 300), (455, 360), box_width=24)
>>> res.fwhm_averaged
>>> v = st.measure_velocity((stack, 10), (stack, 11), delay_ns=10,
...                         pixel_size_um=17.0, axis="down")
>>> v.velocity_m_per_s
"""

__version__ = "1.10.0"

from .annotate import ArrowSpec, TextStyle, apply_overlay, draw_arrow, draw_text_block
from .calibration import (CalibrationResult, calibrate_from_checkerboard,
                          calibrate_from_grid, calibrate_from_line,
                          calibrate_from_period, dark_square_fwhm,
                          half_crossings, saturated_fraction)
from .converter import (ArrowAction, ConverterSettings, CropRotateFlip, ImageConverter,
                        ImageMath, OverlayAction, Resize, SimpleMath, TextAction)
from .display import DisplaySettings, render, render_stack
from .fwhm import FWHMResult, calc_fwhm, smooth_savitzky_golay
from .geometry import (MeasurementBox, angle_between_lines, bilinear_sample, crop, flip,
                       manipulate, pixel_size_from_distance, resample_box, resize,
                       rotate, zoom_out)
from .io import (READERS, ImageStack, frame_files, load, load_stack, read_streamer_bin,
                 read_streamer_txt, save_counts, save_image, write_streamer_bin,
                 write_streamer_txt)
from .palette import Palette, counts_to_rgb, counts_to_unit, resolve_limits
from .pick import (auto_line, auto_measure_width, browse, coordinate_grid,
                   ginput_line, pick_crop, pick_heads, pick_line, pick_velocity,
                   use_inline, use_window)
from .preprocess import (DenoiseSettings, apply_physical_normalization, compute_Mf,
                         estimate_bias, make_dark, preprocess_frame, preprocess_stack,
                         remove_hot_pixels, remove_small_bright_blobs, subtract_dark)
from .tilemaker import (FileEntry, MovieSettings, OriginalSettings, OutputSettings,
                        OutsideComments, OverlaySettings, TileMaker, TileMakerConfig,
                        make_tiles, read_tlmkr, write_tlmkr)
from .velocity import (FrontVelocityResult, HeadPosition, HeadSettings, VelocityResult,
                       detect_head, detect_heads, measure_velocity,
                       measure_velocity_branches, track_head, velocity_frame_to_frame,
                       velocity_from_fronts, velocity_from_heads, velocity_from_series)
from .width import (AnalyzerSettings, StreamerWidthAnalyzer, WidthResult,
                    measure, rgb_overlay)
from .workbench import FILE_MODES, prepare_series, series_file_stem, series_label
from .velocity_window import (VelocityWorkbench, Workbench, open_velocity_window,
                              open_workbench)
from .thickness_window import (ThicknessMeasurement, ThicknessWorkbench,
                               open_thickness_window)

# Submodules, so `streamertools.interactive.diagnose()` and
# `streamertools.plotting.plot_profiles(...)` work after a plain
# `import streamertools`.  (These used to be accessor *functions* of the same
# name, which shadowed the modules -- `st.interactive.diagnose` then raised
# AttributeError.)
from . import calibration, converter, display, fwhm, geometry, interactive  # noqa: E402
from . import io, palette, pick, plotting, preprocess, tilemaker, velocity, width  # noqa: E402
from . import thickness_window, velocity_window, workbench  # noqa: E402
from .interactive import canvas_is_interactive  # noqa: E402
from .interactive import diagnose as diagnose_widgets  # noqa: E402

__all__ = [n for n in dir() if not n.startswith("_")]
