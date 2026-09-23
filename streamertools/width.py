"""
Streamer Width Analyzer  --  Streamertools manual, chapter 3
=============================================================

Measures the width (3.2) or the length (3.2.2) of a bright channel, plus the
extra quantities of the information box (3.4).

How a measurement works
-----------------------
You draw a line over the streamer channel and give the box a *measurement
area width*.  The analyzer resamples the image under that box onto a new,
oblique grid ``B`` (manual fig. 3.3) by bilinear interpolation (fig. 3.4),
and reduces that 2-D patch to one or more 1-D profiles, which the FWHM
routine of appendix A then turns into edge positions.

Three ways to make the profiles (manual 3.2), all computed at once:

``averaged``
    Average the whole box into one profile (eq. 3.1).  Best signal to noise,
    but a curved channel or a badly aligned line makes the width too large.
``per_line``
    One profile per sample along the line, one FWHM each, averaged at the
    end.  Follows the streamer path, so it also works for a channel that is
    not straight -- but it needs a clean image.
``moving``
    The compromise (eq. 3.2): average `averaging_number` neighbouring lines
    before the FWHM.  ``X = 1`` reproduces *per line*, ``X = n`` reproduces
    *averaged*.

Width versus length
-------------------
For a **width** measurement the line is drawn *along* the channel and the
FWHM comes out perpendicular to it.  For a **length** measurement everything
is transposed (manual 3.2.2): draw the line along the streamer, well beyond
the section you want to measure, and make the measurement area width
*smaller* than the channel width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .fwhm import FWHMResult, calc_fwhm, smooth_savitzky_golay
from .geometry import MeasurementBox, angle_between_lines, resample_box
from .io import ImageStack

__all__ = [
    "AnalyzerSettings",
    "WidthResult",
    "StreamerWidthAnalyzer",
    "measure",
    "rgb_overlay",
]

Measure = Literal["width", "length"]


# --------------------------------------------------------------------------
@dataclass
class AnalyzerSettings:
    """The *Settings* tab of the Streamer Width Analyzer."""

    measure: Measure = "width"
    box_width: float = 20.0            # measurement area width, pixels
    averaging_number: int = 8          # X of the moving average (manual 3.2)
    smooth: bool = False               # Savitzky-Golay line smoothing (3.2.1)
    smooth_points: int = 11
    smooth_order: int = 2
    pixel_size_um: Optional[float] = None   # manual 2.4; None -> work in pixels

    @property
    def unit(self) -> str:
        return "um" if self.pixel_size_um else "px"

    @property
    def step(self) -> float:
        """Distance between two profile samples, in the working unit."""
        return float(self.pixel_size_um) if self.pixel_size_um else 1.0


# --------------------------------------------------------------------------
@dataclass
class WidthResult:
    """Everything the information box of the program shows, and then some."""

    settings: AnalyzerSettings
    box: MeasurementBox

    positions: np.ndarray                      # profile axis, in `unit`
    averaged_profile: np.ndarray               # eq. 3.1
    per_line_profiles: np.ndarray              # (n_profile, n_lines)

    averaged: FWHMResult
    per_line: List[FWHMResult] = field(default_factory=list)
    moving: List[FWHMResult] = field(default_factory=list)

    # -- 3.4 other calculations -------------------------------------------
    average_counts_in_area: float = 0.0
    maximum_counts_in_area: float = 0.0
    maximum_counts_centre_area: float = 0.0
    average_maximum_cross_section: float = 0.0

    # -- convenience -------------------------------------------------------
    @staticmethod
    def _stats(results: Sequence[FWHMResult]) -> Tuple[float, float, int]:
        vals = [r.fwhm for r in results if r.valid and r.fwhm > 0]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    @property
    def unit(self) -> str:
        return self.settings.unit

    @property
    def fwhm_averaged(self) -> float:
        """Width from the *of averaged line* method."""
        return self.averaged.fwhm

    @property
    def fwhm_per_line(self) -> float:
        return self._stats(self.per_line)[0]

    @property
    def fwhm_per_line_std(self) -> float:
        return self._stats(self.per_line)[1]

    @property
    def fwhm_moving(self) -> float:
        return self._stats(self.moving)[0]

    @property
    def fwhm_moving_std(self) -> float:
        return self._stats(self.moving)[1]

    @property
    def box_length(self) -> float:
        """Length of the drawn line, in the working unit."""
        return self.box.length_px * self.settings.step

    def edge_points(self, method: str = "per_line") -> Tuple[np.ndarray, np.ndarray]:
        """Edge positions of the channel in *image* coordinates, for overlays.

        Returns ``(left_xy, right_xy)``, each ``(k, 2)`` in original pixels.
        Useful to reproduce figure 3.5.
        """
        results = {"per_line": self.per_line, "moving": self.moving}.get(method)
        if results is None:
            raise ValueError("method must be 'per_line' or 'moving'")
        step = self.settings.step
        n_prof = len(self.positions)
        centre_offset = (n_prof - 1) / 2.0

        ua, uc = self.box.unit_along, self.box.unit_across
        if self.settings.measure == "length":
            ua, uc = uc, ua            # transposed geometry (manual 3.2.2)
        p0 = np.array(self.box.p0, dtype=float)
        if self.settings.measure == "length":
            p0 = p0 - self.box.unit_across * (self.box.width - 1) / 2.0

        n_lines = self.per_line_profiles.shape[1]
        span = max(n_lines - 1, 1)
        stride = span / max(len(results) - 1, 1) if len(results) > 1 else 0.0

        left, right = [], []
        for k, r in enumerate(results):
            if not r.valid:
                continue
            t = k * stride
            for pos, out in ((r.left_pos, left), (r.right_pos, right)):
                off = pos / step - centre_offset
                out.append(p0 + t * ua + off * uc)
        return np.array(left), np.array(right)

    def summary(self) -> Dict[str, object]:
        """Flat dict, ready for a DataFrame row or a csv line."""
        u = self.unit
        return {
            "measure": self.settings.measure,
            "x0": self.box.p0[0], "y0": self.box.p0[1],
            "x1": self.box.p1[0], "y1": self.box.p1[1],
            "box_width_px": self.box.width,
            f"box_length_{u}": self.box_length,
            f"fwhm_averaged_{u}": self.fwhm_averaged,
            f"fwhm_per_line_{u}": self.fwhm_per_line,
            f"fwhm_per_line_std_{u}": self.fwhm_per_line_std,
            f"fwhm_moving_{u}": self.fwhm_moving,
            f"fwhm_moving_std_{u}": self.fwhm_moving_std,
            "averaging_number": self.settings.averaging_number,
            "average_counts_in_area": self.average_counts_in_area,
            "maximum_counts_in_area": self.maximum_counts_in_area,
            "maximum_counts_centre_area": self.maximum_counts_centre_area,
            "average_maximum_cross_section": self.average_maximum_cross_section,
            "valid_per_line": self._stats(self.per_line)[2],
        }

    def __repr__(self) -> str:                                   # pragma: no cover
        u = self.unit
        return (f"<WidthResult {self.settings.measure}  "
                f"averaged={self.fwhm_averaged:.3g} {u}  "
                f"per-line={self.fwhm_per_line:.3g}+-{self.fwhm_per_line_std:.2g} {u}  "
                f"moving(X={self.settings.averaging_number})="
                f"{self.fwhm_moving:.3g}+-{self.fwhm_moving_std:.2g} {u}>")


# --------------------------------------------------------------------------
class StreamerWidthAnalyzer:
    """The analyzer, bound to one image (or one frame of a series).

    Examples
    --------
    >>> ana = StreamerWidthAnalyzer(frame, pixel_size_um=17.0)
    >>> res = ana.measure((410, 300), (455, 360), box_width=24)
    >>> res.fwhm_averaged
    """

    def __init__(self, image, settings: Optional[AnalyzerSettings] = None,
                 pixel_size_um: Optional[float] = None, frame: int = 0):
        if isinstance(image, ImageStack):
            pixel_size_um = pixel_size_um if pixel_size_um is not None else image.pixel_size_um
            self.image = np.asarray(image.data[frame], dtype=np.float32)
            self.stack: Optional[np.ndarray] = image.data
        else:
            arr = np.asarray(image, dtype=np.float32)
            self.stack = arr if arr.ndim == 3 else None
            self.image = arr[frame] if arr.ndim == 3 else arr
        self.settings = settings or AnalyzerSettings()
        if pixel_size_um is not None:
            self.settings.pixel_size_um = pixel_size_um

    # -- the measurement ---------------------------------------------------
    def measure(self, p0: Sequence[float], p1: Sequence[float],
                box_width: Optional[float] = None, **overrides) -> WidthResult:
        """Measure over the line ``p0 -> p1``.

        Parameters
        ----------
        p0, p1 : (x, y)
            The red and green end points, in original pixels.
        box_width : float
            Measurement area width in pixels; defaults to the settings value.
        **overrides
            Any :class:`AnalyzerSettings` field for this one measurement
            (``measure="length"``, ``averaging_number=4``, ``smooth=True``...).
        """
        s = AnalyzerSettings(**{**self.settings.__dict__, **overrides})
        if box_width is not None:
            s.box_width = float(box_width)
        box = MeasurementBox(tuple(map(float, p0)), tuple(map(float, p1)), s.box_width)
        return measure(self.image, box, s)

    # -- 3.4 angle ---------------------------------------------------------
    @staticmethod
    def angle(line_a, line_b) -> float:
        """Angle between the standard line and an extra line, in degrees."""
        return angle_between_lines(line_a, line_b)

    # -- 3.5 RGB overlay ---------------------------------------------------
    def rgb_overlay(self, index: int = 0, **kwargs) -> np.ndarray:
        if self.stack is None:
            raise ValueError("an RGB overlay needs a multi-frame file (manual 3.5)")
        return rgb_overlay(self.stack, index, **kwargs)

    # -- batch (the Automation tab) ---------------------------------------
    def measure_many(self, lines: Sequence[Tuple[Sequence[float], Sequence[float]]],
                     **overrides):
        """Measure a list of lines and return a pandas DataFrame.

        This replaces the *Add data to memo* / *Copy memo to clipboard* loop
        of the original program (manual 3.1).
        """
        import pandas as pd
        rows = []
        for i, (a, b) in enumerate(lines):
            r = self.measure(a, b, **overrides)
            row = r.summary()
            row["line"] = i
            rows.append(row)
        return pd.DataFrame(rows).set_index("line")


# --------------------------------------------------------------------------
def measure(image: np.ndarray, box: MeasurementBox,
            settings: Optional[AnalyzerSettings] = None) -> WidthResult:
    """Core routine: resample the box and run the three FWHM methods."""
    s = settings or AnalyzerSettings()
    img = np.asarray(image, dtype=np.float32)

    # ---- oblique grid B, manual fig. 3.3 --------------------------------
    B = resample_box(img, box)                      # (m across, n along)

    # ---- orientation: width uses B, length uses its transpose (3.2.2) ---
    P = B if s.measure == "width" else B.T          # (n_profile, n_lines)
    n_profile, n_lines = P.shape
    step = s.step
    positions = np.arange(n_profile, dtype=float) * step

    def _fwhm(profile: np.ndarray) -> FWHMResult:
        line = smooth_savitzky_golay(profile, s.smooth_points, s.smooth_order) \
            if s.smooth else profile
        return calc_fwhm(line, pixel_size=step)

    # ---- 1. of averaged line, eq. 3.1 -----------------------------------
    averaged_profile = P.mean(axis=1)
    averaged = _fwhm(averaged_profile)

    # ---- 2. per line ----------------------------------------------------
    per_line = [_fwhm(P[:, j]) for j in range(n_lines)]

    # ---- 3. moving average per X lines, eq. 3.2 -------------------------
    X = int(max(1, min(s.averaging_number, n_lines)))
    if X == 1:
        moving = list(per_line)
    else:
        kernel = np.ones(X, dtype=np.float32) / X
        # cumulative-sum moving average over the along-line axis
        csum = np.cumsum(np.concatenate([np.zeros((n_profile, 1), np.float32), P], axis=1),
                         axis=1)
        blocks = (csum[:, X:] - csum[:, :-X]) / X          # (n_profile, n_lines-X+1)
        moving = [_fwhm(blocks[:, g]) for g in range(blocks.shape[1])]

    # ---- 3.4 other calculations -----------------------------------------
    avg_counts = float(B.mean())
    max_counts = float(B.max())
    # "maximum counts center area": pixels within one pixel of the drawn line
    m = B.shape[0]
    centre = (m - 1) / 2.0
    mask = np.abs(np.arange(m) - centre) <= 1.0
    max_centre = float(B[mask].max()) if mask.any() else max_counts
    avg_max_cross = float(np.mean([r.peak_value for r in per_line])) if per_line else 0.0

    return WidthResult(
        settings=s, box=box, positions=positions,
        averaged_profile=np.asarray(averaged_profile, dtype=np.float32),
        per_line_profiles=np.asarray(P, dtype=np.float32),
        averaged=averaged, per_line=per_line, moving=moving,
        average_counts_in_area=avg_counts,
        maximum_counts_in_area=max_counts,
        maximum_counts_centre_area=max_centre,
        average_maximum_cross_section=avg_max_cross,
    )


# --------------------------------------------------------------------------
def rgb_overlay(stack, index: int = 0, limit_mode: str = "histogram",
                gamma: float = 1.0, **limits) -> np.ndarray:
    """Render frames ``k, k+1, k+2`` as the red, green and blue channel (3.5).

    A quick way to see whether a feature reproduces from frame to frame:
    anything that stays put comes out white, anything that moves is coloured.
    """
    from .palette import counts_to_unit, resolve_limits

    data = getattr(stack, "data", stack)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] < index + 3:
        raise ValueError("need at least three frames from `index` onwards (manual 3.5)")
    trio = data[index:index + 3]
    lo, hi = resolve_limits(trio, limit_mode,
                            limits.get("i_min"), limits.get("i_max"),
                            limits.get("min_hist_percentage", 1.0),
                            limits.get("max_hist_percentage", 99.8))
    unit = counts_to_unit(trio, lo, hi, gamma)
    return (np.moveaxis(unit, 0, -1) * 255).astype(np.uint8)
