"""
Pixel size calibration  --  manual section 2.4
===============================================

Every result in physical units depends on one number: the size of one pixel
*in the image plane* (not on the CCD).  The manual's recipe is to image an
object of known size, measure a known separation in pixels, and divide.

Four ways to do that, in increasing order of how much they get out of one
calibration shot:

:func:`calibrate_from_line`
    Two points you can identify, and the real distance between them.  The
    classic ruler shot.
:func:`calibrate_from_grid`
    One square of a grid target, measured at the mid-level between the
    bright plateau and the dark trough.
:func:`calibrate_from_period`
    *Every* edge along one row of a black-and-white square sheet, fitted as
    a periodic grid.  Roughly ten times more precise than one square,
    because the fit averages over the whole row.
:func:`calibrate_from_checkerboard`
    All the inner corners of a checkerboard at once, via OpenCV's detector.
    Gives the pixel size plus two diagnostics -- anisotropy and corner
    scatter -- that tell you whether the sheet was flat and square to the
    camera.  This is the one to use if you have a checkerboard.

Why the mid-level criterion
---------------------------
For a symmetric point spread function the 50 % crossing sits exactly on the
true edge however defocused the image is: the blur removes as much signal
from inside the edge as it adds outside.  A slightly soft calibration image
therefore costs nothing, and should not be sharpened first.

What actually breaks a calibration
----------------------------------
Not the algorithm.  In order of severity:

1. **The sheet is not in the object plane.**  Pixel size is a property of
   the image plane, so the target has to sit exactly where the discharge
   does, and parallel to the sensor.  A 1 cm depth error at 30 cm object
   distance is a 3 % scale error on everything you measure afterwards.
2. **The printed square is not the size you think.**  Consumer printers are
   off by a few tenths of a percent to a percent, and paper moves with
   humidity.  Measure the sheet with callipers over ten squares, not one,
   and use that number as `square_size_mm`.
3. **The white areas saturate.**  Clipping pulls the bright plateau down and
   drags the mid-level with it, so the measured square is biased -- narrow
   if you bracketed a dark square, wide if you bracketed a bright one.  A
   10 % clip already costs 0.7 %, a 50 % clip several percent.  The
   periodic fit is far less sensitive (the period survives even when the
   individual widths do not) and checkerboard corner detection is immune,
   because clipping does not move the point where four squares meet.  All
   the routines warn when they see it.
4. **Uneven illumination.**  A gradient across the field breaks the symmetry
   the mid-level criterion relies on: a 20 % gradient biases a square by
   about 0.5 %.  Use squares near the centre, or average across the field.

All calculations assume square pixels, as Streamertools does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import pixel_size_from_distance

__all__ = ["CalibrationResult", "calibrate_from_line", "half_crossings",
           "dark_square_fwhm", "calibrate_from_grid", "calibrate_from_period",
           "calibrate_from_checkerboard", "saturated_fraction"]


# --------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    """A pixel size with the evidence behind it."""

    pixel_size_um: float
    square_px: float                      # measured size of one square
    square_size_mm: float
    method: str
    n_features: int = 1                   # edges or corners the fit used
    sigma_px: float = float("nan")        # standard error on square_px
    anisotropy: Optional[float] = None    # (row spacing / column spacing) - 1
    rms_px: Optional[float] = None        # scatter of the individual spacings
    saturated_fraction: float = 0.0
    corners: Optional[np.ndarray] = None  # (rows, cols, 2), checkerboard only

    @property
    def rel_error(self) -> float:
        """Relative standard error of the pixel size, from the fit alone."""
        if not np.isfinite(self.sigma_px) or self.square_px == 0:
            return float("nan")
        return float(self.sigma_px / self.square_px)

    def warnings(self) -> list:
        """Everything about this calibration that deserves a second look."""
        out = []
        if self.saturated_fraction > 0.002:
            if self.corners is not None:
                # corner detection keys on where four squares meet, which
                # clipping does not move -- so this is a note, not a bias
                out.append(f"{100 * self.saturated_fraction:.1f} % of pixels sit at "
                           f"the maximum value; corner detection tolerates that, but "
                           f"check the exposure before trusting anything else in "
                           f"this image")
            else:
                out.append(f"{100 * self.saturated_fraction:.1f} % of pixels sit at "
                           f"the maximum value -- saturation shifts the mid-level and "
                           f"biases the width (narrow for a dark square, wide for a "
                           f"bright one)")
        if self.anisotropy is not None and abs(self.anisotropy) > 0.01:
            out.append(f"row and column spacings differ by "
                       f"{100 * self.anisotropy:+.1f} % -- the sheet is tilted, or "
                       f"the pixels are not square")
        if self.rms_px is not None and self.square_px and self.rms_px / self.square_px > 0.01:
            out.append(f"spacings scatter by {100 * self.rms_px / self.square_px:.1f} % "
                       f"-- perspective, a bent sheet, or a poor detection")
        if np.isfinite(self.rel_error) and self.rel_error > 0.005:
            out.append(f"the fit alone is only good to {100 * self.rel_error:.1f} %; "
                       f"use more squares or a longer baseline")
        return out

    def report(self) -> None:
        print(f"{self.method}: one square = {self.square_px:.3f} px "
              f"= {self.square_size_mm:g} mm  ->  {self.pixel_size_um:.4f} um/px "
              f"({self.n_features} features)")
        for w in self.warnings():
            print(f"  WARNING: {w}")

    def __repr__(self) -> str:                                   # pragma: no cover
        n = len(self.warnings())
        flag = "" if n == 0 else f", {n} warning(s)"
        return (f"<CalibrationResult {self.pixel_size_um:.4f} um/px  "
                f"({self.square_px:.3f} px per {self.square_size_mm:g} mm, "
                f"{self.method}{flag})>")


# --------------------------------------------------------------------------
def saturated_fraction(image: np.ndarray, level: Optional[float] = None) -> float:
    """Fraction of pixels sitting at the maximum value (a clipping check)."""
    a = np.asarray(image, dtype=np.float32)
    top = float(np.nanmax(a)) if level is None else float(level)
    return float(np.mean(a >= top - 1e-6))


def _as_frame(image) -> np.ndarray:
    a = np.asarray(getattr(image, "data", image), dtype=np.float64)
    return a[0] if a.ndim == 3 else a


# --------------------------------------------------------------------------
def calibrate_from_line(p0: Sequence[float], p1: Sequence[float],
                        real_distance_mm: float) -> float:
    """Pixel size (um/px) from two identified points and their real distance."""
    d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return pixel_size_from_distance(d, real_distance_mm)


def half_crossings(x: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """Sub-pixel positions where a profile crosses `level`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.sign(y - level)
    idx = np.where(np.diff(s) != 0)[0]
    return np.array([x[i] + (level - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])
                     for i in idx])


def dark_square_fwhm(profile: np.ndarray, x0: int, x1: int,
                     show: bool = False) -> float:
    """Width in pixels of one dark square of a calibration target.

    Take a row or a column through the target, bracket exactly one dark
    square with ``[x0, x1]`` (the bracket must run bright -> dark -> bright)
    and this returns the distance between the two mid-level crossings.
    """
    x = np.arange(x0, x1)
    seg = np.asarray(profile, dtype=float)[x0:x1]
    if seg.size < 3:
        raise ValueError("bracket is too short")
    half = 0.5 * (seg.max() + seg.min())
    cr = half_crossings(x, seg, half)
    if len(cr) < 2:
        raise RuntimeError("the bracket must span bright -> dark -> bright "
                           "(two mid-level crossings are needed)")
    left, right = float(cr[0]), float(cr[-1])
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(x, seg, color="tab:red")
        plt.axhline(half, color="0.5", ls="--")
        plt.plot([left, right], [half, half], "o", color="tab:blue", ms=8)
        plt.title(f"dark square = {right - left:.2f} px")
        plt.xlabel("position (px)")
        plt.ylabel("counts")
        plt.show()
    return right - left


def calibrate_from_grid(image: np.ndarray, row: Optional[int] = None,
                        col: Optional[int] = None, bracket: Tuple[int, int] = None,
                        square_size_mm: float = 1.0, show: bool = False
                        ) -> Tuple[float, float]:
    """Pixel size from one square of a checkerboard / grid target.

    Returns ``(pixel_size_um, square_width_px)``.  For anything better than
    about half a percent, use :func:`calibrate_from_period` (many squares
    along one row) or :func:`calibrate_from_checkerboard` (the whole board).
    """
    img = _as_frame(image)
    if row is not None:
        profile = img[int(row), :]
    elif col is not None:
        profile = img[:, int(col)]
    else:
        raise ValueError("give either `row` or `col`")
    if bracket is None:
        raise ValueError("give `bracket=(x0, x1)` around exactly one dark square")
    sat = saturated_fraction(img)
    if sat > 0.002:
        print(f"WARNING: {100 * sat:.1f} % of pixels are saturated; the mid-level is "
              f"shifted and this square will be biased (a 50 % clip cost 6 % in "
              f"testing). calibrate_from_checkerboard is immune to this.")
    width_px = dark_square_fwhm(profile, bracket[0], bracket[1], show=show)
    return pixel_size_from_distance(width_px, square_size_mm), width_px


# --------------------------------------------------------------------------
def _dominant_period(profile: np.ndarray) -> Optional[float]:
    """Spacing of the strongest periodic component, in pixels, via the FFT."""
    y = np.asarray(profile, dtype=float)
    n = y.size
    if n < 16:
        return None
    y = y - np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n))   # detrend
    spectrum = np.abs(np.fft.rfft(y * np.hanning(n)))
    spectrum[:2] = 0.0                       # kill DC and the whole-window mode
    k = int(np.argmax(spectrum))
    if k == 0:
        return None
    return float(n / k)                      # one full period = two squares


def _local_level(y: np.ndarray, period: Optional[float]) -> np.ndarray:
    """A baseline that follows illumination and the sheet edge, not the squares.

    Taking the level as ``(max + min) / 2`` of the whole row fails in two very
    ordinary situations: a bright quiet margin around the target raises `max`
    until the level sits above the squares entirely (no crossings at all), and
    an illumination gradient tilts the true mid-level across the frame.  A
    heavily smoothed copy of the profile tracks both and leaves the square
    pattern behind, so crossings against *it* are what we want.
    """
    if period is None or not np.isfinite(period) or period <= 2:
        return np.full_like(y, 0.5 * (y.max() + y.min()))
    # sigma = 2 periods and 'reflect' at the ends: 'nearest' holds the bright
    # margin flat and drags the level near the edges, which pushes the two
    # outermost crossings outward and inflates the fitted period (+1.4 % on a
    # four-square crop; +0.08 % with reflect).
    return gaussian_filter1d(y, sigma=2.0 * float(period), mode="reflect")


def _robust_crossings(profile: np.ndarray, smooth: float = 3.0,
                      min_sep: Optional[float] = None) -> np.ndarray:
    """Mid-level crossings of a periodic profile, noise- and margin-hardened.

    Three filters make this survive a real calibration row: the profile is
    lightly smoothed; crossings are taken against a *local* level (see
    :func:`_local_level`) rather than a single global number; and they must
    alternate in direction and be at least `min_sep` apart, since one spurious
    crossing would renumber every crossing after it and wreck the periodic fit.
    """
    y = gaussian_filter1d(np.asarray(profile, dtype=float), smooth) if smooth \
        else np.asarray(profile, dtype=float)
    x = np.arange(y.size, dtype=float)

    period = _dominant_period(y)
    level = _local_level(y, period)
    delta = y - level
    idx = np.where(np.diff(np.sign(delta)) != 0)[0]
    if idx.size == 0:
        return np.empty(0)

    denominator = delta[idx + 1] - delta[idx]
    denominator = np.where(denominator == 0, 1e-12, denominator)
    pos = x[idx] - delta[idx] / denominator
    direction = np.sign(denominator)

    if min_sep is None and period is not None:
        min_sep = period / 3.0               # half a period = one square

    keep: list = []
    for k in range(len(pos)):
        if keep:
            same_way = direction[k] == direction[keep[-1]]
            too_close = min_sep is not None and (pos[k] - pos[keep[-1]]) < min_sep
            if same_way or too_close:
                continue
        keep.append(k)
    return pos[keep]


def _best_row(image: np.ndarray, axis: int = 0) -> int:
    """The row (or column) whose profile oscillates most strongly.

    Picking the middle of the image is a coin toss: land on the boundary
    between two rows of squares and the blur halves the contrast.  This scores
    every candidate by the height of its dominant FFT peak and takes the best.
    """
    img = np.asarray(image, dtype=float)
    lines = img if axis == 0 else img.T
    n = lines.shape[1]
    scores = []
    for i in range(lines.shape[0]):
        y = lines[i]
        y = y - np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n))
        spectrum = np.abs(np.fft.rfft(y * np.hanning(n)))
        spectrum[:2] = 0.0
        scores.append(spectrum.max())
    return int(np.argmax(scores))


def calibrate_from_period(image, row: Optional[int] = None, col: Optional[int] = None,
                          square_size_mm: float = 1.0,
                          bracket: Optional[Tuple[int, int]] = None,
                          smooth: float = 3.0, min_sep: Optional[float] = None,
                          show: bool = False) -> CalibrationResult:
    """Pixel size from every edge along one row of a square calibration sheet.

    Each edge of the sheet crosses the mid-level once, so edge position is a
    straight line against edge index whose slope is the width of one square.
    Fitting that line uses the whole row instead of one square, which is
    worth about a factor of ten in precision -- and the residuals tell you
    whether the sheet is really periodic.

    Parameters
    ----------
    image : 2-D array or ImageStack
    row, col : int
        Which row (or column) to profile.  Pick one that crosses as many
        squares as possible and is away from any illumination gradient.
    square_size_mm : float
        Real size of one square -- measure it with callipers over ten
        squares, not one.
    bracket : (x0, x1), optional
        Restrict to part of the row.
    min_sep : float, optional
        Minimum distance between consecutive edges, in pixels.  Defaults to
        a third of the median spacing, which rejects noise crossings without
        touching real edges.
    """
    img = _as_frame(image)
    if row is None and col is None:          # pick the strongest row for you
        row = _best_row(img, axis=0)
        print(f"calibrate_from_period: using row {row} (strongest periodic signal)")
    if row is not None:
        profile = img[int(row), :]
    elif col is not None:
        profile = img[:, int(col)]
    offset = 0
    if bracket is not None:
        offset = int(bracket[0])
        profile = profile[int(bracket[0]):int(bracket[1])]

    if min_sep is None:                       # two passes: guess, then refine
        first = _robust_crossings(profile, smooth, None)
        if len(first) >= 3:
            min_sep = 0.33 * float(np.median(np.diff(first)))
    crossings = _robust_crossings(profile, smooth, min_sep)

    if len(crossings) < 4:
        raise RuntimeError(
            f"found only {len(crossings)} edges along this line -- a periodic fit "
            f"needs at least 4. Check that the row really crosses several squares, "
            f"or fall back to calibrate_from_grid on a single square.")

    index = np.arange(len(crossings), dtype=float)
    coeffs, cov = np.polyfit(index, crossings, 1, cov=True)
    square_px = float(coeffs[0])              # one edge-to-edge step = one square
    sigma_px = float(np.sqrt(cov[0, 0]))
    residuals = crossings - np.polyval(coeffs, index)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    result = CalibrationResult(
        pixel_size_um=pixel_size_from_distance(square_px, square_size_mm),
        square_px=square_px, square_size_mm=square_size_mm,
        method=f"periodic fit over {len(crossings)} edges",
        n_features=len(crossings), sigma_px=sigma_px, rms_px=rms,
        saturated_fraction=saturated_fraction(img))

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        x = np.arange(len(profile)) + offset
        ax[0].plot(x, profile, color="tab:red", lw=1)
        for c in crossings:
            ax[0].axvline(c + offset, color="tab:blue", lw=0.8, alpha=0.7)
        ax[0].set_xlabel("position (px)")
        ax[0].set_ylabel("counts")
        ax[0].set_title(f"{len(crossings)} edges found")
        ax[1].plot(index, residuals, "o", ms=4, color="tab:blue")
        ax[1].axhline(0, color="0.6", lw=1)
        ax[1].set_xlabel("edge index")
        ax[1].set_ylabel("residual (px)")
        ax[1].set_title(f"square = {square_px:.3f} +- {sigma_px:.3f} px, "
                        f"rms {rms:.3f} px")
        plt.tight_layout()
        plt.show()
    return result


# --------------------------------------------------------------------------
def calibrate_from_checkerboard(image, pattern_size: Tuple[int, int],
                                square_size_mm: float = 1.0,
                                show: bool = False,
                                refine_window: int = 11) -> CalibrationResult:
    """Pixel size from every inner corner of a black-and-white checkerboard.

    Uses OpenCV's chessboard detector, which finds all inner corners to a
    few hundredths of a pixel, then measures the mean spacing between
    neighbouring corners along the rows and down the columns.

    Parameters
    ----------
    pattern_size : (cols, rows)
        The number of **inner** corners, not squares: a sheet of 10 x 8
        squares has 9 x 7 inner corners.  Count the crossing points where
        four squares meet.
    square_size_mm : float
        The real size of one square.

    Returns
    -------
    CalibrationResult
        With two diagnostics the other methods cannot give you.
        ``anisotropy`` is the fractional difference between the horizontal
        and vertical spacing: a sheet square to the camera with square
        pixels gives ~0, while a tilt shows up immediately (a perspective
        shift of 30 px across a 400 px board reads as -6 %).  ``rms_px`` is
        the scatter of the individual spacings, which grows the same way.
        Both are reported by :meth:`CalibrationResult.report`.
    """
    try:
        import cv2
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError("calibrate_from_checkerboard needs OpenCV "
                          "(pip install opencv-python)") from exc

    img = _as_frame(image)
    sat = saturated_fraction(img)

    # robust stretch to 8-bit: percentiles, so one hot pixel cannot flatten it
    lo, hi = np.percentile(img, [0.5, 99.5])
    u8 = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    u8 = (u8 * 255).astype(np.uint8)

    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    if cols < 3 or rows < 3:
        raise ValueError(
            f"pattern_size {(cols, rows)} is too small: OpenCV needs at least "
            f"3 x 3 inner corners, i.e. a 4 x 4 block of squares fully in view "
            f"with a quiet margin around it. With fewer squares than that use "
            f"calibrate_from_period (needs ~2 squares along a row) or "
            f"calibrate_from_grid (needs 1).")

    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):        # newer, more robust
        try:
            ok, found = cv2.findChessboardCornersSB(
                u8, (cols, rows), flags=getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
                | getattr(cv2, "CALIB_CB_ACCURACY", 0))
            if ok:
                corners = found
        except cv2.error:
            # SB refuses patterns smaller than 3 x 3 outright; the classic
            # detector handles those, so fall through rather than blow up
            corners = None
    if corners is None:
        try:
            ok, found = cv2.findChessboardCorners(
                u8, (cols, rows),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        except cv2.error:
            ok, found = False, None
        if not ok:
            raise RuntimeError(
                f"no {cols} x {rows} checkerboard found. Count the *inner* corners "
                f"(a sheet of N x M squares has (N-1) x (M-1) of them), check that "
                f"the whole board including a light margin is inside the frame, and "
                f"that the white squares are not saturated "
                f"({100 * sat:.1f} % of pixels are at the maximum).")
        w = max(3, int(refine_window))
        corners = cv2.cornerSubPix(
            u8, found, (w, w), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))

    grid = np.asarray(corners, dtype=float).reshape(rows, cols, 2)
    d_row = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()   # along rows
    d_col = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()   # down columns
    spacings = np.concatenate([d_row, d_col])
    square_px = float(spacings.mean())
    sigma_px = float(spacings.std(ddof=1) / np.sqrt(spacings.size))
    anisotropy = float(d_row.mean() / d_col.mean() - 1.0)

    result = CalibrationResult(
        pixel_size_um=pixel_size_from_distance(square_px, square_size_mm),
        square_px=square_px, square_size_mm=square_size_mm,
        method=f"checkerboard, {cols}x{rows} inner corners",
        n_features=int(grid.size // 2), sigma_px=sigma_px,
        anisotropy=anisotropy, rms_px=float(spacings.std(ddof=1)),
        saturated_fraction=sat, corners=grid)

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].imshow(img, cmap="gray")
        ax[0].plot(grid[..., 0].ravel(), grid[..., 1].ravel(), ".",
                   color="tab:cyan", ms=4)
        for r in range(rows):
            ax[0].plot(grid[r, :, 0], grid[r, :, 1], "-", color="tab:cyan", lw=0.6)
        ax[0].set_title(f"{result.n_features} corners")
        ax[0].axis("off")
        ax[1].hist(d_row, bins=20, alpha=0.7, label="along rows")
        ax[1].hist(d_col, bins=20, alpha=0.7, label="down columns")
        ax[1].axvline(square_px, color="k", lw=1)
        ax[1].set_xlabel("corner spacing (px)")
        ax[1].set_ylabel("pairs")
        ax[1].legend(fontsize=8)
        ax[1].set_title(f"{square_px:.3f} +- {sigma_px:.3f} px, "
                        f"anisotropy {100 * anisotropy:+.2f} %")
        plt.tight_layout()
        plt.show()
    return result
