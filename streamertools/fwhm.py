"""
Full Width at Half Maximum  --  manual section 3.2.1 and appendix A
====================================================================

A line-by-line port of the Delphi ``CalcFWHM`` routine printed in appendix A
of the manual.  No function fitting is used, because the shape of a streamer
cross-section is not known in advance; the width is found geometrically:

1. find the absolute maximum ``PeakValue`` at ``PeakPos`` and the overall
   ``Average`` of the line;
2. walk outwards from the peak until the line drops below
   ``(PeakValue + Average) / 2`` -- this gives ``LeftSide`` / ``RightSide``;
3. double that half-width to bracket the peak generously;
4. least-squares fit a straight baseline through everything *outside* the
   bracket and subtract it;
5. redo step 2 on the baseline-corrected line, now with half maximum
   ``PeakValue / 2`` (the baseline is zero);
6. linearly interpolate between the two neighbouring samples at each side to
   get sub-pixel edge positions.

If either edge runs into the end of the line the result is reported as
``fwhm = 0`` and ``valid = False``, exactly like the original: the profile did
not come back down, so no width can be claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["FWHMResult", "calc_fwhm", "smooth_savitzky_golay"]


# --------------------------------------------------------------------------
@dataclass
class FWHMResult:
    """Everything the Delphi ``TFWHM`` record holds, plus a validity flag."""

    fwhm: float                 # width in the units of pixel_size (0 if invalid)
    left_pos: float             # position of the left half-maximum crossing
    right_pos: float            # position of the right half-maximum crossing
    peak_value: float           # peak height above the baseline
    peak_pos: float             # position of the peak
    baseline_a: float           # baseline offset  (counts)
    baseline_b: float           # baseline slope   (counts per unit position)
    valid: bool = True

    @property
    def centre(self) -> float:
        return 0.5 * (self.left_pos + self.right_pos)

    def baseline(self, positions: np.ndarray) -> np.ndarray:
        return self.baseline_a + self.baseline_b * np.asarray(positions, dtype=float)

    def __repr__(self) -> str:                                   # pragma: no cover
        state = "" if self.valid else ", INVALID (edge not found)"
        return (f"FWHMResult(fwhm={self.fwhm:.4g}, centre={self.centre:.4g}, "
                f"peak={self.peak_value:.4g}{state})")


def _ensure_range(value: int, lo: int, hi: int) -> int:
    """Delphi ``EnsureRange``."""
    return int(min(max(value, lo), hi))


# --------------------------------------------------------------------------
def calc_fwhm(line, pixel_size: float = 1.0) -> FWHMResult:
    """Full width at half maximum of a 1-D profile (manual appendix A).

    Parameters
    ----------
    line : array_like
        Count values as a function of position, evenly spaced.
    pixel_size : float
        Distance between two samples.  Pass 1 to get a result in pixels, or
        the calibrated pixel size (manual 2.4) to get it in um / mm.

    Returns
    -------
    FWHMResult
    """
    y = np.asarray(line, dtype=np.float64).ravel()
    n = y.size
    if n < 8:
        return FWHMResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, valid=False)

    # -- 1. peak and overall average -------------------------------------
    peak_pos = int(np.argmax(y))
    peak_value = float(y[peak_pos])
    average = float(y.mean())

    # -- 2. halfway points between Average and PeakValue ------------------
    peak_pos = _ensure_range(peak_pos, 1, n - 2)
    half_first = (peak_value + average) / 2.0
    left, right = peak_pos, peak_pos
    while left > 0 and y[left] > half_first:
        left -= 1
    while right < n - 1 and y[right] > half_first:
        right += 1

    # -- 3. double the width of this peak ---------------------------------
    left_new = _ensure_range(2 * left - peak_pos, 0, peak_pos)
    right_new = _ensure_range(2 * right - peak_pos, peak_pos, n - 1)

    # -- 4. linear baseline through everything outside the bracket --------
    idx = np.arange(n)
    outside = (idx < left_new) | (idx > right_new)
    n_out = int(outside.sum())
    x = idx.astype(np.float64) * float(pixel_size)

    if n_out > 1:
        sx, sy = x[outside].sum(), y[outside].sum()
        sxy = float((x[outside] * y[outside]).sum())
        sxx = float((x[outside] ** 2).sum())
        av_x, av_y = sx / n_out, sy / n_out
        ss_xx = sxx - n_out * av_x ** 2
        ss_xy = sxy - n_out * av_x * av_y
        coef_b = 0.0 if ss_xx == 0 else min(max(ss_xy / ss_xx, -1e12), 1e12)
        lim = abs(average) * 100.0
        coef_a = min(max(av_y - coef_b * av_x, -lim), lim)
    elif n_out == 1:
        coef_b, coef_a = 0.0, float(y[outside][0])
    else:
        coef_a = coef_b = 0.0

    # -- 5. baseline-corrected line, new peak -----------------------------
    new_line = y - (coef_a + coef_b * x)
    peak_pos = int(np.argmax(new_line))
    peak_value = float(new_line[peak_pos])

    hm = peak_value / 2.0
    peak_pos = _ensure_range(peak_pos, 3, n - 4)
    left, right = peak_pos, peak_pos
    while left > 0 and new_line[left] > hm:
        left -= 1
    while right < n - 1 and new_line[right] > hm:
        right += 1

    # -- 6. sub-pixel interpolation of both edges -------------------------
    if new_line[left + 1] == new_line[left]:
        frac = 0.5
    else:
        frac = (hm - new_line[left]) / (new_line[left + 1] - new_line[left])
    frac = min(max(frac, 0.0), 1.0)
    left_pos = (left + frac) * float(pixel_size)

    if new_line[right] == new_line[right - 1]:
        frac = 0.5
    else:
        frac = (hm - new_line[right - 1]) / (new_line[right] - new_line[right - 1])
    frac = min(max(frac, 0.0), 1.0)
    right_pos = (right - 1 + frac) * float(pixel_size)

    # A truncated peak (the walk ran into either end) has no measurable width.
    # So does a degenerate one: if the profile has no peak above its own
    # baseline -- a plateau, a monotonic ramp, a trough fed in the wrong way
    # up -- the two edge walks can collapse onto the same sample and the
    # interpolation then returns right_pos < left_pos.  A negative width is
    # never a result, so it is reported as a failure rather than a number.
    width = right_pos - left_pos
    valid = (left != 0) and (right != n - 1) and (width > 0) and (peak_value > 0)
    if not valid:
        width = 0.0

    return FWHMResult(fwhm=float(width), left_pos=float(left_pos),
                      right_pos=float(right_pos), peak_value=peak_value,
                      peak_pos=float(peak_pos * pixel_size),
                      baseline_a=float(coef_a), baseline_b=float(coef_b),
                      valid=valid)


# --------------------------------------------------------------------------
def smooth_savitzky_golay(line, points: int = 11, order: int = 2) -> np.ndarray:
    """Optional Savitzky-Golay smoothing before the FWHM (manual 3.2.1).

    `points` is forced to be odd and larger than `order`; a profile shorter
    than the window is returned unchanged.
    """
    from scipy.signal import savgol_filter

    y = np.asarray(line, dtype=np.float64).ravel()
    w = int(points)
    if w % 2 == 0:
        w += 1
    if w <= order + 1 or y.size < w:
        return y
    return savgol_filter(y, window_length=w, polyorder=int(order))
