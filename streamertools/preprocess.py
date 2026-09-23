"""
Calibration and denoising of raw camera stacks
==============================================

This is the part that is not in the Streamertools manual but that every ICCD
measurement needs before the manual's algorithms mean anything: remove the
camera bias, optionally put different shots on a common physical brightness
scale, and clean up hot pixels.  It follows the pipeline of the notebook this
framework replaces, with the pieces separated so each one can be tested.

Order matters:

    raw -> dark/bias subtraction -> physical brightness (Mf) -> denoise

Denoising is deliberately a *per frame, on demand* operation: a long kinetic
series of full-sensor frames should never need a second full copy in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter

from .io import ImageStack

__all__ = [
    "estimate_bias",
    "subtract_dark",
    "make_dark",
    "compute_Mf",
    "apply_physical_normalization",
    "remove_small_bright_blobs",
    "remove_hot_pixels",
    "preprocess_frame",
    "preprocess_stack",
    "DenoiseSettings",
]


# --------------------------------------------------------------------------
# dark / bias
# --------------------------------------------------------------------------
def estimate_bias(frame: np.ndarray, corner_frac: float = 0.1) -> float:
    """Median of the four image corners -- a scalar stand-in for a dark frame."""
    f = np.asarray(frame, dtype=np.float32)
    h, w = f.shape[-2:]
    ch, cw = max(1, int(h * corner_frac)), max(1, int(w * corner_frac))
    corners = np.concatenate([f[..., :ch, :cw].ravel(), f[..., :ch, -cw:].ravel(),
                              f[..., -ch:, :cw].ravel(), f[..., -ch:, -cw:].ravel()])
    return float(np.median(corners))


def make_dark(stack: Union[ImageStack, np.ndarray],
              dark_stack: Union[ImageStack, np.ndarray, None] = None,
              from_frames: Optional[int] = None,
              corner_frac: float = 0.1) -> Union[np.ndarray, float]:
    """Pick a dark reference, in the priority order dark file > frames > corners.

    Parameters
    ----------
    dark_stack :
        A separately measured dark/background acquisition; its frames are
        averaged into one dark frame.
    from_frames : int
        Use the mean of the first N frames of `stack` itself -- handy for a
        kinetic series whose first frames are pre-discharge background.
    """
    data = stack.data if isinstance(stack, ImageStack) else np.asarray(stack, np.float32)
    if dark_stack is not None:
        d = dark_stack.data if isinstance(dark_stack, ImageStack) else np.asarray(dark_stack)
        d = np.asarray(d, dtype=np.float32)
        if d.ndim == 3:
            d = d.mean(axis=0)
        if d.shape != data.shape[1:]:
            raise ValueError(f"dark frame {d.shape} does not match data {data.shape[1:]}")
        return d
    if from_frames:
        n = int(from_frames)
        if n >= data.shape[0]:
            raise ValueError(f"from_frames={n} must be smaller than the number of "
                             f"frames ({data.shape[0]})")
        return data[:n].mean(axis=0).astype(np.float32)
    return estimate_bias(data[0], corner_frac)


def subtract_dark(stack: Union[ImageStack, np.ndarray],
                  dark: Union[np.ndarray, float],
                  clip_negative: bool = True, in_place: bool = False):
    """Subtract a scalar bias or a 2-D dark frame from every frame.

    Parameters
    ----------
    in_place : bool
        Overwrite the input instead of allocating a second stack.  A
        40-frame 2560x2160 series is 0.9 GB, so the default (a fresh copy)
        peaks at 1.8 GB; in place it stays at 0.9 GB and is several times
        faster.  Only pass True when you do not need the raw counts again --
        re-loading the file is the way back.
    """
    if isinstance(stack, ImageStack):
        out = subtract_dark(stack.data, dark, clip_negative, in_place)
        return stack if in_place else stack.copy_with(out)
    data = np.asarray(stack, dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    if in_place:
        np.subtract(data, dark, out=data)
        if clip_negative:
            np.clip(data, 0.0, None, out=data)
        return data
    out = data - dark
    return np.clip(out, 0.0, None) if clip_negative else out


# --------------------------------------------------------------------------
# physical brightness normalisation
# --------------------------------------------------------------------------
def compute_Mf(Vg: float, D: float, c_max: float, c_min: float,
               norm_factor: float = 500.0) -> float:
    """Sensitivity factor of one acquisition setting.

    ``Mf = norm_factor * exp(Vg / 68.9) / (D^2 * (Cmax - Cmin))``

    with `Vg` the MCP gain voltage, `D` the f-number of the lens and
    ``Cmax - Cmin`` the count window.  Dividing by Mf makes shots taken at
    different gain / aperture directly comparable.
    """
    denom = float(D) ** 2 * max(float(c_max) - float(c_min), 1e-12)
    return float(norm_factor * np.exp(float(Vg) / 68.9) / denom)


def apply_physical_normalization(stack: Union[ImageStack, np.ndarray],
                                 Vg: float, D: float, c_max: float, c_min: float,
                                 norm_factor: float = 500.0,
                                 Mf_ref: float = 1.0) -> Tuple[object, float]:
    """Scale a stack onto the common brightness scale ``Mf_ref``.

    Returns ``(scaled, Mf)``.  Keep `Mf_ref` identical across every shot you
    want to compare.
    """
    Mf = compute_Mf(Vg, D, c_max, c_min, norm_factor)
    if isinstance(stack, ImageStack):
        out = (stack.data * (Mf_ref / Mf)).astype(np.float32)
        return stack.copy_with(out), Mf
    return (np.asarray(stack, dtype=np.float32) * (Mf_ref / Mf)).astype(np.float32), Mf


# --------------------------------------------------------------------------
# denoising
# --------------------------------------------------------------------------
def remove_small_bright_blobs(frame: np.ndarray, threshold_percentile: float = 90.0,
                              min_component_size: int = 10) -> np.ndarray:
    """Zero bright components smaller than `min_component_size` pixels.

    Strips hot pixels and cosmic rays while keeping larger structures.  This
    does **not** subtract a baseline -- that is the dark subtraction above.
    """
    f = np.asarray(frame, dtype=np.float32)
    nz = f[f > 0]
    if nz.size == 0:
        return f.copy()
    labeled, _ = label(f > np.percentile(nz, threshold_percentile))
    keep = np.bincount(labeled.ravel()) >= int(min_component_size)
    keep[0] = False
    out = f.copy()
    out[~keep[labeled]] = 0.0
    return out


def remove_hot_pixels(frame: np.ndarray, sigma: float = 6.0, size: int = 3) -> np.ndarray:
    """Replace isolated outliers by the local median.

    A pixel is an outlier when it deviates from the median-filtered image by
    more than `sigma` times the robust noise estimate.  Unlike the blob
    filter this keeps the rest of the frame untouched, so it is the safer
    choice before a width measurement.
    """
    f = np.asarray(frame, dtype=np.float32)
    med = median_filter(f, size=size)
    resid = f - med
    mad = np.median(np.abs(resid - np.median(resid)))
    noise = 1.4826 * mad if mad > 0 else float(resid.std()) or 1.0
    bad = resid > sigma * noise
    out = f.copy()
    out[bad] = med[bad]
    return out


@dataclass
class DenoiseSettings:
    """Bundle of the denoising switches, so viewer and export cannot diverge."""
    gaussian: bool = False
    sigma: float = 1.5
    median: bool = False
    median_size: int = 3
    hot_pixels: bool = False
    hot_sigma: float = 6.0
    blobs: bool = False
    blob_percentile: float = 90.0
    blob_min_size: int = 10

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return preprocess_frame(
            frame, do_gaussian=self.gaussian, sigma=self.sigma,
            do_median=self.median, median_size=self.median_size,
            do_hot_pixels=self.hot_pixels, hot_sigma=self.hot_sigma,
            do_blob=self.blobs, threshold_percentile=self.blob_percentile,
            min_component_size=self.blob_min_size)


def preprocess_frame(frame: np.ndarray, do_gaussian: bool = False, sigma: float = 1.5,
                     do_median: bool = False, median_size: int = 3,
                     do_hot_pixels: bool = False, hot_sigma: float = 6.0,
                     do_blob: bool = False, threshold_percentile: float = 90.0,
                     min_component_size: int = 10) -> np.ndarray:
    """Apply the selected denoisers to ONE frame (float32 in, float32 out)."""
    out = np.asarray(frame, dtype=np.float32)
    if do_hot_pixels:
        out = remove_hot_pixels(out, sigma=hot_sigma)
    if do_median:
        out = median_filter(out, size=median_size)
    if do_gaussian:
        out = gaussian_filter(out, sigma=sigma)
    if do_blob:
        out = remove_small_bright_blobs(out, threshold_percentile, min_component_size)
    return out.astype(np.float32)


def preprocess_stack(stack: Union[ImageStack, np.ndarray], **kwargs):
    """Whole-stack version of :func:`preprocess_frame` (allocates a copy)."""
    if isinstance(stack, ImageStack):
        return stack.copy_with(preprocess_stack(stack.data, **kwargs))
    data = np.asarray(stack, dtype=np.float32)
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = preprocess_frame(data[i], **kwargs)
    return out
