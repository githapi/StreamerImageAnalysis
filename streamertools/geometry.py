"""
Geometry: interpolation, oblique sampling, zoom, pixel size
===========================================================

Implements

* the grid interpolation of manual figure 3.4 -- the value of a new pixel is
  the average of the four surrounding original pixels, weighted by the
  distance between the centres (plain bilinear interpolation),
* the oblique measurement grid of figure 3.3, which is what the Streamer
  Width Analyzer builds under the measurement box,
* the integer zoom of manual 2.3 (one output pixel = mean of Z x Z input
  pixels; on screen only -- every calculation stays on the original data),
* crop / rotate / flip / resize for the Image Converter (manual 5.2.4, 5.2.5),
* the pixel size calibration of manual 2.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "bilinear_sample",
    "MeasurementBox",
    "resample_box",
    "zoom_out",
    "crop",
    "rotate",
    "flip",
    "resize",
    "manipulate",
    "pixel_size_from_distance",
    "line_angle_deg",
    "angle_between_lines",
]


# --------------------------------------------------------------------------
# figure 3.4: bilinear interpolation on the original grid
# --------------------------------------------------------------------------
def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray,
                    fill: float = 0.0) -> np.ndarray:
    """Sample `image` at floating point pixel coordinates (manual fig. 3.4).

    ``x`` is the column and ``y`` the row coordinate; integer values refer to
    pixel *centres*.  Samples outside the image get `fill`.
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)

    inside = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)

    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)

    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x1c] * fx * (1 - fy)
         + img[y1c, x0c] * (1 - fx) * fy + img[y1c, x1c] * fx * fy)
    return np.where(inside, v, fill).astype(np.float32)


# --------------------------------------------------------------------------
# figure 3.3: the oblique measurement grid
# --------------------------------------------------------------------------
@dataclass
class MeasurementBox:
    """The line + box of the Streamer Width Analyzer (manual 3.2).

    Attributes
    ----------
    p0, p1 : (x, y)
        The red and green end points of the drawn line, in original pixels.
        Their direction is irrelevant (manual 3.2).
    width : float
        *Measurement area width* in pixels: the box extends ``width/2`` to
        either side of the line.

    Notes
    -----
    Following the manual, the sampled grid ``B`` has shape ``(m, n)`` with

    * ``m`` = the across-line axis (``width`` samples, 1 px apart),
    * ``n`` = the along-line axis (the length of the line, 1 px apart).

    For a **width** measurement the line is drawn *along* the streamer
    channel, so the FWHM comes out perpendicular to it; for a **length**
    measurement the grid is simply transposed (manual 3.2.2).
    """

    p0: Tuple[float, float]
    p1: Tuple[float, float]
    width: float = 20.0

    # -- derived quantities ------------------------------------------------
    @property
    def vector(self) -> np.ndarray:
        return np.array(self.p1, dtype=float) - np.array(self.p0, dtype=float)

    @property
    def length_px(self) -> float:
        """Length of the drawn line in pixels (the *box* length)."""
        return float(np.hypot(*self.vector))

    @property
    def unit_along(self) -> np.ndarray:
        L = self.length_px
        if L < 1e-9:
            raise ValueError("the two end points coincide")
        return self.vector / L

    @property
    def unit_across(self) -> np.ndarray:
        ux, uy = self.unit_along
        return np.array([-uy, ux])

    @property
    def angle_deg(self) -> float:
        return line_angle_deg(self.p0, self.p1)

    @property
    def centre(self) -> np.ndarray:
        return (np.array(self.p0, float) + np.array(self.p1, float)) / 2.0

    def corners(self) -> np.ndarray:
        """The four corners of the box, for drawing (shape (4, 2))."""
        a = np.array(self.p0, float)
        b = np.array(self.p1, float)
        h = self.unit_across * (self.width / 2.0)
        return np.array([a + h, b + h, b - h, a - h])

    def length_mm(self, pixel_size_um: Optional[float]) -> Optional[float]:
        return None if pixel_size_um is None else self.length_px * pixel_size_um / 1000.0

    def moved(self, dx: float, dy: float) -> "MeasurementBox":
        return MeasurementBox((self.p0[0] + dx, self.p0[1] + dy),
                              (self.p1[0] + dx, self.p1[1] + dy), self.width)


def resample_box(image: np.ndarray, box: MeasurementBox,
                 fill: float = 0.0) -> np.ndarray:
    """Build the new oblique image ``B`` under the box (manual fig. 3.3).

    Returns
    -------
    ndarray (m, n)
        ``m`` samples across the line, ``n`` samples along it, spaced one
        original pixel apart in both directions, bilinearly interpolated from
        the original grid.
    """
    img = np.asarray(image, dtype=np.float32)
    n = max(int(round(box.length_px)), 2)          # along the line
    m = max(int(round(box.width)), 1)              # across the line

    along = np.linspace(0.0, box.length_px, n)
    across = (np.arange(m, dtype=float) - (m - 1) / 2.0)

    ua, uc = box.unit_along, box.unit_across
    p0 = np.array(box.p0, dtype=float)

    # coords[i, j] = p0 + along[j]*u_along + across[i]*u_across
    xs = p0[0] + along[None, :] * ua[0] + across[:, None] * uc[0]
    ys = p0[1] + along[None, :] * ua[1] + across[:, None] * uc[1]
    return bilinear_sample(img, xs, ys, fill=fill)


# --------------------------------------------------------------------------
# 2.3 zoom
# --------------------------------------------------------------------------
def zoom_out(image: np.ndarray, factor: int) -> np.ndarray:
    """Integer zoom-out (manual 2.3): one output pixel = mean of Z x Z inputs.

    Only affects what you *look at*; all measurements in this framework are
    done on the original data, exactly as the manual prescribes.
    """
    z = int(factor)
    if z <= 1:
        return np.asarray(image, dtype=np.float32)
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[-2:]
    hh, ww = (h // z) * z, (w // z) * z
    cut = img[..., :hh, :ww]
    new_shape = cut.shape[:-2] + (hh // z, z, ww // z, z)
    return cut.reshape(new_shape).mean(axis=(-3, -1)).astype(np.float32)


# --------------------------------------------------------------------------
# 5.2.4 / 5.2.5 crop, rotate, flip, resize
# --------------------------------------------------------------------------
def crop(image: np.ndarray, left: int = 0, right: int = 0,
         top: int = 0, bottom: int = 0) -> np.ndarray:
    """Crop pixels off the four sides (manual 5.2.4 / tile maker Crop keys)."""
    img = np.asarray(image)
    h, w = img.shape[:2]
    y1 = h - int(bottom) if bottom else h
    x1 = w - int(right) if right else w
    out = img[int(top):y1, int(left):x1]
    if out.size == 0:
        raise ValueError("crop removes the whole image")
    return out


def rotate(image: np.ndarray, angle_deg: float, fill: float = 0.0,
           expand: bool = False) -> np.ndarray:
    """Rotate by an arbitrary angle with the fig. 3.4 interpolation (5.2.4).

    Positive angles rotate counter-clockwise.  With ``expand=True`` the canvas
    grows so nothing is cut off.
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)

    if expand:
        # round first: cos(90 deg) is 6e-17, not 0, and would add a stray pixel
        nw = int(np.ceil(round(abs(w * ca) + abs(h * sa), 9)))
        nh = int(np.ceil(round(abs(w * sa) + abs(h * ca), 9)))
    else:
        nw, nh = w, h

    yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
    cx_o, cy_o = (w - 1) / 2.0, (h - 1) / 2.0
    cx_n, cy_n = (nw - 1) / 2.0, (nh - 1) / 2.0
    dx, dy = xx - cx_n, yy - cy_n
    src_x = cx_o + ca * dx + sa * dy
    src_y = cy_o - sa * dx + ca * dy

    if img.ndim == 2:
        return bilinear_sample(img, src_x, src_y, fill=fill)
    chans = [bilinear_sample(img[..., c], src_x, src_y, fill=fill)
             for c in range(img.shape[2])]
    return np.stack(chans, axis=-1)


def flip(image: np.ndarray, horizontal: bool = False, vertical: bool = False) -> np.ndarray:
    """Mirror the image (manual 5.2.4)."""
    img = np.asarray(image)
    if horizontal:
        img = img[:, ::-1]
    if vertical:
        img = img[::-1]
    return np.ascontiguousarray(img)


def resize(image: np.ndarray, new_width: Optional[int] = None,
           new_height: Optional[int] = None, lock_aspect: bool = False) -> np.ndarray:
    """Resize with bilinear interpolation (manual 5.2.5)."""
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    if new_width is None and new_height is None:
        return img
    if lock_aspect:
        if new_width is None:
            new_width = int(round(w * new_height / h))
        elif new_height is None:
            new_height = int(round(h * new_width / w))
        else:
            new_height = int(round(h * new_width / w))
    new_width = int(new_width if new_width is not None else w)
    new_height = int(new_height if new_height is not None else h)

    # map output pixel centres back onto the input grid
    xs = (np.arange(new_width) + 0.5) * w / new_width - 0.5
    ys = (np.arange(new_height) + 0.5) * h / new_height - 0.5
    X, Y = np.meshgrid(xs, ys)
    if img.ndim == 2:
        return bilinear_sample(img, X, Y)
    return np.stack([bilinear_sample(img[..., c], X, Y) for c in range(img.shape[2])], -1)


def manipulate(image: np.ndarray, mode: int) -> np.ndarray:
    """Tile maker ``ManipulateMode`` (manual 6.2.2, [Original]).

    ``0`` none, ``1/2/3`` rotate clockwise 90/180/270, ``4/5/6/7`` mirror
    horizontally first and then rotate by 0/90/180/270.
    """
    img = np.asarray(image)
    mode = int(mode) % 8
    if mode >= 4:
        img = img[:, ::-1]
        mode -= 4
    if mode:                       # np.rot90 is counter-clockwise -> use -k
        img = np.rot90(img, k=-mode)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------
# 2.4 pixel size
# --------------------------------------------------------------------------
def pixel_size_from_distance(distance_px: float, real_distance_mm: float) -> float:
    """Pixel size in um/px from a known real-life separation (manual 2.4).

    Open an image of a ruler or a calibration target, measure the separation
    of two known points in pixels, and pass both numbers here.  Calculations
    assume square pixels, as in Streamertools.
    """
    if distance_px <= 0:
        raise ValueError("distance_px must be positive")
    return float(real_distance_mm) * 1000.0 / float(distance_px)


# --------------------------------------------------------------------------
# 3.4 angles
# --------------------------------------------------------------------------
def line_angle_deg(p0: Sequence[float], p1: Sequence[float]) -> float:
    """Angle of a line with the +x axis, in degrees, measured anticlockwise
    in image coordinates (y downwards is taken into account)."""
    dx = p1[0] - p0[0]
    dy = -(p1[1] - p0[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def angle_between_lines(line_a: Tuple[Sequence[float], Sequence[float]],
                        line_b: Tuple[Sequence[float], Sequence[float]]) -> float:
    """Angle between two lines in degrees, folded into [0, 180) (manual 3.4)."""
    a = line_angle_deg(*line_a)
    b = line_angle_deg(*line_b)
    return float(abs((a - b) % 360.0 if (a - b) % 360.0 <= 180 else 360 - (a - b) % 360.0))
