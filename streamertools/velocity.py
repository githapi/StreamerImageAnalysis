"""
Streamer propagation velocity
=============================

This is the addition to the Streamertools set of chapters, implementing the
two-image method:

    To measure the streamer propagation velocity, we obtained two images with
    a very short camera exposure time (5 ns) when streamers crossed half of
    the gap.  The two images were taken with a difference in exposure delay
    of 10 ns.  The velocity was determined by dividing the distance between
    streamer head positions in the two images by this delay time.

So the measurement is: find the streamer head in image A, find it again in
image B, divide the displacement by the delay difference.  Everything else
here exists to make those two head positions defensible.

Head positions
--------------
:func:`detect_head` locates the head of the streamer that has protruded
furthest into the gap.  Four definitions are available:

``"front"``     the default, and the definition of the literature this
                package follows: the half-maximum crossing at the leading end
                of that streamer's axial intensity profile (see below)
``"tip"``       furthest pixel above the threshold, refined with an
                intensity-weighted centroid over a small window
``"centroid"``  intensity-weighted centroid of the leading part of the
                channel (the brightest `head_fraction` of the axial extent)
``"max"``       position of the brightest pixel

The ``"front"`` method follows Briels et al (J. Phys. D 41 234004, 2008),
Winands et al (J. Phys. D 41 234001, 2008) and Nijdam (PhD thesis, TU/e 2011,
section 3.4.3):

* only the streamers *protruding furthest* into the gap are evaluated, since
  they propagate closest to the focal plane of the camera (Winands; Briels
  evaluates "the streamers that have propagated furthest");
* lengths along the propagation direction are measured with the same FWHM
  criterion as the diameter (Briels; Nijdam), so the head position is where
  the axial profile through that streamer falls to half its maximum, on the
  leading side.  During a short gate the head is a stripe of light
  (Winands), and this edge is where the head was at the *end* of the gate;
* velocity is the slope of head position against end-of-exposure time over
  a series of delays (Briels; Nijdam fig. 3.18), see
  :func:`velocity_from_fronts`.

You can always bypass detection and pass clicked coordinates instead
(``manual_heads=`` here, or the picker in :mod:`streamertools.interactive`).

Branches
--------
:func:`measure_velocity_branches` detects every connected channel in both
images, matches them one-to-one (globally, by displacement cost) and returns
one velocity per branch, so a tree of streamers gives a distribution rather
than a single number.

Uncertainty
-----------
Every result carries an error bar built from the head-localisation spread,
the timing jitter you quote for the delay, and the relative uncertainty of
the pixel size calibration.  With a 10 ns delay and a ~17 um pixel, a 1-pixel
localisation error is already about 2 x 10^3 m/s -- worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, label

from .io import ImageStack

__all__ = [
    "HeadSettings", "HeadPosition", "VelocityResult",
    "detect_head", "detect_heads", "measure_velocity",
    "measure_velocity_branches", "track_head", "velocity_from_series",
    "velocity_frame_to_frame", "velocity_from_heads",
    "FrontVelocityResult", "velocity_from_fronts",
]

HeadMethod = Literal["front", "tip", "centroid", "max"]
ThresholdMode = Literal["fraction_of_max", "absolute", "percentile", "otsu"]


# --------------------------------------------------------------------------
@dataclass
class HeadSettings:
    """How a streamer head is defined and found.

    The threshold and component settings pick the streamer that protruded
    furthest; `method` then says which point of it is the head (see the
    module notes; ``"front"`` is the literature definition).
    """

    method: HeadMethod = "front"
    threshold_mode: ThresholdMode = "fraction_of_max"
    threshold: float = 0.35            # fraction of max / absolute counts / percentile
    smooth_sigma: float = 1.0          # gaussian pre-smoothing, pixels
    min_component_size: int = 20       # ignore smaller bright blobs (noise)
    refine_window: float = 5.0         # half-size of the centroid window, pixels
    head_fraction: float = 0.15        # for method="centroid": leading part used
    position_uncertainty_px: Optional[float] = None   # override the estimate

    # -- method="front" --------------------------------------------------------
    front_level: float = 0.5           # edge at this fraction from background to maximum
    front_strip_px: float = 5.0        # strip across the channel averaged into the profile
    front_search_px: float = 120.0     # how far behind the tip the head maximum may lie
    front_ahead_px: float = 30.0       # profile ahead of the tip; its far half = background

    # -- "is there a streamer here at all?" --------------------------------
    # A relative threshold always finds *something*, even in a pre-discharge
    # background frame where the brightest pixel is noise.  A head is only
    # accepted when its peak stands `min_snr` robust sigmas above the frame's
    # background.  Set min_snr=0 to disable; min_peak adds an absolute floor
    # in counts when you know what a real streamer looks like.
    min_snr: float = 5.0
    min_peak: Optional[float] = None

    def level(self, image: np.ndarray) -> float:
        img = np.asarray(image, dtype=np.float32)
        if self.threshold_mode == "absolute":
            return float(self.threshold)
        if self.threshold_mode == "fraction_of_max":
            return float(np.nanmax(img)) * float(self.threshold)
        if self.threshold_mode == "percentile":
            return float(np.nanpercentile(img, self.threshold))
        if self.threshold_mode == "otsu":
            return _otsu(img)
        raise ValueError(f"unknown threshold mode {self.threshold_mode!r}")


def _otsu(image: np.ndarray, bins: int = 256) -> float:
    img = np.asarray(image, dtype=np.float64).ravel()
    hist, edges = np.histogram(img, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centres) / w0
        m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.nanargmax(between))])


# --------------------------------------------------------------------------
@dataclass
class HeadPosition:
    """A streamer head in one image."""

    x: float                      # column, sub-pixel
    y: float                      # row, sub-pixel
    axial: float = 0.0            # projection on the propagation axis, pixels
    intensity: float = 0.0        # peak count value near the head
    n_pixels: int = 0             # size of the channel it belongs to
    sigma_px: float = 1.0         # localisation uncertainty, pixels
    label: int = 0                # connected-component id
    method: str = "tip"

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def __repr__(self) -> str:                                   # pragma: no cover
        return (f"HeadPosition(x={self.x:.2f}, y={self.y:.2f}, "
                f"+-{self.sigma_px:.2f} px, {self.method})")


# --------------------------------------------------------------------------
def _resolve_axis(image: np.ndarray, mask: np.ndarray,
                  axis: Union[Sequence[float], str, None],
                  origin: Optional[Sequence[float]]) -> np.ndarray:
    """Unit vector (dx, dy) of the propagation direction, in image coordinates."""
    named = {"down": (0.0, 1.0), "up": (0.0, -1.0),
             "right": (1.0, 0.0), "left": (-1.0, 0.0)}
    if isinstance(axis, str):
        v = np.array(named[axis.lower()], dtype=float)
        return v / np.linalg.norm(v)
    if axis is not None:
        v = np.asarray(axis, dtype=float).ravel()
        if v.size == 4:                      # two points: (x0, y0, x1, y1)
            v = v[2:] - v[:2]
        n = np.linalg.norm(v)
        if n < 1e-9:
            raise ValueError("propagation axis has zero length")
        return v / n

    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        raise ValueError("not enough bright pixels to estimate a propagation axis")
    pts = np.stack([xs, ys], axis=1).astype(float)
    if origin is not None:
        o = np.asarray(origin, dtype=float)
        far = pts[np.argmax(np.linalg.norm(pts - o, axis=1))]
        v = far - o
        return v / np.linalg.norm(v)
    # principal axis of the bright pixels, pointing away from the centre of mass
    w = np.asarray(image, dtype=float)[ys, xs]
    mean = np.average(pts, axis=0, weights=w)
    cov = np.cov((pts - mean).T, aweights=w)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    proj = (pts - mean) @ v
    # point towards the *dim* end: a streamer head is at the far end of the channel
    if np.average(proj, weights=w) > 0:
        v = -v
    return v / np.linalg.norm(v)


def _components(image: np.ndarray, settings: HeadSettings
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed image, boolean mask and labelled components above threshold."""
    img = np.asarray(image, dtype=np.float32)
    smooth = gaussian_filter(img, settings.smooth_sigma) if settings.smooth_sigma else img
    mask = smooth >= settings.level(smooth)
    labels, n = label(mask)
    if n:
        sizes = np.bincount(labels.ravel())
        too_small = np.where(sizes < settings.min_component_size)[0]
        drop = np.isin(labels, too_small)
        labels[drop] = 0
        mask = labels > 0
    return smooth, mask, labels


def _background_sigma(image: np.ndarray) -> Tuple[float, float]:
    """Robust (background level, noise sigma) of a frame, via the MAD."""
    img = np.asarray(image, dtype=np.float32)
    bg = float(np.median(img))
    mad = float(np.median(np.abs(img - bg)))
    noise = 1.4826 * mad
    if noise <= 0:
        noise = float(img.std())
    return bg, noise


def _has_real_signal(smooth: np.ndarray, sel: np.ndarray,
                     settings: HeadSettings) -> bool:
    """Does this component stand far enough above the frame's background?"""
    peak = float(smooth[sel].max())
    if settings.min_peak is not None and peak < float(settings.min_peak):
        return False
    if settings.min_snr and settings.min_snr > 0:
        bg, noise = _background_sigma(smooth)
        if noise > 0 and (peak - bg) < settings.min_snr * noise:
            return False
    return True


def _front_from_pixels(smooth: np.ndarray, sel: np.ndarray, axis: np.ndarray,
                       settings: HeadSettings, label_id: int = 0) -> HeadPosition:
    """The literature head: the half-maximum crossing at the channel's leading end.

    A strip ``front_strip_px`` wide, centred on the channel just behind its
    tip, is resampled along the propagation axis and averaged across into an
    axial profile (the averaging suppresses single-photon noise, as for the
    diameter).  Starting at the head maximum behind the tip, the front is
    where the profile drops below ``front_level`` of the way from the local
    background -- the far part of the profile ahead of the head -- to that
    maximum, and stays below it for three samples, so a one-pixel noise dip
    inside the head does not end it early and a separate blob further ahead
    does not extend it.  The crossing is interpolated linearly.  Its
    uncertainty is the background noise of the profile over the edge slope.

    Falls back to ``method="tip"`` (and says so in ``.method``) when the
    edge does not lie inside the profile, e.g. at the image border.
    """
    import warnings

    from .geometry import MeasurementBox, resample_box

    ys, xs = np.nonzero(sel)
    pts = np.stack([xs, ys], axis=1).astype(float)
    w = smooth[ys, xs].astype(float)
    proj = pts @ axis
    perp = np.array([-axis[1], axis[0]])
    tip = float(proj.max())
    lead = proj >= tip - max(2.0 * settings.refine_window, 2.0)
    lateral = float(np.average(pts[lead] @ perp, weights=np.maximum(w[lead], 1e-12)))

    def fallback():
        return _head_from_pixels(smooth, sel, axis, replace(settings, method="tip"), label_id)

    back = float(np.clip(tip - proj.min(), 3.0 * settings.refine_window,
                         max(settings.front_search_px, 3.0 * settings.refine_window)))
    ahead = float(max(settings.front_ahead_px, 4.0))
    p0 = (tip - back) * axis + lateral * perp
    p1 = (tip + ahead) * axis + lateral * perp
    box = MeasurementBox(tuple(p0), tuple(p1), float(max(settings.front_strip_px, 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns
        profile = np.nanmean(resample_box(smooth, box, fill=np.nan), axis=0)
    pos = np.linspace(0.0, box.length_px, profile.size)          # from p0, along the axis
    finite = np.isfinite(profile)

    behind = finite & (pos <= back)
    if not behind.any():
        return fallback()
    k_max = int(np.flatnonzero(behind)[np.argmax(profile[behind])])
    peak = float(profile[k_max])

    far = finite & (pos >= back + ahead / 2.0)
    if far.sum() >= 3:
        bg = float(np.median(profile[far]))
        noise = 1.4826 * float(np.median(np.abs(profile[far] - bg)))
        noise = noise or float(np.std(profile[far]))
    else:
        bg, noise = _background_sigma(smooth)
        noise /= np.sqrt(box.width)
    if peak <= bg:
        return fallback()
    level = bg + float(settings.front_level) * (peak - bg)

    gap = 3                                                      # samples below = ended
    last_above = k_max
    for i in range(k_max + 1, profile.size):
        if not finite[i]:
            return fallback()                                    # ran off the image
        if profile[i] >= level:
            last_above = i
        elif i - last_above >= gap:
            break
    else:
        return fallback()                                        # never ended inside
    y0, y1 = float(profile[last_above]), float(profile[last_above + 1])
    step = float(pos[last_above + 1] - pos[last_above])
    x = float(pos[last_above]) + step * (y0 - level) / (y0 - y1)
    slope = (y0 - y1) / step

    sigma = noise / slope if slope > 0 else 1.0
    if settings.position_uncertainty_px is not None:
        sigma = float(settings.position_uncertainty_px)
    xy = p0 + x * axis
    return HeadPosition(x=float(xy[0]), y=float(xy[1]), axial=float(xy @ axis),
                        intensity=peak, n_pixels=int(sel.sum()),
                        sigma_px=float(max(sigma, 0.05)), label=int(label_id),
                        method="front")


def _head_from_pixels(smooth: np.ndarray, sel: np.ndarray, axis: np.ndarray,
                      settings: HeadSettings, label_id: int = 0) -> HeadPosition:
    """Locate the head inside one selected component."""
    if settings.method == "front":
        return _front_from_pixels(smooth, sel, axis, settings, label_id)
    ys, xs = np.nonzero(sel)
    pts = np.stack([xs, ys], axis=1).astype(float)
    w = smooth[ys, xs].astype(float)
    proj = pts @ axis

    if settings.method == "max":
        k = int(np.argmax(w))
        cx, cy = pts[k]
        used = np.array([True])
        sigma = 1.0 / np.sqrt(12)                   # a pixel, uniformly
    elif settings.method == "centroid":
        cut = proj.max() - settings.head_fraction * (proj.max() - proj.min())
        used = proj >= cut
        ww = w[used]
        cx, cy = np.average(pts[used], axis=0, weights=ww)
        spread = np.sqrt(np.average((proj[used] - np.average(proj[used], weights=ww)) ** 2,
                                    weights=ww))
        sigma = float(spread / max(np.sqrt(max(used.sum(), 1)), 1.0))
    else:                                            # "tip"
        tip = pts[int(np.argmax(proj))]
        near = np.linalg.norm(pts - tip, axis=1) <= settings.refine_window
        used = near
        ww = w[near]
        if ww.sum() <= 0:
            cx, cy = tip
            sigma = 1.0
        else:
            cx, cy = np.average(pts[near], axis=0, weights=ww)
            p = proj[near]
            m = np.average(p, weights=ww)
            spread = np.sqrt(np.average((p - m) ** 2, weights=ww))
            n_eff = (ww.sum() ** 2) / max(float((ww ** 2).sum()), 1e-12)
            sigma = float(spread / max(np.sqrt(n_eff), 1.0))

    if settings.position_uncertainty_px is not None:
        sigma = float(settings.position_uncertainty_px)
    sigma = float(max(sigma, 0.05))

    return HeadPosition(x=float(cx), y=float(cy),
                        axial=float(np.array([cx, cy]) @ axis),
                        intensity=float(w.max()), n_pixels=int(sel.sum()),
                        sigma_px=sigma, label=int(label_id),
                        method=settings.method)


def detect_head(image, axis: Union[Sequence[float], str, None] = None,
                origin: Optional[Sequence[float]] = None,
                settings: Optional[HeadSettings] = None,
                roi: Optional[Tuple[int, int, int, int]] = None,
                **overrides) -> HeadPosition:
    """Find the leading streamer head in one image.

    Parameters
    ----------
    image : ndarray (H, W)
        A single, preferably dark-subtracted frame.
    axis :
        Propagation direction: ``"down"``/``"up"``/``"left"``/``"right"``,
        a vector ``(dx, dy)``, four numbers ``(x0, y0, x1, y1)``, or ``None``
        to estimate it from the image.
    origin : (x, y), optional
        The electrode tip.  Used to orient an estimated axis, and worth
        giving whenever you know it.
    roi : (x0, y0, x1, y1), optional
        Restrict the search to this box.
    **overrides
        Any :class:`HeadSettings` field.
    """
    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})
    img = np.asarray(getattr(image, "data", image), dtype=np.float32)
    if img.ndim == 3:
        img = img[0]
    x_off = y_off = 0
    if roi:
        x0, y0, x1, y1 = (int(v) for v in roi)
        img = img[y0:y1, x0:x1]
        x_off, y_off = x0, y0

    smooth, mask, labels = _components(img, s)
    if not mask.any():
        raise ValueError("nothing above the threshold -- lower `threshold` or "
                         "check the dark subtraction")
    ax = _resolve_axis(smooth, mask,
                       axis, None if origin is None else
                       (np.asarray(origin, float) - np.array([x_off, y_off])))

    # the component whose tip reaches furthest along the axis: the component of
    # the pixel reaching furthest (lowest label id on a tie).  One pass over the
    # kept pixels -- a noisy frame has thousands of labels before the size
    # filter, and testing each against the whole image took ~10 s per frame.
    ys, xs = np.nonzero(labels)
    proj = np.stack([xs, ys], 1).astype(float) @ ax
    lab_ids = labels[ys, xs]
    best = labels == lab_ids[proj == proj.max()].min()
    if not _has_real_signal(smooth, best, s):
        bg, noise = _background_sigma(smooth)
        raise ValueError(
            f"no streamer in this frame: the brightest structure peaks at "
            f"{float(smooth[best].max()):.1f}, only "
            f"{(float(smooth[best].max()) - bg) / max(noise, 1e-9):.1f} sigma above "
            f"the background. Lower `min_snr` if this frame really does hold a head.")
    head = _head_from_pixels(smooth, best, ax, s, label_id=1)
    head.x += x_off
    head.y += y_off
    head.axial = float(head.xy @ ax)
    return head


def detect_heads(image, axis=None, origin=None, settings: Optional[HeadSettings] = None,
                 max_branches: int = 50, **overrides) -> List[HeadPosition]:
    """One head per connected channel, sorted by how far they have travelled."""
    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})
    img = np.asarray(getattr(image, "data", image), dtype=np.float32)
    if img.ndim == 3:
        img = img[0]
    smooth, mask, labels = _components(img, s)
    if not mask.any():
        return []
    ax = _resolve_axis(smooth, mask, axis, origin)
    heads = []
    for lab in np.unique(labels[labels > 0]):          # only what survived the size filter
        sel = labels == lab
        if sel.sum() < s.min_component_size or not _has_real_signal(smooth, sel, s):
            continue
        heads.append(_head_from_pixels(smooth, sel, ax, s, label_id=lab))
    heads.sort(key=lambda h: h.axial, reverse=True)
    return heads[:max_branches]


# --------------------------------------------------------------------------
@dataclass
class VelocityResult:
    """A propagation velocity with its uncertainty and all its inputs."""

    head_a: HeadPosition
    head_b: HeadPosition
    delay_ns: float
    pixel_size_um: Optional[float]
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    delay_jitter_ns: float = 0.0
    pixel_size_rel_error: float = 0.0
    branch: int = 0

    # -- displacement ------------------------------------------------------
    @property
    def displacement_px(self) -> np.ndarray:
        return self.head_b.xy - self.head_a.xy

    @property
    def distance_px(self) -> float:
        return float(np.hypot(*self.displacement_px))

    @property
    def axial_distance_px(self) -> float:
        """Displacement projected on the propagation axis (can be negative)."""
        return float(self.displacement_px @ self.axis)

    @property
    def lateral_distance_px(self) -> float:
        perp = np.array([-self.axis[1], self.axis[0]])
        return float(self.displacement_px @ perp)

    @property
    def distance_mm(self) -> Optional[float]:
        if self.pixel_size_um is None:
            return None
        return self.distance_px * self.pixel_size_um / 1000.0

    @property
    def sigma_distance_px(self) -> float:
        return float(np.hypot(self.head_a.sigma_px, self.head_b.sigma_px))

    # -- velocity ----------------------------------------------------------
    @property
    def velocity_m_per_s(self) -> Optional[float]:
        """|displacement| / delay.  1 mm/ns = 1e6 m/s."""
        if self.pixel_size_um is None or self.delay_ns == 0:
            return None
        metres = self.distance_px * self.pixel_size_um * 1e-6
        return metres / (self.delay_ns * 1e-9)

    @property
    def axial_velocity_m_per_s(self) -> Optional[float]:
        if self.pixel_size_um is None or self.delay_ns == 0:
            return None
        metres = self.axial_distance_px * self.pixel_size_um * 1e-6
        return metres / (self.delay_ns * 1e-9)

    @property
    def velocity_mm_per_ns(self) -> Optional[float]:
        v = self.velocity_m_per_s
        return None if v is None else v * 1e-6

    @property
    def velocity_px_per_ns(self) -> float:
        return self.distance_px / self.delay_ns if self.delay_ns else float("nan")

    @property
    def sigma_velocity_px_per_ns(self) -> float:
        """Uncertainty of :attr:`velocity_px_per_ns`.

        Head localisation and timing jitter only: the pixel-size calibration
        does not enter a velocity that is still in pixels.
        """
        if not self.delay_ns or self.distance_px == 0:
            return float("nan")
        rel = np.hypot(self.sigma_distance_px / self.distance_px,
                       self.delay_jitter_ns / self.delay_ns)
        return float(abs(self.velocity_px_per_ns) * rel)

    @property
    def sigma_velocity_m_per_s(self) -> Optional[float]:
        """Propagated uncertainty of :attr:`velocity_m_per_s`."""
        v = self.velocity_m_per_s
        if v is None or self.distance_px == 0:
            return None
        rel = np.sqrt((self.sigma_distance_px / self.distance_px) ** 2
                      + (self.delay_jitter_ns / self.delay_ns) ** 2
                      + self.pixel_size_rel_error ** 2)
        return float(abs(v) * rel)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        """Every number of this measurement, ready for a DataFrame row.

        ``velocity_px_per_ns`` is always populated; the metric columns are
        None without a pixel size.  It is the *only* velocity there is in an
        uncalibrated measurement, so it belongs in the table rather than
        leaving a caller with a column of NaN and nothing to plot.
        """
        v = self.velocity_m_per_s
        return {
            "branch": self.branch,
            "x_a": self.head_a.x, "y_a": self.head_a.y,
            "x_b": self.head_b.x, "y_b": self.head_b.y,
            "distance_px": self.distance_px,
            "axial_distance_px": self.axial_distance_px,
            "lateral_distance_px": self.lateral_distance_px,
            "distance_mm": self.distance_mm,
            "delay_ns": self.delay_ns,
            "velocity_px_per_ns": self.velocity_px_per_ns,
            "sigma_velocity_px_per_ns": self.sigma_velocity_px_per_ns,
            "velocity_m_per_s": v,
            "velocity_mm_per_ns": self.velocity_mm_per_ns,
            "sigma_velocity_m_per_s": self.sigma_velocity_m_per_s,
            "axial_velocity_m_per_s": self.axial_velocity_m_per_s,
        }

    def __repr__(self) -> str:                                   # pragma: no cover
        v, s = self.velocity_m_per_s, self.sigma_velocity_m_per_s
        if v is None:
            return (f"<VelocityResult {self.distance_px:.2f} px in "
                    f"{self.delay_ns:g} ns -- no pixel size given>")
        return (f"<VelocityResult v = {v/1e6:.3f} +- {(s or 0)/1e6:.3f} mm/ns "
                f"({v:.3g} m/s), {self.distance_px:.2f} px in {self.delay_ns:g} ns>")


# --------------------------------------------------------------------------
def _as_frame(source, frame: int = 0) -> np.ndarray:
    """Accept an ImageStack, an array, a (stack, index) pair or a file path."""
    from pathlib import Path
    from .io import load
    if isinstance(source, (str, Path)):
        source = load(source)
    if isinstance(source, tuple) and len(source) == 2:
        source, frame = source
    data = np.asarray(getattr(source, "data", source), dtype=np.float32)
    return data[frame] if data.ndim == 3 else data


def measure_velocity(image_a, image_b, delay_ns: float = 10.0,
                     pixel_size_um: Optional[float] = None,
                     axis: Union[Sequence[float], str, None] = None,
                     origin: Optional[Sequence[float]] = None,
                     settings: Optional[HeadSettings] = None,
                     manual_heads: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                     roi: Optional[Tuple[int, int, int, int]] = None,
                     delay_jitter_ns: float = 0.0,
                     pixel_size_rel_error: float = 0.0,
                     frames: Tuple[int, int] = (0, 0),
                     **overrides) -> VelocityResult:
    """Velocity from two short-exposure images taken `delay_ns` apart.

    Parameters
    ----------
    image_a, image_b :
        Two frames, given as arrays, :class:`~streamertools.io.ImageStack`
        objects, file paths, or ``(stack, frame_index)`` pairs.  Passing the
        same stack twice with different indices is the normal way to use two
        frames of one kinetic series.
    delay_ns : float
        Difference in exposure delay between the two images (10 ns in the
        procedure this implements).  The exposure time itself (5 ns) does not
        enter the calculation, but it does set how much the head smears
        during one image -- see :attr:`HeadSettings.position_uncertainty_px`.
    pixel_size_um : float
        Image-plane pixel size (manual 2.4).  Without it the result stays in
        pixels per nanosecond.
    manual_heads : ((xa, ya), (xb, yb)), optional
        Clicked head positions; skips detection entirely.
    delay_jitter_ns, pixel_size_rel_error :
        Fed into the error bar.

    Returns
    -------
    VelocityResult
    """
    a = _as_frame(image_a, frames[0])
    b = _as_frame(image_b, frames[1])
    if a.shape != b.shape:
        raise ValueError(f"the two images differ in size: {a.shape} vs {b.shape}")

    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})

    if manual_heads is not None:
        (xa, ya), (xb, yb) = manual_heads
        sigma = s.position_uncertainty_px if s.position_uncertainty_px is not None else 1.5
        head_a = HeadPosition(float(xa), float(ya), sigma_px=sigma, method="manual")
        head_b = HeadPosition(float(xb), float(yb), sigma_px=sigma, method="manual")
        ax = np.asarray(axis, dtype=float) if axis is not None and not isinstance(axis, str) \
            else None
        if ax is None or ax.size != 2:
            d = head_b.xy - head_a.xy
            n = np.linalg.norm(d)
            ax = d / n if n > 1e-9 else np.array([0.0, 1.0])
        else:
            ax = ax / np.linalg.norm(ax)
    else:
        head_a = detect_head(a, axis=axis, origin=origin, settings=s, roi=roi)
        # reuse the same axis for the second image so the projection is comparable
        smooth, mask, _ = _components(a, s)
        ax = _resolve_axis(smooth, mask, axis, origin)
        head_b = detect_head(b, axis=ax, origin=origin, settings=s, roi=roi)

    head_a.axial = float(head_a.xy @ ax)
    head_b.axial = float(head_b.xy @ ax)
    return VelocityResult(head_a, head_b, float(delay_ns), pixel_size_um, ax,
                          float(delay_jitter_ns), float(pixel_size_rel_error))


# --------------------------------------------------------------------------
def measure_velocity_branches(image_a, image_b, delay_ns: float = 10.0,
                              pixel_size_um: Optional[float] = None,
                              axis=None, origin=None,
                              settings: Optional[HeadSettings] = None,
                              max_match_px: float = 60.0,
                              require_forward: bool = True,
                              frames: Tuple[int, int] = (0, 0),
                              delay_jitter_ns: float = 0.0,
                              pixel_size_rel_error: float = 0.0,
                              **overrides):
    """One velocity per streamer branch.

    Heads are detected in both images and matched one-to-one by minimising
    the total displacement (Hungarian assignment).  Pairs further apart than
    `max_match_px`, or moving backwards when `require_forward`, are dropped.

    Returns
    -------
    (results, dataframe)
        A list of :class:`VelocityResult` and a pandas DataFrame of the same
        data, ready to plot or save.
    """
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    a = _as_frame(image_a, frames[0])
    b = _as_frame(image_b, frames[1])
    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})

    smooth, mask, _ = _components(a, s)
    ax = _resolve_axis(smooth, mask, axis, origin)

    heads_a = detect_heads(a, axis=ax, origin=origin, settings=s)
    heads_b = detect_heads(b, axis=ax, origin=origin, settings=s)
    if not heads_a or not heads_b:
        return [], pd.DataFrame()

    pa = np.array([h.xy for h in heads_a])
    pb = np.array([h.xy for h in heads_b])
    cost = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)

    results: List[VelocityResult] = []
    for k, (i, j) in enumerate(zip(rows, cols)):
        if cost[i, j] > max_match_px:
            continue
        r = VelocityResult(heads_a[i], heads_b[j], float(delay_ns), pixel_size_um,
                           ax, float(delay_jitter_ns), float(pixel_size_rel_error),
                           branch=k)
        if require_forward and r.axial_distance_px <= 0:
            continue
        results.append(r)

    df = pd.DataFrame([r.summary() for r in results])
    return results, df


# --------------------------------------------------------------------------
def _series_axis(data: np.ndarray, settings: HeadSettings, axis, origin):
    """One propagation axis for a whole series.

    Fixed once, from the frame with the most signal: deriving it from frame 0
    would fit noise whenever the series starts before the discharge. Returns
    None when nothing in the series is bright enough, in which case each
    frame falls back to its own estimate.
    """
    ax = np.asarray(axis, dtype=float) if (axis is not None
                                           and not isinstance(axis, str)) else None
    if ax is not None and ax.size == 2:
        return ax / np.linalg.norm(ax)
    brightest = int(np.argmax([np.percentile(f, 99.9) for f in data]))
    smooth, mask, _ = _components(data[brightest], settings)
    if not mask.any():
        return None
    return _resolve_axis(smooth, mask, axis, origin)


def velocity_frame_to_frame(stack, delay_ns: float = 10.0,
                            pixel_size_um: Optional[float] = None,
                            axis=None, origin=None,
                            settings: Optional[HeadSettings] = None,
                            times_ns: Optional[Sequence[float]] = None,
                            delay_jitter_ns: float = 0.0,
                            pixel_size_rel_error: float = 0.0,
                            csv_path=None, **overrides):
    """Apply the two-image method to every consecutive pair of frames.

    The two-image measurement of :func:`measure_velocity`, repeated along a
    kinetic series: frame 0 against 1, 1 against 2, and so on.  Each pair is
    an independent velocity with its own error bar, so the result shows how
    the velocity develops rather than collapsing it to one number.

    The head is detected once per frame, not twice per pair, and the
    propagation axis is fixed once for the whole series so every axial
    projection is comparable.  A frame with no streamer in it (the `min_snr`
    gate of :class:`HeadSettings`) does not raise: the two pairs that use it
    get NaN velocities and the reason in their ``note`` column.

    Parameters
    ----------
    stack : ImageStack or ndarray (frames, H, W)
        The series, preferably dark-subtracted and cropped.
    delay_ns : float
        Delay step between consecutive frames, used when `times_ns` is not
        given.
    times_ns : sequence, optional
        The actual gate delay of each frame.  Preferred over `delay_ns`
        whenever the steps are not perfectly uniform -- each pair then uses
        its own measured ``dt``.
    csv_path : path, optional
        Write the table straight to this CSV file.

    Returns
    -------
    (dataframe, results)
        One row per consecutive pair, and the underlying
        :class:`VelocityResult` objects for the pairs that produced one.
    """
    data = np.asarray(getattr(stack, "data", stack), dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    if len(data) < 2:
        raise ValueError(f"need at least two frames, got {len(data)}")

    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})
    px = pixel_size_um if pixel_size_um is not None else getattr(stack, "pixel_size_um", None)
    ax = _series_axis(data, s, axis, origin)
    times = frame_times(len(data), delay_ns, times_ns)

    # one detection per frame, reused by the two pairs it belongs to
    heads: List[Optional[HeadPosition]] = []
    notes: List[str] = []
    for frame in data:
        try:
            heads.append(detect_head(frame, axis=ax if ax is not None else axis,
                                     origin=origin, settings=s))
            notes.append("")
        except ValueError as exc:
            heads.append(None)
            notes.append(str(exc).split(":")[0])

    df, results = _pair_table(list(range(len(data))), dict(enumerate(heads)),
                              dict(enumerate(notes)), times, px,
                              ax if ax is not None else np.array([0.0, 1.0]),
                              delay_jitter_ns, pixel_size_rel_error)
    if csv_path is not None:
        _write_pair_csv(df, csv_path)
    return df, results


#: columns of every frame-to-frame table, detected or clicked
PAIR_COLUMNS = ["pair", "frame_a", "frame_b", "time_a_ns", "time_b_ns", "dt_ns",
                "x_a", "y_a", "x_b", "y_b", "distance_px", "axial_distance_px",
                "lateral_distance_px", "distance_mm", "velocity_px_per_ns",
                "sigma_velocity_px_per_ns", "velocity_m_per_s",
                "sigma_velocity_m_per_s", "axial_velocity_m_per_s", "note"]


def _pair_table(frames: Sequence[int], heads: dict, notes: dict, times,
                pixel_size_um: Optional[float], axis: np.ndarray,
                delay_jitter_ns: float, pixel_size_rel_error: float,
                bridge_gaps: bool = False):
    """One row per consecutive pair of `frames`.

    Shared by the detected route (:func:`velocity_frame_to_frame`) and the
    clicked one (:func:`velocity_from_heads`), so both give the same table.
    `heads` maps frame -> HeadPosition or None; `times` is indexed by frame.
    With `bridge_gaps`, frames without a head are skipped and each head is
    paired with the previous one that exists, over the longer interval.
    """
    import pandas as pd

    if bridge_gaps:
        frames = [k for k in frames if heads.get(k) is not None]
    rows, results = [], []
    for n, (fa, fb) in enumerate(zip(frames[:-1], frames[1:])):
        ha, hb = heads.get(fa), heads.get(fb)
        dt = float(times[fb] - times[fa])
        row = {"pair": n, "frame_a": fa, "frame_b": fb,
               "time_a_ns": float(times[fa]), "time_b_ns": float(times[fb]),
               "dt_ns": dt}
        if ha is None or hb is None or dt == 0:
            row["note"] = (notes.get(fa) or notes.get(fb)
                           or ("dt is zero" if dt == 0 else ""))
            rows.append(row)                    # the missing columns become NaN
            continue

        r = VelocityResult(ha, hb, dt, pixel_size_um, axis, float(delay_jitter_ns),
                           float(pixel_size_rel_error), branch=n)
        results.append(r)
        summary = r.summary()
        summary.pop("branch", None)
        summary.pop("delay_ns", None)
        row.update(summary)
        row["note"] = ""
        rows.append(row)
    return pd.DataFrame(rows, columns=PAIR_COLUMNS), results


def _write_pair_csv(df, csv_path) -> None:
    from pathlib import Path

    from .io import long_path
    path = Path(csv_path)
    Path(long_path(path.parent)).mkdir(parents=True, exist_ok=True)
    df.to_csv(long_path(path), index=False)
    good = int(df["velocity_px_per_ns"].notna().sum())
    print(f"wrote {path}  ({len(df)} pairs, {good} with a velocity)")


def frame_times(n_frames: int, delay_ns: float = 10.0,
                times_ns: Optional[Sequence[float]] = None) -> np.ndarray:
    """Gate delay of every frame: `times_ns` when given, else uniform steps."""
    if times_ns is None:
        return np.arange(n_frames, dtype=float) * float(delay_ns)
    times = np.asarray(times_ns, dtype=float).ravel()
    if len(times) < n_frames:
        raise ValueError(f"times_ns has {len(times)} entries for {n_frames} frames")
    return times


def velocity_from_heads(heads, delay_ns: float = 10.0,
                        times_ns: Optional[Sequence[float]] = None,
                        pixel_size_um: Optional[float] = None,
                        axis=None, position_uncertainty_px: float = 1.5,
                        delay_jitter_ns: float = 0.0,
                        pixel_size_rel_error: float = 0.0,
                        bridge_gaps: bool = False, csv_path=None):
    """Frame-to-frame velocity from head positions you located yourself.

    The clicked counterpart of :func:`velocity_frame_to_frame`: locate the
    head in frame 17, then 18, then 19 ..., and every consecutive pair
    becomes an independent two-image velocity.  The table has exactly the
    same columns, so :func:`~streamertools.plotting.plot_velocity_series`
    and anything else written for the detected table works unchanged.

    Parameters
    ----------
    heads : mapping {frame index: (x, y) or HeadPosition}
        The located heads.  Frames need not be contiguous; the table runs
        from the first located frame to the last.
    delay_ns, times_ns :
        Uniform frame step, or the measured gate delay of every frame,
        indexed by frame number.
    axis :
        ``"down"``/``"up"``/``"left"``/``"right"`` or ``(dx, dy)``, used for
        the axial and lateral split.  ``None`` takes the direction from the
        first located head to the last.
    position_uncertainty_px : float
        Localisation error of a clicked point (1.5 px, as for
        :func:`measure_velocity` with ``manual_heads``).  A
        :class:`HeadPosition` keeps its own ``sigma_px``.
    bridge_gaps : bool
        False (default): a frame without a head leaves its two pairs NaN,
        as the detected table does.  True: pair each head with the previous
        located one, dividing by the longer interval.
    csv_path : path, optional
        Write the table straight to this CSV file.

    Returns
    -------
    (dataframe, results)
    """
    points = {}
    for frame, head in dict(heads).items():
        if head is None:
            continue
        if not isinstance(head, HeadPosition):
            x, y = head
            head = HeadPosition(float(x), float(y), sigma_px=float(position_uncertainty_px),
                                method="manual")
        points[int(frame)] = head

    if not points:
        import pandas as pd
        return pd.DataFrame(columns=PAIR_COLUMNS), []
    first, last = min(points), max(points)
    times = frame_times(last + 1, delay_ns, times_ns)

    if axis is None:
        d = points[last].xy - points[first].xy
        n = float(np.linalg.norm(d))
        ax = d / n if n > 1e-9 else np.array([0.0, 1.0])
    else:
        ax = _resolve_axis(None, None, axis, None)
    points = {k: replace(h, axial=float(h.xy @ ax)) for k, h in points.items()}

    df, results = _pair_table(list(range(first, last + 1)), points,
                              {k: "no head located" for k in range(first, last + 1)
                               if k not in points},
                              times, pixel_size_um, ax, delay_jitter_ns,
                              pixel_size_rel_error, bridge_gaps=bridge_gaps)
    if csv_path is not None:
        _write_pair_csv(df, csv_path)
    return df, results


# --------------------------------------------------------------------------
def track_head(stack, axis=None, origin=None, settings: Optional[HeadSettings] = None,
               times_ns: Optional[Sequence[float]] = None, **overrides):
    """Follow the head through a whole kinetic series.

    Returns a DataFrame with one row per frame: head position, axial position
    and, when `times_ns` is given, the instantaneous velocity between
    consecutive frames.
    """
    import pandas as pd

    data = np.asarray(getattr(stack, "data", stack), dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})
    ax = _series_axis(data, s, axis, origin)

    rows = []
    for k, frame in enumerate(data):
        try:
            h = detect_head(frame, axis=ax if ax is not None else axis,
                            origin=origin, settings=s)
            rows.append({"frame": k, "x": h.x, "y": h.y, "axial_px": h.axial,
                         "sigma_px": h.sigma_px, "intensity": h.intensity,
                         "n_pixels": h.n_pixels, "note": ""})
        except ValueError as exc:
            rows.append({"frame": k, "x": np.nan, "y": np.nan, "axial_px": np.nan,
                         "sigma_px": np.nan, "intensity": np.nan, "n_pixels": 0,
                         "note": str(exc).split(":")[0]})
    df = pd.DataFrame(rows)
    if times_ns is not None:
        df["time_ns"] = np.asarray(times_ns, dtype=float)[:len(df)]
        pix = getattr(stack, "pixel_size_um", None)
        d = np.hypot(df["x"].diff(), df["y"].diff())
        dt = df["time_ns"].diff()
        df["velocity_px_per_ns"] = d / dt
        if pix:
            df["velocity_m_per_s"] = d * pix * 1e-6 / (dt * 1e-9)
    return df


def velocity_from_series(stack, times_ns: Sequence[float],
                         pixel_size_um: Optional[float] = None,
                         axis=None, origin=None,
                         settings: Optional[HeadSettings] = None, **overrides):
    """Straight-line fit of axial head position against time.

    A cross-check on the two-image method: with more than two frames the
    slope of ``axial position vs. time`` is a less noisy velocity, and the
    residuals show whether the streamer accelerated during the window.

    Returns ``(velocity_m_per_s, sigma, dataframe)``.
    """
    px = pixel_size_um if pixel_size_um is not None else getattr(stack, "pixel_size_um", None)
    df = track_head(stack, axis=axis, origin=origin, settings=settings,
                    times_ns=times_ns, **overrides)
    ok = df["axial_px"].notna()
    if ok.sum() < 2:
        raise ValueError(
            f"need at least two frames with a detected head, found {int(ok.sum())}. "
            f"Frames without one are listed in the 'note' column of the returned "
            f"table -- a series that starts pre-discharge is the usual reason.")
    skipped = int((~ok).sum())
    if skipped:
        print(f"velocity_from_series: fitting {int(ok.sum())} of {len(df)} frames "
              f"({skipped} without a streamer)")
    t = df.loc[ok, "time_ns"].to_numpy(dtype=float)
    y = df.loc[ok, "axial_px"].to_numpy(dtype=float)
    coeffs, cov = np.polyfit(t, y, 1, cov=True)
    slope_px_per_ns, sigma_slope = float(coeffs[0]), float(np.sqrt(cov[0, 0]))
    if px is None:
        return slope_px_per_ns, sigma_slope, df
    scale = px * 1e-6 / 1e-9                       # px/ns -> m/s
    return slope_px_per_ns * scale, sigma_slope * scale, df


# --------------------------------------------------------------------------
def _line_fit(t: np.ndarray, y: np.ndarray, sigma_y: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope and its standard error.

    From the scatter of the points when there are three or more; for two
    points, from their localisation errors.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2 or np.ptp(t) == 0:
        return float("nan"), float("nan")
    tc = t - t.mean()
    sxx = float(tc @ tc)
    slope = float(tc @ (y - y.mean())) / sxx
    if len(t) == 2:
        return slope, float(np.hypot(*sigma_y) / abs(t[1] - t[0]))
    resid = y - (y.mean() + slope * tc)
    return slope, float(np.sqrt(resid @ resid / (len(t) - 2) / sxx))


@dataclass
class FrontVelocityResult:
    """The velocity of one delay series, from :func:`velocity_from_fronts`."""

    track: "object"                    # DataFrame, one row per frame
    pairs: "object"                    # DataFrame, frame to frame (PAIR_COLUMNS)
    velocity_px_per_ns: float          # slope of the fit
    sigma_px_per_ns: float
    pixel_size_um: Optional[float]
    axis: np.ndarray
    fit_frames: List[int] = field(default_factory=list)

    def _metric(self, value: float) -> Optional[float]:
        if self.pixel_size_um is None:
            return None
        return float(value) * self.pixel_size_um * 1e-6 / 1e-9

    @property
    def velocity_m_per_s(self) -> Optional[float]:
        return self._metric(self.velocity_px_per_ns)

    @property
    def sigma_m_per_s(self) -> Optional[float]:
        return self._metric(self.sigma_px_per_ns)

    def summary(self) -> dict:
        """One row for a table of series."""
        ok = self.pairs["distance_px"].notna()
        v = self.pairs.loc[ok, "velocity_px_per_ns"].astype(float)
        return {
            "n_frames": len(self.track),
            "n_fronts": int(self.track["axial_px"].notna().sum()),
            "n_fit": len(self.fit_frames),
            "first_fit_frame": min(self.fit_frames) if self.fit_frames else None,
            "last_fit_frame": max(self.fit_frames) if self.fit_frames else None,
            "fit_velocity_px_per_ns": self.velocity_px_per_ns,
            "fit_sigma_px_per_ns": self.sigma_px_per_ns,
            "fit_velocity_m_per_s": self.velocity_m_per_s,
            "fit_sigma_m_per_s": self.sigma_m_per_s,
            "n_pairs": int(ok.sum()),
            "mean_pair_velocity_px_per_ns": float(v.mean()) if len(v) else float("nan"),
            "std_pair_velocity_px_per_ns": float(v.std()) if len(v) > 1 else float("nan"),
        }

    def __repr__(self) -> str:                                   # pragma: no cover
        v, s = self.velocity_px_per_ns, self.sigma_px_per_ns
        text = f"{v:.4g} +- {s:.2g} px/ns"
        if self.pixel_size_um is not None:
            text += f"  ({self.velocity_m_per_s:.4g} +- {self.sigma_m_per_s:.2g} m/s)"
        return (f"<FrontVelocityResult {text}, fitted over {len(self.fit_frames)} of "
                f"{len(self.track)} frames>")


def velocity_from_fronts(stack, delay_ns: float = 10.0,
                         times_ns: Optional[Sequence[float]] = None,
                         gate_ns: float = 0.0,
                         pixel_size_um: Optional[float] = None,
                         axis=None, origin=None,
                         settings: Optional[HeadSettings] = None,
                         fit_frames=None,
                         delay_jitter_ns: float = 0.0,
                         pixel_size_rel_error: float = 0.0,
                         csv_path=None, **overrides) -> FrontVelocityResult:
    """Streamer velocity from a delay series, the way the literature measures it.

    In every frame the head of the streamer that protruded furthest is
    located with the half-maximum front criterion (``method="front"``, see
    the module notes).  Its position along the propagation axis is plotted
    against the end-of-exposure time, ``t = gate delay + gate width``
    (Nijdam 2011, fig. 3.18), and the velocity is the slope of a straight
    line through those points (Briels et al 2008: position is plotted as a
    function of time for each photograph and an average velocity is
    determined).  The frame-to-frame differences dl/dT come along as well.

    Parameters
    ----------
    stack : ImageStack or ndarray (frames, H, W)
        One delay series, each frame from its own discharge.
    delay_ns, times_ns :
        Uniform delay step, or the measured gate delay of every frame.
    gate_ns : float
        Gate (exposure) width, added to the delay: the front is where the
        head was at the *end* of the exposure.  It shifts every time equally,
        so it changes the times reported, not the velocity.
    origin : (x, y), optional
        The electrode tip.  Orients an estimated axis, and adds the distance
        of each head from the tip ("the maximum distance of the streamer
        fronts") to the track.
    fit_frames : (first, last) or sequence of frame indices, optional
        Restrict the fit to the middle part of the gap, where the velocity is
        constant (Winands et al 2008; Nijdam 2011).  Default: every frame in
        which a front was found.
    csv_path : path, optional
        Write the per-frame track to this CSV file.
    **overrides
        Any :class:`HeadSettings` field.

    Returns
    -------
    FrontVelocityResult
        ``.velocity_px_per_ns`` +- ``.sigma_px_per_ns`` (and m/s with a pixel
        size), ``.track`` and ``.pairs``.  With fewer than two fitted fronts
        the velocity is NaN and ``.track["note"]`` says why.
    """
    import pandas as pd

    data = np.asarray(getattr(stack, "data", stack), dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})
    px = pixel_size_um if pixel_size_um is not None else getattr(stack, "pixel_size_um", None)
    ax = _series_axis(data, s, axis, origin)
    ax_used = ax if ax is not None else np.array([0.0, 1.0])
    t_delay = frame_times(len(data), delay_ns, times_ns)
    t_end = t_delay + float(gate_ns)

    heads, notes, rows = {}, {}, []
    for k, frame in enumerate(data):
        row = {"frame": k, "delay_ns": float(t_delay[k]), "time_ns": float(t_end[k])}
        try:
            h = detect_head(frame, axis=ax if ax is not None else axis,
                            origin=origin, settings=s)
        except ValueError as exc:
            notes[k] = str(exc).split(":")[0]
            row.update(x=np.nan, y=np.nan, axial_px=np.nan, distance_px=np.nan,
                       sigma_px=np.nan, intensity=np.nan, method="", note=notes[k])
            rows.append(row)
            continue
        h = replace(h, axial=float(h.xy @ ax_used))
        heads[k] = h
        dist = (float(np.hypot(*(h.xy - np.asarray(origin, float))))
                if origin is not None else np.nan)
        row.update(x=h.x, y=h.y, axial_px=h.axial, distance_px=dist,
                   sigma_px=h.sigma_px, intensity=h.intensity, method=h.method, note="")
        rows.append(row)
    track = pd.DataFrame(rows)

    ok = track["axial_px"].notna()
    if fit_frames is None:
        chosen = ok
    elif len(fit_frames) == 2 and not isinstance(fit_frames, (list, np.ndarray)):
        chosen = track["frame"].between(int(fit_frames[0]), int(fit_frames[1]))
    else:
        chosen = track["frame"].isin([int(k) for k in fit_frames])
    track["in_fit"] = ok & chosen
    fit = track[track["in_fit"]]
    slope, sigma = _line_fit(fit["time_ns"], fit["axial_px"], fit["sigma_px"])

    pairs, _ = _pair_table(list(range(len(data))), heads, notes, t_end, px, ax_used,
                           delay_jitter_ns, pixel_size_rel_error)
    if csv_path is not None:
        from pathlib import Path

        from .io import long_path
        path = Path(csv_path)
        Path(long_path(path.parent)).mkdir(parents=True, exist_ok=True)
        track.to_csv(long_path(path), index=False)
    return FrontVelocityResult(track, pairs, slope, sigma, px, ax_used,
                               [int(k) for k in fit["frame"]])
