"""
Turning counts into pictures
============================

One canonical rendering function, used by the viewer, the exporter, the image
converter and the tile maker alike, so what you look at on screen and what
lands in the output file can never diverge.

The chain is the manual's (chapter 2.2):

    counts --(Imin, Imax)--> [0,1] --(gamma)--> [0,1] --(palette)--> RGB

with two additions taken from the older notebook, applied *before* the
normalisation so they behave like camera settings rather than like a curve:

* ``contrast`` -- a multiplicative factor on the counts,
* ``brightness`` -- an additive offset in counts.

Limit modes
-----------
``"auto"``       Imin/Imax = min/max of the image (manual AutoMin/AutoMax = 1)
``"histogram"``  percentages of the count histogram (manual AutoMax = 3)
``"manual"``     the numbers you give (manual AutoMin/AutoMax = 0)

Pass ``reference=stack`` (or ``scope="global"``) to compute the limits once
for a whole series -- that is the only way frames of a kinetic series stay
comparable to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

from .palette import LimitMode, Palette, counts_to_unit, resolve_limits

__all__ = ["DisplaySettings", "render", "render_stack"]


@dataclass
class DisplaySettings:
    """Everything that decides how counts become pixels."""

    gamma: float = 1.0                       # manual 2.2.2 (>=10 -> log branch)
    palette: Union[Palette, str, None] = "inferno"
    limit_mode: LimitMode = "histogram"
    i_min: Optional[float] = None            # used when limit_mode == "manual"
    i_max: Optional[float] = None
    min_hist_percentage: float = 1.0         # percentile for Imin
    max_hist_percentage: float = 99.8        # percentile for Imax
    brightness: float = 0.0                  # counts, added
    contrast: float = 1.0                    # counts, multiplied
    scope: str = "frame"                     # "frame" or "global"

    def resolved_palette(self) -> Palette:
        return self.palette if isinstance(self.palette, Palette) else Palette.load(self.palette)

    def limits(self, counts: np.ndarray) -> Tuple[float, float]:
        return resolve_limits(counts, self.limit_mode, self.i_min, self.i_max,
                              self.min_hist_percentage, self.max_hist_percentage)


def render(counts: np.ndarray,
           settings: Optional[DisplaySettings] = None,
           reference: Optional[np.ndarray] = None,
           **overrides) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Render one frame.

    Parameters
    ----------
    counts : ndarray (H, W)
    settings : DisplaySettings, optional
    reference : ndarray, optional
        Array the limits are computed from (e.g. the whole stack).  Ignored
        when ``settings.scope == "frame"``.
    **overrides
        Any :class:`DisplaySettings` field, for one-off changes.

    Returns
    -------
    rgb : ndarray (H, W, 3) uint8
    unit : ndarray (H, W) float32 in [0, 1]  -- the gamma-mapped image
    i_min, i_max : float
    """
    s = settings or DisplaySettings()
    if overrides:
        s = DisplaySettings(**{**s.__dict__, **overrides})

    img = s.contrast * np.asarray(counts, dtype=np.float32) + s.brightness
    ref = img
    if s.scope == "global" and reference is not None:
        ref = s.contrast * np.asarray(reference, dtype=np.float32) + s.brightness

    lo, hi = s.limits(ref)
    unit = counts_to_unit(img, lo, hi, s.gamma)
    rgb = s.resolved_palette().apply(unit)
    return rgb, unit, lo, hi


def render_stack(stack, settings: Optional[DisplaySettings] = None, **overrides):
    """Render every frame of a stack, yielding ``(index, rgb, unit, lo, hi)``.

    With ``scope="global"`` the limits are taken once from the whole stack, so
    brightness is comparable frame to frame.
    """
    data = getattr(stack, "data", stack)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    for i in range(data.shape[0]):
        rgb, unit, lo, hi = render(data[i], settings, reference=data, **overrides)
        yield i, rgb, unit, lo, hi
