"""
Colour representation  --  Streamertools manual, chapter 2.2
=============================================================

Counts are turned into colours in two separate steps, and keeping them apart
is what makes brightness comparable between images:

1. **counts -> [0, 1]**  with :func:`counts_to_unit`, using :math:`I_{min}`,
   :math:`I_{max}` and :math:`\\gamma` (manual 2.2.2)

   .. math::
      I_{out} = \\left(\\frac{I_{in}-I_{min}}{I_{max}-I_{min}}\\right)^{1/\\gamma}
      \\qquad (\\gamma < 10)

      I_{out} = \\log_{10}\\!\\left(1 + 9\\,\\frac{I_{in}-I_{min}}{I_{max}-I_{min}}\\right)
      \\qquad (\\gamma \\geq 10)

   :math:`\\gamma = 1` is linear, :math:`\\gamma = 2` is the old "square root"
   style, :math:`\\gamma = 1/0.7` the old "power 0.7" and :math:`\\gamma \\geq 10`
   the old "logarithm" style (manual 2.2.2, deprecated *Drawing style*).

2. **[0, 1] -> RGB** with a :class:`Palette` (manual 2.2.1): a ``*.pal`` text
   file whose colours are spread homogeneously over [0, 1] and interpolated
   linearly between the two nearest palette entries.

This is the piece the notebook you had was doing differently: it stretched
each frame between two percentiles and then applied ``disp**gamma``.  Here the
percentile stretch only *chooses* Imin/Imax, and the exponent follows the
manual's convention (``1/gamma``, with the logarithmic branch), so a number
you write down as "gamma 1.5" means the same thing as in Streamertools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Palette",
    "THERMAL_PAL",
    "counts_to_unit",
    "resolve_limits",
    "counts_to_rgb",
    "LimitMode",
    "DRAWING_STYLE_GAMMA",
]

LimitMode = Literal["manual", "auto", "histogram"]

#: the deprecated *Drawing style* numbers of the tile maker (manual 2.2.2)
DRAWING_STYLE_GAMMA = {0: 1.0, 1: 2.0, 2: 1.0 / 0.7, 3: 10.0}


# --------------------------------------------------------------------------
# 2.2.2  counts -> [0, 1]
# --------------------------------------------------------------------------
def counts_to_unit(counts: np.ndarray, i_min: float, i_max: float,
                   gamma: float = 1.0, clip: bool = True) -> np.ndarray:
    """Convert count values to display values in [0, 1] (manual eq. 2.1/2.2).

    Parameters
    ----------
    counts : ndarray
    i_min, i_max : float
        Count values that map to 0 and 1.
    gamma : float
        ``1`` is linear.  Values between 1 and 2 make dark regions more
        pronounced.  ``gamma >= 10`` switches to the logarithmic branch
        (eq. 2.2), which is close to gamma 2.2 but with a different profile.
    clip : bool
        Clip the normalised value to [0, 1] before the gamma step, which is
        what the programs do when Imin/Imax do not span the whole image.
    """
    counts = np.asarray(counts, dtype=np.float32)
    span = float(i_max) - float(i_min)
    if abs(span) < 1e-12:
        span = 1e-12
    x = (counts - float(i_min)) / span
    if clip:
        x = np.clip(x, 0.0, 1.0)
    if gamma >= 10.0:                                   # eq. 2.2
        return np.log10(1.0 + 9.0 * np.clip(x, 0.0, None)).astype(np.float32)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return np.power(np.clip(x, 0.0, None), 1.0 / gamma).astype(np.float32)


def resolve_limits(counts: np.ndarray,
                   mode: LimitMode = "auto",
                   i_min: Optional[float] = None,
                   i_max: Optional[float] = None,
                   min_hist_percentage: float = 30.0,
                   max_hist_percentage: float = 99.0) -> Tuple[float, float]:
    """Work out (Imin, Imax) the way the Streamertools do (manual 2.2.2).

    Parameters
    ----------
    mode :
        ``"manual"``    use the given `i_min` / `i_max` verbatim;
        ``"auto"``      lowest and highest count found in the image;
        ``"histogram"`` percentages of the count histogram -- with
                        `max_hist_percentage` = 95 the brightest 5 % of the
                        pixels end up above Imax.
    """
    counts = np.asarray(counts, dtype=np.float32)
    if mode == "manual":
        if i_min is None or i_max is None:
            raise ValueError("mode='manual' needs both i_min and i_max")
        return float(i_min), float(i_max)
    if mode == "auto":
        return float(np.nanmin(counts)), float(np.nanmax(counts))
    if mode == "histogram":
        lo = float(np.nanpercentile(counts, min_hist_percentage))
        hi = float(np.nanpercentile(counts, max_hist_percentage))
        return lo, hi
    raise ValueError(f"unknown limit mode {mode!r}")


# --------------------------------------------------------------------------
# 2.2.1  palette files
# --------------------------------------------------------------------------
#: default palette, written out when no ``*.pal`` file is found -- the
#: black-red-yellow-white ramp Streamertools calls ``thermal.pal``.
THERMAL_PAL = np.array([
    [0, 0, 0], [40, 0, 60], [90, 0, 110], [150, 10, 100], [200, 40, 60],
    [235, 90, 20], [250, 150, 0], [255, 205, 40], [255, 240, 150], [255, 255, 255],
], dtype=np.uint8)


@dataclass
class Palette:
    """A Streamertools ``*.pal`` colour palette (manual 2.2.1).

    The file is plain text: the first line is the number of colours, each
    following line holds ``R G B`` as integers 0-255.  The colours are spread
    homogeneously over [0, 1] and interpolated linearly.
    """

    colours: np.ndarray = None       # (N, 3) uint8
    name: str = "thermal"

    def __post_init__(self):
        if self.colours is None:
            self.colours = THERMAL_PAL.copy()
        self.colours = np.asarray(self.colours, dtype=np.uint8).reshape(-1, 3)
        if len(self.colours) < 2:
            raise ValueError("a palette needs at least two colours")

    # -- io ----------------------------------------------------------------
    @classmethod
    def read(cls, path) -> "Palette":
        path = Path(path)
        tokens = path.read_text().split()
        n = int(tokens[0])
        vals = np.array(tokens[1:1 + 3 * n], dtype=int).reshape(n, 3)
        return cls(vals.astype(np.uint8), name=path.stem)

    @classmethod
    def load(cls, path_or_name: Union[str, Path, None] = None) -> "Palette":
        """Read a ``*.pal`` file, fall back to the built-in thermal palette.

        Also accepts the name of a matplotlib colormap ("inferno", "viridis",
        ...), which is sampled into 256 palette entries.
        """
        if path_or_name is None:
            return cls()
        p = Path(path_or_name)
        if p.exists():
            return cls.read(p)
        if p.suffix.lower() == ".pal":
            # manual 2.2.1: when no palette file is found, the standard
            # 'thermal' palette is used (and can be written out with .write)
            if p.stem.lower() != "thermal":
                print(f"palette file '{p}' not found -- using the built-in thermal palette")
            return cls()
        return cls.from_matplotlib(str(path_or_name))

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [str(len(self.colours))]
        lines += [f"{r} {g} {b}" for r, g, b in self.colours]
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def from_matplotlib(cls, name: str, n: int = 256) -> "Palette":
        import matplotlib
        try:
            cmap = matplotlib.colormaps[name]
        except Exception:                                       # pragma: no cover
            import matplotlib.cm as cm
            cmap = cm.get_cmap(name)
        rgb = (np.asarray(cmap(np.linspace(0, 1, n)))[:, :3] * 255).round()
        return cls(rgb.astype(np.uint8), name=name)

    @classmethod
    def grayscale(cls) -> "Palette":
        return cls(np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8), "gray")

    # -- use ---------------------------------------------------------------
    def apply(self, unit: np.ndarray) -> np.ndarray:
        """Map display values in [0, 1] to RGB uint8 (linear interpolation)."""
        u = np.clip(np.asarray(unit, dtype=np.float32), 0.0, 1.0)
        n = len(self.colours)
        pos = u * (n - 1)
        lo = np.floor(pos).astype(np.int32)
        hi = np.minimum(lo + 1, n - 1)
        frac = (pos - lo)[..., None]
        c = self.colours.astype(np.float32)
        rgb = c[lo] * (1.0 - frac) + c[hi] * frac
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def to_matplotlib(self):
        """A matplotlib ``ListedColormap`` for use in plots."""
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            self.name, self.colours.astype(np.float32) / 255.0)

    def preview(self, width: int = 512, height: int = 40) -> np.ndarray:
        ramp = np.linspace(0, 1, width, dtype=np.float32)[None].repeat(height, 0)
        return self.apply(ramp)


# --------------------------------------------------------------------------
# one-stop helper
# --------------------------------------------------------------------------
def counts_to_rgb(counts: np.ndarray,
                  palette: Union[Palette, str, Path, None] = None,
                  gamma: float = 1.0,
                  limit_mode: LimitMode = "auto",
                  i_min: Optional[float] = None,
                  i_max: Optional[float] = None,
                  min_hist_percentage: float = 30.0,
                  max_hist_percentage: float = 99.0,
                  reference: Optional[np.ndarray] = None,
                  ) -> Tuple[np.ndarray, float, float]:
    """counts -> RGB uint8, following manual 2.2 end to end.

    Parameters
    ----------
    reference : ndarray, optional
        Compute Imin/Imax from *this* array instead of from `counts` -- pass
        the whole stack to give every frame of a series the same scale.

    Returns
    -------
    (rgb uint8, i_min, i_max)
    """
    pal = palette if isinstance(palette, Palette) else Palette.load(palette)
    ref = counts if reference is None else reference
    lo, hi = resolve_limits(ref, limit_mode, i_min, i_max,
                            min_hist_percentage, max_hist_percentage)
    unit = counts_to_unit(counts, lo, hi, gamma)
    return pal.apply(unit), lo, hi
