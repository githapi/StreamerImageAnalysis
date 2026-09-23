"""
Plots
=====

Static matplotlib figures: the width-profile plot of manual figure 3.7, the
count histogram of section 3.3, the measurement overlay of figure 3.5, and a
few plots for the velocity measurement.

Everything returns the matplotlib objects, so you can restyle or save them.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .display import DisplaySettings, render
from .geometry import MeasurementBox

__all__ = ["show_image", "contact_sheet", "plot_profiles", "plot_histogram",
           "plot_measurement", "plot_velocity", "plot_branch_velocities",
           "plot_velocity_series", "plot_head_track", "show_palette"]


def _axes(ax=None, figsize=(8, 6)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# --------------------------------------------------------------------------
def show_image(counts, settings: Optional[DisplaySettings] = None, ax=None,
               box: Optional[MeasurementBox] = None, title: str = "",
               reference=None, colorbar: bool = False, **overrides):
    """Render counts through the manual's colour chain and show them."""
    import matplotlib.pyplot as plt

    data = np.asarray(getattr(counts, "data", counts), dtype=np.float32)
    if data.ndim == 3:
        data = data[0]
    rgb, unit, lo, hi = render(data, settings, reference=reference, **overrides)
    ax = _axes(ax, (7, 7))
    im = ax.imshow(rgb)
    ax.set_title(title or f"Imin = {lo:.4g},  Imax = {hi:.4g}")
    ax.axis("off")
    if box is not None:
        corners = np.vstack([box.corners(), box.corners()[:1]])
        ax.plot(corners[:, 0], corners[:, 1], color="white", lw=1.2)
        ax.plot([box.p0[0], box.p1[0]], [box.p0[1], box.p1[1]], color="white",
                lw=0.8, ls=":")
        ax.plot(*box.p0, "o", color="red", ms=5)
        ax.plot(*box.p1, "o", color="lime", ms=5)
    if colorbar:
        s = settings or DisplaySettings()
        sm = plt.cm.ScalarMappable(cmap=s.resolved_palette().to_matplotlib(),
                                   norm=plt.Normalize(lo, hi))
        plt.colorbar(sm, ax=ax, fraction=0.045, label="counts")
    return ax, im


# --------------------------------------------------------------------------
def contact_sheet(stack, settings: Optional[DisplaySettings] = None,
                  indices=None, ncols: int = 8, scale: float = 1.7,
                  labels=None, **overrides):
    """Every frame of a series as one inline grid -- no widgets needed.

    The quickest way to see what a 40-frame kinetic series contains, and to
    pick the frame worth analysing. Limits are taken once from the whole
    stack, so the panels are comparable to each other.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(getattr(stack, "data", stack), dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    if indices is None:
        indices = range(len(data))
    indices = list(indices)

    s = settings or DisplaySettings()
    s = DisplaySettings(**{**s.__dict__, **overrides, "scope": "global"})

    ncols = max(1, min(ncols, len(indices)))
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(scale * ncols, scale * nrows),
                            squeeze=False)
    reference = data[:, ::4, ::4]              # limits from a subsample: fast
    for ax, k in zip(axs.ravel(), indices):
        rgb, _, _, _ = render(data[k], s, reference=reference)
        ax.imshow(rgb)
        ax.set_title(labels[k] if labels is not None else f"{k}", fontsize=8)
        ax.axis("off")
    for ax in axs.ravel()[len(indices):]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axs


def plot_profiles(result, ax=None, max_grey_lines: int = 25, show_moving: bool = False):
    """The default width plot of the program (manual figure 3.7).

    Green: the averaged profile with its two red edges and the blue baseline.
    Grey: a selection of the *per line* profiles.
    """
    ax = _axes(ax, (8, 5))
    u = result.unit
    pos = result.positions
    P = result.per_line_profiles

    n = P.shape[1]
    step = max(1, n // max(max_grey_lines, 1))
    for j in range(0, n, step):
        ax.plot(pos, P[:, j], color="0.7", lw=0.6, zorder=1)

    ax.plot(pos, result.averaged_profile, color="green", lw=2.2, zorder=3,
            label="averaged profile (eq. 3.1)")
    a = result.averaged
    ax.plot(pos, a.baseline(pos), color="blue", lw=1.2, zorder=2, label="baseline")
    for x, lbl in ((a.left_pos, "edges (FWHM)"), (a.right_pos, None)):
        ax.axvline(x, color="red", lw=1.4, zorder=4, label=lbl)

    if show_moving and result.moving:
        centres = [r.centre for r in result.moving if r.valid]
        for c in centres:
            ax.axvline(c, color="orange", lw=0.4, alpha=0.5, zorder=1)

    ax.set_xlabel(f"position ({u})")
    ax.set_ylabel("counts")
    ax.set_title(f"{result.settings.measure}:  averaged = {result.fwhm_averaged:.3g} {u},  "
                 f"per line = {result.fwhm_per_line:.3g} +- {result.fwhm_per_line_std:.2g} {u}")
    ax.legend(fontsize=8, loc="upper right")
    return ax


def plot_histogram(counts, box: Optional[MeasurementBox] = None, ax=None,
                   bins: int = 200, log: bool = True):
    """Histogram of the count values, of the whole image or inside the box (3.3)."""
    from .geometry import resample_box

    data = np.asarray(getattr(counts, "data", counts), dtype=np.float32)
    if data.ndim == 3:
        data = data[0]
    ax = _axes(ax, (7, 4))
    ax.hist(data.ravel(), bins=bins, color="0.6", label="whole image")
    if box is not None:
        ax.hist(resample_box(data, box).ravel(), bins=bins, color="tab:red",
                alpha=0.7, label="inside the box")
        ax.legend(fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("counts")
    ax.set_ylabel("pixels")
    return ax


def plot_measurement(counts, result, settings: Optional[DisplaySettings] = None,
                     method: str = "per_line", figsize=(13, 5), **overrides):
    """Image with the box and the measured channel edges, plus the profile plot.

    Reproduces figure 3.5 (edges per line drawn on the image) next to the
    profile plot of figure 3.7.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    show_image(counts, settings, ax=ax1, box=result.box, **overrides)
    try:
        left, right = result.edge_points(method)
        for pts, colour in ((left, "cyan"), (right, "cyan")):
            if len(pts):
                ax1.plot(pts[:, 0], pts[:, 1], ".", color=colour, ms=1.5)
    except Exception:
        pass
    ax1.set_title(f"{result.settings.measure} measurement ({method} edges)")
    plot_profiles(result, ax=ax2)
    fig.tight_layout()
    return fig, (ax1, ax2)


# --------------------------------------------------------------------------
def plot_velocity(image_a, image_b, result, settings: Optional[DisplaySettings] = None,
                  figsize=(13, 6), zoom: Optional[float] = None, **overrides):
    """The two short-exposure images with the detected heads and the shift."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=figsize,
                            gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, img, head, name in ((axs[0], image_a, result.head_a, "image A"),
                                (axs[1], image_b, result.head_b, "image B")):
        show_image(img, settings, ax=ax, **overrides)
        ax.plot(head.x, head.y, "+", color="cyan", ms=14, mew=2)
        ax.set_title(f"{name}: head at ({head.x:.1f}, {head.y:.1f})")

    ax = axs[2]
    show_image(image_b, settings, ax=ax, **overrides)
    ax.annotate("", xy=(result.head_b.x, result.head_b.y),
                xytext=(result.head_a.x, result.head_a.y),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
    ax.plot(result.head_a.x, result.head_a.y, "o", color="red", ms=5)
    ax.plot(result.head_b.x, result.head_b.y, "o", color="lime", ms=5)
    v, s = result.velocity_m_per_s, result.sigma_velocity_m_per_s
    label = (f"{result.distance_px:.2f} px / {result.delay_ns:g} ns"
             if v is None else
             f"v = {v/1e6:.3f} +- {(s or 0)/1e6:.3f} mm/ns\n({v:.3g} m/s)")
    ax.set_title(label)
    if zoom:
        cx = 0.5 * (result.head_a.x + result.head_b.x)
        cy = 0.5 * (result.head_a.y + result.head_b.y)
        for a in axs:
            a.set_xlim(cx - zoom, cx + zoom)
            a.set_ylim(cy + zoom, cy - zoom)
    fig.tight_layout()
    return fig, axs


def _velocity_column(df) -> Tuple[str, float, str]:
    """Which velocity column to plot, its scale and its unit label.

    Metric when a pixel size produced one, pixels per nanosecond otherwise.
    Without this, an uncalibrated measurement leaves `velocity_m_per_s` full
    of NaN, and matplotlib's hist() fails deep inside numpy with an
    unhelpful ``'float' object has no attribute 'dtype'``.
    """
    if "velocity_m_per_s" in df and df["velocity_m_per_s"].notna().any():
        return "velocity_m_per_s", 1e-6, "mm/ns"
    return "velocity_px_per_ns", 1.0, "px/ns"


def plot_branch_velocities(df, results, image_b, settings: Optional[DisplaySettings] = None,
                           bins: int = 12, figsize=(12, 4.5), **overrides):
    """The per-branch velocity distribution, next to the branches themselves.

    Takes what :func:`~streamertools.velocity.measure_velocity_branches`
    returns.  Works with or without a pixel size: the histogram falls back to
    px/ns when the measurement was never calibrated.
    """
    import matplotlib.pyplot as plt

    col, scale, unit = _velocity_column(df)
    values = df[col].dropna() * scale if col in df else []

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    if len(values):
        axs[0].hist(values, bins=min(bins, max(len(values), 1)), color="0.6")
        axs[0].set_title(f"mean {values.mean():.3g} +- {values.std():.2g} {unit} "
                         f"over {len(values)} branches")
    else:
        axs[0].set_title("no branch produced a velocity")
    axs[0].set_xlabel(f"velocity ({unit})")
    axs[0].set_ylabel("branches")

    show_image(image_b, settings, ax=axs[1], **overrides)
    for r in results:
        axs[1].annotate("", xy=(r.head_b.x, r.head_b.y),
                        xytext=(r.head_a.x, r.head_a.y),
                        arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
    axs[1].set_title(f"{len(results)} matched branches")
    fig.tight_layout()
    return fig, axs


def plot_velocity_series(df, ax=None, pixel_size_um: Optional[float] = None,
                         show_mean: bool = True, label: str = "frame to frame"):
    """Frame-to-frame velocity against time, with error bars.

    Takes the table from
    :func:`~streamertools.velocity.velocity_frame_to_frame` or
    :func:`~streamertools.velocity.velocity_from_heads`. Pairs without a
    velocity (no streamer in one of their frames) are simply absent, and the
    x position of each point is the midpoint of the pair's interval, which is
    where that velocity actually applies.  Call it twice on one `ax`, with
    different `label`s, to compare clicked and detected heads.
    """
    ax = _axes(ax, (8, 4.5))
    col, scale, unit = _velocity_column(df)
    ok = df[col].notna() if col in df else None
    if ok is None or not ok.any():
        ax.set_title("no pair produced a velocity")
        return ax

    t = 0.5 * (df.loc[ok, "time_a_ns"] + df.loc[ok, "time_b_ns"])
    ylabel = f"velocity ({unit})"
    y = df.loc[ok, col].astype(float) * scale
    sigma_col = "sigma_" + col
    err = (df.loc[ok, sigma_col].astype(float) * scale if sigma_col in df else None)
    ax.errorbar(t, y, yerr=err, fmt="o-", ms=4, lw=1, capsize=2,
                label=label)
    if show_mean and ok.sum() > 1:
        mean, std = float(y.mean()), float(y.std())
        ax.axhline(mean, color="tab:red", lw=1.2,
                   label=f"mean {mean:.3g} +- {std:.2g}")
        ax.axhspan(mean - std, mean + std, color="tab:red", alpha=0.12)
    n_missing = int((~ok).sum())
    ax.set_xlabel("time (ns, midpoint of the pair)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{int(ok.sum())} of {len(df)} pairs"
                 + (f" ({n_missing} without a head)" if n_missing else ""))
    ax.legend(fontsize=8)
    return ax


def plot_head_track(df, ax=None, pixel_size_um: Optional[float] = None):
    """Axial head position against time, with the straight-line fit.

    Takes the track of :func:`~streamertools.velocity.track_head` or
    :func:`~streamertools.velocity.velocity_from_fronts`.  With an ``in_fit``
    column (the latter), only those points are fitted; the others are drawn
    hollow.
    """
    ax = _axes(ax, (7, 4.5))
    ok = df["axial_px"].notna()
    tcol, xlabel = ("time_ns", "time (ns)") if "time_ns" in df else ("frame", "frame")
    scale, ylabel = (1.0, "axial position (px)")
    if pixel_size_um:
        scale, ylabel = (pixel_size_um / 1000.0, "axial position (mm)")
    fitted = ok & df["in_fit"].astype(bool) if "in_fit" in df else ok
    for mask, style, label in ((fitted, dict(fmt="o"), "head position"),
                               (ok & ~fitted, dict(fmt="o", mfc="none"), "not fitted")):
        if mask.any():
            ax.errorbar(df.loc[mask, tcol], df.loc[mask, "axial_px"] * scale,
                        yerr=df.loc[mask, "sigma_px"] * scale, ms=4, capsize=2,
                        color="tab:blue", label=label, **style)
    t = df.loc[fitted, tcol].astype(float)
    y = df.loc[fitted, "axial_px"].astype(float)
    if len(t) > 1:
        c = np.polyfit(t, y * scale, 1)
        ax.plot(t, np.polyval(c, t), "-", color="tab:red",
                label=f"fit: {c[0]:.4g} {'mm' if pixel_size_um else 'px'}/ns")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------
def show_palette(palette, ax=None):
    """Show a palette as a colour ramp (manual 2.2.1)."""
    from .palette import Palette
    pal = palette if isinstance(palette, Palette) else Palette.load(palette)
    ax = _axes(ax, (6, 1))
    ax.imshow(pal.preview())
    ax.set_yticks([])
    ax.set_xticks([0, 255, 511])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_title(f"palette '{pal.name}' ({len(pal.colours)} colours)", fontsize=9)
    return ax
