"""
Getting coordinates without ipywidgets
======================================

The widgets in :mod:`streamertools.interactive` are a convenience, never a
requirement: every measurement in this package is a plain function call that
takes numbers.  The only thing the pickers really do is help you *find* those
numbers.  This module does the same job with nothing but matplotlib, so a
broken widget stack cannot block any analysis.

Three routes, in the order worth trying:

:func:`auto_line` / :func:`auto_measure_width`
    Do not pick at all.  The channel is found the same way the velocity code
    finds a streamer head -- threshold, largest connected component,
    principal axis -- and the measurement line is placed along it.  Two
    passes: a generous box first, then a box three times the width that pass
    found, which is the geometry the FWHM baseline wants (see the box-width
    note in the README).
:func:`ginput_line`
    Click, in a pop-out window.  ``%matplotlib qt`` (or ``tk``) talks to Qt
    directly and does not involve ipywidgets, ipympl, or the notebook
    frontend at all -- so it works when the widget stack is broken.
:func:`coordinate_grid`
    Plain inline image with labelled pixel gridlines and optional markers.
    Read the coordinates off by eye, type them into your script.  Works in
    any backend, including a static PNG in a rendered notebook.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .display import DisplaySettings, render
from .geometry import MeasurementBox
from .velocity import HeadSettings, _components
from .width import AnalyzerSettings, WidthResult, measure

__all__ = ["use_window", "use_inline", "browse", "pick_crop", "pick_line",
           "pick_heads", "pick_velocity", "coordinate_grid", "ginput_line",
           "auto_line", "auto_measure_width"]

#: GUI toolkits, best first, with the import that proves each is installed
_GUI_BACKENDS = (("qt", ("PyQt6", "PySide6", "PyQt5", "PySide2")),
                 ("tk", ("tkinter",)))


def _available(module: str) -> bool:
    """Is `module` importable?  Asked in a way that cannot itself raise.

    ``importlib.util.find_spec`` looks harmless but is not: IPython installs
    an ``ImportDenier`` on ``sys.meta_path`` that *raises* ImportError for
    every Qt binding other than the one already loaded --

        Importing PyQt6 disabled by IPython, which has already imported
        an Incompatible QT Binding: pyqt5

    -- so merely asking "is PyQt6 there?" blew up the whole backend search
    while PyQt5, sitting later in the same tuple, would have worked fine.
    A module already in ``sys.modules`` is obviously available; anything that
    raises while being probed counts as unavailable and we move on.
    """
    import importlib.util
    import sys

    if module in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        # ImportError from IPython's denier, ValueError from a half-initialised
        # namespace package, ModuleNotFoundError for a missing parent package.
        return False


def use_window(prefer: Optional[str] = None, quiet: bool = False) -> str:
    """Switch matplotlib to a pop-out window, bypassing the widget stack.

    A real GUI window talks to Qt or Tk directly: no ipywidgets, no ipympl,
    no notebook frontend, so it works when ``%matplotlib widget`` does not.
    Inside IPython this runs the ``%matplotlib`` magic for you, so the event
    loop is hooked up properly.

    Parameters
    ----------
    prefer : {"qt", "tk"}, optional
        Force one toolkit.  By default Qt is used when a binding is
        installed, otherwise Tk, which ships with Python on Windows.

    Returns
    -------
    str
        The matplotlib backend now in use.
    """
    import matplotlib

    order = [p for p in _GUI_BACKENDS if prefer is None or p[0] == prefer]
    if prefer is not None and not order:
        raise ValueError("prefer must be 'qt' or 'tk'")

    for name, modules in order:
        if not any(_available(m) for m in modules):
            continue
        try:
            shell = None
            try:
                from IPython import get_ipython
                shell = get_ipython()
            except ImportError:
                pass
            if shell is not None:
                shell.run_line_magic("matplotlib", name)
            else:
                matplotlib.use({"qt": "QtAgg", "tk": "TkAgg"}[name])
            backend = matplotlib.get_backend()
            if not quiet:
                print(f"matplotlib backend is now '{backend}' -- plots open in their "
                      f"own window, and clicking works without ipywidgets.")
            return backend
        except Exception as exc:                                # pragma: no cover
            if not quiet:
                print(f"could not start the {name} backend ({exc}); trying the next")

    raise RuntimeError(
        "no GUI toolkit found. Install one of:\n"
        "    pip install PyQt5          (then use_window())\n"
        "Tk normally ships with Python on Windows; if 'import tkinter' fails, "
        "reinstall Python with the tcl/tk option ticked. Meanwhile "
        "auto_measure_width() needs no picking at all.")


def use_inline(quiet: bool = False) -> str:
    """Switch matplotlib back to inline plotting after a pop-out window."""
    import matplotlib
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")
    else:
        matplotlib.use("Agg")
    if not quiet:
        print(f"matplotlib backend is now '{matplotlib.get_backend()}' "
              f"-- plots appear under the cell again.")
    return matplotlib.get_backend()


def _next_index(index: int, key: str, n_frames: int) -> int:
    """Where a key press moves the browser. Pure, so it can be tested."""
    if key in ("right", "down"):
        return min(index + 1, n_frames - 1)
    if key in ("left", "up"):
        return max(index - 1, 0)
    if key == "home":
        return 0
    if key == "end":
        return n_frames - 1
    return index


def browse(stack, settings: Optional[DisplaySettings] = None, start: int = 0,
           figsize=(9, 8), prefer: Optional[str] = None):
    """Step through a stack in a pop-out window, with the arrow keys.

    The widget-free replacement for :class:`~streamertools.interactive.Viewer`:
    a real Qt/Tk window, so nothing depends on ipywidgets or ipympl.

        left / right   previous / next frame
        home / end     first / last frame
        g              cycle gamma 1.0 -> 1.5 -> 2.0 -> log
        q              close (matplotlib's own binding)

    The frame you were last on is returned when the window closes, so you can
    carry it straight into an analysis cell.
    """
    import matplotlib.pyplot as plt

    use_window(prefer)
    data = np.asarray(getattr(stack, "data", stack), dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    base = settings or DisplaySettings()
    gammas = [1.0, 1.5, 2.0, 10.0]
    state = {"index": int(np.clip(start, 0, len(data) - 1)),
             "gamma": gammas.index(base.gamma) if base.gamma in gammas else 1}

    fig, ax = plt.subplots(figsize=figsize)

    def draw():
        s = DisplaySettings(**{**base.__dict__, "gamma": gammas[state["gamma"]]})
        rgb, _, lo, hi = render(data[state["index"]], s)
        ax.clear()
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(f"frame {state['index']} / {len(data) - 1}   "
                     f"gamma {s.gamma:g}   Imin {lo:.0f}  Imax {hi:.0f}\n"
                     f"left/right = frame,  g = gamma,  q = close", fontsize=10)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "g":
            state["gamma"] = (state["gamma"] + 1) % len(gammas)
        else:
            moved = _next_index(state["index"], event.key, len(data))
            if moved == state["index"]:
                return
            state["index"] = moved
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    browse._state = state              # so a test (or you) can drive it
    browse._on_key = on_key
    draw()
    plt.show()
    print(f"last frame shown: {state['index']}")
    return state["index"]


def pick_crop(image, settings: Optional[DisplaySettings] = None, frame: int = 0,
              prefer: Optional[str] = None) -> dict:
    """Click two opposite corners in a pop-out window; get a crop box back.

    The widget-free replacement for :class:`~streamertools.interactive.Cropper`.
    Returns ``{"x0":…, "x1":…, "y0":…, "y1":…}`` and prints the slice line to
    paste into the notebook, so the crop becomes reproducible.
    """
    use_window(prefer)
    points = ginput_line(image, n=2, settings=settings, frame=frame)
    if len(points) != 2:
        raise RuntimeError(f"got {len(points)} points, need 2 opposite corners")
    (xa, ya), (xb, yb) = points
    x0, x1 = sorted((int(round(xa)), int(round(xb))))
    y0, y1 = sorted((int(round(ya)), int(round(yb))))
    box = {"x0": x0, "x1": max(x1, x0 + 1), "y0": y0, "y1": max(y1, y0 + 1)}
    print(f"CROP = {box}")
    print(f"work = stack.copy_with(stack.data[:, {box['y0']}:{box['y1']}, "
          f"{box['x0']}:{box['x1']}])")
    return box


def pick_line(image, box_width: float = 30.0,
              settings: Optional[DisplaySettings] = None,
              frame: int = 0, prefer: Optional[str] = None) -> MeasurementBox:
    """Open a window, click the two end points, get a measurement box back.

    >>> box = st.pick_line(frame, box_width=30)     # window opens, click twice
    >>> res = st.measure(frame, box, st.AnalyzerSettings(pixel_size_um=17.0))
    """
    use_window(prefer)
    points = ginput_line(image, n=2, settings=settings, frame=frame)
    if len(points) != 2:
        raise RuntimeError(f"got {len(points)} points, need 2")
    box = MeasurementBox(points[0], points[1], float(box_width))
    print(f"MeasurementBox({box.p0}, {box.p1}, {box_width})")
    return box


def pick_heads(image_a, image_b, settings: Optional[DisplaySettings] = None,
               prefer: Optional[str] = None) -> Tuple[Tuple[float, float],
                                                      Tuple[float, float]]:
    """Open a window per image, click the streamer head in each.

    Returns the pair ready for
    ``measure_velocity(..., manual_heads=pick_heads(a, b))``.
    """
    use_window(prefer)
    a = ginput_line(image_a, n=1, settings=settings)
    b = ginput_line(image_b, n=1, settings=settings)
    if not a or not b:
        raise RuntimeError("need one point in each image")
    print(f"manual_heads=({a[0]}, {b[0]})")
    return a[0], b[0]


def pick_velocity(image_a, image_b, delay_ns: float = 10.0,
                  pixel_size_um: Optional[float] = None,
                  axis=None, settings: Optional[DisplaySettings] = None,
                  show: bool = True, prefer: Optional[str] = None,
                  back_to_inline: bool = True, **overrides):
    """Click one head per image, then get the result plot back under the cell.

    The whole manual measurement in one call: two pop-out windows to click
    in, the backend switched back to inline, the velocity computed from the
    two clicked points, and the same three-panel figure the automatic route
    draws -- image A with its head, image B with its head, and the
    displacement arrow with the velocity in the title.

    >>> vel = st.pick_velocity(a, b, delay_ns=10.0,
    ...                        pixel_size_um=work.pixel_size_um, axis="down")

    Parameters
    ----------
    image_a, image_b :
        The two frames, in the order they were taken.
    delay_ns, pixel_size_um, axis :
        As in :func:`~streamertools.velocity.measure_velocity`.
    settings : DisplaySettings, optional
        Used both for the clicking windows and for the result figure, so
        what you clicked on is what you see.
    show : bool
        Draw the result figure and print the summary.  Set False to get the
        :class:`~streamertools.velocity.VelocityResult` alone.
    back_to_inline : bool
        Return matplotlib to the inline backend afterwards, so the result
        figure appears under the cell rather than in a fourth window.
    **overrides
        Any :class:`~streamertools.velocity.HeadSettings` field, plus
        ``delay_jitter_ns`` and ``pixel_size_rel_error`` for the error bar.
        ``position_uncertainty_px`` is worth setting here: a clicked point is
        as good as your eye, not as good as a centroid (1.5 px is assumed).

    Returns
    -------
    VelocityResult
    """
    from .velocity import measure_velocity

    heads = pick_heads(image_a, image_b, settings=settings, prefer=prefer)
    if back_to_inline:
        use_inline(quiet=True)

    result = measure_velocity(image_a, image_b, delay_ns=delay_ns,
                              pixel_size_um=pixel_size_um, axis=axis,
                              manual_heads=heads, **overrides)
    if show:
        import matplotlib.pyplot as plt

        from . import plotting as stplot
        print(result)
        stplot.plot_velocity(image_a, image_b, result, settings)
        plt.show()
    return result


def _frame(image) -> np.ndarray:
    data = np.asarray(getattr(image, "data", image), dtype=np.float32)
    return data[0] if data.ndim == 3 else data


# --------------------------------------------------------------------------
def coordinate_grid(image, settings: Optional[DisplaySettings] = None,
                    step: Optional[int] = None, ax=None, figsize=(9, 9),
                    points: Optional[Sequence[Sequence[float]]] = None,
                    box: Optional[MeasurementBox] = None, frame: int = 0):
    """Show the image with labelled pixel gridlines, to read coordinates off.

    Parameters
    ----------
    step : int, optional
        Gridline spacing in pixels; defaults to a round number giving about
        ten lines across the image.
    points : sequence of (x, y), optional
        Marked and labelled, so you can check coordinates you already have.
    box : MeasurementBox, optional
        Drawn as well -- the quickest way to confirm a line you typed in.
    """
    import matplotlib.pyplot as plt

    img = _frame(image) if not hasattr(image, "data") else _frames_at(image, frame)
    h, w = img.shape
    if step is None:
        rough = max(h, w) / 10.0
        step = int(max(10, round(rough / 10.0) * 10))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    rgb, _, lo, hi = render(img, settings)
    ax.imshow(rgb)
    ax.set_xticks(np.arange(0, w + 1, step))
    ax.set_yticks(np.arange(0, h + 1, step))
    ax.set_xticks(np.arange(0, w + 1, step / 2), minor=True)
    ax.set_yticks(np.arange(0, h + 1, step / 2), minor=True)
    ax.grid(which="major", color="cyan", lw=0.6, alpha=0.55)
    ax.grid(which="minor", color="cyan", lw=0.3, alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x / column (px)")
    ax.set_ylabel("y / row (px)")
    ax.set_title(f"{w} x {h} px, gridlines every {step} px")

    for i, p in enumerate(points or []):
        ax.plot(p[0], p[1], "+", color="white", ms=14, mew=2)
        ax.annotate(f"({p[0]:.0f}, {p[1]:.0f})", (p[0], p[1]),
                    textcoords="offset points", xytext=(8, 6),
                    color="white", fontsize=9)
    if box is not None:
        corners = np.vstack([box.corners(), box.corners()[:1]])
        ax.plot(corners[:, 0], corners[:, 1], color="white", lw=1.2)
        ax.plot(*box.p0, "o", color="red", ms=6)
        ax.plot(*box.p1, "o", color="lime", ms=6)
    return ax


def _frames_at(stack, frame: int) -> np.ndarray:
    data = np.asarray(stack.data, dtype=np.float32)
    return data[frame] if data.ndim == 3 else data


# --------------------------------------------------------------------------
def ginput_line(image, n: int = 2, settings: Optional[DisplaySettings] = None,
                timeout: float = 0, figsize=(9, 9), frame: int = 0
                ) -> List[Tuple[float, float]]:
    """Click `n` points on the image and return them. No ipywidgets involved.

    Put ``%matplotlib qt`` (or ``tk``) in a cell first: that opens a real
    window, which bypasses the notebook widget stack completely.  Under an
    inline backend there is nothing to click and this raises, pointing you at
    :func:`coordinate_grid` or :func:`auto_line` instead.

    Returns
    -------
    list of (x, y)
        Ready to paste: ``p0, p1 = ginput_line(frame)``.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if not any(k in backend for k in ("qt", "tk", "gtk", "wx", "macosx",
                                      "ipympl", "widget", "nbagg")):
        raise RuntimeError(
            f"the '{matplotlib.get_backend()}' backend has no window to click in. "
            f"Run '%matplotlib qt' (or tk) first -- that bypasses ipywidgets "
            f"entirely -- or use coordinate_grid() / auto_line() instead.")

    img = _frames_at(image, frame) if hasattr(image, "data") else _frame(image)
    fig, ax = plt.subplots(figsize=figsize)
    rgb, _, _, _ = render(img, settings)
    ax.imshow(rgb)
    ax.set_title(f"click {n} point(s); right-click undoes, middle-click finishes")
    plt.tight_layout()
    picked = plt.ginput(n=n, timeout=timeout)
    plt.close(fig)
    print("p0, p1 = " + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in picked))
    return [(float(x), float(y)) for x, y in picked]


# --------------------------------------------------------------------------
def auto_line(image, settings: Optional[HeadSettings] = None,
              length_frac: float = 0.8, frame: int = 0,
              box_width: Optional[float] = None, **overrides
              ) -> Tuple[MeasurementBox, dict]:
    """Place a measurement line along the brightest channel, automatically.

    The channel is found exactly as a streamer head is (manual-independent,
    but the same machinery): smooth, threshold, keep the largest connected
    component, then take the principal axis of its intensity-weighted pixel
    cloud as the channel direction.  The line is centred on the component's
    centroid and spans `length_frac` of its axial extent, trimmed to the 10th
    and 90th percentile of the projections so the ragged ends do not set the
    length.

    Returns
    -------
    (box, info)
        `box` is ready for :func:`streamertools.width.measure`; `info` carries
        the axis, the centroid, the component size and the perpendicular
        spread used to pick a default box width.

    Notes
    -----
    With several branches this follows the *largest* one. For a tree, pick
    the branch yourself (crop to it, or use :func:`coordinate_grid`) -- an
    automatic choice would be arbitrary, not clever.
    """
    s = HeadSettings(**{**(settings or HeadSettings()).__dict__, **overrides})
    img = _frames_at(image, frame) if hasattr(image, "data") else _frame(image)

    smooth, mask, labels = _components(img, s)
    if not mask.any():
        raise ValueError("nothing above the threshold -- lower `threshold`, or "
                         "check that the frame is dark-subtracted")

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    sel = labels == biggest

    ys, xs = np.nonzero(sel)
    pts = np.stack([xs, ys], axis=1).astype(float)
    weights = smooth[ys, xs].astype(float)
    centroid = np.average(pts, axis=0, weights=weights)

    cov = np.cov((pts - centroid).T, aweights=weights)
    values, vectors = np.linalg.eigh(cov)
    along = vectors[:, int(np.argmax(values))]
    across_spread = float(np.sqrt(max(values.min(), 1e-12)))

    projection = (pts - centroid) @ along
    lo, hi = np.percentile(projection, [10, 90])
    half = 0.5 * length_frac * (hi - lo)

    p0 = centroid - half * along
    p1 = centroid + half * along
    width = box_width if box_width is not None else max(6.0, 6.0 * across_spread)

    info = {"axis": along, "centroid": centroid, "n_pixels": int(sel.sum()),
            "across_spread_px": across_spread, "axial_extent_px": float(hi - lo),
            "threshold": s.level(smooth)}
    return MeasurementBox(tuple(p0), tuple(p1), float(width)), info


def auto_measure_width(image, pixel_size_um: Optional[float] = None,
                       analyzer: Optional[AnalyzerSettings] = None,
                       head: Optional[HeadSettings] = None,
                       length_frac: float = 0.8, frame: int = 0,
                       box_factor: float = 5.0, max_passes: int = 4,
                       tol: float = 0.02, **overrides) -> WidthResult:
    """Find the channel and measure its width -- no clicking, no widgets.

    Runs :func:`auto_line`, then measures repeatedly, each pass sizing the
    box at `box_factor` times the width the previous pass found, until the
    box stops changing by more than `tol`.

    The iteration is not decoration.  The FWHM routine fits its baseline on
    whatever lies outside the doubled peak bracket, so a box barely wider
    than the channel eats into the tails and biases the width low -- and a
    single refinement pass inherits that bias when it sizes its own box.
    Measured on synthetic channels of sigma 4, 6 and 9 px, the bias against
    the box/width ratio is:

    ======  ====================
    ratio   bias
    ======  ====================
    2x      -9 to -11 %
    3x      -0.3 to -2.7 %
    4x      +0.5 to -1.3 %
    5x      +0.3 to -0.6 %
    8x      about 0
    ======  ====================

    Hence the default of 5, not the 3 suggested by a single test case.

    Any :class:`~streamertools.width.AnalyzerSettings` field can be passed as
    a keyword -- ``measure="length"``, ``averaging_number=4``, ``smooth=True``.
    The result carries ``auto_info`` (the detected axis and centroid) and
    ``passes`` (the width after each iteration), so you can see it settle.
    """
    img = _frames_at(image, frame) if hasattr(image, "data") else _frame(image)
    if pixel_size_um is None:
        pixel_size_um = getattr(image, "pixel_size_um", None)

    base = analyzer or AnalyzerSettings()
    a_fields = AnalyzerSettings.__dataclass_fields__
    a_over = {k: v for k, v in overrides.items() if k in a_fields}
    h_over = {k: v for k, v in overrides.items() if k not in a_fields}
    s = AnalyzerSettings(**{**base.__dict__, **a_over})
    if pixel_size_um is not None:
        s.pixel_size_um = pixel_size_um

    box, info = auto_line(img, settings=head, length_frac=length_frac, **h_over)
    limit = float(max(img.shape))
    step = s.step

    result = None
    passes: List[float] = []
    box_px = box.width
    for _ in range(max(1, int(max_passes))):
        current = MeasurementBox(box.p0, box.p1, box_px)
        result = measure(img, current, AnalyzerSettings(**{**s.__dict__,
                                                          "box_width": box_px}))
        passes.append(result.fwhm_averaged)
        width_px = (result.fwhm_averaged / step) if result.fwhm_averaged > 0 \
            else box_px / box_factor
        wanted = float(np.clip(box_factor * width_px, 6.0, limit))
        if abs(wanted - box_px) <= tol * box_px:
            break
        box_px = wanted

    result.auto_info = info                      # type: ignore[attr-defined]
    result.passes = passes                       # type: ignore[attr-defined]
    return result
