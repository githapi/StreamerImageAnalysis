"""
Interactive tools for the notebook
==================================

Every picker exposes the numbers it produced, so an interactive session can
be turned into a reproducible scripted one: place the line by hand once, copy
the printed coordinates, and put them in your batch loop.

Two input modes
---------------
``"click"``
    You click on the image.  Needs an *interactive* matplotlib canvas:
    ``%matplotlib widget`` (the ipympl backend, in the notebook) or
    ``%matplotlib qt`` / ``%matplotlib tk`` (a pop-out window, which works
    even when the widget stack is broken).
``"sliders"``
    Coordinates on sliders, with the same live preview and readout.  Needs
    nothing but ``ipywidgets`` and works under plain ``%matplotlib inline``.

``mode="auto"`` (the default) picks ``"click"`` when the current backend can
deliver mouse events and ``"sliders"`` otherwise, so the pickers keep working
on a machine where ipympl is not set up.  :func:`diagnose` says which you
have and what to fix.

Classes
-------
:class:`Viewer`        frame slider + display settings (works in any backend)
:class:`Cropper`       crop box on sliders (works in any backend)
:class:`LinePicker`    place the two end points, live FWHM readout
:class:`HeadPicker`    place the streamer head in two images -> velocity
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .display import DisplaySettings, render
from .geometry import MeasurementBox
from .palette import Palette
from .preprocess import DenoiseSettings
from .velocity import HeadSettings, VelocityResult, measure_velocity
from .width import AnalyzerSettings, WidthResult, measure

__all__ = ["Viewer", "Cropper", "LinePicker", "HeadPicker",
           "diagnose", "canvas_is_interactive"]

_PALETTES = ["inferno", "magma", "plasma", "viridis", "turbo", "hot", "gray"]

#: backends that deliver mouse events to ``fig.canvas.mpl_connect``
_INTERACTIVE_BACKENDS = ("ipympl", "widget", "nbagg", "notebook", "qt", "tk",
                         "gtk", "wx", "macosx")


def canvas_is_interactive() -> bool:
    """True when the active matplotlib backend can deliver mouse clicks."""
    import matplotlib
    backend = matplotlib.get_backend().lower()
    return any(name in backend for name in _INTERACTIVE_BACKENDS)


def _require_widgets():
    try:
        import ipywidgets                                  # noqa: F401
        import matplotlib.pyplot as plt                    # noqa: F401
    except ImportError as exc:                             # pragma: no cover
        raise ImportError("the interactive tools need ipywidgets:\n"
                          "    pip install ipywidgets\n"
                          "Clicking additionally needs an interactive canvas "
                          "(ipympl, or %matplotlib qt); without one the pickers "
                          "fall back to sliders. Run "
                          "streamertools.interactive.diagnose() for details.") from exc


# --------------------------------------------------------------------------
def diagnose() -> dict:
    """Print what the widget stack looks like here, and what to do about it.

    ``Error displaying widget: model not found`` is a *frontend* message: the
    browser is being asked to render a widget whose model it cannot find in
    the kernel.  In order of how often it is the cause:

    1. the notebook was reopened, or the kernel restarted, with old widget
       output still on screen -- just re-run the cell;
    2. ``ipympl`` is missing from the environment the *kernel* runs in (which
       is not always the one you ran ``pip install`` in);
    3. the frontend extension does not match ``ipywidgets`` -- version 8 needs
       ``jupyterlab_widgets`` 3.x / ``widgetsnbextension`` 4.x, version 7 needs
       1.x / 3.x.  Mixing the two gives exactly this message.
    """
    import importlib.metadata as md

    import matplotlib

    def _version(package):
        try:
            return md.version(package)
        except Exception:
            return None

    from . import __version__

    info = {
        "streamertools": __version__,
        "matplotlib": matplotlib.__version__,
        "backend": matplotlib.get_backend(),
        "interactive_canvas": canvas_is_interactive(),
        "ipywidgets": _version("ipywidgets"),
        "ipympl": _version("ipympl"),
        "jupyterlab_widgets": _version("jupyterlab_widgets"),
        "widgetsnbextension": _version("widgetsnbextension"),
        "jupyterlab": _version("jupyterlab"),
        "notebook": _version("notebook"),
        "python": __import__("sys").executable,
    }

    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")

    print()
    problems = []
    if info["ipywidgets"] is None:
        problems.append("ipywidgets is missing:  pip install ipywidgets")
    else:
        major = int(str(info["ipywidgets"]).split(".")[0])
        lab, nb = info["jupyterlab_widgets"], info["widgetsnbextension"]
        wanted = ("3.x", "4.x") if major >= 8 else ("1.x", "3.x")
        if lab and int(str(lab).split(".")[0]) != int(wanted[0][0]):
            problems.append(f"ipywidgets {major}.x needs jupyterlab_widgets "
                            f"{wanted[0]}, found {lab} -- this is the usual cause "
                            f"of 'model not found'")
        if nb and int(str(nb).split(".")[0]) != int(wanted[1][0]):
            problems.append(f"ipywidgets {major}.x needs widgetsnbextension "
                            f"{wanted[1]}, found {nb}")
    if info["ipympl"] is None:
        problems.append("ipympl is missing, so '%matplotlib widget' cannot work: "
                        "pip install ipympl  (into the kernel's own environment)")
    if not info["interactive_canvas"]:
        problems.append(f"the active backend '{info['backend']}' delivers no mouse "
                        f"events, so the pickers will use sliders. Use "
                        f"'%matplotlib widget', or '%matplotlib qt' for a pop-out "
                        f"window that works without the widget stack.")

    if problems:
        print("What to fix:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("The widget stack looks consistent.")
    print("\nIf the versions are fine and you still see 'model not found', the "
          "output on screen was made by a kernel that no longer exists: restart "
          "the kernel and re-run the cell. Widget output does not survive being "
          "saved and reopened.")
    return info


def _frames(source) -> np.ndarray:
    data = np.asarray(getattr(source, "data", source), dtype=np.float32)
    return data[None] if data.ndim == 2 else data


def _resolve_mode(mode: str) -> str:
    """'auto' -> 'click' if the canvas can deliver mouse events, else 'sliders'."""
    if mode not in ("auto", "click", "sliders"):
        raise ValueError("mode must be 'auto', 'click' or 'sliders'")
    if mode != "auto":
        return mode
    return "click" if canvas_is_interactive() else "sliders"


# --------------------------------------------------------------------------
class Viewer:
    """Frame browser with the manual's display settings live on sliders."""

    def __init__(self, source, settings: Optional[DisplaySettings] = None,
                 denoise: Optional[DenoiseSettings] = None, figsize=(11, 6)):
        _require_widgets()
        self.data = _frames(source)
        self.settings = settings or DisplaySettings()
        self.denoise = denoise or DenoiseSettings()
        self.figsize = figsize
        self._limit_cache: dict = {}
        self._build()

    def _build(self):
        import ipywidgets as w
        import matplotlib.pyplot as plt
        from IPython.display import display

        n = self.data.shape[0]
        vmax = float(np.nanmax(self.data))
        self.w_index = w.IntSlider(0, 0, max(n - 1, 0), description="frame",
                                   continuous_update=False, disabled=n == 1)
        self.w_mode = w.Dropdown(options=["histogram", "auto", "manual"],
                                 value=self.settings.limit_mode, description="limits")
        self.w_scope = w.Dropdown(options=["frame", "global"], value=self.settings.scope,
                                  description="scope")
        self.w_lo = w.FloatSlider(self.settings.min_hist_percentage, min=0, max=20,
                                  step=0.1, description="min %")
        self.w_hi = w.FloatSlider(self.settings.max_hist_percentage, min=80, max=100,
                                  step=0.05, description="max %")
        self.w_imin = w.FloatSlider(0.0, min=0, max=vmax, step=max(vmax / 500, 1e-6),
                                    description="Imin")
        self.w_imax = w.FloatSlider(vmax, min=0, max=vmax, step=max(vmax / 500, 1e-6),
                                    description="Imax")
        self.w_gamma = w.FloatSlider(self.settings.gamma, min=0.2, max=10.0, step=0.05,
                                     description="gamma")
        self.w_log = w.Checkbox(False, description="log (gamma>=10)")
        self.w_pal = w.Dropdown(options=_PALETTES, value="inferno", description="palette")
        self.w_bright = w.FloatSlider(0.0, min=-2000, max=2000, step=10, description="bright")
        self.w_contr = w.FloatSlider(1.0, min=0.1, max=5.0, step=0.05, description="contrast")
        self.w_gauss = w.Checkbox(self.denoise.gaussian, description="gaussian")
        self.w_hot = w.Checkbox(self.denoise.hot_pixels, description="hot pixels")
        self.out = w.Output()

        controls = w.VBox([
            w.HBox([self.w_index, self.w_pal, self.w_gauss, self.w_hot]),
            w.HBox([self.w_mode, self.w_scope, self.w_gamma, self.w_log]),
            w.HBox([self.w_lo, self.w_hi]),
            w.HBox([self.w_imin, self.w_imax]),
            w.HBox([self.w_bright, self.w_contr]),
        ])
        for widget in (self.w_index, self.w_mode, self.w_scope, self.w_lo, self.w_hi,
                       self.w_imin, self.w_imax, self.w_gamma, self.w_log, self.w_pal,
                       self.w_bright, self.w_contr, self.w_gauss, self.w_hot):
            widget.observe(lambda _ : self.render(), names="value")
        display(w.VBox([controls, self.out]))
        self.render()

    def current_settings(self) -> DisplaySettings:
        return DisplaySettings(
            gamma=10.0 if self.w_log.value else self.w_gamma.value,
            palette=self.w_pal.value, limit_mode=self.w_mode.value,
            i_min=self.w_imin.value, i_max=self.w_imax.value,
            min_hist_percentage=self.w_lo.value, max_hist_percentage=self.w_hi.value,
            brightness=self.w_bright.value, contrast=self.w_contr.value,
            scope=self.w_scope.value)

    def current_frame(self) -> np.ndarray:
        d = replace(self.denoise, gaussian=self.w_gauss.value,
                    hot_pixels=self.w_hot.value)
        return d.apply(self.data[self.w_index.value])

    def _global_limits(self, s: DisplaySettings):
        """Imin/Imax over the whole stack, computed once and reused.

        A percentile over a 40-frame 2560x2160 stack takes ~2.7 s, which is
        unusable on a slider.  Two things fix it: the percentiles are taken
        on a 1-in-4 pixel subsample (a display limit does not need every
        pixel), and they are cached -- brightness and contrast are affine,
        so ``percentile(c*x + b) == c*percentile(x) + b`` and the cached raw
        values can be reused as those two sliders move.
        """
        key = (s.limit_mode, s.min_hist_percentage, s.max_hist_percentage)
        if key not in self._limit_cache:
            sample = self.data[:, ::4, ::4]
            self._limit_cache[key] = DisplaySettings(
                limit_mode=s.limit_mode,
                min_hist_percentage=s.min_hist_percentage,
                max_hist_percentage=s.max_hist_percentage).limits(sample)
        raw_lo, raw_hi = self._limit_cache[key]
        return s.contrast * raw_lo + s.brightness, s.contrast * raw_hi + s.brightness

    def render(self):
        import matplotlib.pyplot as plt
        with self.out:
            self.out.clear_output(wait=True)
            frame = self.current_frame()
            s = self.current_settings()
            if s.scope == "global" and s.limit_mode != "manual":
                lo_g, hi_g = self._global_limits(s)
                s = DisplaySettings(**{**s.__dict__, "limit_mode": "manual",
                                       "i_min": lo_g, "i_max": hi_g,
                                       "scope": "frame"})
            rgb, _, lo, hi = render(frame, s, reference=None)
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.imshow(rgb)
            ax.set_title(f"frame {self.w_index.value}   Imin={lo:.4g}  Imax={hi:.4g}  "
                         f"gamma={s.gamma:g}")
            ax.axis("off")
            plt.show()


# --------------------------------------------------------------------------
class Cropper:
    """Pick a crop box with sliders; the result lands in ``.box``."""

    def __init__(self, source, figsize=(12, 5)):
        _require_widgets()
        self.data = _frames(source)
        self.box = {"x0": 0, "y0": 0, "x1": self.data.shape[2], "y1": self.data.shape[1]}
        self.figsize = figsize
        self._build()

    def _build(self):
        import ipywidgets as w
        from IPython.display import display

        h, wd = self.data.shape[1], self.data.shape[2]
        self.w_frame = w.IntSlider(0, 0, max(self.data.shape[0] - 1, 0), description="frame")
        self.w_x0 = w.IntSlider(0, 0, wd, description="x0")
        self.w_x1 = w.IntSlider(wd, 0, wd, description="x1")
        self.w_y0 = w.IntSlider(0, 0, h, description="y0")
        self.w_y1 = w.IntSlider(h, 0, h, description="y1")
        self.out = w.Output()
        for widget in (self.w_frame, self.w_x0, self.w_x1, self.w_y0, self.w_y1):
            widget.observe(lambda _ : self.render(), names="value")
        display(w.VBox([w.HBox([self.w_frame]),
                        w.HBox([self.w_x0, self.w_x1]),
                        w.HBox([self.w_y0, self.w_y1]), self.out]))
        self.render()

    def render(self):
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        x0, x1 = sorted((self.w_x0.value, self.w_x1.value))
        y0, y1 = sorted((self.w_y0.value, self.w_y1.value))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        self.box = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
        frame = self.data[self.w_frame.value]
        rgb, _, _, _ = render(frame)
        with self.out:
            self.out.clear_output(wait=True)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=self.figsize)
            a1.imshow(rgb); a1.axis("off")
            a1.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           ec="cyan", fc="none", lw=1.5))
            a1.set_title("full frame")
            a2.imshow(rgb[y0:y1, x0:x1]); a2.axis("off")
            a2.set_title(f"crop [{x0}:{x1}, {y0}:{y1}]  {x1-x0}x{y1-y0}")
            plt.tight_layout(); plt.show()

    def apply(self, source=None):
        """Apply the current box to a stack (or to the one shown)."""
        data = _frames(source) if source is not None else self.data
        b = self.box
        return data[:, b["y0"]:b["y1"], b["x0"]:b["x1"]]


# --------------------------------------------------------------------------
class LinePicker:
    """Click the two end points of the measurement line (manual 3.2).

    First click places the red point, the second the green one; clicking
    again starts over.  The width is recomputed on every change and the
    result is available as ``.result`` (a :class:`~streamertools.width.WidthResult`).

    >>> picker = LinePicker(frame, pixel_size_um=17.0, box_width=24)
    >>> # ... click twice ...
    >>> picker.result.fwhm_averaged, picker.coordinates
    """

    def __init__(self, source, frame: int = 0,
                 settings: Optional[AnalyzerSettings] = None,
                 display_settings: Optional[DisplaySettings] = None,
                 figsize=(12, 5), mode: str = "auto", **overrides):
        _require_widgets()
        self.data = _frames(source)[frame]
        base = settings or AnalyzerSettings()
        pixel_size = getattr(source, "pixel_size_um", None)
        if base.pixel_size_um is None and pixel_size:
            base.pixel_size_um = pixel_size
        self.settings = AnalyzerSettings(**{**base.__dict__, **overrides})
        self.display = display_settings or DisplaySettings()
        self.figsize = figsize
        self.mode = _resolve_mode(mode)
        self.points: List[Tuple[float, float]] = []
        self.result: Optional[WidthResult] = None
        self._build()

    # -- ui ----------------------------------------------------------------
    def _build(self):
        import ipywidgets as w
        import matplotlib.pyplot as plt
        from IPython.display import display

        self.w_mode = w.ToggleButtons(options=["width", "length"],
                                      value=self.settings.measure, description="measure")
        self.w_box = w.FloatSlider(self.settings.box_width, min=2, max=200, step=1,
                                   description="box width", continuous_update=False)
        self.w_avg = w.IntSlider(self.settings.averaging_number, min=1, max=50,
                                 description="X (moving)", continuous_update=False)
        self.w_smooth = w.Checkbox(self.settings.smooth, description="Savitzky-Golay")
        self.w_clear = w.Button(description="clear points", icon="eraser")
        self.readout = w.HTML()
        self.out = w.Output()
        rows = [w.HBox([self.w_mode, self.w_box, self.w_avg,
                        self.w_smooth, self.w_clear])]

        if self.mode == "sliders":
            h, wd = self.data.shape
            common = dict(continuous_update=False)
            self.w_x0 = w.IntSlider(int(wd * 0.4), min=0, max=wd - 1, description="x0", **common)
            self.w_y0 = w.IntSlider(int(h * 0.4), min=0, max=h - 1, description="y0", **common)
            self.w_x1 = w.IntSlider(int(wd * 0.6), min=0, max=wd - 1, description="x1", **common)
            self.w_y1 = w.IntSlider(int(h * 0.6), min=0, max=h - 1, description="y1", **common)
            for widget in (self.w_x0, self.w_y0, self.w_x1, self.w_y1):
                widget.observe(lambda _: self._from_sliders(), names="value")
            rows.append(w.HBox([self.w_x0, self.w_y0]))
            rows.append(w.HBox([self.w_x1, self.w_y1]))
            rows.append(w.HTML("<i>No interactive canvas, so the end points are on "
                               "sliders. Use <code>%matplotlib widget</code> or "
                               "<code>%matplotlib qt</code> to click instead; "
                               "<code>streamertools.interactive.diagnose()</code> "
                               "says what is missing.</i>"))

        self.w_clear.on_click(lambda _: (self.points.clear(), self._draw()))
        for widget in (self.w_mode, self.w_box, self.w_avg, self.w_smooth):
            widget.observe(lambda _ : self._recompute(), names="value")

        display(w.VBox(rows + [self.readout, self.out]))

        if self.mode == "click":
            with self.out:
                self.fig, (self.ax_img, self.ax_plot) = plt.subplots(
                    1, 2, figsize=self.figsize, gridspec_kw={"width_ratios": [1, 1.2]})
                self.fig.canvas.mpl_connect("button_press_event", self._on_click)
                plt.show()
            self._draw()
        else:
            self._from_sliders()

    def _from_sliders(self):
        self.points = [(float(self.w_x0.value), float(self.w_y0.value)),
                       (float(self.w_x1.value), float(self.w_y1.value))]
        self._recompute()

    def _on_click(self, event):
        if event.inaxes is not self.ax_img or event.xdata is None:
            return
        if len(self.points) >= 2:
            self.points.clear()
        self.points.append((float(event.xdata), float(event.ydata)))
        self._recompute()

    # -- work --------------------------------------------------------------
    @property
    def coordinates(self):
        """``(p0, p1)`` of the current line, for pasting into a script."""
        return tuple(self.points) if len(self.points) == 2 else None

    @property
    def box(self) -> Optional[MeasurementBox]:
        if len(self.points) != 2:
            return None
        return MeasurementBox(self.points[0], self.points[1], self.w_box.value)

    def _recompute(self):
        box = self.box
        if box is not None:
            s = AnalyzerSettings(**{**self.settings.__dict__,
                                    "measure": self.w_mode.value,
                                    "box_width": self.w_box.value,
                                    "averaging_number": self.w_avg.value,
                                    "smooth": self.w_smooth.value})
            self.result = measure(self.data, box, s)
        self._draw()

    def _draw(self):
        import matplotlib.pyplot as plt
        if self.mode == "click":
            self.ax_img.clear()
            self.ax_plot.clear()
            self._render_axes()
            self.fig.canvas.draw_idle()
            return
        # slider mode: a fresh figure into the Output, so plain inline works
        with self.out:
            self.out.clear_output(wait=True)
            self.fig, (self.ax_img, self.ax_plot) = plt.subplots(
                1, 2, figsize=self.figsize, gridspec_kw={"width_ratios": [1, 1.2]})
            self._render_axes()
            plt.tight_layout()
            plt.show()

    def _render_axes(self):
        rgb, _, lo, hi = render(self.data, self.display)
        self.ax_img.imshow(rgb)
        self.ax_img.axis("off")
        self.ax_img.set_title("click the two end points of the line"
                              if self.mode == "click" else "end points on the sliders")
        for (x, y), colour in zip(self.points, ("red", "lime")):
            self.ax_img.plot(x, y, "o", color=colour, ms=6)

        r = self.result
        if r is not None and len(self.points) == 2:
            corners = np.vstack([r.box.corners(), r.box.corners()[:1]])
            self.ax_img.plot(corners[:, 0], corners[:, 1], color="white", lw=1.2)
            try:
                left, right = r.edge_points("per_line")
                for pts in (left, right):
                    if len(pts):
                        self.ax_img.plot(pts[:, 0], pts[:, 1], ".", color="cyan", ms=1.5)
            except Exception:
                pass
            from .plotting import plot_profiles
            plot_profiles(r, ax=self.ax_plot)
            u = r.unit
            self.readout.value = (
                f"<b>{r.settings.measure}</b> &nbsp; "
                f"averaged <b>{r.fwhm_averaged:.4g} {u}</b> &nbsp;|&nbsp; "
                f"per line {r.fwhm_per_line:.4g} &plusmn; {r.fwhm_per_line_std:.2g} {u} "
                f"&nbsp;|&nbsp; moving(X={r.settings.averaging_number}) "
                f"{r.fwhm_moving:.4g} &plusmn; {r.fwhm_moving_std:.2g} {u}<br>"
                f"box length {r.box_length:.4g} {u} &nbsp;|&nbsp; "
                f"avg counts {r.average_counts_in_area:.1f} &nbsp;|&nbsp; "
                f"max counts {r.maximum_counts_in_area:.1f} &nbsp;|&nbsp; "
                f"avg max cross sect. {r.average_maximum_cross_section:.1f}<br>"
                f"<code>p0={tuple(round(v,1) for v in self.points[0])}, "
                f"p1={tuple(round(v,1) for v in self.points[1])}, "
                f"box_width={self.w_box.value:g}</code>")
        else:
            self.ax_plot.text(0.5, 0.5, "click two points on the image",
                              ha="center", va="center", transform=self.ax_plot.transAxes)
            self.ax_plot.axis("off")


# --------------------------------------------------------------------------
class HeadPicker:
    """Click the streamer head in two images and read off the velocity.

    Click once in the left image and once in the right image; the velocity
    updates immediately.  Press *detect* to let
    :func:`~streamertools.velocity.detect_head` place both points instead,
    then drag them by clicking again if you disagree.
    """

    def __init__(self, image_a, image_b, delay_ns: float = 10.0,
                 pixel_size_um: Optional[float] = None,
                 head_settings: Optional[HeadSettings] = None,
                 display_settings: Optional[DisplaySettings] = None,
                 axis=None, origin=None, figsize=(12, 6), mode: str = "auto"):
        _require_widgets()
        self.mode = _resolve_mode(mode)
        self.a = _frames(image_a)[0]
        self.b = _frames(image_b)[0]
        self.delay_ns = float(delay_ns)
        self.pixel_size_um = (pixel_size_um if pixel_size_um is not None
                              else getattr(image_a, "pixel_size_um", None))
        self.head_settings = head_settings or HeadSettings()
        self.display = display_settings or DisplaySettings()
        self.axis, self.origin = axis, origin
        self.figsize = figsize
        self.pa: Optional[Tuple[float, float]] = None
        self.pb: Optional[Tuple[float, float]] = None
        self.result: Optional[VelocityResult] = None
        self._build()

    def _build(self):
        import ipywidgets as w
        import matplotlib.pyplot as plt
        from IPython.display import display

        self.w_delay = w.FloatText(self.delay_ns, description="delay (ns)")
        self.w_pixel = w.FloatText(self.pixel_size_um or 0.0, description="um/px")
        self.w_detect = w.Button(description="detect heads", icon="magic")
        self.w_clear = w.Button(description="clear", icon="eraser")
        self.readout = w.HTML()
        self.out = w.Output()
        rows = [w.HBox([self.w_delay, self.w_pixel, self.w_detect, self.w_clear])]

        if self.mode == "sliders":
            ha, wa = self.a.shape
            common = dict(continuous_update=False)
            self.w_xa = w.IntSlider(wa // 2, min=0, max=wa - 1, description="x A", **common)
            self.w_ya = w.IntSlider(ha // 2, min=0, max=ha - 1, description="y A", **common)
            self.w_xb = w.IntSlider(wa // 2, min=0, max=wa - 1, description="x B", **common)
            self.w_yb = w.IntSlider(ha // 2, min=0, max=ha - 1, description="y B", **common)
            for widget in (self.w_xa, self.w_ya, self.w_xb, self.w_yb):
                widget.observe(lambda _: self._from_sliders(), names="value")
            rows.append(w.HBox([self.w_xa, self.w_ya]))
            rows.append(w.HBox([self.w_xb, self.w_yb]))
            rows.append(w.HTML("<i>No interactive canvas, so the heads are on "
                               "sliders. Press <b>detect heads</b> to place them "
                               "automatically, then nudge. "
                               "<code>streamertools.interactive.diagnose()</code> "
                               "says what is missing for clicking.</i>"))

        self.w_detect.on_click(lambda _: self._detect())
        self.w_clear.on_click(lambda _: self._clear())
        for widget in (self.w_delay, self.w_pixel):
            widget.observe(lambda _ : self._recompute(), names="value")

        display(w.VBox(rows + [self.readout, self.out]))
        if self.mode == "click":
            with self.out:
                self.fig, self.axs = plt.subplots(1, 2, figsize=self.figsize)
                self.fig.canvas.mpl_connect("button_press_event", self._on_click)
                plt.show()
            self._draw()
        else:
            self._from_sliders()

    def _from_sliders(self):
        self.pa = (float(self.w_xa.value), float(self.w_ya.value))
        self.pb = (float(self.w_xb.value), float(self.w_yb.value))
        self._recompute()

    def _clear(self):
        self.pa = self.pb = self.result = None
        self._draw()

    def _on_click(self, event):
        if event.xdata is None:
            return
        if event.inaxes is self.axs[0]:
            self.pa = (float(event.xdata), float(event.ydata))
        elif event.inaxes is self.axs[1]:
            self.pb = (float(event.xdata), float(event.ydata))
        else:
            return
        self._recompute()

    def _detect(self):
        from .velocity import detect_head
        try:
            ha = detect_head(self.a, axis=self.axis, origin=self.origin,
                             settings=self.head_settings)
            hb = detect_head(self.b, axis=self.axis, origin=self.origin,
                             settings=self.head_settings)
            self.pa, self.pb = (ha.x, ha.y), (hb.x, hb.y)
            if self.mode == "sliders":       # move the sliders to the detection
                for widget, value in ((self.w_xa, ha.x), (self.w_ya, ha.y),
                                      (self.w_xb, hb.x), (self.w_yb, hb.y)):
                    widget.unobserve_all()
                    widget.value = int(round(value))
                    widget.observe(lambda _: self._from_sliders(), names="value")
            self._recompute()
        except ValueError as exc:
            self.readout.value = f"<span style='color:tomato'>{exc}</span>"

    def _recompute(self):
        if self.pa and self.pb:
            px = self.w_pixel.value or None
            self.result = measure_velocity(
                self.a, self.b, delay_ns=self.w_delay.value, pixel_size_um=px,
                axis=self.axis, origin=self.origin, settings=self.head_settings,
                manual_heads=(self.pa, self.pb))
        self._draw()

    def _draw(self):
        import matplotlib.pyplot as plt
        if self.mode == "sliders":
            with self.out:
                self.out.clear_output(wait=True)
                self.fig, self.axs = plt.subplots(1, 2, figsize=self.figsize)
                self._render_axes()
                plt.tight_layout()
                plt.show()
            return
        self._render_axes()
        self.fig.canvas.draw_idle()

    def _render_axes(self):
        for ax, img, point, name in ((self.axs[0], self.a, self.pa, "image A"),
                                     (self.axs[1], self.b, self.pb, "image B")):
            ax.clear()
            rgb, _, _, _ = render(img, self.display)
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(f"{name} -- " + ("click the streamer head"
                                          if self.mode == "click" else "head on the sliders"))
            if point:
                ax.plot(*point, "+", color="cyan", ms=16, mew=2)
        r = self.result
        if r is not None:
            v, s = r.velocity_m_per_s, r.sigma_velocity_m_per_s
            phys = ("give a pixel size for m/s" if v is None else
                    f"<b>v = {v/1e6:.4g} &plusmn; {(s or 0)/1e6:.2g} mm/ns</b> "
                    f"({v:.4g} m/s)")
            self.readout.value = (
                f"{phys} &nbsp;|&nbsp; displacement {r.distance_px:.2f} px "
                f"(axial {r.axial_distance_px:.2f}, lateral {r.lateral_distance_px:.2f}) "
                f"in {r.delay_ns:g} ns<br><code>manual_heads="
                f"({tuple(round(c,1) for c in self.pa)}, "
                f"{tuple(round(c,1) for c in self.pb)})</code>")
