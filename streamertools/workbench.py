"""
The analysis windows: what they share
=====================================

Two permanent windows run next to the V3 notebook, each driven by its own
configuration cell and never merged into one:

* :mod:`.velocity_window` -- locate the streamer head frame by frame; the
  frame-to-frame velocity is computed in the window;
* :mod:`.thickness_window` -- place lines on straight channel sections; the
  diameter (FWHM) is computed in the window.

This module holds the common part: the model base class
:class:`SeriesWorkbench`, the Qt window it is shown in, loading of one series
(:func:`prepare_series`) and the hook into the kernel's Qt event loop.

Why a Qt window and not a pyplot figure
---------------------------------------
Switching matplotlib's backend closes every open figure, which is why V2's
pickers kept popping up and vanishing.  These windows are Qt main windows
with an embedded matplotlib canvas, run by the kernel's Qt event loop (what
``%gui qt`` installs).  The notebook stays on ``%matplotlib inline``, and the
windows stay live between cells.

Series
------
A window works through a list of series.  In ``"multi"`` file mode a series
is one multi-frame file; in ``"single"`` file mode it is a set of
single-frame files, one per delay (a folder, a glob pattern or a list).
Each is loaded with :func:`prepare_series` (rotated and, when switched on,
dark-subtracted as the notebook's load cell does).  Three buttons drive it:

*Save mean ...*   one row per series in ``<out_dir>/<kind>_summary.csv``
                  (saving a series again replaces its row), plus its detail
                  table
*Next series*     moves on, asking first when the current series changed
                  since its mean was last saved
*Save and close*  saves the current series and closes the window, at any time

Individual clicks are written to ``<out_dir>/<file stem>_<...>.csv`` after
every change and restored when a series is opened again, so nothing is lost
to a closed window or a restarted kernel.

Settings and state
------------------
Settings (display, timing, units, ...) are applied every time the
configuration cell runs.  State the window changes itself -- series, frame,
crop, tool -- is applied only when its value in the cell changed since the
cell last ran, so re-running the cell to adjust the gamma does not throw away
where you are.

Coordinates
-----------
Everything shown and returned is in the coordinates of the *cropped* stack,
:attr:`SeriesWorkbench.work`.  Results are stored in uncropped coordinates
underneath, so changing the crop never invalidates them.
"""

from __future__ import annotations

import re
import sys
import traceback
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .display import DisplaySettings, render
from .io import ImageStack, load, load_stack, long_path
from .preprocess import apply_physical_normalization, make_dark
from .preprocess import subtract_dark as _subtract_dark
from .velocity import frame_times

__all__ = ["SeriesWorkbench", "prepare_series", "series_label", "series_file_stem",
           "enable_gui", "FILE_MODES"]

_DISPLAY_FIELDS = tuple(DisplaySettings.__dataclass_fields__)
_RENDER_CACHE = 12                    # rendered frames kept, for stepping back and forth


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _same(a, b) -> bool:
    """Equality that survives numpy arrays and objects holding them."""
    if a is b:
        return True
    try:
        eq = a == b
        if isinstance(eq, np.ndarray):
            return np.shape(a) == np.shape(b) and bool(eq.all())
        return bool(eq)
    except Exception:
        return False


def _snapshot(value):
    import copy
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _clip_crop(crop: dict, height: int, width: int) -> dict:
    """A crop box as integer slice bounds inside the frame, never empty."""
    missing = {"x0", "x1", "y0", "y1"} - set(crop)
    if missing:
        raise ValueError(f"crop needs x0, x1, y0 and y1; missing {sorted(missing)}")
    x0, x1 = sorted((int(round(crop["x0"])), int(round(crop["x1"]))))
    y0, y1 = sorted((int(round(crop["y0"])), int(round(crop["y1"]))))
    x0 = int(np.clip(x0, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    y1 = int(np.clip(y1, y0 + 1, height))
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _registry() -> dict:
    """Open windows and loaded series, kept where ``%autoreload`` cannot wipe them.

    autoreload clears a module's namespace before re-running it, so a module
    variable would forget the open window and the next configuration run
    would open a second one.  The ``sys`` module is never reloaded.
    """
    return sys.__dict__.setdefault("_streamertools_workbench", {})


FILE_MODES = ("multi", "single")


def _is_file_spec(item) -> bool:
    """A path, a pattern, or a list of paths -- anything that is loaded from disk."""
    if isinstance(item, (str, Path)):
        return True
    return isinstance(item, (list, tuple)) and bool(item) \
        and all(isinstance(p, (str, Path)) for p in item)


def _series_key(item) -> str:
    """What identifies a series: its file(s), or the object itself."""
    if isinstance(item, (list, tuple)) and _is_file_spec(item):
        return "|".join(str(Path(p).resolve()).lower() for p in item)
    path = item if isinstance(item, (str, Path)) else getattr(item, "path", None)
    if path is not None:
        return str(Path(path).resolve()).lower()
    return f"id:{id(item)}"


def series_label(item, index: int = 0) -> str:
    """A short name for a series: the file, the folder, the pattern.

    A list of single-frame files is named after its first file and its size,
    e.g. ``shot_001.img+9``.
    """
    if isinstance(item, (list, tuple)) and _is_file_spec(item):
        first = Path(item[0]).name
        return first if len(item) == 1 else f"{first}+{len(item) - 1}"
    path = item if isinstance(item, (str, Path)) else getattr(item, "path", None)
    return Path(path).name if path is not None else f"series{index}"


_series_name = series_label


def series_file_stem(item, index: int = 0) -> str:
    """The series' name as used in output file names (no wildcards or slashes)."""
    if isinstance(item, (list, tuple)) and _is_file_spec(item):
        stem = Path(item[0]).stem + (f"+{len(item) - 1}" if len(item) > 1 else "")
    else:
        stem = Path(series_label(item, index)).stem
    return re.sub(r'[<>:"/\\|?*]+', "_", stem).strip(" .") or f"series{index}"


def _write_csv(df, path: Path) -> Path:
    Path(long_path(path.parent)).mkdir(parents=True, exist_ok=True)
    df.to_csv(long_path(path), index=False)
    return path


def _read_csv(path: Path):
    """The CSV at `path`, or None when there is none (long paths included)."""
    import pandas as pd
    if not Path(long_path(path)).is_file():
        return None
    return pd.read_csv(long_path(path))


def enable_gui():
    """Run Qt's event loop inside the kernel, leaving matplotlib's backend alone.

    This is the hook ``%matplotlib qt`` installs, without pointing pyplot at
    windows, so inline figures stay inline.  IPython removes the hook on
    every ``%matplotlib`` magic; opening or configuring a window installs it
    again, so re-running the window cell always revives the window.

    Returns the QApplication.
    """
    # choose the binding through matplotlib first, so IPython follows it
    from matplotlib.backends.qt_compat import QtWidgets

    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None:
        try:
            shell.enable_gui("qt")
        except Exception as exc:
            raise RuntimeError(
                f"could not run Qt inside this kernel ({exc}). Another GUI event "
                f"loop is probably active, e.g. from '%matplotlib tk': restart the "
                f"kernel and open the window before any such magic.") from exc
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["streamertools"])
        # outside IPython nothing else holds it, and a collected QApplication
        # takes the process down with "Must construct a QApplication first"
        _registry()["app"] = app
    return app


def _load_dark(dark_path, rot90: int) -> Tuple[ImageStack, str]:
    """The dark acquisition(s) at `dark_path`, and how to name them.

    One file (all its frames), or a folder, pattern or list of files.
    """
    if isinstance(dark_path, (str, Path)) and Path(long_path(dark_path)).is_file():
        dark = load(dark_path, rot90=rot90)
        name = Path(dark_path).name
    else:
        dark = load_stack(dark_path, rot90=rot90)
        name = series_label(dark_path)
    if dark.n_frames > 1:
        one_each = len(dark.metadata.get("sources", ())) == dark.n_frames
        name += f", mean of {dark.n_frames} {'files' if one_each else 'frames'}"
    return dark, name


def prepare_series(path, rot90: int = 0, pixel_size_um: Optional[float] = None,
                   dark_path=None, dark_from_frames: Optional[int] = None,
                   physical_norm: Optional[dict] = None,
                   mode: str = "multi", subtract_dark: bool = True) -> ImageStack:
    """Load one series the way the notebook's load cell does.

    Rotate on load, subtract the dark (see `subtract_dark`), and optionally
    put it on the common brightness scale.  What was subtracted is kept in
    ``stack.metadata["dark"]``.

    Parameters
    ----------
    path :
        With ``mode="multi"``: one multi-frame file (a kinetic series).
        With ``mode="single"``: the single-frame files of one series, one per
        delay -- a folder, a glob pattern such as ``"run3/*.img"``, or a list
        of files -- read in name order (see
        :func:`~streamertools.io.frame_files`).
    mode : {"multi", "single"}
        Whether a series is one file with many frames, or many files with one
        frame each.
    subtract_dark : bool
        The dark/background switch.  False leaves the counts as recorded
        (`dark_path` and `dark_from_frames` are then not used).  True
        subtracts

        * ``mode="single"``: the dark at `dark_path`, which must be given --
          the shots of a single-frame series are separate discharges, so
          none of them stands in for the background;
        * ``mode="multi"``: the dark at `dark_path` if given, otherwise the
          mean of the first `dark_from_frames` frames, otherwise the median
          of the image corners.
    dark_path :
        The dark/background acquisition: one file (its frames are averaged),
        or a folder, glob pattern or list of dark files (all averaged).
    dark_from_frames : int, optional
        ``mode="multi"`` only: use the mean of the first frames of the
        series when there is no `dark_path`.
    physical_norm : dict, optional
        Keyword arguments of
        :func:`~streamertools.preprocess.apply_physical_normalization`
        (``Vg, D, c_max, c_min, norm_factor, Mf_ref``).  The factor used is
        kept in ``stack.metadata["Mf"]``.

    A series that is still held somewhere -- by the notebook or by the other
    window -- is not loaded a second time: the same object comes back.
    """
    if mode not in FILE_MODES:
        raise ValueError(f"mode must be one of {FILE_MODES}, got {mode!r}")
    if mode == "multi" and not isinstance(path, (str, Path)):
        raise TypeError("mode='multi' takes one file per series; for a list of "
                        "single-frame files use mode='single'")
    if subtract_dark and mode == "single" and not dark_path:
        raise ValueError(
            "dark subtraction is on for single-frame files, but no dark was given: "
            "set DARK_PATH (dark_path=...) to the dark/background shot(s), or switch "
            "the subtraction off with SUBTRACT_DARK = False (subtract_dark=False)")
    if not subtract_dark:                      # only what is used identifies the load
        dark_path = dark_from_frames = None
    elif mode == "single" or dark_path:
        dark_from_frames = None
    norm = tuple(sorted(physical_norm.items())) if physical_norm else None
    dark_key = None if not dark_path else (
        _series_key(dark_path) if _is_file_spec(dark_path) else str(dark_path))
    key = (mode, _series_key(path), int(rot90) % 4, pixel_size_um, bool(subtract_dark),
           dark_key, dark_from_frames, norm)
    cache = _registry().setdefault("stacks", weakref.WeakValueDictionary())
    stack = cache.get(key)
    if stack is not None:
        return stack
    if mode == "single":
        stack = load_stack(path, pixel_size_um=pixel_size_um, rot90=rot90)
    else:
        stack = load(path, pixel_size_um=pixel_size_um, rot90=rot90)
    if not subtract_dark:
        stack.metadata["dark"] = "off"
    else:
        dark_stack, note = _load_dark(dark_path, rot90) if dark_path else (None, None)
        if dark_stack is not None and dark_stack.shape[1:] != stack.shape[1:]:
            raise ValueError(f"the dark {note} has frames of {dark_stack.shape[1:]} px, "
                             f"the series {stack.shape[1:]} px (both after rot90={rot90})")
        dark = make_dark(stack, dark_stack=dark_stack, from_frames=dark_from_frames)
        _subtract_dark(stack, dark, in_place=True)          # the raw counts are not kept
        if note is None:
            n = int(dark_from_frames or 0)
            note = (f"mean of the first {n} frame{'s' if n > 1 else ''}" if n
                    else f"median of the corners, {float(dark):.1f} counts")
        stack.metadata["dark"] = note
    if physical_norm:
        stack, mf = apply_physical_normalization(stack, **physical_norm)
        stack.metadata["Mf"] = mf
    cache[key] = stack
    return stack


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class SeriesWorkbench:
    """One permanent window onto a list of series, driven from the notebook.

    The base of :class:`~streamertools.velocity_window.VelocityWorkbench` and
    :class:`~streamertools.thickness_window.ThicknessWorkbench`; it knows
    about series, frames, the crop and the display, and about saving
    per-series results, but nothing about what is measured.  All of it works
    without Qt; the window only shows it and forwards clicks.

    Subclasses define the class attributes below and the result hooks
    (``_init_results`` ... ``series_summary``).
    """

    KIND = "series"                            # registry key and file names
    TOOLS: Tuple[str, ...] = ("view", "crop")
    MEAN_LABEL = "Save mean"
    RESULTS_SUFFIX = "results"                 # <stem>_<suffix>.csv, rewritten per change
    DETAIL_SUFFIX: Optional[str] = None        # <stem>_<suffix>.csv, written with the mean
    HINTS = {
        "view": "wheel = zoom, middle-drag = pan, toolbar = zoom rectangle / pan / home;  "
                "left/right = frame",
        "crop": "drag a rectangle to crop to it;  'Full frame' undoes the crop",
    }
    #: applied every time the configuration cell runs
    SETTINGS = {
        "display": None,                       # DisplaySettings; any of its fields works too
        "delay_ns": 10.0,
        "times_ns": None,                      # measured gate delay per frame
        "pixel_size_um": None,                 # None -> the stack's own
        "out_dir": None,                       # where results and summaries are kept
    }
    #: applied only when changed in the cell, because the window changes them too
    STATE: Tuple[str, ...] = ("series_index", "frame", "crop", "tool")

    def __init__(self, series, prepare: Optional[dict] = None, show: bool = True,
                 **config):
        self._view = None
        self._render_cache: "OrderedDict" = OrderedDict()
        self._limit_cache: dict = {}
        self._applied: dict = {}
        self._batch_depth = 0
        self._pending: Optional[bool] = None
        self._stash: Dict[str, tuple] = {}     # results of series not shown
        self._summary_rows: Dict[str, dict] = {}
        self.last_error: Optional[str] = None
        self.last_message: Optional[str] = None
        self.dirty = False

        self.frame, self.tool = 0, "view"
        self._crop: Optional[dict] = None
        self.stack: Optional[ImageStack] = None
        self._series: list = []
        self._prepare: dict = {}
        self.series_index = 0
        self._out_dir: Optional[Path] = None
        for key, value in self.SETTINGS.items():
            if key not in ("display", "out_dir"):
                setattr(self, key, value)
        self.display = DisplaySettings(scope="global")
        self._init_defaults()
        self._init_results()

        self.set_out_dir(config.get("out_dir"))
        start = config.get("series_index") or 0
        self.set_series(series, prepare, index=start)
        self.configure(**config)
        if show:
            self.show()

    # -- hooks for subclasses ---------------------------------------------------
    def _init_defaults(self):
        """Default values of subclass settings and state."""

    def _init_results(self):
        """Empty result containers for one series."""

    def _export_results(self):
        """The current series' results, to be stashed while another is shown."""

    def _import_results(self, results):
        """Restore stashed results (None: start empty)."""
        self._init_results()

    @property
    def has_results(self) -> bool:
        return False

    def results_table(self):
        """The current series' results as a DataFrame (the autosave file)."""
        import pandas as pd
        return pd.DataFrame()

    def _load_results_table(self, df) -> int:
        """Merge a saved results table in; returns how many entries were added."""
        return 0

    def series_summary(self) -> Optional[dict]:
        """The numbers saved by *Save mean*, or None when there is nothing yet."""
        return None

    def detail_table(self):
        return None

    def describe_saved(self, row: dict) -> str:
        return "saved"

    def click(self, x: float, y: float, button: str):
        """A click on the image, in window coordinates ("left" / "right")."""

    def _set_setting(self, key: str, value):
        setattr(self, key, value)

    def _validate(self, config: dict):
        if config.get("times_ns") is not None and self.stack is not None:
            frame_times(self.n_frames, times_ns=config["times_ns"])     # raises if short

    def _state_setters(self) -> dict:
        return {"series_index": self.goto_series, "frame": self.goto,
                "crop": self.set_crop, "tool": self.set_tool}

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def configure(self, force: bool = False, **config) -> "SeriesWorkbench":
        """Apply the configuration cell.

        Everything in ``SETTINGS`` (and any :class:`DisplaySettings` field,
        such as ``gamma``) is applied every time.  Everything in ``STATE`` is
        applied when its value differs from the previous run of the cell (or
        always, with ``force=True``), because the window changes it as well;
        ``None`` leaves it to the window (except for ``crop``, where it means
        the full frame).  A misspelt keyword raises instead of being ignored.
        """
        valid = set(self.SETTINGS) | set(self.STATE) | set(_DISPLAY_FIELDS)
        unknown = sorted(set(config) - valid)
        if unknown:
            raise TypeError(
                f"unknown {self.KIND} window setting(s) {unknown}. Valid: "
                f"{', '.join(sorted(set(self.SETTINGS) | set(self.STATE)))}, or any "
                f"DisplaySettings field ({', '.join(_DISPLAY_FIELDS)})")
        if config.get("tool") not in self.TOOLS + (None,):
            raise ValueError(f"tool must be one of {self.TOOLS}, got {config['tool']!r}")
        self._validate(config)

        kept = []
        with self._batch():
            display_over = {k: config.pop(k) for k in list(config) if k in _DISPLAY_FIELDS}
            if "display" in config or display_over:
                self.set_display(config.pop("display", None), **display_over)
            out_dir = config.pop("out_dir", self._out_dir)
            for key in [k for k in config if k in self.SETTINGS]:
                self._set_setting(key, config.pop(key))

            setters = self._state_setters()
            for key in self.STATE:
                if key not in config:
                    continue
                value = config[key]
                if value is None and key != "crop":
                    continue
                if force or key not in self._applied or not _same(self._applied[key], value):
                    setters[key](value)
                elif not _same(self._state(key), value):
                    kept.append(key)
                self._applied[key] = _snapshot(value)

            self.set_out_dir(out_dir)
            self._changed()

        if kept:
            now = ", ".join(f"{k}={self._state(k)!r}" for k in kept)
            print(f"{self.KIND} window: kept its own {now} -- unchanged in this cell "
                  f"since it last ran. goto(...), goto_series(...), set_crop(...) or "
                  f"configure(..., force=True) override it.")
        return self

    def _state(self, key: str):
        return self.crop if key == "crop" else getattr(self, key)

    def set_display(self, settings: Optional[DisplaySettings] = None, **overrides):
        """Change how counts are drawn (any :class:`DisplaySettings` field)."""
        new = DisplaySettings(**{**(settings or self.display).__dict__, **overrides})
        if not _same(new.__dict__, self.display.__dict__):
            self.display = new
            self._render_cache.clear()
        self._changed()

    def set_tool(self, tool: str):
        if tool not in self.TOOLS:
            raise ValueError(f"tool must be one of {self.TOOLS}, got {tool!r}")
        self.tool = tool
        self._changed()

    # ------------------------------------------------------------------
    # series
    # ------------------------------------------------------------------
    def set_series(self, series, prepare: Optional[dict] = None, index: int = 0):
        """Work through these series, loaded with `prepare`.

        With ``prepare["mode"] == "multi"`` (the default) every item is one
        multi-frame file (or a stack).  With ``"single"`` every item is one
        series of single-frame files: a folder, a glob pattern or a list of
        files; a plain list of existing files is taken as *one* such series.

        The series shown stays when it is still in the list (and loaded the
        same way); otherwise the window opens series `index`.
        """
        prepare = dict(prepare or {})
        if prepare.get("mode", "multi") not in FILE_MODES:
            raise ValueError(f"prepare['mode'] must be one of {FILE_MODES}")
        items = list(series) if isinstance(series, (list, tuple)) else [series]
        if prepare.get("mode") == "single" and items \
                and all(isinstance(i, (str, Path)) and Path(long_path(i)).is_file()
                        for i in items):
            items = [items]                            # one series of frame files
        if not items:
            raise ValueError("the list of series is empty")
        new_keys = [_series_key(i) for i in items]
        old_keys = [_series_key(i) for i in self._series]
        same_prepare = _same(prepare, self._prepare)
        if self.stack is not None and new_keys == old_keys and same_prepare:
            old_item = self._series[self.series_index]
            self._series = items
            if items[self.series_index] is not old_item and not _is_file_spec(old_item):
                self._open_series(self.series_index, force=True, keep_results=True)
            return
        current = old_keys[self.series_index] if self.stack is not None else None
        if current in new_keys and same_prepare:
            self._series, self._prepare = items, prepare
            self.series_index = new_keys.index(current)
            self._changed()
            return
        old = (self._series, self._prepare, self.series_index)
        self._series, self._prepare = items, prepare
        try:
            if current in new_keys:
                self._open_series(new_keys.index(current), force=True, keep_results=True)
            else:
                # the series shown leaves the list: keep its results under its own key
                self._open_series(index, force=True, leaving_key=current)
        except Exception:
            self._series, self._prepare, self.series_index = old
            raise

    @property
    def n_series(self) -> int:
        return len(self._series)

    @property
    def series(self) -> list:
        """The names of the series, in order."""
        return [_series_name(item, k) for k, item in enumerate(self._series)]

    @property
    def series_name(self) -> str:
        return _series_name(self._series[self.series_index], self.series_index)

    @property
    def series_stem(self) -> str:
        """The series' name in output file names."""
        return series_file_stem(self._series[self.series_index], self.series_index)

    @property
    def file_mode(self) -> str:
        """``"multi"`` (one file per series) or ``"single"`` (one file per frame)."""
        return self._prepare.get("mode", "multi")

    @property
    def series_key(self) -> str:
        return _series_key(self._series[self.series_index])

    def goto_series(self, index: int):
        """Open series `index` (negative counts from the end)."""
        index = int(index)
        if index < 0:
            index += self.n_series
        if not 0 <= index < self.n_series:
            raise IndexError(f"series {index} does not exist (0 .. {self.n_series - 1})")
        self._open_series(index)

    def next_series(self) -> bool:
        """Move to the next series; False when this is the last one."""
        if self.series_index + 1 >= self.n_series:
            self._message(f"{self.series_name} is the last series -- "
                          f"'Save and close' when you are done")
            return False
        self._open_series(self.series_index + 1)
        return True

    def _open_series(self, index: int, force: bool = False, keep_results: bool = False,
                     leaving_key: Optional[str] = None):
        """Show series `index`.  The results of the series being left are
        stashed under `leaving_key` (default: its key in the current list)."""
        if not force and index == self.series_index and self.stack is not None:
            return
        item = self._series[index]
        if _is_file_spec(item):                # load first: a failure changes nothing
            stack = prepare_series(item, **self._prepare)
        else:
            stack = item if isinstance(item, ImageStack) else ImageStack(np.asarray(item))
        if self.stack is not None and not keep_results:
            key = leaving_key if leaving_key is not None else self.series_key
            self._stash[key] = (self._export_results(), self.dirty)
        self.series_index = index
        self.stack = stack
        self._render_cache.clear()
        self._limit_cache.clear()
        if self._crop is not None:
            self._crop = _clip_crop(self._crop, stack.height, stack.width)
        start = int(self._applied.get("frame", self.frame) or 0)
        if start < 0:
            start += self.n_frames
        self.frame = int(np.clip(start, 0, self.n_frames - 1))
        if not keep_results:
            results, dirty = self._stash.pop(self.series_key, (None, False))
            self._import_results(results)
            self.dirty = dirty
            self._load_series_results()
        self._changed(reset_view=True)

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------
    @property
    def out_dir(self) -> Optional[Path]:
        return self._out_dir

    def set_out_dir(self, out_dir):
        """Keep results in this folder; what is already there is read back."""
        path = None if out_dir is None else Path(out_dir)
        if path == self._out_dir:
            return
        self._out_dir = path
        if path is None:
            return
        Path(long_path(path)).mkdir(parents=True, exist_ok=True)
        self._load_summary()
        if self.stack is not None:
            self._load_series_results()
            if self.has_results:
                self._write_results()

    def results_path(self) -> Optional[Path]:
        if self._out_dir is None:
            return None
        return self._out_dir / f"{self.series_stem}_{self.RESULTS_SUFFIX}.csv"

    def summary_path(self) -> Optional[Path]:
        return None if self._out_dir is None else self._out_dir / f"{self.KIND}_summary.csv"

    def _write_results(self):
        path = self.results_path()
        if path is None:
            return
        try:
            _write_csv(self.results_table(), path)
        except OSError as exc:                     # e.g. OneDrive holding the file
            self._report(f"could not write {path}: {exc}")

    def _results_changed(self):
        self.dirty = True
        self._write_results()

    def _load_series_results(self):
        path = self.results_path()
        if path is None:
            return
        try:
            df = _read_csv(path)
            if df is None:
                return
            added = self._load_results_table(df)
        except Exception as exc:                   # noqa: BLE001
            self._report(f"could not read {path.name}: {exc}")
            return
        if added:
            self._message(f"{self.series_name}: restored {added} from {path.name}")

    def _load_summary(self):
        path = self.summary_path()
        if path is None:
            return
        try:
            df = _read_csv(path)
        except Exception as exc:                   # noqa: BLE001
            self._report(f"could not read {path.name}: {exc}")
            return
        if df is None:
            return
        for rec in df.to_dict("records"):
            key = rec.get("source")
            if isinstance(key, str):
                self._summary_rows.setdefault(key, rec)

    def _write_summary(self):
        path = self.summary_path()
        if path is None or not self._summary_rows:
            return
        try:
            _write_csv(self.summary, path)
        except OSError as exc:
            self._report(f"could not write {path}: {exc}")

    @property
    def summary(self):
        """One row per saved series, as a DataFrame."""
        import pandas as pd
        df = pd.DataFrame(list(self._summary_rows.values()))
        return df.sort_values("series", kind="stable").reset_index(drop=True) \
            if "series" in df else df

    def saved_row(self) -> Optional[dict]:
        """The saved summary row of the series shown, if any."""
        return self._summary_rows.get(self.series_key)

    def save_mean(self) -> Optional[dict]:
        """Save the mean of the series shown: a summary row and its detail table."""
        row = self.series_summary()
        if row is None:
            self._report(f"{self.series_name}: nothing to average yet")
            return None
        row = {"series": self.series_index, "file": self.series_name,
               "source": self.series_key, **row,
               "saved_at": datetime.now().isoformat(timespec="seconds")}
        self._summary_rows[self.series_key] = row
        self._write_results()
        self._write_summary()
        detail = self.detail_table()
        if detail is not None and self._out_dir is not None and self.DETAIL_SUFFIX:
            try:
                _write_csv(detail, self._out_dir / f"{self.series_stem}_{self.DETAIL_SUFFIX}.csv")
            except OSError as exc:
                self._report(f"could not write the detail table: {exc}")
        self.dirty = False
        where = "" if self._out_dir is not None else "  (no out_dir: kept in memory only)"
        self._message(f"{self.series_name}: {self.describe_saved(row)}{where}")
        self._changed()
        return row

    def save_and_close(self):
        """Save the series shown (its mean too, if it has results) and close.

        Returns the summary of every saved series.
        """
        if self.has_results and (self.dirty or self.saved_row() is None):
            self.save_mean()
        self._write_results()
        self._write_summary()
        self.close()
        return self.summary

    # ------------------------------------------------------------------
    # the stack, the crop, the frame
    # ------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.stack.n_frames

    @property
    def pixel_size_um(self) -> Optional[float]:
        """The configured pixel size, or the stack's own when none was given."""
        if self._pixel_size_um is not None:
            return self._pixel_size_um
        return None if self.stack is None else self.stack.pixel_size_um

    @pixel_size_um.setter
    def pixel_size_um(self, value: Optional[float]):
        self._pixel_size_um = value

    @property
    def crop(self) -> Optional[dict]:
        """The crop box in uncropped pixels, ready to paste as ``CROP = ...``."""
        return None if self._crop is None else dict(self._crop)

    @property
    def offset(self) -> Tuple[int, int]:
        """(x0, y0) of the crop: add it to a window coordinate for the full frame."""
        return (0, 0) if self._crop is None else (self._crop["x0"], self._crop["y0"])

    @property
    def work_data(self) -> np.ndarray:
        """The cropped (frames, H, W) array -- a view, not a copy."""
        c = self._crop
        data = self.stack.data
        return data if c is None else data[:, c["y0"]:c["y1"], c["x0"]:c["x1"]]

    @property
    def work(self) -> ImageStack:
        """The series as the window shows it, for the rest of the notebook."""
        return self.stack.copy_with(self.work_data, pixel_size_um=self.pixel_size_um)

    @property
    def times(self) -> np.ndarray:
        """Gate delay of every frame, ns."""
        return frame_times(self.n_frames, self.delay_ns, self.times_ns)

    def set_crop(self, crop: Optional[dict]):
        """Crop to ``{"x0", "x1", "y0", "y1"}`` in uncropped pixels; None = whole frame."""
        self._crop = None if crop is None else _clip_crop(crop, self.stack.height,
                                                         self.stack.width)
        self._render_cache.clear()
        self._changed(reset_view=True)

    def crop_to(self, x0: float, y0: float, x1: float, y1: float):
        """Crop to a rectangle given in the *current* (cropped) coordinates."""
        ox, oy = self.offset
        self.set_crop({"x0": ox + x0, "x1": ox + x1, "y0": oy + y0, "y1": oy + y1})

    def goto(self, frame: int):
        """Show `frame` (negative counts from the end)."""
        frame = int(frame)
        if frame < 0:
            frame += self.n_frames
        self.frame = int(np.clip(frame, 0, self.n_frames - 1))
        self._changed()

    def step(self, n: int = 1):
        self.goto(int(np.clip(self.frame + n, 0, self.n_frames - 1)))

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _crop_key(self):
        c = self._crop
        return None if c is None else (c["x0"], c["x1"], c["y0"], c["y1"])

    def _global_limits(self, s: DisplaySettings) -> Tuple[float, float]:
        """Imin/Imax over the whole cropped series, from a 1-in-4 subsample.

        Cached per series, crop and limit setting; brightness and contrast
        are affine, so the cached values are simply transformed.
        """
        key = (self._crop_key(), s.limit_mode, s.min_hist_percentage, s.max_hist_percentage)
        if key not in self._limit_cache:
            sample = self.work_data[:, ::4, ::4]
            self._limit_cache[key] = DisplaySettings(
                limit_mode=s.limit_mode, min_hist_percentage=s.min_hist_percentage,
                max_hist_percentage=s.max_hist_percentage).limits(sample)
        lo, hi = self._limit_cache[key]
        return s.contrast * lo + s.brightness, s.contrast * hi + s.brightness

    def image(self, frame: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
        """The frame as the window draws it: ``(rgb, Imin, Imax)``."""
        k = self.frame if frame is None else int(frame)
        key = (self.series_key, k, self._crop_key())
        if key in self._render_cache:
            self._render_cache.move_to_end(key)
            return self._render_cache[key]
        s = self.display
        if s.scope == "global" and s.limit_mode != "manual":
            lo, hi = self._global_limits(s)
            s = replace(s, limit_mode="manual", i_min=lo, i_max=hi, scope="frame")
        rgb, _, lo, hi = render(self.work_data[k], s)
        self._render_cache[key] = (rgb, lo, hi)
        while len(self._render_cache) > _RENDER_CACHE:
            self._render_cache.popitem(last=False)
        return rgb, lo, hi

    # ------------------------------------------------------------------
    # the window
    # ------------------------------------------------------------------
    @contextmanager
    def _batch(self):
        """Collect redraws, and draw once at the end."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._pending is not None:
                reset, self._pending = self._pending, None
                self._changed(reset_view=reset)

    def _changed(self, reset_view: bool = False):
        if self._batch_depth:
            self._pending = bool(self._pending) or reset_view
            return
        if self._view is not None:
            self._view.refresh(reset_view=reset_view)

    def _report(self, message: str):
        self.last_error = message
        if self._view is not None:
            self._view.show_error(message)
        else:
            print(f"{self.KIND} window: {message}")

    def _message(self, message: str):
        self.last_message = message
        if self._view is not None:
            self._view.show_message(message)
        else:
            print(f"{self.KIND} window: {message}")

    def _view_class(self):
        raise NotImplementedError

    def show(self):
        """Open the window, or bring it back after it was closed."""
        enable_gui()
        if self._view is None:
            self._view = self._view_class()(self)
        self._view.show()
        self._view.raise_()
        self._view.activateWindow()
        return self

    def close(self):
        if self._view is not None:
            self._view.close()

    @property
    def is_open(self) -> bool:
        return self._view is not None and self._view.isVisible()

    def wait(self):
        """Block until the window is closed -- for scripts run outside Jupyter."""
        if not self.is_open:
            return
        from matplotlib.backends.qt_compat import QtCore
        loop = QtCore.QEventLoop()
        self._view.on_close(loop.quit)
        loop.exec() if hasattr(loop, "exec") else loop.exec_()

    def __repr__(self) -> str:
        h, w = self.work_data.shape[1:]
        crop = f", {self.crop}" if self._crop else ", full frame"
        return (f"<{type(self).__name__} series {self.series_index + 1}/{self.n_series} "
                f"'{self.series_name}', frame {self.frame}/{self.n_frames - 1}, "
                f"{w}x{h} px{crop}, tool '{self.tool}', "
                f"{len(self._summary_rows)} series saved, "
                f"window {'open' if self.is_open else 'closed'}>")


def _open_window(cls, series, prepare, show, config):
    """Open `cls`'s window, or update the one already open."""
    registry = _registry()
    wb = registry.get(cls.KIND)
    if wb is None:
        wb = cls(series, prepare=prepare, show=show, **config)
    else:
        wb.set_series(series, prepare)
        wb.configure(**config)
        if show:
            wb.show()
    registry[cls.KIND] = wb
    return wb


# --------------------------------------------------------------------------
# the Qt window -- built on first use, so importing the package never needs Qt
# --------------------------------------------------------------------------
_BASE_WINDOW = None


def _base_window_class():
    global _BASE_WINDOW
    if _BASE_WINDOW is None:
        _BASE_WINDOW = _build_base_window()
    return _BASE_WINDOW


def _build_base_window():
    from matplotlib.backend_bases import MouseButton
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
    from matplotlib.figure import Figure
    from matplotlib.widgets import RectangleSelector

    Qt = QtCore.Qt
    QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
    # a class body cannot read a same-named variable of this function
    qt_namespace, qt_widgets, figure_class = Qt, QtWidgets, Figure

    class Toolbar(NavigationToolbar2QT):
        # 'Subplots' fights the constrained layout ("layout engine that is
        # incompatible with subplots_adjust") and 'Customize' has nothing to
        # offer here: home, back, forward, pan, zoom and save remain
        toolitems = [t for t in NavigationToolbar2QT.toolitems
                     if t[0] not in ("Subplots", "Customize")]

    class SeriesWindow(QtWidgets.QMainWindow):
        """The view: draws a :class:`SeriesWorkbench` and forwards input to it.

        Subclasses add their own panel (``_build_extra``), tool options
        (``_build_tool_extras``), keys (``_extra_keys``) and drawing
        (``_refresh_extra``).
        """

        Qt = qt_namespace
        QtWidgets = qt_widgets
        Figure = figure_class
        FigureCanvas = FigureCanvasQTAgg

        def __init__(self, wb: SeriesWorkbench):
            super().__init__()
            self.wb = wb
            self._close_callbacks = []
            self._pan = None
            self.image_artist = None
            self.overlays = []
            self._series_names = None
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.resize(min(1500, int(0.92 * screen.width())),
                        min(1000, int(0.92 * screen.height())))
            self._build()
            self.refresh(reset_view=True)

        # -- error handling: a slot must never take the kernel down --------
        def _guard(self, fn):
            def run(*args):
                try:
                    return fn(*args)
                except Exception as exc:                        # noqa: BLE001
                    self.wb._report(f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
            return run

        def _slot(self, fn):
            """A Qt slot that ignores the signal's arguments."""
            guarded = self._guard(fn)
            return lambda *_: guarded()

        @contextmanager
        def _busy(self, message: str):
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            self.show_message(message)
            QtWidgets.QApplication.processEvents()
            try:
                yield
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def button(self, text: str, fn, tip: str = ""):
            b = QtWidgets.QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(self._slot(fn))
            return b

        # -- layout --------------------------------------------------------
        def _build(self):
            W = QtWidgets

            self.fig = Figure(figsize=(8, 8), layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setMinimumSize(360, 360)
            self.ax = self.fig.add_subplot()
            self.ax.tick_params(labelsize=8)
            self.toolbar = Toolbar(self.canvas, self, coordinates=False)
            self.addToolBar(self.toolbar)
            for name, handler in (("button_press_event", self._on_press),
                                  ("button_release_event", self._on_release),
                                  ("motion_notify_event", self._on_move),
                                  ("scroll_event", self._on_scroll)):
                self.canvas.mpl_connect(name, self._guard(handler))
            self.selector = RectangleSelector(
                self.ax, self._guard(self._on_rectangle), useblit=True, button=[1],
                minspanx=3, minspany=3, spancoords="data", interactive=False,
                props=dict(edgecolor="cyan", facecolor="none", lw=1.5))
            self.selector.set_active(False)

            panel = W.QWidget()
            lay = W.QVBoxLayout(panel)

            # series
            box = W.QGroupBox("Series")
            g = W.QGridLayout(box)
            self.series_combo = W.QComboBox()
            self.series_combo.activated.connect(self._guard(self._on_series_chosen))
            self.btn_save_mean = self.button(self.wb.MEAN_LABEL, self._save_mean,
                                             "average this series and save it (Ctrl+S)")
            self.btn_next_series = self.button("Next series >", self._next_series,
                                               "move on to the next series")
            self.btn_save_close = self.button("Save and close", self._save_and_close,
                                              "save this series and close the window")
            self.lbl_saved = W.QLabel()
            self.lbl_saved.setWordWrap(True)
            g.addWidget(self.series_combo, 0, 0, 1, 3)
            g.addWidget(self.btn_save_mean, 1, 0)
            g.addWidget(self.btn_next_series, 1, 1)
            g.addWidget(self.btn_save_close, 1, 2)
            g.addWidget(self.lbl_saved, 2, 0, 1, 3)
            lay.addWidget(box)

            # frame
            box = W.QGroupBox("Frame")
            g = W.QGridLayout(box)
            self.btn_frame_prev = self.button("<", lambda: self.wb.step(-1))
            self.btn_frame_next = self.button(">", lambda: self.wb.step(1))
            for b in (self.btn_frame_prev, self.btn_frame_next):
                b.setMaximumWidth(44)
            self.spin = W.QSpinBox()
            self.spin.setKeyboardTracking(False)
            self.spin.setMinimumWidth(64)
            self.slider = W.QSlider(Qt.Horizontal)
            self.slider.setTracking(False)          # render on release, not per pixel
            self.lbl_time = W.QLabel()
            g.addWidget(self.btn_frame_prev, 0, 0)
            g.addWidget(self.spin, 0, 1)
            g.addWidget(self.btn_frame_next, 0, 2)
            g.addWidget(self.lbl_time, 0, 3)
            g.addWidget(self.slider, 1, 0, 1, 4)
            g.setColumnStretch(3, 1)
            self.spin.valueChanged.connect(self._guard(lambda v: self.wb.goto(v)))
            self.slider.valueChanged.connect(self._guard(lambda v: self.wb.goto(v)))
            lay.addWidget(box)

            # tool and crop
            box = W.QGroupBox("Tool and crop")
            v = W.QVBoxLayout(box)
            row = W.QHBoxLayout()
            self.tool_buttons = {}
            for name in self.wb.TOOLS:
                b = W.QRadioButton(name)
                b.clicked.connect(self._slot(lambda name=name: self.wb.set_tool(name)))
                row.addWidget(b)
                self.tool_buttons[name] = b
            v.addLayout(row)
            self._build_tool_extras(v)
            row = W.QHBoxLayout()
            row.addWidget(self.button("Crop to view", self._crop_to_view,
                                      "crop to the current zoom"))
            row.addWidget(self.button("Full frame", lambda: self.wb.set_crop(None)))
            v.addLayout(row)
            self.crop_edit = W.QLineEdit()
            self.crop_edit.setReadOnly(True)
            self.crop_edit.setToolTip("paste this into the configuration cell")
            v.addWidget(self.crop_edit)
            lay.addWidget(box)

            self._build_extra(lay)

            # scroll vertically on a small screen, but never clip the panel sideways
            scroll = W.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(panel)
            width = max(panel.minimumSizeHint().width(), 400)
            scroll.setMinimumWidth(width + scroll.verticalScrollBar().sizeHint().width() + 4)

            split = W.QSplitter(Qt.Horizontal)
            split.addWidget(self.canvas)
            split.addWidget(scroll)
            split.setStretchFactor(0, 1)
            split.setCollapsible(1, False)
            split.setSizes([1000, 450])
            self.setCentralWidget(split)

            self.lbl_hint = W.QLabel()
            self.lbl_cursor = W.QLabel()
            self.statusBar().addWidget(self.lbl_hint, 1)
            self.statusBar().addPermanentWidget(self.lbl_cursor)
            self.statusBar().messageChanged.connect(
                lambda text: text or self.statusBar().setStyleSheet(""))

            keys = [(("Right", "."), lambda: self.wb.step(1)),
                    (("Left", ","), lambda: self.wb.step(-1)),
                    (("Home",), lambda: self.wb.goto(0)),
                    (("End",), lambda: self.wb.goto(-1)),
                    (("Ctrl+S",), self._save_mean),
                    (("V",), lambda: self.wb.set_tool("view")),
                    (("C",), lambda: self.wb.set_tool("crop"))] + self._extra_keys()
            self.shortcuts = {}
            for names, fn in keys:
                for name in names:
                    sc = QShortcut(QtGui.QKeySequence(name), self)
                    sc.activated.connect(self._slot(fn))
                    self.shortcuts[name] = sc

        # -- subclass hooks ------------------------------------------------
        def _build_tool_extras(self, layout):
            pass

        def _build_extra(self, layout):
            pass

        def _extra_keys(self):
            return []

        def _refresh_extra(self):
            pass

        def add_overlay(self, artists):
            self.overlays.extend(artists if isinstance(artists, (list, tuple)) else [artists])

        # -- drawing -------------------------------------------------------
        def refresh(self, reset_view: bool = False):
            self._sync_widgets()
            self._draw_image(reset_view)
            for artist in self.overlays:
                artist.remove()
            self.overlays = []
            try:
                self._refresh_extra()
            except Exception as exc:                            # noqa: BLE001
                self.show_error(f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            self.canvas.draw_idle()
            # the next frame is usually the one wanted next: have it ready
            QtCore.QTimer.singleShot(30, self._guard(self._prefetch))

        def _prefetch(self):
            if self.wb.frame + 1 < self.wb.n_frames:
                self.wb.image(self.wb.frame + 1)

        def _sync_widgets(self):
            wb = self.wb
            dark = wb.stack.metadata.get("dark") if wb.stack is not None else None
            self.setWindowTitle(f"streamertools {wb.KIND} - {wb.series_name} "
                                f"({wb.series_index + 1}/{wb.n_series}, "
                                f"{wb.file_mode}-frame files"
                                f"{'' if dark is None else ', dark: ' + dark})")
            names = wb.series
            if names != self._series_names:
                self.series_combo.blockSignals(True)
                self.series_combo.clear()
                self.series_combo.addItems([f"{k + 1}/{len(names)}  {n}"
                                            for k, n in enumerate(names)])
                self.series_combo.blockSignals(False)
                self._series_names = names
            self.series_combo.setCurrentIndex(wb.series_index)
            self.btn_next_series.setEnabled(wb.series_index + 1 < wb.n_series)
            self.btn_save_mean.setEnabled(wb.has_results)
            row = wb.saved_row()
            text = "not saved yet" if row is None else wb.describe_saved(row)
            if wb.dirty and wb.has_results:
                text += "  -- changed since"
            self.lbl_saved.setText(text)

            for w in (self.spin, self.slider):
                w.blockSignals(True)
                w.setRange(0, wb.n_frames - 1)
                w.setValue(wb.frame)
                w.blockSignals(False)
            try:
                self.lbl_time.setText(f"delay {wb.times[wb.frame]:g} ns")
            except ValueError:
                self.lbl_time.setText("")
            for tool, b in self.tool_buttons.items():
                b.setChecked(tool == wb.tool)
            self.crop_edit.setText(f"CROP = {wb.crop}")
            self.selector.set_active(wb.tool == "crop")
            if wb.tool != "view":
                self._leave_toolbar_mode()
            self.lbl_hint.setText(wb.HINTS.get(wb.tool, ""))

        def _leave_toolbar_mode(self):
            """Clicks belong to the tool, so switch the toolbar's pan/zoom off."""
            mode = self.ax.get_navigate_mode()
            if mode == "PAN":
                self.toolbar.pan()
            elif mode == "ZOOM":
                self.toolbar.zoom()

        def _full_limits(self):
            h, w = self.wb.work_data.shape[1:]
            return (-0.5, w - 0.5), (h - 0.5, -0.5)

        def _draw_image(self, reset_view: bool):
            wb = self.wb
            rgb, lo, hi = wb.image()
            fresh = (self.image_artist is None or reset_view
                     or self.image_artist.get_array().shape[:2] != rgb.shape[:2])
            if fresh:
                if self.image_artist is not None:
                    self.image_artist.remove()
                self.image_artist = self.ax.imshow(rgb)
                xlim, ylim = self._full_limits()
                self.ax.set_xlim(*xlim)
                self.ax.set_ylim(*ylim)
                self.ax.set_autoscale_on(False)
                self.toolbar.update()
                self.toolbar.push_current()
            else:
                self.image_artist.set_data(rgb)
            s = wb.display
            self.ax.set_title(
                f"{wb.series_name}   frame {wb.frame} / {wb.n_frames - 1}    "
                f"gamma {s.gamma:g}   Imin {lo:.4g}   Imax {hi:.4g}    tool: {wb.tool}",
                fontsize=9)

        def show_error(self, message: str):
            self.statusBar().setStyleSheet("color: #c0392b; font-weight: bold")
            self.statusBar().showMessage(message, 15000)

        def show_message(self, message: str):
            self.statusBar().setStyleSheet("color: #1e7d32; font-weight: bold")
            self.statusBar().showMessage(message, 10000)

        # -- series buttons --------------------------------------------------
        def ask_unsaved(self) -> str:
            """'save', 'discard' or 'cancel' -- replaceable, e.g. in a test."""
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Mean not saved")
            box.setText(f"{self.wb.series_name} changed since its mean was last saved.")
            save = box.addButton("Save mean and continue",
                                 QtWidgets.QMessageBox.AcceptRole)
            discard = box.addButton("Continue without saving",
                                    QtWidgets.QMessageBox.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec() if hasattr(box, "exec") else box.exec_()
            clicked = box.clickedButton()
            return "save" if clicked is save else "discard" if clicked is discard \
                else "cancel"

        def _resolve_unsaved(self) -> bool:
            wb = self.wb
            if not (wb.dirty and wb.has_results):
                return True
            answer = self.ask_unsaved()
            if answer == "save":
                return wb.save_mean() is not None
            return answer == "discard"

        def _save_mean(self):
            self.wb.save_mean()

        def _next_series(self):
            wb = self.wb
            if wb.series_index + 1 >= wb.n_series or not self._resolve_unsaved():
                return
            with self._busy(f"loading {wb.series[wb.series_index + 1]} ..."):
                wb.next_series()

        def _on_series_chosen(self, index: int):
            wb = self.wb
            if index == wb.series_index:
                return
            if not self._resolve_unsaved():
                self.series_combo.setCurrentIndex(wb.series_index)
                return
            with self._busy(f"loading {wb.series[index]} ..."):
                wb.goto_series(index)

        def _save_and_close(self):
            with self._busy("saving ..."):
                self.wb.save_and_close()

        # -- mouse ---------------------------------------------------------
        def _on_press(self, event):
            if event.inaxes is not self.ax or event.xdata is None or event.dblclick:
                return
            if self.ax.get_navigate_mode() is not None:
                return                                   # the toolbar owns the mouse
            if event.button == MouseButton.MIDDLE:
                self._pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim(),
                             self.ax.transData.inverted().frozen())
                return
            button = {MouseButton.LEFT: "left", MouseButton.RIGHT: "right"}.get(event.button)
            if button is not None:
                self.wb.click(float(event.xdata), float(event.ydata), button)

        def _on_release(self, event):
            self._pan = None

        def _on_move(self, event):
            if self._pan is not None and event.x is not None:
                x0, y0, xlim, ylim, inverse = self._pan
                (ax0, ay0), (ax1, ay1) = inverse.transform([(x0, y0), (event.x, event.y)])
                self.ax.set_xlim(xlim[0] - (ax1 - ax0), xlim[1] - (ax1 - ax0))
                self.ax.set_ylim(ylim[0] - (ay1 - ay0), ylim[1] - (ay1 - ay0))
                self.canvas.draw_idle()
            if event.inaxes is not self.ax or event.xdata is None:
                self.lbl_cursor.setText("")
                return
            data = self.wb.work_data
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            text = f"x {event.xdata:.1f}   y {event.ydata:.1f}"
            if 0 <= iy < data.shape[1] and 0 <= ix < data.shape[2]:
                text += f"   counts {float(data[self.wb.frame, iy, ix]):.0f}"
            self.lbl_cursor.setText(text)

        def _on_scroll(self, event):
            if event.inaxes is not self.ax or event.xdata is None:
                return
            factor = 1 / 1.3 if event.button == "up" else 1.3
            (fx0, fx1), (fy0, fy1) = self._full_limits()
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            if factor > 1 and abs(x1 - x0) * factor >= abs(fx1 - fx0) \
                    and abs(y1 - y0) * factor >= abs(fy1 - fy0):
                self.ax.set_xlim(fx0, fx1)
                self.ax.set_ylim(fy0, fy1)
            else:
                x, y = event.xdata, event.ydata
                self.ax.set_xlim(x + (x0 - x) * factor, x + (x1 - x) * factor)
                self.ax.set_ylim(y + (y0 - y) * factor, y + (y1 - y) * factor)
            self.canvas.draw_idle()

        def _on_rectangle(self, press, release):
            xs = sorted((press.xdata, release.xdata))
            ys = sorted((press.ydata, release.ydata))
            # pixel k spans k-0.5 .. k+0.5, so an edge e starts slice round(e+0.5)
            self.selector.clear()
            self.wb.crop_to(xs[0] + 0.5, ys[0] + 0.5, xs[1] + 0.5, ys[1] + 0.5)

        def _crop_to_view(self):
            xs = sorted(self.ax.get_xlim())
            ys = sorted(self.ax.get_ylim())
            self.wb.crop_to(xs[0] + 0.5, ys[0] + 0.5, xs[1] + 0.5, ys[1] + 0.5)

        # -- lifetime ------------------------------------------------------
        def on_close(self, callback):
            self._close_callbacks.append(callback)

        def closeEvent(self, event):
            callbacks, self._close_callbacks = self._close_callbacks, []
            for callback in callbacks:
                callback()
            super().closeEvent(event)

    return SeriesWindow


def small_font_axes(ax, size: int = 7):
    """Shrink an axes' text for the side panel of a window."""
    ax.tick_params(labelsize=size)
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)
    ax.title.set_size(size + 1)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(size)
