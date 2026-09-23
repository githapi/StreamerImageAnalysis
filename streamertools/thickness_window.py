"""
The thickness window
====================

Measure the streamer diameter on straight channel sections, series after
series.  Tools: ``"view"``, ``"crop"`` and ``"line"``.  Kept separate from the
velocity window on purpose: each has its own configuration cell, its own
summary and its own files.

>>> twin = st.open_thickness_window(SERIES, prepare=PREP, out_dir=OUT_DIR,
...                                 tool="line", box_width=75)
>>> twin.lines                     # the series shown
>>> twin.summary                   # one row per saved series

The diameter is defined as in Briels et al (J. Phys. D 41 234004, 2008) and
Nijdam (PhD thesis, TU/e 2011, section 3.4.3): select a straight channel
section, take a perpendicular cross section per pixel along it, average them
into one cross section, and take its full width at half maximum.  That is the
*averaged* method of the width analyzer (manual eq. 3.1), used unchanged.
Nijdam requires at least 10 px across for a reliable value; narrower lines
are flagged (``min_diameter_px``).  Briels evaluates three to ten streamers
per photograph and averages over several photographs -- place a line on
every suitable section, on as many frames as you like; *Save mean
thickness* averages all of them.

Each click pair adds one line: the first click sets one end, the second the
other, and the line is measured straight away.  Right-click cancels a half
placed line, or removes the selected one.  *Detect channel* places a line
automatically with :func:`~streamertools.pick.auto_measure_width`.

Keys: left/right frames, l = line tool, d = detect, delete = remove the
selected line, escape = cancel a half-placed line, Ctrl+S = save the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .geometry import MeasurementBox
from .width import AnalyzerSettings, WidthResult, measure
from .workbench import SeriesWorkbench, _base_window_class, _open_window, small_font_axes

__all__ = ["ThicknessMeasurement", "ThicknessWorkbench", "open_thickness_window"]

_LINE_COLUMNS = ["line", "frame", "x0", "y0", "x1", "y1", "x0_full", "y0_full",
                 "x1_full", "y1_full", "box_width", "diameter_px", "diameter_um",
                 "per_line_px", "per_line_std_px", "valid", "below_min", "method"]


@dataclass
class ThicknessMeasurement:
    """One line on one frame, and the diameter under it."""

    frame: int
    p0: Tuple[float, float]          # uncropped pixels
    p1: Tuple[float, float]
    box_width: float
    diameter_px: float               # FWHM of the averaged cross section
    per_line_px: float               # mean of the per-line FWHMs, for comparison
    per_line_std_px: float
    valid: bool
    method: str = "manual"


class ThicknessWorkbench(SeriesWorkbench):
    """The thickness window's model: measured lines, per series.

    Usually made by :func:`open_thickness_window`.  See
    :class:`~streamertools.workbench.SeriesWorkbench` for series, frames,
    crop, display and saving.
    """

    KIND = "thickness"
    TOOLS = ("view", "crop", "line")
    MEAN_LABEL = "Save mean thickness"
    RESULTS_SUFFIX = "thickness_lines"
    DETAIL_SUFFIX = None                        # the lines file is the detail
    HINTS = {**SeriesWorkbench.HINTS,
             "line": "click both ends of a straight channel section;  right-click "
                     "cancels / removes the selected line;  d = detect"}
    SETTINGS = {**SeriesWorkbench.SETTINGS,
                "box_width": 30.0,              # measurement box, ~5x the diameter
                "averaging_number": 8,          # X of the moving average (manual 3.2)
                "min_diameter_px": 10.0}        # Nijdam (2011): at least 10 px across

    # -- hooks ---------------------------------------------------------------
    def _init_results(self):
        self._lines: List[ThicknessMeasurement] = []
        self._first: Optional[Tuple[float, float]] = None     # uncropped, half placed
        self.selected: Optional[int] = None

    def _export_results(self):
        return list(self._lines)

    def _import_results(self, results):
        self._init_results()
        self._lines = list(results or [])

    def _view_class(self):
        return _window_class()

    @property
    def has_results(self) -> bool:
        return any(m.valid for m in self._lines)

    def results_table(self):
        return self.lines

    def _load_results_table(self, df) -> int:
        missing = {"frame", "x0_full", "y0_full", "x1_full", "y1_full", "diameter_px"} \
            - set(df.columns)
        if missing:
            raise ValueError(f"not a thickness table: missing {sorted(missing)}")
        if self._lines:
            return 0                                 # the window wins
        for rec in df.to_dict("records"):
            k = int(rec["frame"])
            if not 0 <= k < self.n_frames:
                continue
            method = rec.get("method")
            self._lines.append(ThicknessMeasurement(
                frame=k, p0=(float(rec["x0_full"]), float(rec["y0_full"])),
                p1=(float(rec["x1_full"]), float(rec["y1_full"])),
                box_width=float(rec.get("box_width", self.box_width)),
                diameter_px=float(rec["diameter_px"]),
                per_line_px=float(rec.get("per_line_px", np.nan)),
                per_line_std_px=float(rec.get("per_line_std_px", np.nan)),
                valid=bool(rec.get("valid", True)),
                method=method if isinstance(method, str) else "manual"))
        self._changed()
        return len(self._lines)

    def click(self, x, y, button):
        if self.tool != "line":
            return
        if button == "left":
            self.add_point(x, y)
        elif button == "right":
            if self._first is not None:
                self.cancel_point()
            else:
                self.remove_line()

    # ------------------------------------------------------------------
    # lines
    # ------------------------------------------------------------------
    def _to_full(self, p) -> Tuple[float, float]:
        ox, oy = self.offset
        return float(p[0]) + ox, float(p[1]) + oy

    def _to_work(self, p) -> Tuple[float, float]:
        ox, oy = self.offset
        return float(p[0]) - ox, float(p[1]) - oy

    def add_point(self, x: float, y: float) -> Optional[ThicknessMeasurement]:
        """The next end of a line (window coordinates); the second one measures it."""
        if self._first is None:
            self._first = self._to_full((x, y))
            self._changed()
            return None
        p0 = self._to_work(self._first)
        self._first = None
        return self.add_line(p0, (x, y))

    def cancel_point(self):
        self._first = None
        self._changed()

    def _analyzer(self, box_width: float, **overrides) -> AnalyzerSettings:
        # measured in pixels; converted once, in the tables
        return AnalyzerSettings(**{"measure": "width", "box_width": float(box_width),
                                   "averaging_number": int(self.averaging_number),
                                   "pixel_size_um": None, **overrides})

    def _store(self, frame: int, result: WidthResult, method: str) -> ThicknessMeasurement:
        box = result.box
        m = ThicknessMeasurement(
            frame=int(frame), p0=self._to_full(box.p0), p1=self._to_full(box.p1),
            box_width=float(box.width), diameter_px=float(result.fwhm_averaged),
            per_line_px=float(result.fwhm_per_line),
            per_line_std_px=float(result.fwhm_per_line_std),
            valid=bool(result.averaged.valid and result.fwhm_averaged > 0),
            method=method)
        self._lines.append(m)
        self.selected = len(self._lines) - 1
        self._results_changed()
        self._changed()
        if not m.valid:
            self._report(f"frame {frame}: no half-maximum edges under this line -- "
                         f"is it along the channel, with a wide enough box?")
        return m

    def add_line(self, p0, p1, frame: Optional[int] = None,
                 box_width: Optional[float] = None) -> ThicknessMeasurement:
        """Measure the diameter under the line p0-p1 (window coordinates) and keep it."""
        k = self.frame if frame is None else int(frame)
        bw = self.box_width if box_width is None else box_width
        box = MeasurementBox(tuple(map(float, p0)), tuple(map(float, p1)), float(bw))
        return self._store(k, measure(self.work_data[k], box, self._analyzer(bw)), "manual")

    def detect_line(self, frame: Optional[int] = None) -> ThicknessMeasurement:
        """Find the largest channel of a frame and measure it (auto_measure_width)."""
        from .pick import auto_measure_width
        k = self.frame if frame is None else int(frame)
        result = auto_measure_width(self.work_data[k], pixel_size_um=None,
                                    analyzer=self._analyzer(self.box_width))
        return self._store(k, result, "auto")

    def remove_line(self, index: Optional[int] = None):
        """Remove a line: `index`, else the selected one, else the last on this frame."""
        if index is None:
            index = self.selected
        if index is None:
            on_frame = [i for i, m in enumerate(self._lines) if m.frame == self.frame]
            index = on_frame[-1] if on_frame else None
        if index is None or not 0 <= index < len(self._lines):
            return
        del self._lines[index]
        self.selected = None
        self._results_changed()
        self._changed()

    def clear_lines(self):
        self._lines.clear()
        self._first = None
        self.selected = None
        self._results_changed()
        self._changed()

    def select(self, index: Optional[int]):
        """Select a line and show its frame."""
        if index is not None and not 0 <= index < len(self._lines):
            raise IndexError(f"line {index} does not exist")
        self.selected = index
        if index is not None and self._lines[index].frame != self.frame:
            self.goto(self._lines[index].frame)
        else:
            self._changed()

    @property
    def lines(self):
        """Every line of the series shown, as a DataFrame."""
        import pandas as pd
        px = self.pixel_size_um
        rows = []
        for i, m in enumerate(self._lines):
            (x0, y0), (x1, y1) = self._to_work(m.p0), self._to_work(m.p1)
            rows.append({"line": i, "frame": m.frame, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "x0_full": m.p0[0], "y0_full": m.p0[1],
                         "x1_full": m.p1[0], "y1_full": m.p1[1],
                         "box_width": m.box_width, "diameter_px": m.diameter_px,
                         "diameter_um": m.diameter_px * px if px else np.nan,
                         "per_line_px": m.per_line_px, "per_line_std_px": m.per_line_std_px,
                         "valid": m.valid,
                         "below_min": m.valid and m.diameter_px < self.min_diameter_px,
                         "method": m.method})
        return pd.DataFrame(rows, columns=_LINE_COLUMNS)

    def measurement(self, index: Optional[int] = None) -> WidthResult:
        """The full :class:`WidthResult` of a line, for ``stplot.plot_measurement``."""
        index = self.selected if index is None else index
        if index is None:
            raise ValueError("no line selected")
        m = self._lines[index]
        box = MeasurementBox(self._to_work(m.p0), self._to_work(m.p1), m.box_width)
        return measure(self.work_data[m.frame], box, self._analyzer(m.box_width))

    # the line API the width section of the notebook uses
    @property
    def line(self):
        """``(p0, p1)`` of the selected line, in window coordinates."""
        if self.selected is None:
            return None
        m = self._lines[self.selected]
        return self._to_work(m.p0), self._to_work(m.p1)

    @property
    def box(self) -> Optional[MeasurementBox]:
        """The selected line as a :class:`MeasurementBox`."""
        line = self.line
        if line is None:
            return None
        return MeasurementBox(line[0], line[1], self._lines[self.selected].box_width)

    def measure_line(self, frame: Optional[int] = None, **analyzer) -> WidthResult:
        """Measure under the selected line, on `frame` (default: the line's own)."""
        if self.selected is None:
            raise ValueError("no line yet: choose the 'line' tool and click two points")
        m = self._lines[self.selected]
        k = m.frame if frame is None else int(frame)
        analyzer = {"pixel_size_um": self.pixel_size_um, **analyzer}
        return measure(self.work_data[k], self.box, self._analyzer(m.box_width, **analyzer))

    # ------------------------------------------------------------------
    # the mean
    # ------------------------------------------------------------------
    def series_summary(self) -> Optional[dict]:
        good = [m for m in self._lines if m.valid]
        if not good:
            return None
        d = np.array([m.diameter_px for m in good])
        n = len(d)
        std = float(np.std(d, ddof=1)) if n > 1 else float("nan")
        px = self.pixel_size_um
        metric = (lambda value: None if not px else float(value) * px)
        return {
            "n_lines": n, "n_invalid": len(self._lines) - n,
            "n_frames": len({m.frame for m in good}),
            "n_below_min": int((d < self.min_diameter_px).sum()),
            "mean_diameter_px": float(d.mean()), "std_diameter_px": std,
            "sem_diameter_px": std / np.sqrt(n) if n > 1 else float("nan"),
            "min_diameter_px": float(d.min()), "max_diameter_px": float(d.max()),
            "pixel_size_um": px,
            "mean_diameter_um": metric(d.mean()), "std_diameter_um": metric(std),
            "sem_diameter_um": metric(std / np.sqrt(n)) if n > 1 else None,
            "mean_diameter_mm": None if not px else float(d.mean()) * px / 1000.0,
        }

    def describe_saved(self, row: dict) -> str:
        um = row.get("mean_diameter_um")
        metric = um is not None and np.isfinite(float(um))
        key, unit = ("um", "um") if metric else ("px", "px")
        mean, std = row.get(f"mean_diameter_{key}"), row.get(f"std_diameter_{key}")
        text = f"saved d = {float(mean):.4g}"
        if std is not None and np.isfinite(float(std)):
            text += f" +- {float(std):.2g}"
        text += f" {unit} ({row.get('n_lines')} lines on {row.get('n_frames')} frames)"
        if row.get("n_below_min"):
            text += f";  {row['n_below_min']} below {self.min_diameter_px:g} px"
        return text

    def readout(self) -> str:
        """What the window says about the selected line."""
        if self._first is not None:
            return "first end placed -- click the other end (right-click cancels)"
        if self.selected is None:
            n = sum(m.frame == self.frame for m in self._lines)
            return f"frame {self.frame}: {n} line(s)  |  {len(self._lines)} in this series"
        m = self._lines[self.selected]
        px = self.pixel_size_um
        d = f"{m.diameter_px:.4g} px" + (f" = {m.diameter_px * px:.4g} um" if px else "")
        flag = "" if m.valid else "   (no valid edges)"
        if m.valid and m.diameter_px < self.min_diameter_px:
            flag = f"   (below {self.min_diameter_px:g} px: unreliable)"
        (xa, ya), (xb, yb) = self._to_work(m.p0), self._to_work(m.p1)
        return (f"line {self.selected}, frame {m.frame}:  d = {d}{flag}\n"
                f"per line {m.per_line_px:.4g} +- {m.per_line_std_px:.2g} px   "
                f"P0, P1 = ({xa:.1f}, {ya:.1f}), ({xb:.1f}, {yb:.1f})")


def open_thickness_window(series, prepare: Optional[dict] = None, show: bool = True,
                          **config) -> ThicknessWorkbench:
    """Open the thickness window, or update the one already open.

    Put this at the end of the thickness window's configuration cell.
    `series` and `prepare` as for
    :func:`~streamertools.velocity_window.open_velocity_window`; the two
    windows share loaded series but nothing else.
    """
    return _open_window(ThicknessWorkbench, series, prepare, show, config)


# --------------------------------------------------------------------------
_WINDOW_CLASS = None


def _window_class():
    global _WINDOW_CLASS
    if _WINDOW_CLASS is None:
        _WINDOW_CLASS = _build_window_class()
    return _WINDOW_CLASS


def _build_window_class():
    Base = _base_window_class()
    Qt, W = Base.Qt, Base.QtWidgets

    class ThicknessWindow(Base):
        """Lines, their diameters, and the diameters of the series."""

        def _build_extra(self, layout):
            box = W.QGroupBox("Lines and diameter")
            v = W.QVBoxLayout(box)
            g = W.QGridLayout()
            buttons = (("Detect channel", self._detect,
                        "find the largest channel of this frame and measure it (d)"),
                       ("Delete line", lambda: self.wb.remove_line(),
                        "remove the selected line (Delete, or right-click)"),
                       ("Clear all lines", self._clear_lines, "asks first"))
            for i, (text, fn, tip) in enumerate(buttons):
                g.addWidget(self.button(text, fn, tip), 0, i)
            v.addLayout(g)
            self.lbl_readout = W.QLabel()
            self.lbl_readout.setWordWrap(True)
            font = self.lbl_readout.font()
            font.setBold(True)
            self.lbl_readout.setFont(font)
            v.addWidget(self.lbl_readout)

            self.table = W.QTableWidget(0, 5)
            self.table.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(W.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(W.QAbstractItemView.SingleSelection)
            self.table.setFocusPolicy(Qt.NoFocus)
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.cellClicked.connect(self._guard(self._on_table_click))
            self.table.setMinimumHeight(90)
            self.dfig = self.Figure(figsize=(3.4, 2.2), layout="constrained")
            self.dcanvas = self.FigureCanvas(self.dfig)
            self.dcanvas.setMinimumHeight(170)
            self.dax = self.dfig.add_subplot()
            vsplit = W.QSplitter(Qt.Vertical)
            vsplit.addWidget(self.table)
            vsplit.addWidget(self.dcanvas)
            vsplit.setChildrenCollapsible(False)
            v.addWidget(vsplit, stretch=1)
            layout.addWidget(box, stretch=1)

        def _extra_keys(self):
            return [(("L",), lambda: self.wb.set_tool("line")),
                    (("D",), self._detect),
                    (("Delete", "Backspace"), lambda: self.wb.remove_line()),
                    (("Escape",), lambda: self.wb.cancel_point())]

        # -- drawing -------------------------------------------------------
        def _refresh_extra(self):
            self._draw_lines()
            self._draw_table()
            self._draw_diameters()
            self.lbl_readout.setText(self.wb.readout())
            self.dcanvas.draw_idle()

        def _draw_lines(self):
            wb, ax, add = self.wb, self.ax, self.add_overlay
            for i, m in enumerate(wb._lines):
                if m.frame != wb.frame:
                    continue
                box = MeasurementBox(wb._to_work(m.p0), wb._to_work(m.p1), m.box_width)
                corners = np.vstack([box.corners(), box.corners()[:1]])
                chosen = i == wb.selected
                colour = "cyan" if chosen else ("white" if m.valid else "tomato")
                add(ax.plot(corners[:, 0], corners[:, 1], color=colour,
                            lw=1.3 if chosen else 0.8))
                add(ax.plot([box.p0[0], box.p1[0]], [box.p0[1], box.p1[1]],
                            color=colour, lw=0.7, ls=":"))
                add(ax.annotate(str(i), box.p1, textcoords="offset points",
                                xytext=(5, 5), color=colour, fontsize=8))
                if chosen and m.valid:
                    try:
                        left, right = wb.measurement(i).edge_points("per_line")
                        for pts in (left, right):
                            if len(pts):
                                add(ax.plot(pts[:, 0], pts[:, 1], ".", color="cyan", ms=1.5))
                    except Exception:                           # noqa: BLE001
                        pass
            if wb._first is not None:
                x, y = wb._to_work(wb._first)
                add(ax.plot(x, y, "o", color="red", ms=6))

        def _draw_table(self):
            wb = self.wb
            lines = wb.lines
            px = wb.pixel_size_um
            unit = "um" if px else "px"
            self.table.setHorizontalHeaderLabels(
                ["#", "frame", f"d ({unit})", "per line (px)", "note"])
            self.table.setRowCount(len(lines))
            for r, rec in enumerate(lines.to_dict("records")):
                d = rec["diameter_um"] if px else rec["diameter_px"]
                note = ("no edges" if not rec["valid"] else
                        f"< {wb.min_diameter_px:g} px" if rec["below_min"] else "")
                if rec["method"] != "manual":
                    note = (note + " auto").strip()
                texts = (str(rec["line"]), str(rec["frame"]), f"{d:.4g}",
                         f"{rec['per_line_px']:.4g} +- {rec['per_line_std_px']:.2g}", note)
                for c, text in enumerate(texts):
                    self.table.setItem(r, c, W.QTableWidgetItem(text))
            self.table.resizeColumnsToContents()
            self.table.clearSelection()
            if wb.selected is not None and wb.selected < len(lines):
                self.table.selectRow(wb.selected)
                self.table.scrollToItem(self.table.item(wb.selected, 0))

        def _draw_diameters(self):
            wb, ax = self.wb, self.dax
            ax.clear()
            lines = wb.lines
            px = wb.pixel_size_um
            col, unit = ("diameter_um", "um") if px else ("diameter_px", "px")
            valid = lines["valid"].astype(bool)
            good = lines[valid]
            if len(good):
                ax.plot(good["frame"], good[col], "o", ms=4, label="lines")
                mean, std = float(good[col].mean()), float(good[col].std())
                if len(good) > 1:
                    ax.axhline(mean, color="tab:red", lw=1.2,
                               label=f"mean {mean:.3g} +- {std:.2g}")
                    ax.axhspan(mean - std, mean + std, color="tab:red", alpha=0.12)
                if wb.selected is not None and wb.selected < len(lines) \
                        and valid.iloc[wb.selected]:
                    rec = lines.iloc[wb.selected]
                    ax.plot(rec["frame"], rec[col], "o", mfc="none", mec="cyan",
                            ms=9, mew=1.5)
                ax.legend(fontsize=7)
                ax.set_title(f"{len(good)} lines on {good['frame'].nunique()} frames")
            else:
                ax.set_title("no line measured yet")
            ax.axvline(wb.frame, color="0.6", lw=0.8, ls="--")
            ax.set_xlabel("frame")
            ax.set_ylabel(f"diameter ({unit})")
            small_font_axes(ax)

        # -- buttons -------------------------------------------------------
        def _on_table_click(self, row, column):
            item = self.table.item(row, 0)
            if item is not None:
                self.wb.select(int(item.text()))

        def _detect(self):
            wb = self.wb
            try:
                with self._busy("finding the channel ..."):
                    wb.detect_line()
            except ValueError as exc:
                wb._report(f"frame {wb.frame}: {exc}")

        def _clear_lines(self):
            wb = self.wb
            if not wb._lines:
                return
            where = (f"\nThe file {wb.results_path().name} is emptied too."
                     if wb.results_path() is not None else "")
            answer = W.QMessageBox.question(
                self, "Clear all lines",
                f"Remove all {len(wb._lines)} lines of {wb.series_name}?{where}")
            if answer == W.QMessageBox.Yes:
                wb.clear_lines()

    return ThicknessWindow
