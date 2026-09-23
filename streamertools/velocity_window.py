"""
The velocity window
===================

Locate the streamer head frame by frame -- frame 17, 18, 19 ... one click
each -- and the frame-to-frame velocity is computed in the window after
every click.  Tools: ``"view"``, ``"crop"`` and ``"head"``.

>>> vwin = st.open_velocity_window(SERIES, prepare=PREP, out_dir=OUT_DIR,
...                                frame=17, tool="head", delay_ns=10.0,
...                                gate_ns=5.0, axis="down")
>>> vwin.velocity_table()          # the series shown
>>> vwin.summary                   # one row per saved series

*Save mean velocity* stores, per series, the mean of the frame-to-frame
velocities and the slope of head position against end-of-exposure time (the
literature's "average velocity", see :func:`~streamertools.velocity.velocity_from_fronts`).
*Detect* places heads with ``head_settings`` -- by default the literature
front of the furthest-protruding streamer -- as a first pass to correct by
clicking.

Keys: left/right (or , and .) frames, home/end, h = head tool, d = detect,
delete = remove this frame's head, Ctrl+S = save the mean.

The shared machinery (series, crop, saving, the Qt window) is in
:mod:`streamertools.workbench`; the thickness window is separate, in
:mod:`streamertools.thickness_window`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

from .velocity import (HeadPosition, HeadSettings, VelocityResult, _line_fit,
                       detect_head, velocity_from_heads)
from .workbench import SeriesWorkbench, _base_window_class, _open_window, small_font_axes

__all__ = ["VelocityWorkbench", "open_velocity_window", "Workbench", "open_workbench"]

_HEAD_COLUMNS = ["frame", "time_ns", "time_end_ns", "x", "y", "x_full", "y_full",
                 "sigma_px", "method"]


class VelocityWorkbench(SeriesWorkbench):
    """The velocity window's model: heads per frame, per series.

    Usually made by :func:`open_velocity_window`.  See
    :class:`~streamertools.workbench.SeriesWorkbench` for series, frames,
    crop, display and saving.
    """

    KIND = "velocity"
    TOOLS = ("view", "crop", "head")
    MEAN_LABEL = "Save mean velocity"
    RESULTS_SUFFIX = "clicked_heads"
    DETAIL_SUFFIX = "velocity_clicked"
    HINTS = {**SeriesWorkbench.HINTS,
             "head": "click the streamer head (the window moves on);  right-click removes "
                     "it;  left/right = frame,  d = detect"}
    SETTINGS = {**SeriesWorkbench.SETTINGS,
                "gate_ns": 0.0,                # exposure: t_end = delay + gate
                "axis": None,                  # propagation direction
                "origin": None,                # electrode tip, for detection
                "delay_jitter_ns": 0.0,
                "pixel_size_rel_error": 0.0,
                "click_uncertainty_px": 1.5,
                "bridge_gaps": False,
                "head_settings": None,         # HeadSettings, or a dict of its fields
                "show_track": True}
    STATE = SeriesWorkbench.STATE + ("auto_advance",)

    # -- hooks ---------------------------------------------------------------
    def _init_defaults(self):
        self.head_settings = HeadSettings()
        self.auto_advance = True

    def _init_results(self):
        self._heads: Dict[int, HeadPosition] = {}     # uncropped coordinates

    def _export_results(self):
        return dict(self._heads)

    def _import_results(self, results):
        self._heads = dict(results or {})

    def _set_setting(self, key, value):
        if key == "head_settings":
            value = HeadSettings(**value) if isinstance(value, dict) else (value or HeadSettings())
        setattr(self, key, value)

    def _state_setters(self):
        return {**super()._state_setters(), "auto_advance": self.set_auto_advance}

    def _view_class(self):
        return _window_class()

    @property
    def has_results(self) -> bool:
        return bool(self._heads)

    def results_table(self):
        return self.heads

    def _load_results_table(self, df) -> int:
        missing = {"frame", "x_full", "y_full"} - set(df.columns)
        if missing:
            raise ValueError(f"not a head table: missing {sorted(missing)}")
        added = 0
        for rec in df.to_dict("records"):
            k = int(rec["frame"])
            if not 0 <= k < self.n_frames or k in self._heads:
                continue                               # the window wins
            sigma = rec.get("sigma_px", np.nan)
            method = rec.get("method")
            self._heads[k] = HeadPosition(
                float(rec["x_full"]), float(rec["y_full"]),
                sigma_px=float(sigma) if np.isfinite(sigma) else self.click_uncertainty_px,
                method=method if isinstance(method, str) else "manual")
            added += 1
        self._changed()
        return added

    def click(self, x, y, button):
        if self.tool != "head":
            return
        if button == "left":
            self.set_head(x, y, advance=self.auto_advance)
        elif button == "right":
            self.remove_head()

    def set_auto_advance(self, on: bool):
        self.auto_advance = bool(on)
        self._changed()

    # ------------------------------------------------------------------
    # heads
    # ------------------------------------------------------------------
    def set_head(self, x: float, y: float, frame: Optional[int] = None,
                 sigma_px: Optional[float] = None, method: str = "manual",
                 advance: bool = False):
        """Put the head of `frame` at (x, y), in window (cropped) coordinates.

        With `advance`, the window moves on to the next frame -- the click
        workflow of the head tool.
        """
        k = self.frame if frame is None else int(frame)
        ox, oy = self.offset
        sigma = self.click_uncertainty_px if sigma_px is None else sigma_px
        self._heads[k] = HeadPosition(float(x) + ox, float(y) + oy,
                                      sigma_px=float(sigma), method=method)
        self._results_changed()
        if advance and k == self.frame and self.frame < self.n_frames - 1:
            self.frame += 1
        self._changed()

    def remove_head(self, frame: Optional[int] = None):
        self._heads.pop(self.frame if frame is None else int(frame), None)
        self._results_changed()
        self._changed()

    def clear_heads(self):
        self._heads.clear()
        self._results_changed()
        self._changed()

    def head(self, frame: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """(x, y) of a frame's head in window coordinates, or None."""
        h = self._heads.get(self.frame if frame is None else int(frame))
        if h is None:
            return None
        ox, oy = self.offset
        return h.x - ox, h.y - oy

    def detect_head(self, frame: Optional[int] = None) -> HeadPosition:
        """Detect a frame's head automatically and place it; correct it by clicking.

        Uses ``head_settings`` (the literature front by default), ``axis``
        and ``origin``.  Raises ValueError, with the reason, when the frame
        holds no streamer.
        """
        k = self.frame if frame is None else int(frame)
        head = self._detect(k)
        self._results_changed()
        self._changed()
        return head

    def detect_heads(self, frames=None, overwrite: bool = False) -> Dict[int, str]:
        """Detect the head in every frame of `frames` that has none yet.

        Returns ``{frame: reason}`` for the frames where nothing was found,
        so a pre-discharge frame is reported rather than raising.
        """
        failed = {}
        with self._batch():
            for k in (range(self.n_frames) if frames is None else frames):
                if k in self._heads and not overwrite:
                    continue
                try:
                    self._detect(int(k))
                except ValueError as exc:
                    failed[int(k)] = str(exc).split(":")[0]
            self._results_changed()
            self._changed()
        return failed

    def _detect(self, k: int) -> HeadPosition:
        head = detect_head(self.work_data[k], axis=self.axis, origin=self.origin,
                           settings=self.head_settings)
        ox, oy = self.offset
        self._heads[k] = replace(head, x=head.x + ox, y=head.y + oy)
        return head

    def _work_heads(self) -> Dict[int, HeadPosition]:
        ox, oy = self.offset
        return {k: replace(h, x=h.x - ox, y=h.y - oy) for k, h in self._heads.items()}

    def _previous_head(self, frame: int) -> Optional[int]:
        """The frame whose head the velocity of `frame` is measured from."""
        if self.bridge_gaps:
            earlier = [k for k in self._heads if k < frame]
            return max(earlier) if earlier else None
        return frame - 1 if (frame - 1) in self._heads else None

    @property
    def heads(self):
        """The located heads as a DataFrame, in window and uncropped coordinates."""
        import pandas as pd
        ox, oy = self.offset
        try:
            t = self.times
        except ValueError:
            t = np.full(self.n_frames, np.nan)
        rows = [{"frame": k, "time_ns": float(t[k]),
                 "time_end_ns": float(t[k]) + float(self.gate_ns),
                 "x": h.x - ox, "y": h.y - oy, "x_full": h.x, "y_full": h.y,
                 "sigma_px": h.sigma_px, "method": h.method}
                for k, h in sorted(self._heads.items())]
        return pd.DataFrame(rows, columns=_HEAD_COLUMNS)

    # kept from V3, for code written against the first workbench
    def save_heads(self, path):
        from pathlib import Path

        from .workbench import _write_csv
        return _write_csv(self.heads, Path(path))

    def load_heads(self, path) -> int:
        from pathlib import Path

        from .workbench import _read_csv
        df = _read_csv(Path(path))
        if df is None:
            raise FileNotFoundError(path)
        return self._load_results_table(df)

    # ------------------------------------------------------------------
    # velocity
    # ------------------------------------------------------------------
    def velocity(self, csv_path=None):
        """``(table, results)`` of the frame-to-frame velocity from the located heads.

        Times are end-of-exposure times, ``delay + gate_ns``, as in the table
        of :func:`~streamertools.velocity.velocity_from_fronts`; the
        intervals, and so the velocities, do not depend on the gate.
        """
        return velocity_from_heads(
            self._work_heads(), times_ns=self.times + float(self.gate_ns),
            pixel_size_um=self.pixel_size_um, axis=self.axis,
            position_uncertainty_px=self.click_uncertainty_px,
            delay_jitter_ns=self.delay_jitter_ns,
            pixel_size_rel_error=self.pixel_size_rel_error,
            bridge_gaps=self.bridge_gaps, csv_path=csv_path)

    def velocity_table(self, csv_path=None):
        """The frame-to-frame table, same columns as ``velocity_frame_to_frame``."""
        return self.velocity(csv_path)[0]

    def detail_table(self):
        return self.velocity_table()

    def pair_result(self, frame_a: Optional[int] = None) -> VelocityResult:
        """The :class:`VelocityResult` of the pair starting at `frame_a`.

        By default, the pair that ends at the frame the window shows. Ready
        for ``stplot.plot_velocity(work[a], work[b], result)``.
        """
        df, results = self.velocity()
        ok = df["distance_px"].notna()
        pairs = dict(zip(zip(df.loc[ok, "frame_a"], df.loc[ok, "frame_b"]), results))
        for (a, b), result in pairs.items():
            if (frame_a is None and b == self.frame) or (frame_a is not None and a == frame_a):
                return result
        where = f"starting at frame {frame_a}" if frame_a is not None \
            else f"ending at frame {self.frame}"
        raise KeyError(f"no measured pair {where}; pairs: {sorted(pairs)}")

    def velocity_unit(self) -> Tuple[str, float, str]:
        """(table column, scale, unit) the window reports velocities in."""
        if self.pixel_size_um:
            return "velocity_m_per_s", 1e-6, "mm/ns"
        return "velocity_px_per_ns", 1.0, "px/ns"

    def series_summary(self) -> Optional[dict]:
        """Mean frame-to-frame velocity, and the slope of position against time.

        The slope uses every located head: axial position against the
        end-of-exposure time ``delay + gate_ns``, as in Nijdam (2011, fig. 3.18).
        """
        df, results = self.velocity()
        ok = df["distance_px"].notna()
        if not ok.any():
            return None
        v = df.loc[ok, "velocity_px_per_ns"].astype(float).to_numpy()
        n = len(v)
        std = float(np.std(v, ddof=1)) if n > 1 else float("nan")

        heads = self._work_heads()
        frames = sorted(heads)
        axis = results[0].axis
        t_end = self.times[frames] + float(self.gate_ns)
        axial = np.array([heads[k].xy @ axis for k in frames])
        sigma = np.array([heads[k].sigma_px for k in frames])
        fit, fit_sigma = _line_fit(t_end, axial, sigma)

        px = self.pixel_size_um
        scale = None if not px else px * 1e-6 / 1e-9                # px/ns -> m/s
        metric = (lambda value: None if scale is None else float(value) * scale)
        return {
            "n_heads": len(frames), "n_detected": sum(heads[k].method != "manual"
                                                      for k in frames),
            "n_pairs": n, "first_frame": frames[0], "last_frame": frames[-1],
            "mean_velocity_px_per_ns": float(v.mean()),
            "std_velocity_px_per_ns": std,
            "sem_velocity_px_per_ns": std / np.sqrt(n) if n > 1 else float("nan"),
            "fit_velocity_px_per_ns": fit, "fit_sigma_px_per_ns": fit_sigma,
            "pixel_size_um": px,
            "mean_velocity_m_per_s": metric(v.mean()),
            "std_velocity_m_per_s": metric(std),
            "sem_velocity_m_per_s": metric(std / np.sqrt(n)) if n > 1 else None,
            "fit_velocity_m_per_s": metric(fit), "fit_sigma_m_per_s": metric(fit_sigma),
            "delay_ns": self.delay_ns, "gate_ns": self.gate_ns,
            "bridge_gaps": self.bridge_gaps,
        }

    def describe_saved(self, row: dict) -> str:
        def fmt(value, sigma):
            if value is None or not np.isfinite(float(value)):
                return "n/a"
            text = f"{float(value) * scale:.4g}"
            if sigma is not None and np.isfinite(float(sigma)):
                text += f" +- {float(sigma) * scale:.2g}"
            return text
        metric = row.get("mean_velocity_m_per_s") is not None \
            and np.isfinite(float(row.get("mean_velocity_m_per_s") or np.nan))
        scale, unit = (1e-6, "mm/ns") if metric else (1.0, "px/ns")
        key = "m_per_s" if metric else "px_per_ns"
        fit_sigma = row.get("fit_sigma_m_per_s" if metric else "fit_sigma_px_per_ns")
        return (f"saved v = {fmt(row.get(f'mean_velocity_{key}'), row.get(f'std_velocity_{key}'))}"
                f" {unit} (mean of {row.get('n_pairs')} pairs);  fit "
                f"{fmt(row.get(f'fit_velocity_{key}'), fit_sigma)} {unit}")

    def readout(self, table=None) -> str:
        """One line about the frame shown: time, head, velocity from the previous head."""
        df = self.velocity_table() if table is None else table
        k = self.frame
        try:
            parts = [f"frame {k}", f"delay {self.times[k]:g} ns"]
        except ValueError:
            parts = [f"frame {k}"]
        head = self.head(k)
        parts.append("no head" if head is None else f"head ({head[0]:.1f}, {head[1]:.1f})")
        col, scale, unit = self.velocity_unit()
        row = df[(df["frame_b"] == k) & df["distance_px"].notna()]
        if len(row):
            r = row.iloc[0]
            sigma = r.get("sigma_" + col)
            text = f"v({int(r['frame_a'])}->{k}) = {float(r[col]) * scale:.4g}"
            if sigma is not None and np.isfinite(float(sigma)):
                text += f" +- {float(sigma) * scale:.2g}"
            parts.append(f"{text} {unit}")
        return "   |   ".join(parts)


def open_velocity_window(series, prepare: Optional[dict] = None, show: bool = True,
                         **config) -> VelocityWorkbench:
    """Open the velocity window, or update the one already open.

    Put this at the end of the velocity window's configuration cell: the
    first run opens the window, later runs update it in place, and a closed
    window comes back.

    Parameters
    ----------
    series : path, list of paths, ImageStack or list of stacks
        The series to work through, in order.
    prepare : dict, optional
        How a file is loaded: keyword arguments of
        :func:`~streamertools.workbench.prepare_series` (``rot90``,
        ``dark_path``, ``dark_from_frames``, ``physical_norm``, ...).
    **config
        See :meth:`~streamertools.workbench.SeriesWorkbench.configure` and
        ``VelocityWorkbench.SETTINGS`` / ``STATE``.
    """
    return _open_window(VelocityWorkbench, series, prepare, show, config)


# names of the first V3 notebook
Workbench = VelocityWorkbench


def open_workbench(stack, show: bool = True, **config) -> VelocityWorkbench:
    """The V3 name of :func:`open_velocity_window` for one stack.

    V3's ``autosave=<dir>/<stem>_clicked_heads.csv`` is the file the new
    window writes into ``out_dir=<dir>``, so it maps straight across.
    """
    from pathlib import Path
    autosave = config.pop("autosave", None)
    if autosave is not None and config.get("out_dir") is None:
        config["out_dir"] = Path(autosave).parent
    return open_velocity_window(stack, show=show, **config)


# --------------------------------------------------------------------------
_WINDOW_CLASS = None


def _window_class():
    global _WINDOW_CLASS
    if _WINDOW_CLASS is None:
        _WINDOW_CLASS = _build_window_class()
    return _WINDOW_CLASS


def _build_window_class():
    from .plotting import plot_velocity_series

    Base = _base_window_class()
    Qt, W = Base.Qt, Base.QtWidgets

    class VelocityWindow(Base):
        """Heads, the frame-to-frame table and a live v(t) plot."""

        def _build_tool_extras(self, layout):
            self.chk_advance = W.QCheckBox("next frame after each head click")
            self.chk_advance.clicked.connect(
                self._slot(lambda: self.wb.set_auto_advance(self.chk_advance.isChecked())))
            layout.addWidget(self.chk_advance)

        def _build_extra(self, layout):
            box = W.QGroupBox("Heads and frame-to-frame velocity")
            v = W.QVBoxLayout(box)
            g = W.QGridLayout()
            buttons = (("Detect this frame", self._detect_one,
                        "place this frame's head automatically (d)"),
                       ("Detect from here", self._detect_rest,
                        "detect the head in every later frame that has none yet"),
                       ("Delete this head", lambda: self.wb.remove_head(),
                        "remove this frame's head (Delete, or right-click)"),
                       ("Clear all heads", self._clear_heads, "asks first"))
            for i, (text, fn, tip) in enumerate(buttons):
                g.addWidget(self.button(text, fn, tip), i // 2, i % 2)
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
            self.vfig = self.Figure(figsize=(3.4, 2.2), layout="constrained")
            self.vcanvas = self.FigureCanvas(self.vfig)
            self.vcanvas.setMinimumHeight(170)
            self.vax = self.vfig.add_subplot()
            vsplit = W.QSplitter(Qt.Vertical)           # drag to trade table for plot
            vsplit.addWidget(self.table)
            vsplit.addWidget(self.vcanvas)
            vsplit.setChildrenCollapsible(False)
            v.addWidget(vsplit, stretch=1)
            layout.addWidget(box, stretch=1)

        def _extra_keys(self):
            return [(("H",), lambda: self.wb.set_tool("head")),
                    (("D",), self._detect_one),
                    (("Delete", "Backspace"), lambda: self.wb.remove_head())]

        def _sync_widgets(self):
            super()._sync_widgets()
            self.chk_advance.setChecked(self.wb.auto_advance)

        # -- drawing -------------------------------------------------------
        def _refresh_extra(self):
            wb = self.wb
            self._draw_heads()
            table, _ = wb.velocity()
            self._draw_table(table)
            self._draw_velocity(table)
            parts = wb.readout(table).split("   |   ")
            self.lbl_readout.setText("   ".join(parts[:2]) + "\n"
                                     + "\n".join(["   ".join(parts[2:3])] + parts[3:]))
            self.vcanvas.draw_idle()

        def _draw_heads(self):
            wb, ax, add = self.wb, self.ax, self.add_overlay
            heads = wb._work_heads()
            if wb.show_track and len(heads) > 1:
                ks = sorted(heads)
                xs, ys = [heads[k].x for k in ks], [heads[k].y for k in ks]
                add(ax.plot(xs, ys, "-", color="white", lw=0.8, alpha=0.5))
                add(ax.plot(xs, ys, ".", color="white", ms=4, alpha=0.8))
            prev = wb._previous_head(wb.frame)
            cur = heads.get(wb.frame)
            if prev is not None:
                p = heads[prev]
                add(ax.plot(p.x, p.y, "o", mfc="none", mec="red", ms=13, mew=1.5))
                add(ax.annotate(f"{prev}", (p.x, p.y), textcoords="offset points",
                                xytext=(9, -12), color="red", fontsize=8))
            if cur is not None:
                add(ax.plot(cur.x, cur.y, "+", color="cyan", ms=20, mew=2))
                if prev is not None:
                    add(ax.annotate("", xy=(cur.x, cur.y), xytext=(p.x, p.y),
                                    arrowprops=dict(arrowstyle="->", color="cyan", lw=1.4)))

        def _draw_table(self, table):
            wb = self.wb
            col, scale, unit = wb.velocity_unit()
            ok = table["distance_px"].notna()
            v_by_end = {int(fb): (float(v) * scale, s)
                        for fb, v, s in zip(table.loc[ok, "frame_b"], table.loc[ok, col],
                                            table.loc[ok, "sigma_" + col])}
            heads = wb.heads
            self.table.setHorizontalHeaderLabels(
                ["frame", "delay (ns)", "x", "y", f"v from prev ({unit})"])
            self.table.setRowCount(len(heads))
            selected = None
            for r, rec in enumerate(heads.to_dict("records")):
                k = int(rec["frame"])
                v = v_by_end.get(k)
                vtext = ""
                if v is not None:
                    vtext = f"{v[0]:.4g}"
                    if v[1] is not None and np.isfinite(float(v[1])):
                        vtext += f" +- {float(v[1]) * scale:.2g}"
                texts = (str(k), f"{rec['time_ns']:g}", f"{rec['x']:.1f}",
                         f"{rec['y']:.1f}", vtext)
                for c, text in enumerate(texts):
                    item = W.QTableWidgetItem(text)
                    if rec["method"] != "manual":
                        item.setToolTip(f"detected ({rec['method']}); click to correct")
                        f = item.font()
                        f.setItalic(True)
                        item.setFont(f)
                    self.table.setItem(r, c, item)
                if k == wb.frame:
                    selected = r
            self.table.resizeColumnsToContents()
            self.table.clearSelection()
            if selected is not None:
                self.table.selectRow(selected)
                self.table.scrollToItem(self.table.item(selected, 0))

        def _draw_velocity(self, table):
            wb = self.wb
            self.vax.clear()
            plot_velocity_series(table, ax=self.vax, label="clicked")
            ok = table["distance_px"].notna() & (table["frame_b"] == wb.frame)
            if ok.any():
                r = table[ok].iloc[0]
                self.vax.axvline(0.5 * (r["time_a_ns"] + r["time_b_ns"]),
                                 color="cyan", lw=1.0, ls="--")
            small_font_axes(self.vax)

        # -- buttons -------------------------------------------------------
        def _on_table_click(self, row, column):
            item = self.table.item(row, 0)
            if item is not None:
                self.wb.goto(int(item.text()))

        def _detect_one(self):
            try:
                self.wb.detect_head()
            except ValueError as exc:
                self.wb._report(f"frame {self.wb.frame}: {exc}")

        def _detect_rest(self):
            wb = self.wb
            with self._busy("detecting heads ..."):
                failed = wb.detect_heads(range(wb.frame, wb.n_frames))
            if failed:
                wb._report(f"no head found in frame(s) {sorted(failed)}: "
                           f"{next(iter(failed.values()))}")

        def _clear_heads(self):
            wb = self.wb
            if not wb._heads:
                return
            where = (f"\nThe file {wb.results_path().name} is emptied too."
                     if wb.results_path() is not None else "")
            answer = W.QMessageBox.question(
                self, "Clear all heads",
                f"Remove all {len(wb._heads)} heads of {wb.series_name}?{where}")
            if answer == W.QMessageBox.Yes:
                wb.clear_heads()

    return VelocityWindow
