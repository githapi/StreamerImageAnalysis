"""
Tile maker  --  Streamertools manual, chapter 6
================================================

Combines a number of images into one compound matrix image (figure 6.1), or
into a movie, with borders, per-image comments, dimension arrows, an overlay
and comments outside the matrix.

Two ways to configure it:

* **Python** -- build a :class:`TileMakerConfig` (or just pass keyword
  arguments to :func:`make_tiles`).  This is the primary interface.
* **``*.tlmkr``** -- :func:`read_tlmkr` parses the original ini-style script
  files described in manual 6.2, so existing scripts keep working, and
  :func:`write_tlmkr` writes one back out.

Naming note: following the pairing in manual 6.2.2 (*"Extra horizontal border
per X rows"*), the **horizontal border** is the stripe *between rows* and the
**vertical border** the stripe *between columns*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .annotate import (ArrowSpec, TextStyle, apply_overlay, draw_arrow,
                       draw_text_block, expand_placeholders, parse_colour)
from .geometry import crop as crop_image
from .geometry import manipulate, zoom_out
from .io import load, save_image
from .palette import DRAWING_STYLE_GAMMA, Palette, counts_to_unit, resolve_limits

__all__ = [
    "OriginalSettings", "OutputSettings", "OutsideComments", "OverlaySettings",
    "MovieSettings", "FileEntry", "TileMakerConfig", "TileMaker",
    "make_tiles", "read_tlmkr", "write_tlmkr",
]


# --------------------------------------------------------------------------
# configuration dataclasses, one per *.tlmkr section
# --------------------------------------------------------------------------
@dataclass
class OriginalSettings:
    """``[Original]`` -- how the input images are read and scaled (6.2.2)."""

    use_gray_values: bool = True     # 1 = count based, 0 = keep the RGB colours
    manipulate_mode: int = 0         # 1-3 rotate cw 90/180/270; 4-7 mirror first
    crop_left: int = 0
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0

    # Imax (only relevant for use_gray_values = True)
    auto_max: int = 1                # 0 MaxVal, 1 per-image max, 2 MaxValPerPic, 3 histogram
    max_val: float = 4095.0
    auto_max_percentage: float = 100.0
    max_hist_percentage: float = 99.0
    # Imin
    auto_min: int = 1                # 0 MinVal, 1 per-image min, 3 histogram
    min_val: float = 0.0
    min_hist_percentage: float = 30.0

    gamma: float = 1.0               # 6.2.2 / 2.2.2
    drawing_style: Optional[int] = None   # deprecated; overrides gamma when set
    show_min_max: bool = False

    def effective_gamma(self) -> float:
        if self.drawing_style is not None:
            return DRAWING_STYLE_GAMMA.get(int(self.drawing_style), 1.0)
        return float(self.gamma)

    def limits(self, counts: np.ndarray, max_val_per_pic: Optional[float] = None
               ) -> Tuple[float, float]:
        """Imin / Imax for one image, following AutoMin / AutoMax."""
        if self.auto_max == 0:
            i_max = float(self.max_val)
        elif self.auto_max == 1:
            i_max = float(np.nanmax(counts)) * float(self.auto_max_percentage) / 100.0
        elif self.auto_max == 2:
            if max_val_per_pic is None:
                raise ValueError("AutoMax=2 needs a [MaxValPerPic] entry for every file")
            i_max = float(max_val_per_pic)
        elif self.auto_max == 3:
            i_max = float(np.nanpercentile(counts, self.max_hist_percentage))
        else:
            raise ValueError(f"AutoMax must be 0..3, got {self.auto_max}")

        if self.auto_min == 0:
            i_min = float(self.min_val)
        elif self.auto_min == 1:
            i_min = float(np.nanmin(counts))
        elif self.auto_min == 3:
            i_min = float(np.nanpercentile(counts, self.min_hist_percentage))
        else:
            raise ValueError(f"AutoMin must be 0, 1 or 3, got {self.auto_min}")
        return i_min, i_max


@dataclass
class OutputSettings:
    """``[Output]`` -- layout and lettering of the compound image (6.2.2)."""

    rows: int = 1
    columns: int = 1
    zoom_factor: int = 1             # integer zoom-out, manual 2.3

    horizontal_border: int = 4       # stripe between rows
    vertical_border: int = 4         # stripe between columns
    extra_horizontal_border_per_x_rows: int = 0
    extra_vertical_border_per_x_columns: int = 0
    extra_horizontal_border: int = 0
    extra_vertical_border: int = 0

    comment_type: int = 0            # 0 none, 1 one unified comment, 2 per image
    unified_comment: str = ""
    text_from_border: int = 10
    text_font: Optional[str] = None
    text_size: int = 20
    text_shadow: bool = True
    arrow_shadow: bool = True
    shadow_offset: int = 2
    text_transparent_bg: bool = True

    background_color: Union[str, int] = "#000000"
    text_color: Union[str, int] = "#FFFFFF"
    text_shadow_color: Union[str, int] = "#000000"
    text_background_color: Union[str, int] = "#000000"

    palette: Union[str, Path, Palette, None] = None    # default: thermal
    stack_mode: int = 0              # 0 none, 1 RGB, 2-7 two-frame combinations
    stack_reversed: bool = False

    def text_style(self) -> TextStyle:
        return TextStyle(font=self.text_font, size=self.text_size,
                         colour=self.text_color, shadow=self.text_shadow,
                         shadow_colour=self.text_shadow_color,
                         shadow_offset=self.shadow_offset,
                         transparent_background=self.text_transparent_bg,
                         background_colour=self.text_background_color,
                         from_border=self.text_from_border)


@dataclass
class OutsideComments:
    """``[OutsideComments]`` -- text left of the rows / above the columns."""
    show_row_comments: bool = False
    show_column_comments: bool = False
    text_size: int = 20
    text_color: Union[str, int] = "#FFFFFF"


@dataclass
class OverlaySettings:
    """``[Overlay]`` (6.2.2)."""
    use_overlay: bool = False
    overlay_file: Union[str, Path, None] = None
    transparent_color: Union[str, int] = "#000000"
    on_all_images: bool = True
    overlay_row: Optional[int] = None
    overlay_col: Optional[int] = None


@dataclass
class MovieSettings:
    """``[Movie]`` (6.2.2).  AVI is replaced here by GIF / MP4."""
    record_movie: bool = False
    fps: float = 8.0
    row_column_layout: bool = False
    last_frame_double: bool = False
    filename: Optional[str] = None
    filetype: str = "gif"            # "gif" or "mp4"


@dataclass
class FileEntry:
    """One line of the ``[Filenames]`` section."""
    path: Union[str, Path]
    frame: Optional[int] = None          # ``file.rtv?4``
    frames: Optional[Tuple[int, int]] = None   # ``file.rtv?3:9``
    all_frames: bool = False             # ``file.rtv?*``
    group: int = 0                       # ``file.rtv|1``
    empty: bool = False                  # the literal "none=" placeholder


@dataclass
class TileMakerConfig:
    """The complete ``*.tlmkr`` content."""

    original: OriginalSettings = field(default_factory=OriginalSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    outside: OutsideComments = field(default_factory=OutsideComments)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    movie: MovieSettings = field(default_factory=MovieSettings)

    folders: List[Union[str, Path]] = field(default_factory=list)   # [Folders]
    files: List[FileEntry] = field(default_factory=list)            # [Filenames]
    comments: List[str] = field(default_factory=list)               # [Comments]
    max_val_per_pic: List[float] = field(default_factory=list)      # [MaxValPerPic]
    row_comments: List[str] = field(default_factory=list)
    column_comments: List[str] = field(default_factory=list)
    arrows: Dict[str, ArrowSpec] = field(default_factory=dict)      # [Arrow] etc.
    groups: Dict[int, Dict[str, object]] = field(default_factory=dict)  # [Group1] ...

    # -- group handling ----------------------------------------------------
    def settings_for(self, group: int) -> OriginalSettings:
        """[Original] with the [GroupN] overrides applied (and ParentGroup)."""
        chain, g = [], int(group)
        seen = set()
        while g and g in self.groups and g not in seen:
            seen.add(g)
            chain.append(self.groups[g])
            g = int(self.groups[g].get("parent_group", 0) or 0)
        settings = self.original
        for overrides in reversed(chain):
            kwargs = {k: v for k, v in overrides.items()
                      if k != "parent_group" and hasattr(settings, k)}
            settings = replace(settings, **kwargs)
        return settings

    def resolve(self, path: Union[str, Path]) -> Path:
        """Find a file, using the [Folders] base folders (manual 6.2.2)."""
        p = Path(path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p
        for folder in self.folders:
            cand = Path(folder) / p
            if cand.exists():
                return cand
            hits = sorted(Path(folder).rglob(p.name))
            if hits:
                return hits[0]
        return p


# --------------------------------------------------------------------------
# stacking colours (manual 6.2.2, StackMode)
# --------------------------------------------------------------------------
_STACK_COLOURS = {
    1: [(255, 0, 0), (0, 255, 0), (0, 0, 255)],       # red / green / blue
    2: [(255, 0, 0), (0, 255, 0)],                    # red / green
    3: [(255, 0, 0), (0, 0, 255)],                    # red / blue
    4: [(255, 0, 0), (0, 255, 255)],                  # red / cyan
    5: [(255, 0, 255), (0, 255, 0)],                  # purple / green
    6: [(0, 0, 255), (255, 255, 0)],                  # blue / yellow
    7: [(128, 128, 128), (255, 255, 255)],            # grey / white
}


# --------------------------------------------------------------------------
class TileMaker:
    """Build the compound image, or the movie, from a configuration."""

    def __init__(self, config: Optional[TileMakerConfig] = None):
        self.config = config or TileMakerConfig()
        self._cache: Dict[Tuple[str, bool], object] = {}

    # -- one input image ---------------------------------------------------
    def _load_cached(self, path, keep_colour: bool):
        """Read a file once per build.

        A multi-frame file used with ``?*`` becomes one tile per frame, and
        without this every tile would re-read the whole file -- 40 reads of a
        22 MB .sif instead of one.
        """
        key = (str(path), bool(keep_colour))
        if key not in self._cache:
            self._cache[key] = load(path, keep_colour=keep_colour)
        return self._cache[key]

    def _prepare_counts(self, entry: FileEntry, frame: int = 0
                        ) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
        """Read, manipulate, crop and zoom one input image."""
        cfg = self.config
        s = cfg.settings_for(entry.group)
        stack = self._load_cached(cfg.resolve(entry.path), keep_colour=not s.use_gray_values)
        counts = stack.data[min(frame, stack.n_frames - 1)]
        colour = None
        if stack.colour is not None:
            colour = stack.colour[min(frame, stack.n_frames - 1)]

        if s.manipulate_mode:
            counts = manipulate(counts, s.manipulate_mode)
            colour = manipulate(colour, s.manipulate_mode) if colour is not None else None
        if any((s.crop_left, s.crop_right, s.crop_top, s.crop_bottom)):
            counts = crop_image(counts, s.crop_left, s.crop_right, s.crop_top, s.crop_bottom)
            if colour is not None:
                colour = crop_image(colour, s.crop_left, s.crop_right,
                                    s.crop_top, s.crop_bottom)
        z = int(cfg.output.zoom_factor)
        if z > 1:
            counts = zoom_out(counts, z)
            if colour is not None:
                colour = np.stack([zoom_out(colour[..., c], z) for c in range(3)], -1)
                colour = np.clip(colour, 0, 255).astype(np.uint8)
        return counts, colour, s

    def _render_tile(self, entry: FileEntry, frame: int = 0,
                     max_val_per_pic: Optional[float] = None) -> np.ndarray:
        """One input image as RGB, scaled per AutoMin/AutoMax and gamma."""
        counts, colour, s = self._prepare_counts(entry, frame)
        if not s.use_gray_values and colour is not None:
            return colour
        i_min, i_max = s.limits(counts, max_val_per_pic)
        unit = counts_to_unit(counts, i_min, i_max, s.effective_gamma())
        pal = self.config.output.palette
        pal = pal if isinstance(pal, Palette) else Palette.load(pal)
        rgb = pal.apply(unit)
        if s.show_min_max:
            rgb = draw_text_block(rgb, f"min {i_min:g}  max {i_max:g}", "top-right",
                                  self.config.output.text_style())
        return rgb

    # -- tiles -------------------------------------------------------------
    def _expand_entries(self) -> List[Tuple[FileEntry, int]]:
        """Turn the [Filenames] list into concrete (entry, frame) pairs."""
        out: List[Tuple[FileEntry, int]] = []
        for e in self.config.files:
            if e.empty:
                out.append((e, 0))
            elif e.all_frames or e.frames:
                stack = self._load_cached(self.config.resolve(e.path), False)
                lo, hi = e.frames if e.frames else (0, stack.n_frames - 1)
                out.extend((e, k) for k in range(lo, min(hi, stack.n_frames - 1) + 1))
            else:
                out.append((e, e.frame or 0))
        return out

    def _tiles(self) -> List[Optional[np.ndarray]]:
        """Render every entry, applying the stack mode if one is set."""
        cfg = self.config
        pairs = self._expand_entries()
        rendered: List[Optional[np.ndarray]] = []
        for i, (entry, frame) in enumerate(pairs):
            if entry.empty:
                rendered.append(None)
                continue
            mv = cfg.max_val_per_pic[i] if i < len(cfg.max_val_per_pic) else None
            rendered.append(self._render_tile(entry, frame, mv))

        mode = int(cfg.output.stack_mode)
        if mode == 0:
            return rendered
        colours = list(_STACK_COLOURS[mode])
        if cfg.output.stack_reversed:
            colours = colours[::-1]
        n = len(colours)
        stacked: List[Optional[np.ndarray]] = []
        for start in range(0, len(rendered), n):
            group = [g for g in rendered[start:start + n] if g is not None]
            if not group:
                stacked.append(None)
                continue
            shape = group[0].shape
            acc = np.zeros(shape, dtype=np.float32)
            for img, col in zip(group, colours):
                lum = np.asarray(img, dtype=np.float32).mean(axis=2, keepdims=True) / 255.0
                acc += lum * np.asarray(col, dtype=np.float32)
            stacked.append(np.clip(acc, 0, 255).astype(np.uint8))
        return stacked

    # -- comments ----------------------------------------------------------
    def _comment_for(self, index: int, entry: FileEntry, frame: int,
                     tile: np.ndarray) -> str:
        cfg = self.config
        if cfg.output.comment_type == 1:
            raw = cfg.output.unified_comment
        elif cfg.output.comment_type == 2:
            raw = cfg.comments[index] if index < len(cfg.comments) else ""
        else:
            return ""
        if raw.strip().lower() == "nocomment":
            return ""
        return raw

    def _draw_comment(self, tile: np.ndarray, raw: str, entry: FileEntry,
                      frame: int) -> np.ndarray:
        """Draw a comment; ``/b`` splits it into a top-left and a bottom-left part."""
        if not raw:
            return tile
        style = self.config.output.text_style()
        name = str(entry.path)
        top, _, bottom = raw.partition("/b")
        if not _:
            top, bottom = "", raw
        for text, pos in ((top, "top-left"), (bottom, "bottom-left")):
            text = expand_placeholders(text, filename=name, frame=frame,
                                       max_count=float(np.asarray(tile).max()))
            if text.strip():
                tile = draw_text_block(tile, text, pos, style)
        return tile

    # -- assembly ----------------------------------------------------------
    def build(self) -> np.ndarray:
        """Render the compound matrix image (manual 6.1)."""
        cfg = self.config
        out = cfg.output
        tiles = self._tiles()
        pairs = self._expand_entries()

        # comments, arrows and overlay, per tile
        prepared: List[Optional[np.ndarray]] = []
        for i, tile in enumerate(tiles):
            if tile is None:
                prepared.append(None)
                continue
            entry, frame = pairs[i] if i < len(pairs) else (FileEntry(""), 0)
            tile = self._draw_comment(tile, self._comment_for(i, entry, frame, tile),
                                      entry, frame)
            r, c = divmod(i, max(out.columns, 1))
            for spec in cfg.arrows.values():
                if not spec.show:
                    continue
                default_pos = spec.row is None and spec.col is None
                here = (spec.row == r and spec.col == c) or (default_pos and i == out.columns - 1)
                if here:
                    style = out.text_style()
                    style.shadow = out.arrow_shadow
                    tile = draw_arrow(tile, spec, style)
            ov = cfg.overlay
            if ov.use_overlay and ov.overlay_file is not None:
                if ov.on_all_images or (ov.overlay_row == r and ov.overlay_col == c) \
                        or (ov.overlay_row is None and ov.overlay_col is None
                            and i == out.columns - 1):
                    tile = apply_overlay(tile, cfg.resolve(ov.overlay_file),
                                         ov.transparent_color)
            prepared.append(tile)

        shapes = {t.shape[:2] for t in prepared if t is not None}
        if not shapes:
            raise ValueError("no input images")
        if len(shapes) > 1:
            raise ValueError("all images must have the same resolution after "
                             f"cropping and zooming; found {shapes}")
        th, tw = shapes.pop()

        rows, cols = int(out.rows), int(out.columns)
        if rows * cols < len([t for t in prepared]):
            print(f"WARNING: {len(prepared)} images do not fit in "
                  f"{rows} x {cols} tiles; the surplus is dropped")

        # borders
        gap_r, gap_c = int(out.horizontal_border), int(out.vertical_border)
        extra_r_every = int(out.extra_horizontal_border_per_x_rows or 0)
        extra_c_every = int(out.extra_vertical_border_per_x_columns or 0)
        extra_r, extra_c = int(out.extra_horizontal_border), int(out.extra_vertical_border)

        def _offsets(n, size, gap, every, extra):
            offs, pos = [], 0
            for k in range(n):
                offs.append(pos)
                pos += size + gap
                if every and (k + 1) % every == 0 and k != n - 1:
                    pos += extra
            total = pos - gap if n else 0
            return offs, total

        y_off, grid_h = _offsets(rows, th, gap_r, extra_r_every, extra_r)
        x_off, grid_w = _offsets(cols, tw, gap_c, extra_c_every, extra_c)

        # room for the outside comments
        oc = cfg.outside
        left_pad = top_pad = 0
        oc_style = TextStyle(font=out.text_font, size=oc.text_size, colour=oc.text_color,
                             shadow=out.text_shadow, shadow_colour=out.text_shadow_color,
                             shadow_offset=out.shadow_offset,
                             transparent_background=True,
                             background_colour=out.background_color,
                             from_border=out.text_from_border)
        right_pad = 0
        if (oc.show_row_comments and cfg.row_comments) or \
                (oc.show_column_comments and cfg.column_comments):
            from PIL import Image, ImageDraw
            probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
            font = oc_style.pil_font()

            def _size(text):
                b = probe.multiline_textbbox((0, 0), text.replace("/n", "\n"), font=font)
                return b[2] - b[0], b[3] - b[1]

            if oc.show_row_comments and cfg.row_comments:
                left_pad = max(_size(t)[0] for t in cfg.row_comments) \
                    + 2 * out.text_from_border
            if oc.show_column_comments and cfg.column_comments:
                top_pad = max(_size(t)[1] for t in cfg.column_comments) \
                    + 2 * out.text_from_border
                # centred column texts may stick out beyond the outer tiles
                over_l = over_r = 0
                for c, text in enumerate(cfg.column_comments[:cols]):
                    half = _size(text)[0] / 2.0
                    cx = x_off[c] + tw / 2.0
                    over_l = max(over_l, int(np.ceil(half - cx)))
                    over_r = max(over_r, int(np.ceil(cx + half - grid_w)))
                left_pad += max(over_l, 0)
                right_pad = max(over_r, 0)

        canvas = np.zeros((grid_h + top_pad, grid_w + left_pad + right_pad, 3),
                          dtype=np.uint8)
        canvas[:] = parse_colour(out.background_color, (0, 0, 0))

        for i, tile in enumerate(prepared):
            r, c = divmod(i, cols)
            if r >= rows or tile is None:
                continue
            y, x = y_off[r] + top_pad, x_off[c] + left_pad
            canvas[y:y + th, x:x + tw] = tile

        # outside comments (manual [OutsideComments] / [RowComments] / [ColumnComments])
        if oc.show_column_comments and cfg.column_comments:
            from PIL import Image, ImageDraw
            im = Image.fromarray(canvas)
            draw = ImageDraw.Draw(im)
            font = oc_style.pil_font()
            for c, text in enumerate(cfg.column_comments[:cols]):
                cx = left_pad + x_off[c] + tw // 2
                draw.text((cx, top_pad - out.text_from_border),
                          text.replace("/n", "\n"), font=font,
                          fill=parse_colour(oc.text_color), anchor="ms", align="center")
            canvas = np.asarray(im, dtype=np.uint8)
        if oc.show_row_comments and cfg.row_comments:
            from PIL import Image, ImageDraw
            im = Image.fromarray(canvas)
            draw = ImageDraw.Draw(im)
            font = oc_style.pil_font()
            for r, text in enumerate(cfg.row_comments[:rows]):
                cy = top_pad + y_off[r] + th // 2
                draw.text((left_pad - out.text_from_border, cy),
                          text.replace("/n", "\n"), font=font,
                          fill=parse_colour(oc.text_color), anchor="rm", align="right")
            canvas = np.asarray(im, dtype=np.uint8)
        return canvas

    # -- movie -------------------------------------------------------------
    def build_movie_frames(self) -> List[np.ndarray]:
        """Every input image as one movie frame (manual [Movie]).

        With ``row_column_layout = True`` the row/column settings are honoured
        and each movie frame is a small matrix, which is how you animate data
        from several source files at once.
        """
        cfg = self.config
        if not cfg.movie.row_column_layout:
            frames = [t for t in self._tiles() if t is not None]
        else:
            per = max(cfg.output.rows * cfg.output.columns, 1)
            all_files = list(cfg.files)
            frames = []
            for start in range(0, len(all_files), per):
                sub = TileMakerConfig(**{**cfg.__dict__, "files": all_files[start:start + per]})
                frames.append(TileMaker(sub).build())
        if cfg.movie.last_frame_double and frames:
            frames.append(frames[-1])
        return frames

    def save_movie(self, path, frames: Optional[List[np.ndarray]] = None) -> Path:
        """Write a GIF (always available) or an MP4 (needs ``imageio-ffmpeg``).

        The manual's AVI export exists because of Delphi/Windows codecs; a GIF
        or MP4 is the sane modern equivalent and needs no codec install.
        """
        from PIL import Image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = frames if frames is not None else self.build_movie_frames()
        if not frames:
            raise ValueError("no frames to write")
        fps = float(self.config.movie.fps) or 8.0
        if path.suffix.lower() == ".gif":
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(path, save_all=True, append_images=imgs[1:],
                         duration=int(1000 / fps), loop=0)
        else:
            import imageio.v2 as imageio
            imageio.mimwrite(path, frames, fps=fps)
        return path

    # -- convenience -------------------------------------------------------
    def save(self, path) -> Path:
        cfg = self.config
        if cfg.movie.record_movie:
            return self.save_movie(path)
        return save_image(path, self.build())


# --------------------------------------------------------------------------
def make_tiles(files: Sequence[Union[str, Path, FileEntry]], rows: int, columns: int,
               output: Optional[Union[str, Path]] = None, **kwargs) -> np.ndarray:
    """Quick entry point: a matrix from a list of files.

    Any :class:`OriginalSettings` or :class:`OutputSettings` field can be
    passed as a keyword; ``comments``, ``row_comments`` and ``column_comments``
    are taken as lists.

    >>> make_tiles(paths, rows=3, columns=4, gamma=1.5, comment_type=2,
    ...            comments=["13 kV", "16 kV", ...], output="figure.png")
    """
    orig_fields = OriginalSettings.__dataclass_fields__
    out_fields = OutputSettings.__dataclass_fields__
    orig = OriginalSettings(**{k: v for k, v in kwargs.items() if k in orig_fields})
    out = OutputSettings(rows=rows, columns=columns,
                         **{k: v for k, v in kwargs.items() if k in out_fields
                            and k not in ("rows", "columns")})
    cfg = TileMakerConfig(
        original=orig, output=out,
        files=[f if isinstance(f, FileEntry) else FileEntry(f) for f in files],
        comments=list(kwargs.get("comments", [])),
        row_comments=list(kwargs.get("row_comments", [])),
        column_comments=list(kwargs.get("column_comments", [])),
        max_val_per_pic=list(kwargs.get("max_val_per_pic", [])),
        arrows=dict(kwargs.get("arrows", {})),
        folders=list(kwargs.get("folders", [])),
        outside=kwargs.get("outside", OutsideComments(
            show_row_comments=bool(kwargs.get("row_comments")),
            show_column_comments=bool(kwargs.get("column_comments")))),
        overlay=kwargs.get("overlay", OverlaySettings()),
        movie=kwargs.get("movie", MovieSettings()),
    )
    tm = TileMaker(cfg)
    image = tm.build()
    if output:
        save_image(output, image)
    return image


# --------------------------------------------------------------------------
# *.tlmkr  (manual 6.2)
# --------------------------------------------------------------------------
_KEY_MAP = {
    # [Original]
    "usegrayvalues": ("original", "use_gray_values", bool),
    "manipulatemode": ("original", "manipulate_mode", int),
    "cropleft": ("original", "crop_left", int),
    "cropright": ("original", "crop_right", int),
    "croptop": ("original", "crop_top", int),
    "cropbottom": ("original", "crop_bottom", int),
    "automax": ("original", "auto_max", int),
    "maxval": ("original", "max_val", float),
    "automaxpercentage": ("original", "auto_max_percentage", float),
    "maxhistpercentage": ("original", "max_hist_percentage", float),
    "automin": ("original", "auto_min", int),
    "minval": ("original", "min_val", float),
    "minhistpercentage": ("original", "min_hist_percentage", float),
    "gamma": ("original", "gamma", float),
    "drawingstyle": ("original", "drawing_style", int),
    "showminmax": ("original", "show_min_max", bool),
    "basefolder": ("original", "_base_folder", str),
    # [Output]
    "rows": ("output", "rows", int),
    "columns": ("output", "columns", int),
    "zoomfactor": ("output", "zoom_factor", int),
    "horizontalborder": ("output", "horizontal_border", int),
    "verticalborder": ("output", "vertical_border", int),
    "extrahorizontalborderperxrows": ("output", "extra_horizontal_border_per_x_rows", int),
    "extraverticalborderperxcolums": ("output", "extra_vertical_border_per_x_columns", int),
    "extraverticalborderperxcolumns": ("output", "extra_vertical_border_per_x_columns", int),
    "extrahorizontalborder": ("output", "extra_horizontal_border", int),
    "extraverticalborder": ("output", "extra_vertical_border", int),
    "commenttype": ("output", "comment_type", int),
    "usecomments": ("output", "comment_type", int),
    "unifiedcomment": ("output", "unified_comment", str),
    "textfromborder": ("output", "text_from_border", int),
    "textfont": ("output", "text_font", str),
    "textsize": ("output", "text_size", int),
    "textshadow": ("output", "text_shadow", bool),
    "arrowshadow": ("output", "arrow_shadow", bool),
    "shadowoffset": ("output", "shadow_offset", int),
    "texttransparentbg": ("output", "text_transparent_bg", bool),
    "backgroundcolor": ("output", "background_color", str),
    "textcolor": ("output", "text_color", str),
    "textshadowcolour": ("output", "text_shadow_color", str),
    "textshadowcolor": ("output", "text_shadow_color", str),
    "textbackgroundcolor": ("output", "text_background_color", str),
    "palette": ("output", "palette", str),
    "stackmode": ("output", "stack_mode", int),
    "stackreversed": ("output", "stack_reversed", bool),
    # [OutsideComments]
    "showrowcomments": ("outside", "show_row_comments", bool),
    "showcolumncomments": ("outside", "show_column_comments", bool),
    # [Overlay]
    "useoverlay": ("overlay", "use_overlay", bool),
    "overlayfile": ("overlay", "overlay_file", str),
    "transparentcolor": ("overlay", "transparent_color", str),
    "onallimages": ("overlay", "on_all_images", bool),
    "overlayrow": ("overlay", "overlay_row", int),
    "overlaycol": ("overlay", "overlay_col", int),
    # [Movie]
    "recordmovie": ("movie", "record_movie", bool),
    "fps": ("movie", "fps", float),
    "rowcolumnlayout": ("movie", "row_column_layout", bool),
    "lastframedouble": ("movie", "last_frame_double", bool),
    "filename": ("movie", "filename", str),
    "filetype": ("movie", "filetype", str),
}

_ARROW_KEYS = {
    "showarrow": ("show", bool), "showarrowtopright": ("show", bool),
    "showendbars": ("end_bars", bool),
    "arrowtop": ("start", int), "arrowleft": ("start", int),
    "arrowlength": ("length", int), "arrowfromside": ("from_side", int),
    "arrowtext": ("text", str), "toptext": ("top_text", str),
    "bottomtext": ("bottom_text", str),
    "arrowrow": ("row", int), "arrowcol": ("col", int),
}

_LIST_SECTIONS = {"filenames", "comments", "maxvalperpic", "rowcomments", "columncomments"}
_FILE_RE = re.compile(r"^(?P<path>.*?)(?:\?(?P<frame>\*|\d+(?::\d+)?))?(?:\|(?P<group>\d+))?$")


def _cast(value: str, kind):
    v = value.strip()
    if kind is bool:
        return v not in ("0", "", "false", "False")
    if kind is int:
        return int(float(v))
    if kind is float:
        return float(v)
    return v


def _parse_file_line(raw: str) -> FileEntry:
    text = raw.strip()
    if text.lower() == "none":
        return FileEntry("", empty=True)
    m = _FILE_RE.match(text)
    path = m.group("path")
    frame_spec, group = m.group("frame"), int(m.group("group") or 0)
    if frame_spec == "*":
        return FileEntry(path, all_frames=True, group=group)
    if frame_spec and ":" in frame_spec:
        a, b = frame_spec.split(":")
        return FileEntry(path, frames=(int(a), int(b)), group=group)
    return FileEntry(path, frame=int(frame_spec) if frame_spec else None, group=group)


def read_tlmkr(path) -> TileMakerConfig:
    """Parse a Tile maker ``*.tlmkr`` script (manual 6.2).

    Unknown keys are ignored with a warning, missing keys keep their default,
    and lines starting with ``;`` are comments.
    """
    path = Path(path)
    cfg = TileMakerConfig()
    section = ""
    base_folder: Optional[str] = None
    unknown: List[str] = []

    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        if line.strip().startswith("[") and line.strip().endswith("]"):
            section = line.strip()[1:-1].strip().lower()
            continue
        if section in _LIST_SECTIONS:
            entry = line.rstrip()
            entry = entry[:-1] if entry.endswith("=") else entry.split("=", 1)[0]
            if section == "filenames":
                cfg.files.append(_parse_file_line(entry))
            elif section == "comments":
                cfg.comments.append(entry)
            elif section == "maxvalperpic":
                cfg.max_val_per_pic.append(float(entry or 0))
            elif section == "rowcomments":
                cfg.row_comments.append(entry)
            else:
                cfg.column_comments.append(entry)
            continue
        if section == "folders":
            cfg.folders.append(line.rstrip().rstrip("="))
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = key.strip().lower().replace(" ", "")

        if section == "outsidecomments":
            # 'Text Size' / 'Text Color' also exist in [Output]; here they
            # belong to the outside comments (manual 6.2.2)
            if k == "textsize":
                cfg.outside.text_size = int(float(value))
                continue
            if k in ("textcolor", "textcolour"):
                cfg.outside.text_color = value.strip()
                continue

        if section.startswith("arrow") or section.endswith("arrow"):
            spec = cfg.arrows.setdefault(section, ArrowSpec(show=False))
            spec.horizontal = "hor" in section
            if k in _ARROW_KEYS:
                attr, kind = _ARROW_KEYS[k]
                setattr(spec, attr, _cast(value, kind))
            continue

        if section.startswith("group"):
            n = int(re.sub(r"\D", "", section) or 0)
            grp = cfg.groups.setdefault(n, {})
            if k == "parentgroup":
                grp["parent_group"] = int(float(value))
            elif k in _KEY_MAP and _KEY_MAP[k][0] == "original":
                _, attr, kind = _KEY_MAP[k]
                grp[attr] = _cast(value, kind)
            continue

        if k in _KEY_MAP:
            target, attr, kind = _KEY_MAP[k]
            if attr == "_base_folder":
                base_folder = value.strip()
                continue
            setattr(getattr(cfg, target), attr, _cast(value, kind))
        else:
            unknown.append(key.strip())

    if base_folder:
        cfg.folders.insert(0, base_folder)
    if cfg.output.palette:
        cfg.output.palette = str(cfg.output.palette).strip().strip("'\"")
    if unknown:
        print("tlmkr: ignored unknown keys:", ", ".join(sorted(set(unknown))))
    return cfg


def write_tlmkr(path, cfg: TileMakerConfig) -> Path:
    """Write a configuration back out as a ``*.tlmkr`` script."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rev = {}
    for k, (target, attr, kind) in _KEY_MAP.items():
        rev.setdefault((target, attr), k)

    lines: List[str] = []
    for section, obj, title in (("original", cfg.original, "[Original]"),
                                ("output", cfg.output, "[Output]"),
                                ("outside", cfg.outside, "[OutsideComments]"),
                                ("overlay", cfg.overlay, "[Overlay]"),
                                ("movie", cfg.movie, "[Movie]")):
        lines.append(title)
        for attr, value in obj.__dict__.items():
            key = rev.get((section, attr))
            if key is None or value is None:
                continue
            if isinstance(value, bool):
                value = int(value)
            lines.append(f"{key}={value}")
        if section == "outside":
            lines.append(f"TextSize={cfg.outside.text_size}")
            lines.append(f"TextColor={cfg.outside.text_color}")
        lines.append("")

    for name, spec in cfg.arrows.items():
        title = {"arrow": "[Arrow]", "secondarrow": "[SecondArrow]",
                 "horarrow": "[HorArrow]", "secondhorarrow": "[SecondHorArrow]"}.get(
                     name.lower(), f"[{name}]")
        start_key = "ArrowLeft" if spec.horizontal else "ArrowTop"
        lines += [title,
                  f"ShowArrow={int(spec.show)}",
                  f"ShowEndBars={int(spec.end_bars)}",
                  f"{start_key}={spec.start}",
                  f"ArrowLength={spec.length}"]
        if spec.from_side is not None:
            lines.append(f"ArrowFromSide={spec.from_side}")
        for key, value in (("ArrowText", spec.text), ("TopText", spec.top_text),
                           ("BottomText", spec.bottom_text)):
            if value:
                lines.append(f"{key}={value}")
        if spec.row is not None:
            lines.append(f"ArrowRow={spec.row}")
        if spec.col is not None:
            lines.append(f"ArrowCol={spec.col}")
        lines.append("")

    if cfg.folders:
        lines += ["[Folders]"] + [f"{f}=" for f in cfg.folders] + [""]
    lines += ["[Filenames]"]
    for e in cfg.files:
        if e.empty:
            lines.append("none=")
            continue
        s = str(e.path)
        if e.all_frames:
            s += "?*"
        elif e.frames:
            s += f"?{e.frames[0]}:{e.frames[1]}"
        elif e.frame is not None:
            s += f"?{e.frame}"
        if e.group:
            s += f"|{e.group}"
        lines.append(s + "=")
    lines.append("")
    for title, items in (("[Comments]", cfg.comments),
                         ("[MaxValPerPic]", cfg.max_val_per_pic),
                         ("[RowComments]", cfg.row_comments),
                         ("[ColumnComments]", cfg.column_comments)):
        if items:
            lines += [title] + [f"{i}=" for i in items] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
