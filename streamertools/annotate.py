"""
Text, arrows and overlays on rendered images
============================================

Shared by the Image Converter (manual 5.2.6 - 5.2.8) and the Tile maker
(6.2.2, sections ``[Arrow]``, ``[Overlay]``, ``[Comments]``).  All of these
work on **colour** images only, exactly as the manual states.

Colours follow the tile-maker convention (manual 6.2.2): an html hex triplet
such as ``"#FFAA00"``, a plain colour name, an ``(r, g, b)`` tuple, or the
old Streamertools integer ``1*R + 256*G + 65536*B``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "parse_colour", "TextStyle", "ArrowSpec", "resolve_font",
    "draw_text_block", "draw_arrow", "apply_overlay", "expand_placeholders",
]

Colour = Union[str, int, Sequence[int]]


# --------------------------------------------------------------------------
def parse_colour(value: Colour, default=(255, 255, 255)) -> Tuple[int, int, int]:
    """Accept ``"#RRGGBB"``, a colour name, ``(r,g,b)`` or the old integer."""
    if value is None:
        return tuple(default)
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(int(v) for v in list(value)[:3])
    if isinstance(value, (int, np.integer)):
        v = int(value)                      # 1*R + 256*G + 65536*B
        return (v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)
    s = str(value).strip()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    if s.isdigit():
        return parse_colour(int(s), default)
    from PIL import ImageColor
    try:
        return ImageColor.getrgb(s)[:3]
    except ValueError:
        return tuple(default)


def resolve_font(name: Optional[str], size: int):
    """Find a truetype font by name, falling back to a bundled default."""
    from PIL import ImageFont
    import matplotlib.font_manager as fm

    candidates = [name] if name else []
    candidates += ["DejaVu Sans", "Arial", "Liberation Sans"]
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        try:
            if p.suffix.lower() in (".ttf", ".otf") and p.exists():
                return ImageFont.truetype(str(p), size)
            return ImageFont.truetype(fm.findfont(cand, fallback_to_default=False), size)
        except Exception:
            continue
    try:
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:                                            # pragma: no cover
        return ImageFont.load_default()


# --------------------------------------------------------------------------
@dataclass
class TextStyle:
    """Font settings shared by comments and arrow texts (manual 5.2.6/6.2.2)."""

    font: Optional[str] = None
    size: int = 20
    colour: Colour = "#FFFFFF"
    shadow: bool = True
    shadow_colour: Colour = "#000000"
    shadow_offset: int = 2
    transparent_background: bool = True
    background_colour: Colour = "#000000"
    from_border: int = 10          # distance text - image edge, in pixels

    def pil_font(self):
        return resolve_font(self.font, int(self.size))


def expand_placeholders(text: str, filename: str = "", secondary: str = "",
                        frame: Optional[int] = None, counter: Optional[int] = None,
                        max_count: Optional[float] = None) -> str:
    """Substitute the manual's special symbols in a comment or arrow text.

    ``/f`` input filename, ``/s`` secondary filename, ``/m`` frame number,
    ``/c`` custom counter, ``/max`` maximum count of the image, ``/file`` the
    filename (tile maker spelling), ``/frame`` the frame number, ``/n`` a
    line break.  ``/b`` is handled by the caller (it selects the position).
    """
    if not text:
        return ""
    out = text.replace("/b", "")
    pairs = [("/max", "" if max_count is None else f"{max_count:g}"),
             ("/file", Path(filename).name if filename else ""),
             ("/frame", "" if frame is None else str(frame)),
             ("/f", Path(filename).stem if filename else ""),
             ("/s", Path(secondary).stem if secondary else ""),
             ("/m", "" if frame is None else str(frame)),
             ("/c", "" if counter is None else str(counter)),
             ("/n", "\n")]
    for token, value in pairs:
        out = out.replace(token, value)
    return out


# --------------------------------------------------------------------------
def draw_text_block(rgb: np.ndarray, text: str, position: str = "bottom",
                    style: Optional[TextStyle] = None, grow: bool = False,
                    align: str = "left") -> np.ndarray:
    """Draw a line (or several, via ``/n``) of text on a colour image.

    Parameters
    ----------
    position : {"top", "bottom", "top-left", "top-right",
                "bottom-left", "bottom-right", "centre"}
    grow : bool
        Add a strip to the image instead of drawing over it -- used by the
        tile maker for outside comments.
    """
    from PIL import Image, ImageDraw

    if not text:
        return rgb
    style = style or TextStyle()
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    font = style.pil_font()
    colour = parse_colour(style.colour)
    shadow_colour = parse_colour(style.shadow_colour)
    margin = int(style.from_border)

    tmp = ImageDraw.Draw(img)
    bbox = tmp.multiline_textbbox((0, 0), text, font=font, align=align)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if grow:
        pad = th + 2 * margin
        canvas = Image.new("RGB", (img.width, img.height + pad),
                           parse_colour(style.background_colour, (0, 0, 0)))
        top_strip = position.startswith("top")
        canvas.paste(img, (0, pad if top_strip else 0))
        img = canvas

    W, H = img.size
    pos_map = {
        "top": ((W - tw) // 2, margin),
        "bottom": ((W - tw) // 2, H - th - margin - bbox[1]),
        "top-left": (margin, margin),
        "top-right": (W - tw - margin, margin),
        "bottom-left": (margin, H - th - margin - bbox[1]),
        "bottom-right": (W - tw - margin, H - th - margin - bbox[1]),
        "centre": ((W - tw) // 2, (H - th) // 2),
    }
    x, y = pos_map.get(position, pos_map["bottom"])

    draw = ImageDraw.Draw(img)
    if not style.transparent_background:
        pad = max(2, int(style.size * 0.15))
        draw.rectangle([x - pad, y - pad + bbox[1], x + tw + pad, y + th + pad + bbox[1]],
                       fill=parse_colour(style.background_colour, (0, 0, 0)))
    if style.shadow:
        off = int(style.shadow_offset)
        draw.multiline_text((x + off, y + off), text, font=font,
                            fill=shadow_colour, align=align)
    draw.multiline_text((x, y), text, font=font, fill=colour, align=align)
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------
@dataclass
class ArrowSpec:
    """A dimension arrow (manual 5.2.7 and tile maker ``[Arrow]``)."""

    show: bool = True
    horizontal: bool = False       # False -> vertical arrow on the right side
    start: int = 30                # ArrowTop / ArrowLeft, pixels from the edge
    length: int = 200              # ArrowLength, pixels
    from_side: Optional[int] = None  # ArrowFromSide; None -> 30 px from right/bottom
    text: str = ""                 # printed halfway the arrow
    top_text: str = ""             # LeftText for a horizontal arrow
    bottom_text: str = ""          # RightText for a horizontal arrow
    end_bars: bool = False
    shadow: bool = True
    row: Optional[int] = None      # tile maker: which image to draw it on
    col: Optional[int] = None


def draw_arrow(rgb: np.ndarray, spec: ArrowSpec,
               style: Optional[TextStyle] = None) -> np.ndarray:
    """Draw a double-headed dimension arrow with optional texts."""
    from PIL import Image, ImageDraw

    if not spec.show:
        return rgb
    style = style or TextStyle()
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    colour = parse_colour(style.colour)
    shadow_colour = parse_colour(style.shadow_colour)
    lw = max(1, int(style.size / 12))
    head = max(4, int(style.size / 2))
    font = resolve_font(style.font, max(6, int(style.size * 0.8)))   # 80 %, manual 6.2.2

    if spec.horizontal:
        y = H - 30 if spec.from_side is None else int(spec.from_side)
        x0, x1 = int(spec.start), int(spec.start) + int(spec.length)
        line = [(x0, y), (x1, y)]
        heads = [[(x0, y), (x0 + head, y - head // 2), (x0 + head, y + head // 2)],
                 [(x1, y), (x1 - head, y - head // 2), (x1 - head, y + head // 2)]]
        bars = [[(x0, y - head), (x0, y + head)], [(x1, y - head), (x1, y + head)]]
        mid, anchor = ((x0 + x1) // 2, y - head - 2), "ms"
        ends = [((x0, y + head + 2), spec.top_text, "lt"),
                ((x1, y + head + 2), spec.bottom_text, "rt")]
    else:
        x = W - 30 if spec.from_side is None else int(spec.from_side)
        y0, y1 = int(spec.start), int(spec.start) + int(spec.length)
        line = [(x, y0), (x, y1)]
        heads = [[(x, y0), (x - head // 2, y0 + head), (x + head // 2, y0 + head)],
                 [(x, y1), (x - head // 2, y1 - head), (x + head // 2, y1 - head)]]
        bars = [[(x - head, y0), (x + head, y0)], [(x - head, y1), (x + head, y1)]]
        mid, anchor = (x - head - 4, (y0 + y1) // 2), "rm"
        ends = [((x - head - 4, y0), spec.top_text, "rt"),
                ((x - head - 4, y1), spec.bottom_text, "rs")]

    def _paint(col, dx=0, dy=0):
        draw.line([(p[0] + dx, p[1] + dy) for p in line], fill=col, width=lw)
        for h in heads:
            draw.polygon([(p[0] + dx, p[1] + dy) for p in h], fill=col)
        if spec.end_bars:
            for b in bars:
                draw.line([(p[0] + dx, p[1] + dy) for p in b], fill=col, width=lw)

    off = int(style.shadow_offset)
    if spec.shadow and style.shadow:
        _paint(shadow_colour, off, off)
    _paint(colour)

    for point, text, anc in [(mid, spec.text, anchor)] + ends:
        if not text:
            continue
        if spec.shadow and style.shadow:
            draw.text((point[0] + off, point[1] + off), text, font=font,
                      fill=shadow_colour, anchor=anc)
        draw.text(point, text, font=font, fill=colour, anchor=anc)
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------
def apply_overlay(rgb: np.ndarray, overlay: Union[str, Path, np.ndarray],
                  transparent_colour: Colour = "#000000",
                  tolerance: int = 0) -> np.ndarray:
    """Paste an overlay image, one of whose colours is transparent (5.2.8).

    The overlay must have the same resolution as the image it is drawn on
    (after cropping and zooming, manual 6.2.2 ``OverlayFile``).
    """
    from PIL import Image

    base = np.asarray(rgb, dtype=np.uint8)
    if isinstance(overlay, (str, Path)):
        ov = np.asarray(Image.open(overlay).convert("RGB"), dtype=np.uint8)
    else:
        ov = np.asarray(overlay, dtype=np.uint8)
    if ov.shape[:2] != base.shape[:2]:
        raise ValueError(f"overlay {ov.shape[:2]} does not match image {base.shape[:2]}")
    key = np.array(parse_colour(transparent_colour, (0, 0, 0)), dtype=np.int16)
    diff = np.abs(ov.astype(np.int16) - key).max(axis=2)
    opaque = diff > int(tolerance)
    out = base.copy()
    out[opaque] = ov[opaque]
    return out
