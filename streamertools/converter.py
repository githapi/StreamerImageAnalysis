"""
Image Converter  --  Streamertools manual, chapter 5
=====================================================

Converts between all readable formats and performs, in the fixed order of
the manual (5.2), a chain of actions on the way:

===  ==============================  ===========================================
1    Simple math (5.2.1)             arithmetic with a fixed number A
2    Image math (5.2.2)              arithmetic with a second image
3    Calc temps (5.2.3)              disabled in the original, and here
4    Crop, rotate, flip (5.2.4)      in that order, bilinear interpolation
5    Resize (5.2.5)                  arbitrary, optionally aspect-locked
6    Add text (5.2.6)                colour output only
7    Add arrow (5.2.7)               colour output only
8    Overlay (5.2.8)                 colour output only
===  ==============================  ===========================================

Actions 1-5 work on the **counts**; 6-8 work on the rendered colour image, so
they are refused for a count-based output type, exactly as in the manual.

When the output is a colour file the Imin/Imax/gamma of the
:class:`~streamertools.display.DisplaySettings` are applied; when it is a
count based file no display setting is applied at all (manual 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import geometry
from .annotate import ArrowSpec, TextStyle, apply_overlay, draw_arrow, draw_text_block, expand_placeholders
from .display import DisplaySettings, render
from .io import COLOUR_EXTENSIONS, ImageStack, load, save_counts, save_image

__all__ = ["SimpleMath", "ImageMath", "CropRotateFlip", "Resize", "TextAction",
           "ArrowAction", "OverlayAction", "ConverterSettings", "ImageConverter"]


# --------------------------------------------------------------------------
# the seven actions
# --------------------------------------------------------------------------
@dataclass
class SimpleMath:
    """5.2.1 -- arithmetic between every count value and a fixed number A."""
    enabled: bool = False
    operation: str = "subtract_from"       # add|subtract|multiply|divide|subtract_from
    value: float = 0.0

    def apply(self, counts: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return counts
        a = float(self.value)
        op = self.operation
        if op == "add":
            return counts + a
        if op == "subtract":
            return counts - a
        if op == "multiply":
            return counts * a
        if op == "divide":
            return counts / (a if a != 0 else 1e-12)
        if op == "subtract_from":           # A - source, the '1000 - Source' case
            return a - counts
        raise ValueError(f"unknown operation {op!r}")


@dataclass
class ImageMath:
    """5.2.2 -- arithmetic with the pixels of a secondary image.

    Typical uses from the manual: subtracting a background measurement from a
    whole series, or multiplying with a prepared mask to draw an overlay.
    """
    enabled: bool = False
    operation: str = "subtract"            # add|subtract|multiply|divide
    secondary: Union[str, Path, np.ndarray, None] = None
    frame: int = 0

    def _second(self, shape) -> np.ndarray:
        if isinstance(self.secondary, (str, Path)):
            arr = load(self.secondary).data[self.frame]
        else:
            arr = np.asarray(self.secondary, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[self.frame]
        if arr.shape != tuple(shape):
            raise ValueError(f"secondary image {arr.shape} must match the input {shape}")
        return arr.astype(np.float32)

    def apply(self, counts: np.ndarray) -> np.ndarray:
        if not self.enabled or self.secondary is None:
            return counts
        b = self._second(counts.shape)
        op = self.operation
        if op == "add":
            return counts + b
        if op == "subtract":
            return counts - b
        if op == "multiply":
            return counts * b
        if op == "divide":
            return counts / np.where(b == 0, 1e-12, b)
        raise ValueError(f"unknown operation {op!r}")


@dataclass
class CropRotateFlip:
    """5.2.4 -- crop, then rotate by an arbitrary angle, then mirror."""
    enabled: bool = False
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    angle_deg: float = 0.0
    expand: bool = False
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def apply(self, counts: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return counts
        out = counts
        if any((self.left, self.right, self.top, self.bottom)):
            out = geometry.crop(out, self.left, self.right, self.top, self.bottom)
        if self.angle_deg:
            out = geometry.rotate(out, self.angle_deg, expand=self.expand)
        if self.flip_horizontal or self.flip_vertical:
            out = geometry.flip(out, self.flip_horizontal, self.flip_vertical)
        return out


@dataclass
class Resize:
    """5.2.5 -- resize with the fig. 3.4 interpolation."""
    enabled: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    lock_aspect: bool = True

    def apply(self, counts: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return counts
        return geometry.resize(counts, self.width, self.height, self.lock_aspect)


@dataclass
class TextAction:
    """5.2.6 -- a line of text at the top and/or the bottom (colour only)."""
    enabled: bool = False
    top: str = ""
    bottom: str = ""
    style: TextStyle = field(default_factory=TextStyle)
    counter_start: int = 0
    counter_step: int = 1

    def apply(self, rgb: np.ndarray, **ctx) -> np.ndarray:
        if not self.enabled:
            return rgb
        out = rgb
        for text, pos in ((self.top, "top"), (self.bottom, "bottom")):
            if text:
                out = draw_text_block(out, expand_placeholders(text, **ctx), pos, self.style)
        return out


@dataclass
class ArrowAction:
    """5.2.7 -- a dimension arrow at the right-hand side (colour only)."""
    enabled: bool = False
    arrow: ArrowSpec = field(default_factory=ArrowSpec)
    style: TextStyle = field(default_factory=TextStyle)

    def apply(self, rgb: np.ndarray, **ctx) -> np.ndarray:
        if not self.enabled:
            return rgb
        spec = ArrowSpec(**{**self.arrow.__dict__,
                            "text": expand_placeholders(self.arrow.text, **ctx)})
        return draw_arrow(rgb, spec, self.style)


@dataclass
class OverlayAction:
    """5.2.8 -- paste an image with one transparent colour (colour only)."""
    enabled: bool = False
    file: Union[str, Path, np.ndarray, None] = None
    transparent_colour: str = "#000000"
    tolerance: int = 0

    def apply(self, rgb: np.ndarray, **ctx) -> np.ndarray:
        if not self.enabled or self.file is None:
            return rgb
        return apply_overlay(rgb, self.file, self.transparent_colour, self.tolerance)


# --------------------------------------------------------------------------
@dataclass
class ConverterSettings:
    """The whole *Actions* panel plus the output colour settings."""

    simple_math: SimpleMath = field(default_factory=SimpleMath)
    image_math: ImageMath = field(default_factory=ImageMath)
    crop_rotate_flip: CropRotateFlip = field(default_factory=CropRotateFlip)
    resize: Resize = field(default_factory=Resize)
    text: TextAction = field(default_factory=TextAction)
    arrow: ArrowAction = field(default_factory=ArrowAction)
    overlay: OverlayAction = field(default_factory=OverlayAction)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    grayscale_output: bool = False          # 'Grayscale image' radio button
    jpeg_quality: int = 90


class ImageConverter:
    """Run the action chain on one frame, one file, or a folder of files."""

    def __init__(self, settings: Optional[ConverterSettings] = None):
        self.settings = settings or ConverterSettings()
        self._counter = None

    # -- the chain ---------------------------------------------------------
    def apply_count_actions(self, counts: np.ndarray) -> np.ndarray:
        """Actions 1-5, which all work on count values."""
        s = self.settings
        out = np.asarray(counts, dtype=np.float32)
        out = s.simple_math.apply(out)
        out = s.image_math.apply(out)
        # 3. Calc temps is disabled in the original program (manual 5.2.3)
        out = s.crop_rotate_flip.apply(out)
        out = s.resize.apply(out)
        return np.asarray(out, dtype=np.float32)

    def apply_colour_actions(self, rgb: np.ndarray, **ctx) -> np.ndarray:
        """Actions 6-8, which need a colour image."""
        s = self.settings
        out = s.text.apply(rgb, **ctx)
        out = s.arrow.apply(out, **ctx)
        out = s.overlay.apply(out, **ctx)
        return out

    # -- previewing --------------------------------------------------------
    def preview(self, source, frame: int = 0, as_colour: bool = True,
                reference: Optional[np.ndarray] = None, **ctx):
        """Run everything and return ``(image, counts, info)`` without saving.

        ``image`` is RGB uint8 when `as_colour`, else the processed counts.
        """
        stack = source if isinstance(source, ImageStack) else (
            load(source) if isinstance(source, (str, Path)) else ImageStack(source))
        counts = self.apply_count_actions(stack.data[frame])
        info = {
            "input_shape": stack.data.shape[1:],
            "output_shape": counts.shape,
            "input_range": (float(stack.data[frame].min()), float(stack.data[frame].max())),
            "output_range": (float(counts.min()), float(counts.max())),
        }
        if not as_colour:
            return counts, counts, info
        rgb, _, lo, hi = render(counts, self.settings.display, reference=reference)
        if self.settings.grayscale_output:
            rgb = np.repeat((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                             + 0.114 * rgb[..., 2]).astype(np.uint8)[..., None], 3, axis=2)
        ctx.setdefault("filename", str(getattr(stack, "path", "") or ""))
        ctx.setdefault("frame", frame)
        rgb = self.apply_colour_actions(rgb, **ctx)
        info.update(i_min=lo, i_max=hi)
        return rgb, counts, info

    # -- single file -------------------------------------------------------
    def convert(self, source, output, frame: int = 0,
                reference: Optional[np.ndarray] = None, **ctx) -> Path:
        """Convert one frame and write it to `output` (type from its suffix)."""
        output = Path(output)
        colour_out = output.suffix.lower() in COLOUR_EXTENSIONS
        s = self.settings
        if not colour_out and any((s.text.enabled, s.arrow.enabled, s.overlay.enabled)):
            raise ValueError("adding text, arrows or an overlay needs a colour output "
                             "file type (manual 5.2.6 - 5.2.8)")
        image, counts, _ = self.preview(source, frame, as_colour=colour_out,
                                        reference=reference, **ctx)
        if colour_out:
            return save_image(output, image, jpeg_quality=s.jpeg_quality)
        return save_counts(output, counts)

    # -- multiple files ----------------------------------------------------
    def convert_files(self, sources: Iterable[Union[str, Path]], out_dir,
                      out_ext: str = ".png", all_frames: bool = False,
                      subfolder: Optional[str] = None,
                      progress: Optional[Callable[[int, int], None]] = None
                      ) -> List[Path]:
        """Batch mode (manual 5.1).

        Output files keep the input filename, with the new extension; frames
        of a movie file get the frame number appended, as in the original.
        """
        sources = [Path(p) for p in sources]
        out_dir = Path(out_dir) / subfolder if subfolder else Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        counter = self.settings.text.counter_start

        for i, src in enumerate(sources):
            stack = load(src)
            indices = range(stack.n_frames) if all_frames else [0]
            for k in indices:
                name = src.stem + (f"_{k:04d}" if all_frames and stack.n_frames > 1 else "")
                target = out_dir / f"{name}{out_ext}"
                self.convert(stack, target, frame=k, reference=stack.data,
                             filename=str(src), counter=counter,
                             max_count=float(stack.data[k].max()))
                written.append(target)
                counter += self.settings.text.counter_step
            if progress:
                progress(i + 1, len(sources))
        return written
