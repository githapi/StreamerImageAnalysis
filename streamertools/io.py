"""
File input/output  --  Streamertools manual, chapter 2.1
========================================================

Two families of files exist (manual 2.1):

* **count based**  -- one count value per pixel (ICCD cameras).  Internally we
  always keep these as ``float32``, exactly as Streamertools does.
* **colour based** -- RGB(A) images.  They can be *read* as count based files,
  in which case the channels are averaged (manual 2.1), or kept in colour
  (Tile maker ``UseGrayValues = 0``).

Everything is returned as a 3-D stack ``(frames, H, W)``; a single image is
simply ``frames == 1``.  That keeps a single image and a kinetic series on the
same code path.

Supported here -- every type of manual 2.1
------------------------------------------
======================  ====================================================
``.img``                Stanford Computer Optics, one frame, 16/32 bit
``.rtv``                Stanford Computer Optics movie, 16 bit       movie
``.spe``                Princeton Instruments, SPE 2.x and 3.0       movie
``.sif``                Andor (needs ``sif_parser``)                 movie
``.st2k``               SBIG, compressed or not
``.tsv``                CWI model output: tab-separated grids
``.fit .fits``          FITS (needs ``astropy``)                     movie
``.cih`` / ``.raw``     Photron: text header + raw frames            movie
``.dat``                Shimadzu, 312x260 only (manual 2.1.1)        movie
``.im7``                LaVision DaVis, uncompressed or zlib         movie
``.txt``                Streamertools text file (2.1.3)        read + write
``.streamer``           Streamertools binary file (2.1.3)      read + write
``.avi``                video; uncompressed read directly,           movie
                        anything else through OpenCV's codecs
``.jpg .png .bmp``      colour images, RGB averaged (2.1.2)    read + write
``.tif``                colour images; multi-page = movie      read + write
``.npy``, ``.gif``      convenience                                  movie
======================  ====================================================

"movie" marks the multi-frame types of manual 2.1.4.  Formats the manual
calls undocumented follow the readers StreamerTools itself ships (its Matlab
toolbox, ``T2DFileImage``) or the vendor's own code; where the manual's
support is partial, so is this one, and the error says what is missing:

* Agema/FLIR thermal ``.img`` shares the extension with Stanford, but its
  layout is not documented anywhere available -- such a file is refused
  with that explanation;
* LaVision ``.im7``: images stored uncompressed or zlib-packed (the usual
  DaVis settings); the old IMX packing, the fixed 12-bit packing and
  vector or sparse buffers are refused.

Everything comes back as ``(frames, H, W)``.  Several single-frame files
become one series with :func:`load_stack` (a folder, a glob pattern or a
list), which is how a delay series of one-shot images is read.

Add a reader to ``READERS`` and it becomes available everywhere in the
framework.
"""

from __future__ import annotations

import glob
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ImageStack",
    "load",
    "load_stack",
    "frame_files",
    "long_path",
    "save_counts",
    "save_image",
    "read_streamer_txt",
    "write_streamer_txt",
    "read_streamer_bin",
    "write_streamer_bin",
    "COUNT_EXTENSIONS",
    "COLOUR_EXTENSIONS",
    "MOVIE_EXTENSIONS",
]

COUNT_EXTENSIONS = {".img", ".rtv", ".spe", ".sif", ".st2k", ".tsv", ".fit", ".fits",
                    ".cih", ".raw", ".dat", ".im7", ".txt", ".streamer", ".avi", ".npy"}
COLOUR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------
@dataclass
class ImageStack:
    """A (frames, H, W) count-based stack plus its provenance.

    Parameters
    ----------
    data : ndarray (frames, H, W) float32
        Count values.  Always 3-D, even for a single image.
    metadata : dict
        Whatever the reader could extract from the file header.
    path : Path or None
    pixel_size_um : float or None
        Size of one pixel *in the image plane* in micrometre (manual 2.4).
        ``None`` means "not calibrated"; distances then come out in pixels.
    colour : ndarray (frames, H, W, 3) uint8 or None
        Original RGB data, kept only when the file was colour based and it was
        read with ``keep_colour=True`` (Tile maker ``UseGrayValues = 0``).
    """

    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None
    pixel_size_um: Optional[float] = None
    colour: Optional[np.ndarray] = None

    # -- basics ------------------------------------------------------------
    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim == 2:
            self.data = self.data[None]
        if self.data.ndim != 3:
            raise ValueError(f"expected a 2-D or 3-D array, got {self.data.shape}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        return self.data[index]

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def is_movie(self) -> bool:
        """True for a multi-frame (movie / kinetic series) file, manual 2.1.4."""
        return self.n_frames > 1

    def frame(self, index: int = 0) -> np.ndarray:
        return self.data[index]

    def copy_with(self, data: np.ndarray, **overrides) -> "ImageStack":
        """New ImageStack with the same provenance but different pixel data."""
        kwargs = dict(metadata=dict(self.metadata), path=self.path,
                      pixel_size_um=self.pixel_size_um, colour=None)
        kwargs.update(overrides)
        return ImageStack(data=data, **kwargs)

    def describe(self) -> str:
        kind = "single image" if self.n_frames == 1 else f"{self.n_frames}-frame series"
        px = ("uncalibrated" if self.pixel_size_um is None
              else f"{self.pixel_size_um:g} um/px")
        return (f"{kind}  {self.height}x{self.width}  "
                f"counts {float(self.data.min()):.1f} -> {float(self.data.max()):.1f}  "
                f"({px})")

    # -- metadata ----------------------------------------------------------
    def meta(self, *name_substrings: str, default=None):
        """Case-insensitive fuzzy lookup in the file metadata."""
        return metadata_get(self.metadata, *name_substrings, default=default)

    def show_metadata(self, max_len: int = 100, key_filter: Optional[str] = None) -> None:
        show_metadata(self.metadata, max_len=max_len, key_filter=key_filter)


# --------------------------------------------------------------------------
# metadata helpers
# --------------------------------------------------------------------------
def metadata_get(metadata, *name_substrings: str, default=None):
    """Fuzzy, case-insensitive lookup of a header field."""
    try:
        items = metadata.items()
    except AttributeError:
        return default
    for key, val in items:
        if any(s.lower() in str(key).lower() for s in name_substrings):
            return val
    return default


def show_metadata(metadata, max_len: int = 100, key_filter: Optional[str] = None) -> None:
    """Pretty-print every field of a header dict (arrays truncated)."""
    try:
        items = list(metadata.items())
    except AttributeError:
        print(metadata)
        return
    if key_filter:
        items = [(k, v) for k, v in items if key_filter.lower() in str(k).lower()]
    width = max((len(str(k)) for k, _ in items), default=0)
    for k, v in items:
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v).ravel()
                head = np.array2string(arr[:5], precision=4, separator=", ")
                s = f"<{type(v).__name__} n={arr.size}> {head}" + (" ..." if arr.size > 5 else "")
            except (ValueError, TypeError):
                seq = list(v)
                s = f"<{type(v).__name__} n={len(seq)}> {seq[:3]}" + (" ..." if len(seq) > 3 else "")
        else:
            s = str(v)
        print(f"{str(k):<{width}} : {s[:max_len] + ' ...' if len(s) > max_len else s}")


# --------------------------------------------------------------------------
# Streamertools own formats (manual 2.1.3)
# --------------------------------------------------------------------------
_TXT_MAGIC = "Streamer tools text file."
_BIN_MAGIC = "Streamer tools binary file."
_BIN_DATA_OFFSET = 256


def read_streamer_txt(path) -> np.ndarray:
    """Read a Streamertools ``*.txt`` count file -> (H, W) float32."""
    path = Path(path)
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    if not lines or _TXT_MAGIC.rstrip(".").lower() not in lines[0].lower():
        raise ValueError(f"{path.name}: missing '{_TXT_MAGIC}' header line")
    n_x, n_y = int(lines[1].strip()), int(lines[2].strip())
    rows = [np.fromstring(ln, sep="\t") for ln in lines[3:3 + n_y]]
    data = np.asarray(rows, dtype=np.float32)
    if data.shape != (n_y, n_x):
        raise ValueError(f"{path.name}: header says {n_y}x{n_x}, read {data.shape}")
    return data


def write_streamer_txt(path, image: np.ndarray, fmt: str = "%.9g") -> Path:
    """Write a 2-D count image as a Streamertools ``*.txt`` file."""
    path, image = Path(path), np.asarray(image, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_y, n_x = image.shape
    with open(path, "w") as fh:
        fh.write(_TXT_MAGIC + "\n")
        fh.write(f"{n_x}\n{n_y}\n")
        for row in image:
            fh.write("\t".join(fmt % v for v in row) + "\n")
    return path


def read_streamer_bin(path) -> np.ndarray:
    """Read a Streamertools ``*.streamer`` binary count file -> (H, W) float32."""
    path = Path(path)
    raw = path.read_bytes()
    head = raw[:_BIN_DATA_OFFSET].split(b"\x00")[0].decode("latin-1", "ignore")
    lines = head.splitlines()
    if not lines or _BIN_MAGIC.rstrip(".").lower() not in lines[0].lower():
        raise ValueError(f"{path.name}: missing '{_BIN_MAGIC}' header line")
    n_x, n_y = int(lines[1].strip()), int(lines[2].strip())
    values = np.frombuffer(raw, dtype="<f4", count=n_x * n_y, offset=_BIN_DATA_OFFSET)
    return values.reshape(n_y, n_x).astype(np.float32)


def write_streamer_bin(path, image: np.ndarray) -> Path:
    """Write a 2-D count image as a Streamertools ``*.streamer`` binary file."""
    path, image = Path(path), np.asarray(image, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    n_y, n_x = image.shape
    header = f"{_BIN_MAGIC}\n{n_x}\n{n_y}\n".encode("latin-1")
    if len(header) > _BIN_DATA_OFFSET:
        raise ValueError("header does not fit in the first 256 bytes")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(b"\x00" * (_BIN_DATA_OFFSET - len(header)))
        fh.write(image.tobytes(order="C"))
    return path


# --------------------------------------------------------------------------
# individual readers -> (data (frames,H,W) float32, metadata dict, colour|None)
# --------------------------------------------------------------------------
def _read_sif(path, **kw):
    try:
        import sif_parser
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError("reading .sif needs the 'sif_parser' package "
                          "(pip install sif_parser)") from exc
    data, metadata = sif_parser.np_open(str(path))
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    return data, dict(metadata) if hasattr(metadata, "items") else {"raw": metadata}, None


def _read_txt(path, **kw):
    return read_streamer_txt(path)[None], {}, None


def _read_bin(path, **kw):
    return read_streamer_bin(path)[None], {}, None


def _read_npy(path, **kw):
    data = np.load(path)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    return data, {}, None


def _read_picture(path, keep_colour: bool = False, **kw):
    """Colour based file (manual 2.1.2).  RGB is averaged into counts."""
    from PIL import Image
    im = Image.open(path)
    frames, colours = [], []
    try:                                       # multi-page tiff = movie file
        n = getattr(im, "n_frames", 1)
    except Exception:
        n = 1
    for i in range(n):
        im.seek(i) if n > 1 else None
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
        frames.append(arr.mean(axis=2))        # average RGB -> counts
        if keep_colour:
            colours.append(np.asarray(rgb, dtype=np.uint8))
    data = np.stack(frames).astype(np.float32)
    colour = np.stack(colours) if keep_colour else None
    return data, {"mode": im.mode, "n_frames": n}, colour


def _read_fits(path, **kw):
    try:
        from astropy.io import fits
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError("reading FITS needs 'astropy'") from exc
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = np.asarray(hdu.data, dtype=np.float32)
        meta = {k: hdu.header[k] for k in hdu.header}
    if data.ndim == 2:
        data = data[None]
    return data, meta, None


def _sibling(path: Path, suffix: str) -> Optional[Path]:
    """The file next to `path` with another extension, in either case."""
    for candidate in (path.with_suffix(suffix.lower()), path.with_suffix(suffix.upper())):
        if candidate.exists():
            return candidate
    return None


def _read_photron_cih(path, **kw):
    """Photron ``.cih`` (plain-text header) + ``.raw`` frames (manual 2.1.1).

    The frames are in ``<name>.raw``, or -- when the camera wrote one file
    per frame -- in ``<name>*.raw``, which are then taken in name order.
    """
    path = Path(path)
    meta = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    w = int(meta.get("Image Width", 0))
    h = int(meta.get("Image Height", 0))
    n = int(meta.get("Total Frame", 0))
    bits = int(meta.get("Color Bit", 16))
    if w <= 0 or h <= 0:
        raise ValueError(f"{path.name}: no 'Image Width' / 'Image Height' in the header")
    single = _sibling(path, ".raw")
    raws = [single] if single is not None else sorted(
        (p for p in path.parent.glob(glob.escape(path.stem) + "*")
         if p.suffix.lower() == ".raw"), key=_natural_key)
    if not raws:
        raise FileNotFoundError(f"no {path.stem}.raw next to {path.name} "
                                f"(the .cih only holds the header)")
    dtype = np.uint8 if bits <= 8 else "<u2"
    values = np.concatenate([np.fromfile(p, dtype=dtype) for p in raws])
    n = n or values.size // (w * h)
    if values.size < n * w * h:
        raise ValueError(f"{path.name}: the header promises {n} frames of {w}x{h}, "
                         f"the .raw data holds {values.size // (w * h)}")
    data = values[:n * w * h].reshape(n, h, w).astype(np.float32)
    meta["raw_files"] = [p.name for p in raws]
    return data, meta, None


def _read_photron_raw(path, **kw):
    """A Photron ``.raw`` opened directly: its ``.cih`` holds the header."""
    path = Path(path)
    cih = _sibling(path, ".cih")
    if cih is None:
        stem = path.stem
        while stem and cih is None:                # frame files: <name>000123.raw
            stem = stem[:-1]
            cih = _sibling(path.with_name(stem + ".raw"), ".cih") if stem else None
    if cih is None:
        raise FileNotFoundError(f"{path.name}: a Photron .raw needs its .cih header "
                                f"file next to it")
    return _read_photron_cih(cih, **kw)


def _read_shimadzu_dat(path, **kw):
    """Shimadzu ``.dat``: only 312x260 is supported (manual 2.1.1)."""
    w, h = 312, 260
    values = np.fromfile(path, dtype="<u2")
    n = values.size // (w * h)
    if n == 0:
        raise ValueError("file too small for a 312x260 Shimadzu frame")
    data = values[:n * w * h].reshape(n, h, w).astype(np.float32)
    return data, {"assumed_resolution": (w, h), "frames": n}, None


_SPE_DTYPES = {0: "<f4", 1: "<i4", 2: "<i2", 3: "<u2", 5: "<f8", 6: "u1", 8: "<u4"}
_SPE_PIXEL_FORMATS = {"MonochromeUnsigned16": "<u2", "MonochromeUnsigned32": "<u4",
                      "MonochromeFloating32": "<f4"}


def _spe3_layout(raw: bytes, offset: int) -> Optional[dict]:
    """Frame layout from the XML footer of an SPE 3.0 file, if it has one."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(raw[offset:].split(b"\x00")[0])
    except ET.ParseError:
        return None
    blocks = [e for e in root.iter() if e.tag.split("}")[-1] == "DataBlock"]
    frame = next((e for e in blocks if e.get("type") == "Frame"), None)
    region = next((e for e in blocks if e.get("type") == "Region"), None)
    if frame is None or region is None:
        return None
    return {"count": int(frame.get("count", 0)),
            "stride": int(frame.get("stride") or frame.get("size") or 0),
            "width": int(region.get("width", 0)), "height": int(region.get("height", 0)),
            "dtype": _SPE_PIXEL_FORMATS.get(region.get("pixelFormat"))}


def _read_spe(path, **kw):
    """Princeton Instruments ``.spe``, versions 2.x and 3.0.

    The 4100-byte header gives width (byte 42), height (656), pixel type
    (108) and frame count (1446).  A 3.0 file adds an XML footer (its offset
    at byte 678) whose frame *stride* includes any per-frame metadata; when
    present it decides the layout, so such files are not misread.
    """
    raw = Path(path).read_bytes()
    w = struct.unpack_from("<H", raw, 42)[0]
    h = struct.unpack_from("<H", raw, 656)[0]
    dtype_code = struct.unpack_from("<h", raw, 108)[0]
    n = struct.unpack_from("<i", raw, 1446)[0]
    version = struct.unpack_from("<f", raw, 1992)[0]
    xml_offset = struct.unpack_from("<Q", raw, 678)[0]
    dtype = _SPE_DTYPES.get(dtype_code)
    stride = None
    layout = _spe3_layout(raw, xml_offset) if version >= 3 and 4100 < xml_offset < len(raw) \
        else None
    if layout is not None:
        w, h = layout["width"] or w, layout["height"] or h
        n = layout["count"] or n
        dtype = layout["dtype"] or dtype
        stride = layout["stride"] or None
    if dtype is None:
        raise ValueError(f"unsupported SPE pixel type {dtype_code}")
    n = max(n, 1)
    item = np.dtype(dtype).itemsize
    stride = stride or w * h * item
    end = 4100 + (n - 1) * stride + w * h * item
    if w <= 0 or h <= 0 or end > len(raw):
        raise ValueError(f"{Path(path).name}: header says {n} frames of {w}x{h}, "
                         f"the file is too short for that")
    data = np.stack([np.frombuffer(raw, dtype=dtype, count=w * h,
                                   offset=4100 + k * stride).reshape(h, w)
                     for k in range(n)]).astype(np.float32)
    meta = {"width": w, "height": h, "frames": n, "spe_datatype": dtype_code,
            "spe_version": round(float(version), 2)}
    return data, meta, None


# -- StreamerTools' undocumented types, as its own Matlab toolbox reads them -
def _read_stanford_img(path, **kw):
    """Stanford Computer Optics ``.img``, one frame (manual 2.1.1).

    Byte 0 is 254 for 16-bit counts, anything else means 32-bit; width and
    height are 16-bit integers at byte 218; the counts start at byte 258,
    left to right and top to bottom (StreamerTools' ``T2DFileImage``).
    Agema/FLIR thermal cameras write ``.img`` files too, in a layout that is
    not documented, so a file whose header does not fit is refused.
    """
    path = Path(path)
    raw = path.read_bytes()
    bits = 16 if raw[:1] == b"\xfe" else 32
    w, h = struct.unpack_from("<hh", raw, 218) if len(raw) >= 222 else (0, 0)
    if w <= 0 or h <= 0 or len(raw) < 258 + w * h * bits // 8:
        raise ValueError(
            f"{path.name} is not a Stanford Computer Optics .img (its header gives "
            f"{w}x{h} at {bits} bit for a {len(raw)}-byte file). Agema/FLIR thermal "
            f".img files use the same extension but an undocumented layout, and are "
            f"not supported -- a sample file would make them possible.")
    dtype = "<i2" if bits == 16 else "<i4"
    data = np.frombuffer(raw, dtype=dtype, count=w * h, offset=258).reshape(1, h, w)
    meta = {"format": "Stanford Computer Optics IMG", "bits_per_pixel": bits,
            "width": w, "height": h}
    return data.astype(np.float32), meta, None


def _read_stanford_rtv(path, **kw):
    """Stanford Computer Optics ``.rtv`` movie, 16-bit (manual 2.1.1).

    Width and height are the first and third 16-bit integers at byte 266,
    the frame count the one at byte 278.  Frames start at byte 8192, each
    in a block of whole 4096-byte pages plus one more page
    (StreamerTools' ``T2DFileImage``).
    """
    path = Path(path)
    head = np.fromfile(path, dtype=np.uint8, count=8192)
    if head.size < 280:
        raise ValueError(f"{path.name}: too short for an .rtv header")
    w, _, h = struct.unpack_from("<hhh", head.tobytes(), 266)
    n = struct.unpack_from("<h", head.tobytes(), 278)[0]
    frame_bytes = w * h * 2
    if w <= 0 or h <= 0:
        raise ValueError(f"{path.name}: the header gives a {w}x{h} frame")
    block = 4096 * ((1 if frame_bytes % 4096 == 0 else 2) + frame_bytes // 4096)
    size = path.stat().st_size
    fits = max(0, (size - 8192 - frame_bytes) // block + 1) if size >= 8192 + frame_bytes \
        else 0
    if n <= 0 or n > fits:
        n = fits                                   # a truncated file: what is there
    if n == 0:
        raise ValueError(f"{path.name}: no complete {w}x{h} frame in the file")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    data = np.stack([np.frombuffer(mm, dtype="<i2", count=w * h,
                                   offset=8192 + k * block).reshape(h, w)
                     for k in range(n)]).astype(np.float32)
    del mm
    meta = {"format": "Stanford Computer Optics RTV", "bits_per_pixel": 16,
            "width": w, "height": h, "frames": n}
    return data, meta, None


def _sbig_row(buf: bytes, width: int) -> np.ndarray:
    """One compressed SBIG row: a 16-bit first pixel, then signed byte deltas,
    where 0x80 escapes a full 16-bit value."""
    b = np.frombuffer(buf, dtype=np.uint8)
    first = int(b[0]) | (int(b[1]) << 8)
    codes = b[2:]
    absolute = []
    escapes = np.flatnonzero(codes == 0x80)
    skip_until = 0
    token_mask = np.ones(len(codes), dtype=bool)
    for e in escapes:                               # payload bytes are not tokens
        if e < skip_until:
            continue
        token_mask[e + 1:e + 3] = False
        skip_until = e + 3
        absolute.append(e)
    tokens = np.flatnonzero(token_mask)
    if len(tokens) != width - 1 or (absolute and absolute[-1] + 3 > len(codes)):
        raise ValueError("an SBIG row does not decode to the image width")
    is_abs = np.isin(tokens, absolute)
    delta = codes[tokens].astype(np.int8).astype(np.int64)
    delta[is_abs] = 0
    abs_vals = np.array([int(codes[e + 1]) | (int(codes[e + 2]) << 8) for e in absolute],
                        dtype=np.int64)
    segment = np.cumsum(is_abs)                     # 0 = after the first pixel
    base = np.concatenate([[first], abs_vals])
    cs = np.cumsum(delta)
    cs_at_start = np.concatenate([[0], cs[is_abs]])
    values = base[segment] + cs - cs_at_start[segment]
    return np.concatenate([[first], values]) % 65536


def _read_sbig(path, **kw):
    """SBIG CCD camera image (``.st2k`` and the other CCDOPS types).

    A 2048-byte text header ("<camera> Image" or "<camera> Compressed
    Image", then ``Key = value`` lines), then 16-bit little-endian rows,
    top to bottom.  In a compressed file each row starts with its byte
    length; a length of twice the width means the row is stored as is.
    (SBIG's own CSBIGImg class.)
    """
    path = Path(path)
    raw = path.read_bytes()
    text = raw[:2048].split(b"\x00")[0].decode("latin-1", "ignore")
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    kind = lines[0] if lines else ""
    if not kind.endswith("Image"):
        raise ValueError(f"{path.name}: not an SBIG image (first header line {kind!r})")
    meta = {"camera": kind.rsplit(" ", 2 if kind.endswith("Compressed Image") else 1)[0]}
    for ln in lines[1:]:
        if "=" in ln:
            k, v = ln.split("=", 1)
            meta[k.strip()] = v.strip()
    w, h = int(meta.get("Width", 0)), int(meta.get("Height", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"{path.name}: no Width/Height in the SBIG header")
    if not kind.endswith("Compressed Image"):
        if len(raw) < 2048 + 2 * w * h:
            raise ValueError(f"{path.name}: too short for {w}x{h} pixels")
        data = np.frombuffer(raw, dtype="<u2", count=w * h, offset=2048).reshape(h, w)
    else:
        data = np.empty((h, w), dtype=np.int64)
        pos = 2048
        for row in range(h):
            if pos + 2 > len(raw):
                raise ValueError(f"{path.name}: the data ends at row {row} of {h}")
            length = raw[pos] | (raw[pos + 1] << 8)
            pos += 2
            if not w + 1 <= length <= 2 * w or pos + length > len(raw):
                raise ValueError(f"{path.name}: row {row} has an impossible length {length}")
            chunk = raw[pos:pos + length]
            data[row] = (np.frombuffer(chunk, dtype="<u2") if length == 2 * w
                         else _sbig_row(chunk, w))
            pos += length
    meta.update(width=w, height=h, compressed=kind.endswith("Compressed Image"))
    return data[None].astype(np.float32), meta, None


def _read_tsv(path, **kw):
    """Tab-separated numbers, such as the grids of CWI's streamer models.

    One image row per line; lines that are not all numbers (headers,
    comments) are skipped.  A table of exactly three columns whose first two
    span a full regular grid is read as ``x, y, value`` instead.  CWI's
    compound output stores refined sub-grids in files of their own; each
    loads like this.
    """
    path = Path(path)
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = [p for p in re.split(r"\t" if "\t" in line else r"[ ,;]+", line.strip()) if p]
        if not parts:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{path.name}: no numeric rows")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path.name}: rows of different lengths {sorted(widths)}")
    table = np.asarray(rows, dtype=np.float64)
    meta = {"format": "tab-separated values", "layout": "matrix"}
    if table.shape[1] == 3 and len(table) > 3:
        xs, xi = np.unique(table[:, 0], return_inverse=True)
        ys, yi = np.unique(table[:, 1], return_inverse=True)
        if len(xs) * len(ys) == len(table) and len(xs) > 1 and len(ys) > 1:
            grid = np.full((len(ys), len(xs)), np.nan)
            grid[yi, xi] = table[:, 2]
            if not np.isnan(grid).any():
                meta.update(layout="x y value", x=xs, y=ys)
                return grid[None].astype(np.float32), meta, None
    return table[None].astype(np.float32), meta, None


_IM7_WORD, _IM7_FLOAT = "<u2", "<f4"


def _read_lavision_im7(path, **kw):
    """LaVision DaVis ``.im7`` image (manual 2.1.1: specific subtypes only).

    Layout from LaVision's ReadIMX: a 256-byte header (version, pack type,
    buffer format, sparse flag, then sizeX, sizeY, sizeZ, sizeF), then
    ``sizeY*sizeZ*sizeF`` rows of ``sizeX`` values -- uncompressed (pack
    type 0; 16-bit, bytes, or 32-bit float) or as one zlib block preceded
    by its byte length (pack type 2).  Frames follow each other.  Older
    IMX-style files, pack types 1 and 3, sparse and vector buffers are
    refused rather than guessed.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 256:
        raise ValueError(f"{path.name}: too short for an IM7 header")
    version, pack, fmt, sparse = struct.unpack_from("<4h", raw, 0)
    nx, ny, nz, nf = struct.unpack_from("<4i", raw, 8)
    if 18 <= version <= 23:
        raise NotImplementedError(f"{path.name} is an older LaVision IMX-type file "
                                  f"(type {version}); only IM7 images are supported")
    if fmt > 0:
        raise ValueError(f"{path.name} holds a vector field, not an image")
    if sparse:
        raise NotImplementedError(f"{path.name}: sparse IM7 buffers are not supported")
    element = {-4: _IM7_WORD, 0: _IM7_WORD, -2: "u1", -3: _IM7_FLOAT}.get(fmt)
    if element is None:
        raise NotImplementedError(f"{path.name}: IM7 buffer format {fmt} (colour or "
                                  f"double) is not supported")
    nz, nf = max(nz, 1), max(nf, 1)
    if nx <= 0 or ny <= 0:
        raise ValueError(f"{path.name}: header gives a {nx}x{ny} image")
    count = nx * ny * nz * nf
    if pack == 0:
        need = 256 + count * np.dtype(element).itemsize
        if len(raw) < need:
            raise ValueError(f"{path.name}: {len(raw)} bytes, {need} needed")
        values = np.frombuffer(raw, dtype=element, count=count, offset=256)
    elif pack == 2:
        length = struct.unpack_from("<i", raw, 256)[0]
        unpacked = zlib.decompress(raw[260:260 + length])
        dtype = _IM7_FLOAT if element == _IM7_FLOAT else _IM7_WORD   # rows are words
        values = np.frombuffer(unpacked, dtype=dtype, count=count)
    else:
        raise NotImplementedError(f"{path.name}: IM7 pack type {pack} "
                                  f"({'old IMX' if pack == 1 else 'fixed-bit'} "
                                  f"packing) is not supported")
    data = values.reshape(nf * nz, ny, nx).astype(np.float32)
    meta = {"format": "LaVision IM7", "im7_version": version, "pack_type": pack,
            "buffer_format": fmt, "size": (nx, ny, nz, nf)}
    return data, meta, None


# -- AVI -------------------------------------------------------------------
def _riff_chunks(buf: bytes, start: int, end: int):
    """(id, data start, size) of the chunks between `start` and `end`."""
    pos = start
    while pos + 8 <= end:
        cid = buf[pos:pos + 4]
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        yield cid, pos + 8, size
        pos += 8 + size + (size & 1)


_AVI_GRAY = {b"Y800", b"Y8  ", b"GREY"}


def _avi_uncompressed(raw: bytes):
    """Frames of an uncompressed AVI as (rgb list, meta), or None if compressed."""
    fmt, frames, fourcc = None, [], None
    video_id = None
    streams = [-1]                                   # counted across the whole file

    def walk(start, end):
        nonlocal fmt, fourcc, video_id
        for cid, data, size in _riff_chunks(raw, start, end):
            if cid in (b"RIFF", b"LIST"):
                walk(data + 4, data + size)
            elif cid == b"strh":
                streams[0] += 1
                if raw[data:data + 4] == b"vids" and video_id is None:
                    video_id = f"{streams[0]:02d}".encode()
            elif cid == b"strf" and fmt is None and video_id is not None:
                bw, bh, _, bits, comp = struct.unpack_from("<iiHHI", raw, data + 4)
                palette = None
                header = struct.unpack_from("<I", raw, data)[0]
                if bits <= 8:
                    n_colours = min(struct.unpack_from("<I", raw, data + 32)[0]
                                    or 1 << bits, max(0, (size - header) // 4))
                    if n_colours:
                        pal = np.frombuffer(raw, dtype=np.uint8, count=4 * n_colours,
                                            offset=data + header)
                        palette = pal.reshape(-1, 4)[:, 2::-1]      # BGRx -> RGB
                fmt = (bw, bh, bits, comp, palette)
                fourcc = struct.pack("<I", comp)
            elif video_id is not None and cid[:2] == video_id and cid[2:] in (b"db", b"dc"):
                frames.append((data, size))

    walk(0, len(raw))
    if fmt is None:
        raise ValueError("no video stream in the AVI")
    w, h, bits, comp, palette = fmt
    gray = fourcc in _AVI_GRAY
    if comp not in (0, 0x20424944) and not gray:      # BI_RGB or 'DIB '
        return None
    rgb_frames = []
    for data, size in frames:
        if size == 0:                                    # a dropped frame repeats
            if rgb_frames:
                rgb_frames.append(rgb_frames[-1])
            continue
        if gray:
            img = np.frombuffer(raw, np.uint8, count=w * abs(h), offset=data)
            img = img.reshape(abs(h), w)
            rgb_frames.append(np.repeat(img[..., None], 3, axis=2))
            continue
        stride = ((w * bits + 31) // 32) * 4
        rows = np.frombuffer(raw, np.uint8, count=stride * abs(h),
                             offset=data).reshape(abs(h), stride)
        if bits == 8 and palette is not None:
            img = palette[rows[:, :w]]
        elif bits in (24, 32):
            img = rows[:, :w * bits // 8].reshape(abs(h), w, bits // 8)[..., 2::-1]
        else:
            return None                                  # 16-bit RGB: leave to OpenCV
        rgb_frames.append(img[::-1] if h > 0 else img)   # positive height = bottom-up
    return rgb_frames, {"codec": "uncompressed", "bits_per_pixel": bits}


def _read_avi(path, keep_colour: bool = False, **kw):
    """AVI video (manual 2.1.1).  RGB is averaged into counts.

    Uncompressed AVIs (RGB, palette or 8-bit grey) are decoded here, bit for
    bit.  Anything else goes through OpenCV's video reader, which brings its
    own codecs -- the manual's "make sure the codec is installed".
    """
    path = Path(path)
    decoded = _avi_uncompressed(path.read_bytes())
    if decoded is None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("this AVI is compressed; reading it needs OpenCV "
                              "(pip install opencv-python)") from exc
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"{path.name}: OpenCV cannot open this AVI (codec missing?)")
        rgb_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb_frames.append(frame[..., ::-1] if frame.ndim == 3 else
                              np.repeat(frame[..., None], 3, axis=2))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        meta = {"codec": struct.pack("<I", codec).decode("latin-1", "replace")}
    else:
        rgb_frames, meta = decoded
    if not rgb_frames:
        raise ValueError(f"{path.name}: no frames could be read")
    rgb = np.stack(rgb_frames)
    meta["frames"] = len(rgb_frames)
    return (rgb.astype(np.float32).mean(axis=3), meta,
            rgb.astype(np.uint8) if keep_colour else None)


READERS: Dict[str, Callable] = {
    # count based (manual 2.1.1)
    ".img": _read_stanford_img,
    ".rtv": _read_stanford_rtv,
    ".spe": _read_spe,
    ".sif": _read_sif,
    ".st2k": _read_sbig,
    ".tsv": _read_tsv,
    ".fit": _read_fits, ".fits": _read_fits,
    ".cih": _read_photron_cih, ".raw": _read_photron_raw,
    ".dat": _read_shimadzu_dat,
    ".im7": _read_lavision_im7,
    ".txt": _read_txt,
    ".streamer": _read_bin,
    ".avi": _read_avi,
    # colour based (manual 2.1.2)
    ".png": _read_picture, ".jpg": _read_picture, ".jpeg": _read_picture,
    ".bmp": _read_picture, ".tif": _read_picture, ".tiff": _read_picture,
    # convenience
    ".gif": _read_picture,
    ".npy": _read_npy,
}

#: the multi-frame types of manual 2.1.4 (plus the convenience ones)
MOVIE_EXTENSIONS = {".cih", ".raw", ".avi", ".dat", ".rtv", ".spe", ".tif", ".tiff",
                    ".sif", ".im7", ".fit", ".fits", ".gif", ".npy"}


def long_path(path) -> str:
    """`path` as a string Windows can open even beyond 260 characters.

    Without long-path support switched on in Windows, a file whose full path
    reaches 260 characters cannot be created or found -- easy to hit with a
    long camera file name inside a OneDrive folder.  The ``\\\\?\\`` prefix
    lifts that limit; elsewhere the path is returned unchanged.
    """
    import os
    text = str(Path(path).absolute())
    if os.name != "nt" or len(text) < 240 or text.startswith("\\\\?\\"):
        return str(path)
    if text.startswith("\\\\"):                          # a network share
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + os.path.normpath(text)


def _natural_key(path) -> list:
    """Sort key that puts shot2 before shot10."""
    name = Path(path).name.lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------
# public loading API
# --------------------------------------------------------------------------
def load(path, *, pixel_size_um: Optional[float] = None, keep_colour: bool = False,
         frames: Optional[slice] = None, rot90: int = 0) -> ImageStack:
    """Load any supported file into an :class:`ImageStack`.

    Parameters
    ----------
    path : str or Path
    pixel_size_um : float, optional
        Image-plane size of one pixel, manual 2.4.  Needed for every result in
        physical units; leave ``None`` to work in pixels.
    keep_colour : bool
        Keep the original RGB data next to the averaged counts (Tile maker's
        ``UseGrayValues = 0``).
    frames : slice, optional
        Select a subset of the frames of a movie file (manual 2.1.4).
    rot90 : int
        Number of counter-clockwise quarter turns applied on load.
    """
    path = Path(path)
    ext = path.suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"unsupported file type '{ext}'. Known: "
                         f"{', '.join(sorted(READERS))}")
    data, meta, colour = reader(Path(long_path(path)), keep_colour=keep_colour)
    if frames is not None:
        data = data[frames]
        colour = colour[frames] if colour is not None else None
    if rot90 % 4:
        data = np.rot90(data, k=rot90, axes=(1, 2))
        if colour is not None:
            colour = np.rot90(colour, k=rot90, axes=(1, 2))
    meta.setdefault("source_file", path.name)
    return ImageStack(np.ascontiguousarray(data, dtype=np.float32), meta, path,
                      pixel_size_um, np.ascontiguousarray(colour) if colour is not None else None)


def frame_files(spec) -> List[Path]:
    """The files of a series stored one frame per file, in order.

    `spec` is a folder (every readable file in it), a glob pattern such as
    ``"shots/run3_*.img"``, or an explicit list of files (kept in the order
    given).  Folders and patterns are sorted by name, numbers numerically,
    so ``shot2`` comes before ``shot10``.  In a folder, a Photron ``.raw``
    that belongs to a ``.cih`` is not listed separately.
    """
    def exists(p):
        return Path(long_path(p)).is_file()

    def plain(p):                                        # drop the long-path prefix
        text = str(p)
        if text.startswith("\\\\?\\UNC\\"):
            return Path("\\\\" + text[8:])
        return Path(text[4:] if text.startswith("\\\\?\\") else text)

    if isinstance(spec, (list, tuple)):
        files = [Path(p) for p in spec]
        missing = [str(p) for p in files if not exists(p)]
        if missing:
            raise FileNotFoundError(f"missing: {', '.join(missing[:5])}"
                                    + (" ..." if len(missing) > 5 else ""))
        return files
    spec = Path(spec)
    if Path(long_path(spec)).is_dir():
        files = [plain(p) for p in Path(long_path(spec)).iterdir()
                 if p.is_file() and p.suffix.lower() in READERS]
        cih_stems = {p.stem.lower() for p in files if p.suffix.lower() == ".cih"}
        files = [p for p in files if not (p.suffix.lower() == ".raw"
                                          and any(p.stem.lower().startswith(s)
                                                  for s in cih_stems))]
        where = f"folder {spec}"
    else:
        files = [Path(p) for p in glob.glob(str(spec)) if exists(p)]
        where = f"pattern {spec}"
    if not files:
        raise FileNotFoundError(f"no readable image files in {where}")
    return sorted(files, key=_natural_key)


def load_stack(paths, **kwargs) -> ImageStack:
    """Several files as one series -- single-frame files, one per delay.

    `paths` is anything :func:`frame_files` accepts: a folder, a glob
    pattern or a list.  Each file contributes its frames in order (one,
    normally).  Keyword arguments go to :func:`load`.  The metadata keeps
    every file name, and ``frames_per_file`` shows where each file starts.
    """
    files = frame_files(paths)
    stacks = [load(p, **kwargs) for p in files]
    shapes = {s.data.shape[1:] for s in stacks}
    if len(shapes) != 1:
        sizes = {f"{h}x{w}" for h, w in shapes}
        raise ValueError(f"the files differ in size ({', '.join(sorted(sizes))}); "
                         f"a series needs one frame size")
    data = np.concatenate([s.data for s in stacks], axis=0)
    first = stacks[0]
    meta = dict(first.metadata)
    meta.update(sources=[str(s.path) for s in stacks],
                frames_per_file=[s.n_frames for s in stacks],
                source_file=f"{first.path.name} (+{len(stacks) - 1} files)")
    colour = (np.concatenate([s.colour for s in stacks])
              if all(s.colour is not None for s in stacks) else None)
    return ImageStack(data, meta, first.path, first.pixel_size_um, colour)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def save_counts(path, image: np.ndarray) -> Path:
    """Write a 2-D count image to one of the count-based output types.

    Streamertools can only *write* its own two count formats (manual 2.1.3);
    ``.npy`` is added here because it is convenient inside Python.
    """
    path = Path(path)
    ext = path.suffix.lower()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("save_counts expects a single 2-D frame")
    if ext == ".txt":
        return write_streamer_txt(path, image)
    if ext == ".streamer":
        return write_streamer_bin(path, image)
    if ext == ".npy":
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, image)
        return path
    raise ValueError(f"'{ext}' is not a count-based output type "
                     f"(use .txt, .streamer or .npy)")


def save_image(path, rgb_or_gray: np.ndarray, jpeg_quality: int = 90) -> Path:
    """Write an 8-bit image (H,W) or (H,W,3) to png/jpg/bmp/tif.

    JPEG quality 90 matches Streamertools (manual 2.1.2).
    """
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb_or_gray)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.convert("RGB").save(path, quality=jpeg_quality)
    else:
        im.save(path)
    return path
