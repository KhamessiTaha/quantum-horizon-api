"""Read the pixels around a position from one SPHEREx Level 2 spectral image.

Primary path: byte-range reads of only the needed rows from the public S3 mirror (about 6x faster than the cutout
service in the spike, identical pixels). Fallback: the IRSA cutout service. Each region is cached on disk.
"""

import io
import json
import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import requests
from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS

from . import cache
from .config import settings
from .models import ImageRecord

warnings.simplefilter("ignore", AstropyWarning)
astropy_log.setLevel("WARNING")
log = logging.getLogger(__name__)

REJECT_FLAGS = {"TRANSIENT", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD", "OUTLIER", "STREAK", "SNOWBALL"}


class OffDetector(Exception):
    pass


@dataclass
class Region:
    image: np.ndarray
    variance: np.ndarray
    flags: np.ndarray
    header: fits.Header
    flag_names: dict[int, str]
    wavelength_um: float | None
    flags_at_target: list[str]
    source: str

    @property
    def wcs(self) -> WCS:
        return WCS(self.header).celestial

    def bad_mask(self) -> np.ndarray:
        bits = 0
        for bit, name in self.flag_names.items():
            if name in REJECT_FLAGS:
                bits |= 1 << bit
        return (self.flags & bits) != 0


def s3_url(uri: str) -> str:
    marker = "/spherex/"
    index = uri.find(marker)
    if index < 0:
        raise ValueError(f"cannot map {uri} to the S3 mirror")
    return f"s3://{settings.s3_bucket}/{uri[index + len(marker):]}"


def cutout_url(uri: str, center: SkyCoord, size_deg: float) -> str:
    base = uri if uri.startswith("http") else settings.irsa_root + uri.lstrip("/")
    return f"{base}?center={center.ra.deg:.7f},{center.dec.deg:.7f}&size={size_deg:.4f}"


def flag_names(header: fits.Header) -> dict[int, str]:
    names = {}
    for card in header.cards:
        if card.keyword.startswith("MSKN") and card.keyword[4:].isdigit():
            names[int(card.keyword[4:])] = str(card.value).strip()
    return names


def _names(mask: int, table: dict[int, str]) -> list[str]:
    return [table.get(bit, f"BIT{bit}") for bit in range(32) if mask >> bit & 1]


def _wavelength(header: fits.Header, hdul: fits.HDUList, x: float, y: float) -> float | None:
    try:
        spectral = WCS(header=header, fobj=hdul, key="W")
        spectral.sip = None
        wavelength, _ = spectral.pixel_to_world(x, y)
        value = float(wavelength.to_value("um"))
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _extract(hdul: fits.HDUList, center: SkyCoord, target: SkyCoord, half_px: int, source: str) -> Region:
    image_hdu = hdul["IMAGE"]
    header = image_hdu.header.copy()
    sky = WCS(header).celestial
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    cx, cy = (float(v) for v in sky.world_to_pixel(center))
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise OffDetector("position does not project onto this image")
    xi, yi = int(round(cx)), int(round(cy))
    x0, x1 = max(xi - half_px, 0), min(xi + half_px + 1, nx)
    y0, y1 = max(yi - half_px, 0), min(yi + half_px + 1, ny)
    if x1 - x0 < half_px or y1 - y0 < half_px:
        raise OffDetector("position is at or beyond the detector edge")

    image = np.asarray(image_hdu.section[y0:y1, x0:x1], dtype=np.float32)
    flags = np.asarray(hdul["FLAGS"].section[y0:y1, x0:x1]).astype(np.int64)
    variance = np.asarray(hdul["VARIANCE"].section[y0:y1, x0:x1], dtype=np.float32)
    zodi = np.asarray(hdul["ZODI"].section[y0:y1, x0:x1], dtype=np.float32)
    names = flag_names(hdul["FLAGS"].header)

    tx, ty = (float(v) for v in sky.world_to_pixel(target))
    at_target: list[str] = []
    wavelength = None
    if np.isfinite(tx) and np.isfinite(ty):
        wavelength = _wavelength(header, hdul, tx, ty)
        lx, ly = int(round(tx)) - x0, int(round(ty)) - y0
        box = flags[max(ly - 1, 0):ly + 2, max(lx - 1, 0):lx + 2]
        if box.size:
            at_target = _names(int(np.bitwise_or.reduce(box.ravel())), names)

    header["CRPIX1"] = float(header["CRPIX1"]) - x0
    header["CRPIX2"] = float(header["CRPIX2"]) - y0
    header["NAXIS1"], header["NAXIS2"] = x1 - x0, y1 - y0
    return Region(
        image=image - zodi,
        variance=variance,
        flags=flags,
        header=header,
        flag_names=names,
        wavelength_um=wavelength,
        flags_at_target=at_target,
        source=source,
    )


def _read_s3(record: ImageRecord, center: SkyCoord, target: SkyCoord, half_px: int) -> Region:
    options = {"anon": True, "default_block_size": settings.s3_block_bytes}
    with fits.open(s3_url(record.uri), use_fsspec=True, fsspec_kwargs=options, lazy_load_hdus=True) as hdul:
        return _extract(hdul, center, target, half_px, "s3")


def _read_cutout(record: ImageRecord, center: SkyCoord, target: SkyCoord, half_px: int, pixel_arcsec: float) -> Region:
    size_deg = (2 * half_px + 3) * pixel_arcsec / 3600
    response = requests.get(cutout_url(record.uri, center, size_deg), timeout=120)
    if response.status_code != 200 or response.content[:6] != b"SIMPLE":
        raise OSError(f"cutout service HTTP {response.status_code}")
    with fits.open(io.BytesIO(response.content)) as hdul:
        return _extract(hdul, center, target, half_px, "cutout")


def _save(name: str, region: Region) -> None:
    cache.write_arrays(
        "regions",
        name,
        image=region.image,
        variance=region.variance,
        flags=region.flags,
        header=np.array(region.header.tostring()),
        meta=np.array(
            json.dumps(
                {
                    "flag_names": {str(k): v for k, v in region.flag_names.items()},
                    "wavelength_um": region.wavelength_um,
                    "flags_at_target": region.flags_at_target,
                    "source": region.source,
                }
            )
        ),
    )


def _load(name: str) -> Region | None:
    stored = cache.read_arrays("regions", name)
    if stored is None:
        return None
    meta = json.loads(str(stored["meta"]))
    return Region(
        image=stored["image"],
        variance=stored["variance"],
        flags=stored["flags"],
        header=fits.Header.fromstring(str(stored["header"])),
        flag_names={int(k): v for k, v in meta["flag_names"].items()},
        wavelength_um=meta["wavelength_um"],
        flags_at_target=list(meta["flags_at_target"]),
        source="cache",
    )


def read_region(record: ImageRecord, center: SkyCoord, target: SkyCoord, half_px: int, pixel_arcsec: float) -> Region:
    name = cache.key(
        "region", record.uri, f"{center.ra.deg:.6f}", f"{center.dec.deg:.6f}", half_px, f"{target.ra.deg:.6f}", f"{target.dec.deg:.6f}"
    )
    cached = _load(name)
    if cached is not None:
        return cached

    errors = []
    for attempt in range(settings.read_attempts):
        try:
            region = _read_s3(record, center, target, half_px)
            _save(name, region)
            return region
        except OffDetector:
            raise
        except Exception as error:
            errors.append(f"s3: {type(error).__name__}: {str(error)[:120]}")
            time.sleep(0.5 * (attempt + 1))
    try:
        region = _read_cutout(record, center, target, half_px, pixel_arcsec)
        _save(name, region)
        return region
    except OffDetector:
        raise
    except Exception as error:
        errors.append(f"cutout: {type(error).__name__}: {str(error)[:120]}")
    raise OSError(" | ".join(errors))
