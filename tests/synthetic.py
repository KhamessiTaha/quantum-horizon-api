"""Synthetic SPHEREx-like Level 2 files: same HDU names and FLAGS header cards as the real ones, no real data."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from app.models import ImageRecord, Resolved
from app.resolve import position_at

FLAG_BITS = {0: "TRANSIENT", 1: "OVERFLOW", 6: "NONFUNC", 21: "SOURCE"}
PASS_STARTS = (60910.0, 61115.0, 61270.0)
FRAMES_PER_PASS = 6
SIZE = 160
SCALE_ARCSEC = 6.15


def barnard() -> Resolved:
    return Resolved(
        query="Barnard's Star", name="Barnard's Star", simbad_id=None, ra=269.452077, dec=4.693365,
        pmra=-801.6, pmdec=10362.4, parallax=546.9759, epoch="J2000.0", source="SIMBAD (built-in copy)", featured="barnard",
    )


def _frame_wcs(center, rng) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.ra.deg + rng.normal(0, 0.01), center.dec.deg + rng.normal(0, 0.01)]
    wcs.wcs.crpix = [SIZE / 2 + rng.uniform(-20, 20), SIZE / 2 + rng.uniform(-20, 20)]
    angle = np.radians(rng.uniform(0, 360))
    s = SCALE_ARCSEC / 3600
    wcs.wcs.cd = [[-s * np.cos(angle), s * np.sin(angle)], [s * np.sin(angle), s * np.cos(angle)]]
    return wcs


def _psf(shape, x, y, flux, sigma=1.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return flux / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def write_dataset(folder: Path, seed: int = 3, off_detector_one: bool = True) -> tuple[list[ImageRecord], dict[str, Path]]:
    rng = np.random.default_rng(seed)
    target = barnard()
    reference = position_at(target, 61041.0)
    field = [(rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(20, 300)) for _ in range(25)]
    records, files = [], {}
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for p, start in enumerate(PASS_STARTS):
        for k in range(FRAMES_PER_PASS):
            mjd = start + 1.3 * k
            star = position_at(target, mjd)
            wcs = _frame_wcs(reference, rng)
            if off_detector_one and count == 0:
                wcs.wcs.crval = [reference.ra.deg + 2.0, reference.dec.deg]
            image = rng.normal(0, 1.0, (SIZE, SIZE)) + 5.0
            for dra, ddec, flux in field:
                x, y = wcs.world_to_pixel_values(reference.ra.deg + dra / np.cos(np.radians(reference.dec.deg)), reference.dec.deg + ddec)
                image += _psf(image.shape, float(x), float(y), flux)
            sx, sy = (float(v) for v in wcs.world_to_pixel(star))
            image += _psf(image.shape, sx, sy, 2000.0)
            flags = np.zeros((SIZE, SIZE), dtype=np.int32)
            if np.isfinite(sx) and 0 <= sx < SIZE and 0 <= sy < SIZE:
                flags[int(round(sy)), int(round(sx))] |= 1 << 21
            flags[rng.integers(0, SIZE, 30), rng.integers(0, SIZE, 30)] |= 1 << 0
            header = wcs.to_header()
            flag_header = fits.Header()
            for bit, name in FLAG_BITS.items():
                flag_header[f"MSKN{bit:04d}"] = name
                flag_header[f"MSKM{bit:04d}"] = 1 << bit
            hdul = fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(image.astype(np.float32), header, name="IMAGE"),
                fits.ImageHDU(flags, flag_header, name="FLAGS"),
                fits.ImageHDU(np.ones((SIZE, SIZE), dtype=np.float32), name="VARIANCE"),
                fits.ImageHDU(np.full((SIZE, SIZE), 5.0, dtype=np.float32), name="ZODI"),
            ])
            uri = f"ibe/data/spherex/qr{2 if p < 2 else 3}/level2/synthetic/{count:03d}/level2_{count:03d}_D2_spx.fits"
            path = folder / f"frame_{count:03d}.fits"
            hdul.writeto(path, overwrite=True)
            files[uri] = path
            records.append(ImageRecord(plane_id=f"plane{count:03d}", mjd=mjd, detector="D2", release=f"qr{2 if p < 2 else 3}", uri=uri))
            count += 1
    return records, files
