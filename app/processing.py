"""Alignment, pass grouping and stacking.

Every frame is reprojected onto one tangent-plane grid before anything is compared; otherwise WCS differences between
frames would look like motion. SPHEREx frames within one pass sample different wavelengths (linear variable filters),
so frames are stacked per detector per pass, never mixed across detectors.
"""

import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from reproject import reproject_interp

PASS_GAP_DAYS = 30.0
PASS_MAX_SPAN_DAYS = 40.0


def common_grid(center: SkyCoord, pixel_arcsec: float, fov_deg: float) -> tuple[WCS, tuple[int, int]]:
    npix = int(round(fov_deg * 3600 / pixel_arcsec))
    npix += 1 - npix % 2
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [center.ra.deg, center.dec.deg]
    wcs.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    wcs.wcs.cdelt = [-pixel_arcsec / 3600, pixel_arcsec / 3600]
    return wcs, (npix, npix)


def align(image: np.ndarray, bad: np.ndarray, wcs: WCS, grid: WCS, shape: tuple[int, int]) -> np.ndarray:
    data = np.where(bad, np.nan, image).astype(np.float64)
    out, footprint = reproject_interp((data, wcs), grid, shape_out=shape)
    out[footprint <= 0] = np.nan
    return out.astype(np.float32)


def group_passes(mjds: list[float]) -> list[list[int]]:
    order = sorted(range(len(mjds)), key=lambda i: mjds[i])
    groups: list[list[int]] = []
    for i in order:
        if groups:
            last = mjds[groups[-1][-1]]
            first = mjds[groups[-1][0]]
            if mjds[i] - last <= PASS_GAP_DAYS and mjds[i] - first <= PASS_MAX_SPAN_DAYS:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def background(image: np.ndarray) -> float:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    for _ in range(3):
        mad = 1.4826 * float(np.median(np.abs(values - median)))
        if mad <= 0:
            break
        keep = np.abs(values - median) < 3 * mad
        if keep.sum() < 10:
            break
        values = values[keep]
        median = float(np.median(values))
    return median


def stack(frames: list[np.ndarray]) -> np.ndarray:
    cube = np.stack(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stacked = np.nanmedian(cube, axis=0)
    return (stacked - background(stacked)).astype(np.float32)
