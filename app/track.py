"""Blink an asteroid: every SPHEREx exposure that caught it, each cut out around JPL's predicted position.

IRSA's Moving Object Search Tool (MOST) lists the exposures that contain a body and where JPL Horizons puts it at
each exposure's time. The pixels come from the same S3 mirror as the stars. Each frame gets its own grid centred on
the predicted position, so the asteroid holds still in the middle while the stars stream past. Where the body is
detected, its offset from the prediction is measured, which tests the JPL orbit against SPHEREx.

The MOST lists are stored in cache/track, so bodies that have been looked up open without IRSA.
"""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time

from . import cache, processing, render
from .config import settings
from .discovery import release_of
from .measure import centroid, photometry
from .models import ImageRecord
from .pixels import OffDetector, read_region

log = logging.getLogger(__name__)

PIXEL_ARCSEC = 6.15
FOV_DEG = 0.08
MAX_FRAMES = 16
MOST_URL = "https://irsa.ipac.caltech.edu/cgi-bin/MOST/nph-most"
DETECTORS = ["D1", "D2", "D3", "D4", "D5", "D6"]

BODIES: dict[str, int] = {
    "Ceres": 1,
    "Pallas": 2,
    "Juno": 3,
    "Vesta": 4,
    "Hebe": 6,
    "Iris": 7,
    "Flora": 8,
    "Hygiea": 10,
    "Eunomia": 15,
    "Psyche": 16,
    "Eros": 433,
}


class TrackError(Exception):
    pass


@dataclass
class Sighting:
    ra: float
    dec: float
    mjd: float
    detector: str
    uri: str
    plane_id: str
    vmag: float | None


def _columns(header: str) -> list[tuple[str, int, int]]:
    bars = [i for i, ch in enumerate(header) if ch == "|"]
    return [(header[a + 1:b].strip(), a + 1, b) for a, b in zip(bars, bars[1:])]


def parse_most(text: str) -> list[Sighting]:
    lines = text.splitlines()
    header = next((l for l in lines if l.startswith("|") and "ra_obj" in l), None)
    if header is None:
        raise TrackError("MOST returned no table")
    columns = _columns(header)
    start = lines.index(header) + 1
    sightings = []
    for line in lines[start:]:
        if not line.strip() or line.startswith("|") or line.startswith("\\"):
            continue
        row = {name: line[a:b + 1].strip() for name, a, b in columns}
        try:
            mjd = (float(row["mjd_start"]) + float(row["mjd_end"])) / 2
            vmag = float(row["vmag"]) if row.get("vmag") not in (None, "", "null") else None
            sightings.append(
                Sighting(float(row["ra_obj"]), float(row["dec_obj"]), mjd, row["productid"], row["uri"], row["planeid"], vmag)
            )
        except (KeyError, ValueError):
            continue
    return sorted(sightings, key=lambda s: s.mjd)


def query_most(number: int, end: str) -> list[Sighting]:
    params = {
        "catalog": "spherex",
        "input_type": "name_input",
        "obj_name": f"{20000000 + number}:NID",
        "obs_begin": "2025-03-12",
        "obs_end": end,
        "output_mode": "Brief",
    }
    response = requests.get(MOST_URL, params=params, timeout=600)
    response.raise_for_status()
    return parse_most(response.text)


def stored(name: str) -> list[Sighting] | None:
    data = cache.read_json("track", name.lower())
    return [Sighting(**s) for s in data["sightings"]] if data else None


def store(name: str, sightings: list[Sighting]) -> None:
    cache.write_json("track", name.lower(), {"name": name, "sightings": [s.__dict__ for s in sightings]})


def sightings_for(name: str) -> list[Sighting]:
    found = stored(name)
    if found is not None:
        return found
    if name not in BODIES:
        raise TrackError(f"{name} is not one of the asteroids that can be blinked yet")
    try:
        found = query_most(BODIES[name], Time.now().iso[:10])
    except Exception as error:
        raise TrackError(f"NASA's moving-object search (IRSA MOST) is not answering: {type(error).__name__}") from error
    store(name, found)
    return found


def _even(sightings: list[Sighting], limit: int) -> list[Sighting]:
    if len(sightings) <= limit:
        return sightings
    mjds = np.array([s.mjd for s in sightings])
    picks = np.unique(np.linspace(0, len(sightings) - 1, limit).round().astype(int))
    return [sightings[i] for i in picks if np.isfinite(mjds[i])]


def _frame(sighting: Sighting) -> dict | None:
    centre = SkyCoord(sighting.ra, sighting.dec, unit="deg")
    record = ImageRecord(plane_id=sighting.plane_id, mjd=sighting.mjd, detector=sighting.detector, release=release_of(sighting.uri), uri=sighting.uri)
    half_px = int(math.ceil(FOV_DEG * 3600 / PIXEL_ARCSEC / 2 * math.sqrt(2))) + 2
    try:
        region = read_region(record, centre, centre, half_px, PIXEL_ARCSEC)
    except OffDetector:
        return None
    grid, shape = processing.common_grid(centre, PIXEL_ARCSEC, FOV_DEG)
    aligned = processing.align(region.image, region.bad_mask(), region.wcs, grid, shape)
    aligned = aligned - processing.background(aligned)
    return {"sighting": sighting, "image": aligned, "wavelength_um": region.wavelength_um, "flags": region.flags_at_target}


def track(name: str, detector: str) -> dict:
    if detector not in DETECTORS:
        raise TrackError("unknown detector")
    every = sightings_for(name)
    on_detector = [s for s in every if s.detector == detector]
    if not on_detector:
        raise TrackError(f"SPHEREx has no {detector} exposures of {name} yet")
    chosen = _even(on_detector, MAX_FRAMES)
    with ThreadPoolExecutor(max_workers=max(settings.read_workers, 1)) as pool:
        frames = [f for f in pool.map(_frame, chosen) if f is not None]
    if not frames:
        raise TrackError(f"none of the {detector} exposures of {name} could be read")

    low, high, softening = render.shared_limits([f["image"] for f in frames])
    centre_px = (frames[0]["image"].shape[1] - 1) / 2
    shifts: list[tuple[float, float]] = []
    saturated_hits = 0
    out = []
    for f in frames:
        image = f["image"]
        found = centroid(image, centre_px, centre_px, half=3)
        offset = snr = None
        measured_at = None
        if found is not None:
            _, _, snr = photometry(image, found[0], found[1])
            if snr >= 5:
                dx = (found[0] - centre_px) * PIXEL_ARCSEC
                dy = (found[1] - centre_px) * PIXEL_ARCSEC
                offset = math.hypot(dx, dy)
                shifts.append((dx, dy))
                measured_at = [round(found[0], 2), round(found[1], 2)]
        saturated = any(flag in {"OVERFLOW", "BLOOM", "NONLINEAR"} for flag in f["flags"])
        if offset is not None and saturated:
            saturated_hits += 1
        s = f["sighting"]
        out.append({
            "date": Time(s.mjd, format="mjd").iso[:16],
            "mjd": s.mjd,
            "ra": s.ra,
            "dec": s.dec,
            "vmag": s.vmag,
            "wavelength_um": f["wavelength_um"],
            "offset_arcsec": round(offset, 2) if offset is not None else None,
            "snr": round(snr, 1) if snr is not None else None,
            "saturated": saturated,
            "measured_px": measured_at,
            "size_px": int(image.shape[1]),
            "image": render.data_uri(render.grayscale(image, low, high, softening)),
        })

    count = len(shifts)
    mean = sigma = None
    if count >= 3:
        arr = np.array(shifts)
        centre_shift = arr.mean(axis=0)
        error = arr.std(axis=0, ddof=1) / math.sqrt(count)
        mean = float(math.hypot(*centre_shift))
        sigma = float(math.hypot(*error))
        agree = mean <= 3 * sigma or mean < PIXEL_ARCSEC
        verdict = "the orbit and the image agree" if agree else "the image sits off the predicted orbit"
        summary = (
            f"Detected in {count} of {len(out)} frames. On average it sits {mean:.1f} ± {sigma:.1f}″ from JPL's predicted "
            f"position ({mean / PIXEL_ARCSEC:.2f} SPHEREx pixels): {verdict}."
        )
        if saturated_hits > count / 2:
            label = "indicative"
            summary += f" {name} saturates SPHEREx on {detector}, so positions come from the edges of the blob and are indicative."
        else:
            label = "measured" if count >= 5 else "marginal"
    elif count:
        label = "marginal"
        summary = f"Detected in only {count} of {len(out)} frames; too few to test the orbit."
    else:
        label = "not-measurable"
        summary = f"Not detected above five sigma on {detector}. Try a shorter-wavelength detector, where reflected sunlight is brightest."

    span_days = on_detector[-1].mjd - on_detector[0].mjd
    return {
        "name": name,
        "detector": detector,
        "exposures_total": len(every),
        "exposures_on_detector": len(on_detector),
        "span_days": round(span_days, 1),
        "frames": out,
        "fov_deg": FOV_DEG,
        "pixel_arcsec": PIXEL_ARCSEC,
        "measurement": {"mean_offset_arcsec": mean, "sigma_arcsec": sigma, "label": label, "summary": summary},
        "source": "Exposure list and predicted positions: IRSA Moving Object Search Tool with JPL Horizons. Pixels: SPHEREx via AWS Open Data.",
    }


def available() -> list[str]:
    return [name for name in BODIES if stored(name) is not None]
