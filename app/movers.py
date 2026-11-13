"""Search a field for anything that moved between passes.

This is the Pluto method: find every source in each pass, link them into tracks, fit a straight line in time, and keep
only the tracks whose motion is significant. The rules below come from running the detector over 150 synthetic fields
in the research phase, where they brought false alarms down from 0.21 to 0.01 per field:

- a position error floor of 0.1 pixel, so a bright source cannot claim impossible precision
- blended sources are set aside rather than measured
- a track lying on a catalogued star that is not moving is dropped, because that is the star, not a discovery
- nothing is called unexplained without a catalogue check having actually run

Detection limit on one year of data: about 6 arcseconds per year for a bright source, about 12 for a faint one. A
distant Solar System body would not be found this way, because the Earth's own orbit swings it far outside the search
box between passes; what this finds is nearby stars and brown dwarfs.
"""

import logging
import math
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import OptimizeWarning, curve_fit

from . import catalog as gaia
from .view import PIXEL_ARCSEC, ViewData

warnings.simplefilter("ignore", OptimizeWarning)
log = logging.getLogger(__name__)

PSF_SIGMA = 1.0
DETECT_SNR = 5.0
LINK_RADIUS_PX = 4.0
PREDICT_RADIUS_PX = 1.5
MOTION_SIGMA = 5.0
CATALOG_MATCH_PX = 1.5
CATALOG_PM_TOLERANCE = 0.35
POSITION_FLOOR_PX = 0.1
STATIC_PX_PER_YR = 0.1
ISOLATION_PX = 2.5
MIN_PASSES = 3
MAX_TRACKS = 40


def _gaussian(coords, amplitude, xc, yc, offset):
    x, y = coords
    return amplitude * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * PSF_SIGMA**2)) + offset


def detect(image: np.ndarray) -> list[dict]:
    """Every source in one pass stack, with a fitted position and its uncertainty."""
    finite = np.isfinite(image)
    if finite.sum() < 25:
        return []
    filled = np.where(finite, image, np.nanmedian(image[finite]))
    noise = 1.4826 * float(np.median(np.abs(filled - np.median(filled))))
    if noise <= 0:
        return []
    smooth = gaussian_filter(filled, PSF_SIGMA)
    snr_map = smooth / (noise / (2 * math.sqrt(math.pi) * PSF_SIGMA))
    peaks = (snr_map == maximum_filter(snr_map, size=3)) & (snr_map > DETECT_SNR)

    sources: list[dict] = []
    half = 3
    for yi, xi in zip(*np.nonzero(peaks)):
        if xi < half or yi < half or xi >= image.shape[1] - half or yi >= image.shape[0] - half:
            continue
        box = filled[yi - half : yi + half + 1, xi - half : xi + half + 1]
        yy, xx = np.mgrid[yi - half : yi + half + 1, xi - half : xi + half + 1]
        try:
            params, covariance = curve_fit(
                _gaussian,
                (xx.ravel(), yy.ravel()),
                box.ravel(),
                p0=[float(box.max()), float(xi), float(yi), 0.0],
                sigma=np.full(box.size, noise),
                absolute_sigma=True,
                bounds=([0, xi - 1.5, yi - 1.5, -np.inf], [np.inf, xi + 1.5, yi + 1.5, np.inf]),
                maxfev=2000,
            )
        except Exception:
            continue
        errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        sources.append(
            {
                "x": float(params[1]),
                "y": float(params[2]),
                "ex": float(math.hypot(errors[1], POSITION_FLOOR_PX)),
                "ey": float(math.hypot(errors[2], POSITION_FLOOR_PX)),
                "flux": float(params[0] * 2 * math.pi * PSF_SIGMA**2),
                "snr": float(snr_map[yi, xi]),
                "blanked": bool(~finite[yi, xi]),
            }
        )
    return sources


def _nearest(sources: list[dict], x: float, y: float, radius: float, used: set[int]) -> int | None:
    best, best_distance = None, radius
    for index, source in enumerate(sources):
        if index in used:
            continue
        distance = math.hypot(source["x"] - x, source["y"] - y)
        if distance < best_distance:
            best, best_distance = index, distance
    return best


def fit_motion(points: list[tuple[float, float, float, float, float]]) -> dict:
    """Weighted straight-line fit of position against time, in pixels per year."""
    t = np.array([p[0] for p in points])
    result: dict = {}
    chi2 = 0.0
    for axis, value_index, error_index in (("x", 1, 3), ("y", 2, 4)):
        values = np.array([p[value_index] for p in points])
        weights = 1 / np.array([p[error_index] for p in points]) ** 2
        design = np.vstack([np.ones_like(t), t]).T
        covariance = np.linalg.inv(design.T @ (design * weights[:, None]))
        coefficients = covariance @ (design.T @ (weights * values))
        chi2 += float(np.sum(weights * (values - design @ coefficients) ** 2))
        result[axis] = float(coefficients[0])
        result[f"v{axis}"] = float(coefficients[1])
        result[f"ev{axis}"] = float(math.sqrt(covariance[1, 1]))
    speed = math.hypot(result["vx"], result["vy"])
    speed_error = math.hypot(result["vx"] * result["evx"], result["vy"] * result["evy"]) / max(speed, 1e-9)
    result.update(
        {
            "speed": speed,
            "speed_error": speed_error,
            "significance": speed / max(speed_error, 1e-9),
            "chi2_dof": chi2 / max(2 * len(points) - 4, 1),
        }
    )
    return result


def link(detections: list[list[dict]], years: list[float]) -> list[dict]:
    """Chain detections across passes: pair the first two, then follow the line into every later pass."""
    tracks: list[dict] = []
    used = [set() for _ in detections]
    for first in detections[0]:
        second_index = _nearest(detections[1], first["x"], first["y"], LINK_RADIUS_PX, used[1])
        if second_index is None:
            continue
        second = detections[1][second_index]
        step = years[1] - years[0]
        if step <= 0:
            continue
        vx = (second["x"] - first["x"]) / step
        vy = (second["y"] - first["y"]) / step
        chain = [(0, first), (1, second)]
        matched = {1: second_index}
        for later in range(2, len(detections)):
            gap = years[later] - years[0]
            index = _nearest(
                detections[later], first["x"] + vx * gap, first["y"] + vy * gap, PREDICT_RADIUS_PX, used[later]
            )
            if index is None:
                continue
            chain.append((later, detections[later][index]))
            matched[later] = index
        if len(chain) < MIN_PASSES:
            continue
        for pass_index, source_index in matched.items():
            used[pass_index].add(source_index)

        crowded = any(
            other is not source
            and math.hypot(other["x"] - source["x"], other["y"] - source["y"]) < ISOLATION_PX
            for pass_index, source in chain
            for other in detections[pass_index]
        )
        points = [(years[pass_index], s["x"], s["y"], s["ex"], s["ey"]) for pass_index, s in chain]
        motion = fit_motion(points)
        motion.update(
            {
                "crowded": crowded,
                "snr": float(np.median([s["snr"] for _, s in chain])),
                "passes": [pass_index + 1 for pass_index, _ in chain],
                "positions": {pass_index + 1: (s["x"], s["y"]) for pass_index, s in chain},
                "fluxes": [s["flux"] for _, s in chain],
            }
        )
        tracks.append(motion)
    return tracks


def _catalogue_tracks(view: ViewData, years: list[float], mids: list[float]) -> tuple[list[dict], str, str | None]:
    """Every Gaia source as a straight line in grid pixels, fitted the same way as the detections."""
    radius = view.fov_deg * 0.75
    try:
        rows, provider = gaia.sources(view.center.ra.deg, view.center.dec.deg, radius)
    except gaia.CatalogError as error:
        return [], "none", str(error)

    entries: list[dict] = []
    per_pass = [gaia.at_epoch(rows, mid) for mid in mids]
    for order, row in enumerate(rows):
        xs, ys = [], []
        for coords in per_pass:
            if not len(coords):
                continue
            x, y = (float(v) for v in view.grid.world_to_pixel(coords[order]))
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            continue
        slope_x = np.polyfit(years[: len(xs)], xs, 1)
        slope_y = np.polyfit(years[: len(ys)], ys, 1)
        entries.append(
            {
                "source_id": row["source_id"],
                "g_mag": row["g_mag"],
                "x": float(slope_x[1]),
                "y": float(slope_y[1]),
                "vx": float(slope_x[0]),
                "vy": float(slope_y[0]),
            }
        )
    return entries, provider, None


def search(view: ViewData) -> dict:
    """Find everything that moved in this field, and say honestly what each one is."""
    if len(view.groups) < MIN_PASSES:
        return {
            "candidates": [],
            "catalogue": "none",
            "notes": [
                f"{len(view.groups)} pass covers this position; the search needs at least {MIN_PASSES} to tell motion "
                "from noise."
            ],
            "limit_arcsec_per_year": None,
            "searched": 0,
            "rejected": {},
        }

    mids = [float(np.mean([view.records[i].mjd for i in group])) for group in view.groups]
    years = [(mid - mids[0]) / 365.25 for mid in mids]
    baseline = years[-1] or 1.0
    detections = [detect(stack) for stack in view.stacks]
    found = sum(len(d) for d in detections)
    tracks = link(detections, years)
    entries, provider, catalogue_error = _catalogue_tracks(view, years, mids)

    rejected = {"below the motion threshold": 0, "blended with a neighbour": 0, "a catalogued star that is not moving": 0}
    candidates: list[dict] = []
    mid_year = float(np.mean(years))

    for track in tracks:
        if track["significance"] < MOTION_SIGMA or track["chi2_dof"] > 9:
            rejected["below the motion threshold"] += 1
            continue
        if track["crowded"]:
            rejected["blended with a neighbour"] += 1
            continue
        mx = track["x"] + track["vx"] * mid_year
        my = track["y"] + track["vy"] * mid_year
        if any(
            math.hypot(entry["vx"], entry["vy"]) < STATIC_PX_PER_YR
            and math.hypot(entry["x"] + entry["vx"] * mid_year - mx, entry["y"] + entry["vy"] * mid_year - my)
            < CATALOG_MATCH_PX
            for entry in entries
        ):
            rejected["a catalogued star that is not moving"] += 1
            continue

        match = None
        for entry in entries:
            follows = all(
                math.hypot(
                    entry["x"] + entry["vx"] * year - (track["x"] + track["vx"] * year),
                    entry["y"] + entry["vy"] * year - (track["y"] + track["vy"] * year),
                )
                < CATALOG_MATCH_PX
                for year in years
            )
            entry_speed = math.hypot(entry["vx"], entry["vy"])
            if follows and abs(entry_speed - track["speed"]) < CATALOG_PM_TOLERANCE * max(entry_speed, 0.5):
                match = entry
                break

        if match is not None:
            label, explanation = "known", f"Gaia DR3 {match['source_id']}"
        elif provider == "none":
            label, explanation = "unchecked", "No catalogue check ran, so this is not claimed as unexplained."
        else:
            label, explanation = "unexplained", "No Gaia DR3 source follows this track."

        sky = view.sky(mx, my)
        candidates.append(
            {
                "label": label,
                "explanation": explanation,
                "catalogue_match": None if match is None else match["source_id"],
                "catalogue_g_mag": None if match is None else match["g_mag"],
                "speed_arcsec_per_year": round(track["speed"] * PIXEL_ARCSEC, 3),
                "speed_error_arcsec_per_year": round(track["speed_error"] * PIXEL_ARCSEC, 3),
                "catalogue_speed_arcsec_per_year": (
                    None if match is None else round(math.hypot(match["vx"], match["vy"]) * PIXEL_ARCSEC, 3)
                ),
                "position_angle_deg": round((math.degrees(math.atan2(-track["vx"], track["vy"])) + 360) % 360, 1),
                "significance": round(track["significance"], 1),
                "snr": round(track["snr"], 1),
                "ra": round(float(sky.ra.deg), 6),
                "dec": round(float(sky.dec.deg), 6),
                "passes": track["passes"],
                "track_px": {
                    str(index): (
                        round(track["x"] + track["vx"] * years[index - 1], 2),
                        round(view.shape[0] - 1 - (track["y"] + track["vy"] * years[index - 1]), 2),
                    )
                    for index in range(1, len(view.groups) + 1)
                },
            }
        )

    candidates.sort(key=lambda c: (c["label"] != "unexplained", -c["speed_arcsec_per_year"]))
    limit = round(6.0 / max(baseline, 0.2), 1)
    notes = [
        f"Tracks are kept when their motion is at least {MOTION_SIGMA:.0f} times its own uncertainty.",
        f"Over this {baseline:.2f} year baseline the search finds objects moving faster than roughly {limit}″ per year "
        "when bright, and about twice that when faint. Slower objects are invisible until SPHEREx has observed for "
        "longer.",
        "A distant Solar System body would not appear here: the Earth's own orbit moves it hundreds of arcseconds "
        "between passes, far outside this field. What this finds is nearby stars and brown dwarfs.",
    ]
    if provider == "none":
        notes.append(f"Gaia could not be reached, so nothing is labelled unexplained. {catalogue_error or ''}".strip())
    return {
        "candidates": candidates[:MAX_TRACKS],
        "catalogue": provider,
        "notes": notes,
        "limit_arcsec_per_year": limit,
        "searched": found,
        "rejected": {reason: count for reason, count in rejected.items() if count},
    }
