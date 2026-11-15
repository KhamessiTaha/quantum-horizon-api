"""Assemble the viewer payload: pass stacks as PNGs, the difference image, and the measurements.

Measurement rules come from the research spikes on real data: positions are fitted per frame and combined, a pass whose
frames disagree is reported as uncertain rather than silently averaged, and a shift smaller than three times its own
uncertainty is reported as no detection rather than as motion.
"""

import logging
import time

import numpy as np
from astropy.coordinates import SkyCoord

from . import cache, measure, render
from .models import (
    DetectorId,
    FramesResponse,
    Grid,
    Measurement,
    Motion,
    PassInfo,
    PassMeasurement,
    Resolved,
)
from .resolve import position_at
from .view import NoCoverage, PIXEL_ARCSEC, ViewData, frames_in, iso, prepare

log = logging.getLogger(__name__)

__all__ = ["NoCoverage", "build_view", "measure_view"]

ORDER = ["measured", "marginal", "not-measurable"]


def _grid_xy(view: ViewData, coord: SkyCoord) -> tuple[float, float] | None:
    x, y = (float(v) for v in view.grid.world_to_pixel(coord))
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def _pass_mid(view: ViewData, group: list[int]) -> float:
    return float(np.mean([view.records[i].mjd for i in group]))


def _predicted(view: ViewData, mjd: float) -> SkyCoord:
    if view.target.source == "user coordinates":
        return view.center
    return position_at(view.target, mjd)


def _summary(motion: Motion, first_date: str, last_date: str, name: str) -> str:
    if motion.confidence == "measured":
        agree = ""
        if motion.difference_arcsec is not None and motion.sigma_arcsec:
            agree = (
                " That agrees with the catalog within the uncertainty."
                if abs(motion.difference_arcsec) <= 2 * motion.sigma_arcsec
                else " That differs from the catalog by more than twice the uncertainty, so treat it with caution."
            )
        return (
            f"{name} moved {motion.measured_arcsec:.2f} ± {motion.sigma_arcsec:.2f}″ between {first_date} and "
            f"{last_date} ({motion.baseline_years:.2f} yr, {motion.pixels:.2f} pixels). The catalog predicts "
            f"{motion.predicted_arcsec:.2f}″.{agree}"
        )
    if motion.confidence == "below-noise":
        return (
            f"No motion detected above the noise: the shift between {first_date} and {last_date} is "
            f"{motion.measured_arcsec:.2f} ± {motion.sigma_arcsec:.2f}″, less than three times its uncertainty. "
            f"The catalog predicts {motion.predicted_arcsec:.2f}″, which this baseline cannot yet resolve."
        )
    if motion.confidence == "marginal":
        return (
            "Marginal detection: the source is too faint in single frames for a trustworthy position, so no motion is "
            f"claimed. The catalog predicts {motion.predicted_arcsec:.2f}″ over this baseline."
        )
    return (
        "Not measurable: the source is not detected in individual frames on this detector, so no position and no "
        "motion can be reported. Try another detector, or a wider field of view."
    )


def measure_view(view: ViewData) -> Measurement:
    fits: list[measure.PassFit] = []
    results: list[PassMeasurement] = []

    for index, group in enumerate(view.groups, start=1):
        mid = _pass_mid(view, group)
        predicted = _predicted(view, mid)
        guess = _grid_xy(view, predicted)
        if guess is None:
            guess = ((view.shape[1] - 1) / 2, (view.shape[0] - 1) / 2)
        fit = measure.fit_frames(frames_in(view, group), group, guess[0], guess[1])
        fits.append(fit)

        measured_sky = view.sky(fit.x, fit.y) if fit.x is not None else None
        offset = (
            round(float(measured_sky.separation(predicted).arcsec), 3)
            if measured_sky is not None and view.target.source != "user coordinates"
            else None
        )
        results.append(
            PassMeasurement(
                index=index,
                confidence=fit.confidence,
                measured_px=view.png_px(measured_sky) if measured_sky is not None else None,
                measured_ra=round(float(measured_sky.ra.deg), 6) if measured_sky is not None else None,
                measured_dec=round(float(measured_sky.dec.deg), 6) if measured_sky is not None else None,
                sigma_arcsec=round(fit.sigma_px * PIXEL_ARCSEC, 3) if fit.sigma_px is not None else None,
                scatter_arcsec=round(fit.scatter_px * PIXEL_ARCSEC, 3) if fit.scatter_px is not None else None,
                snr=round(fit.snr, 2) if fit.snr is not None else None,
                flux=round(fit.flux, 5) if fit.flux is not None else None,
                flux_err=round(fit.flux_err, 5) if fit.flux_err is not None else None,
                frames_fit=fit.frames_fit,
                frames_rejected=fit.frames_rejected,
                offset_from_predicted_arcsec=offset,
            )
        )

    located = [i for i, fit in enumerate(fits) if fit.x is not None]
    motion: Motion | None = None
    if len(located) >= 2:
        first_i, last_i = located[0], located[-1]
        first, last = fits[first_i], fits[last_i]
        first_mid, last_mid = _pass_mid(view, view.groups[first_i]), _pass_mid(view, view.groups[last_i])
        baseline = (last_mid - first_mid) / 365.25
        found = measure.separation(first, last, PIXEL_ARCSEC)
        confidence = measure.grade_motion(found, first, last)
        predicted_first, predicted_last = _predicted(view, first_mid), _predicted(view, last_mid)
        predicted_arcsec = float(predicted_first.separation(predicted_last).arcsec)
        predicted_pa = float(predicted_first.position_angle(predicted_last).deg)
        distance, sigma, angle = found if found else (None, None, None)
        motion = Motion(
            confidence=confidence,
            baseline_years=round(baseline, 3),
            from_pass=first_i + 1,
            to_pass=last_i + 1,
            measured_arcsec=round(distance, 3) if distance is not None else None,
            sigma_arcsec=round(sigma, 3) if sigma is not None else None,
            measured_pa_deg=round(angle, 1) if angle is not None else None,
            measured_arcsec_per_year=round(distance / baseline, 3) if distance is not None and baseline > 0 else None,
            predicted_arcsec=round(predicted_arcsec, 3),
            predicted_pa_deg=round(predicted_pa, 1),
            predicted_arcsec_per_year=round(predicted_arcsec / baseline, 3) if baseline > 0 else None,
            difference_arcsec=round(distance - predicted_arcsec, 3) if distance is not None else None,
            significance=round(distance / sigma, 2) if distance is not None and sigma else None,
            pixels=round(distance / PIXEL_ARCSEC, 3) if distance is not None else None,
            summary="",
        )
        motion.summary = _summary(
            motion,
            iso(first_mid),
            iso(last_mid),
            view.target.name if view.target.source != "user coordinates" else "The source at this position",
        )

    saturation = view.saturation_flags
    snrs = [r.snr for r in results if r.snr is not None]
    if saturation:
        photometry_confidence = "indicative"
        note = (
            f"Saturation flags on the source ({', '.join(saturation)}): brightness is indicative only, and the true "
            "change may be larger than measured."
        )
    elif snrs and max(snrs) >= measure.MEASURED_SNR:
        photometry_confidence = "measured"
        note = None
    elif snrs and max(snrs) >= measure.MARGINAL_SNR:
        photometry_confidence = "marginal"
        note = "Low signal-to-noise: brightness changes below about 20 per cent are not meaningful here."
    else:
        photometry_confidence = "not-measurable"
        note = "No usable detection, so no brightness is reported."

    return Measurement(
        passes=results,
        motion=motion,
        photometry_confidence=photometry_confidence,
        photometry_note=note,
        aperture_arcsec=round(measure.APERTURE_PX * PIXEL_ARCSEC, 2),
        method=(
            f"Positions fitted on each aligned frame inside a {2 * measure.BOX_HALF_PX + 1} pixel box, frames below "
            f"S/N {measure.MIN_FRAME_SNR:.0f} or disagreeing by more than {measure.CLIP_SIGMA:.0f}σ rejected, the pass "
            "position taken as the median and its uncertainty from the scatter of the surviving frames."
        ),
    )


def build_view(target: Resolved, detector: DetectorId, fov_deg: float, refresh: bool = False) -> FramesResponse:
    started = time.perf_counter()
    view = prepare(target, detector, fov_deg, refresh=refresh)

    cached = None if refresh else cache.read_json("views", view.key)
    if cached:
        response = FramesResponse(**cached)
        response.from_cache = True
        response.elapsed_s = round(time.perf_counter() - started, 3)
        return response

    notes = list(view.notes)
    low, high, softening = render.shared_limits(view.stacks)
    passes: list[PassInfo] = []
    for index, (group, image) in enumerate(zip(view.groups, view.stacks), start=1):
        mjds = [view.records[i].mjd for i in group]
        waves = [view.wavelengths[i] for i in group if view.wavelengths[i] is not None]
        mid = _pass_mid(view, group)
        name = f"{view.key}_pass{index}.png"
        png = render.grayscale(image, low, high, softening)
        cache.write_bytes("images", name, png)
        passes.append(
            PassInfo(
                index=index,
                start=iso(min(mjds)),
                end=iso(max(mjds)),
                mid_mjd=round(mid, 5),
                mid_date=iso(mid),
                frames_used=len(group),
                frames_total=sum(1 for r in view.records if min(mjds) <= r.mjd <= max(mjds)),
                wavelength_min_um=round(min(waves), 4) if waves else None,
                wavelength_max_um=round(max(waves), 4) if waves else None,
                predicted_px=view.png_px(_predicted(view, mid)) if view.target.source != "user coordinates" else None,
                image=render.data_uri(png),
            )
        )

    difference_image = None
    if len(view.stacks) >= 2:
        name = f"{view.key}_diff.png"
        png = render.difference(view.stacks[-1], view.stacks[0])
        cache.write_bytes("images", name, png)
        difference_image = render.data_uri(png)
    else:
        notes.append("Only one pass covers this position so far, so there is nothing to compare yet.")

    measurement = measure_view(view)
    if measurement.photometry_note:
        notes.append(measurement.photometry_note)
    if view.target.source != "user coordinates" and (view.target.pmra or view.target.pmdec):
        notes.append(f"Predicted positions use {view.target.source} proper motion, propagated from {view.target.epoch} to each pass.")

    response = FramesResponse(
        target=view.target,
        detector=view.detector,
        grid=Grid(
            ra=view.center.ra.deg,
            dec=view.center.dec.deg,
            width_px=view.shape[1],
            height_px=view.shape[0],
            pixel_arcsec=PIXEL_ARCSEC,
            fov_deg=fov_deg,
        ),
        passes=passes,
        difference_image=difference_image,
        frames=view.infos,
        releases=view.releases,
        notes=notes,
        measurement=measurement,
        elapsed_s=round(time.perf_counter() - started, 3),
        from_cache=False,
    )
    cache.write_json("views", view.key, response.model_dump())
    return response
