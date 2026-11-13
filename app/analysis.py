"""Spectra and light curves from the aligned cube.

SPHEREx images one position many times per pass, each time through a different part of its linear variable filter, so a
pass holds a whole spectrum rather than one brightness. Two consequences shape everything here: a time series must be
built inside narrow wavelength bands, otherwise the filter's wavelength change masquerades as variability, and a
spectrum can be rebuilt for each pass and compared across a year.
"""

import math
import time

import numpy as np

from . import flags as flags_module, measure
from .models import (
    LightCurveBand,
    LightCurvePoint,
    LightCurveResponse,
    SpectrumPoint,
    SpectrumResponse,
    SpectrumSeries,
)
from .view import PIXEL_ARCSEC, ViewData, iso

SPECTRUM_BINS = 24
LIGHTCURVE_BANDS = 4
MIN_BAND_POINTS = 3
SUSPECT_MASKED_FRACTION = 0.3


def _aperture_mask(shape: tuple[int, int], x: float, y: float) -> np.ndarray:
    ys, xs = np.indices(shape)
    return np.hypot(xs - x, ys - y) <= measure.APERTURE_PX


def _point_flux(frame: np.ndarray, x: float, y: float) -> tuple[float, float, float, float]:
    """Aperture flux, error, S/N and the masked fraction of the aperture at a fixed position."""
    inside = _aperture_mask(frame.shape, x, y)
    total = int(inside.sum())
    if total == 0:
        return 0.0, 0.0, 0.0, 1.0
    masked = float(np.mean(~np.isfinite(frame[inside])))
    flux, error, snr = measure.photometry(frame, x, y)
    return flux, error, snr, masked


def _target_px(view: ViewData, pass_index: int | None = None) -> tuple[float, float]:
    """Where the target sits on the grid: the centre, which is the catalog position at the pass midpoint."""
    if pass_index is not None and pass_index < len(view.groups):
        from .resolve import position_at

        mid = float(np.mean([view.records[i].mjd for i in view.groups[pass_index]]))
        if view.target.source != "user coordinates":
            x, y = (float(v) for v in view.grid.world_to_pixel(position_at(view.target, mid)))
            if np.isfinite(x) and np.isfinite(y):
                return x, y
    return (view.shape[1] - 1) / 2, (view.shape[0] - 1) / 2


def spectrum(view: ViewData, png_x: float | None = None, png_y: float | None = None) -> SpectrumResponse:
    started = time.perf_counter()
    notes: list[str] = []
    if png_x is None or png_y is None:
        x, y = _target_px(view, 0)
        notes.append("Measured at the target's catalog position.")
    else:
        x, y = view.grid_px(png_x, png_y)
        notes.append("Measured at the position you picked, held fixed across every pass.")

    series: list[SpectrumSeries] = []
    for index, group in enumerate(view.groups, start=1):
        points: list[tuple[float, float, float, float]] = []
        for i in group:
            wavelength = view.wavelengths[i]
            if wavelength is None:
                continue
            flux, error, _, blanked = _point_flux(view.aligned[i], x, y)
            if blanked >= 1.0:
                continue
            points.append((wavelength, flux, error, blanked))
        if not points:
            continue
        points.sort()
        waves = np.array([p[0] for p in points])
        fluxes = np.array([p[1] for p in points])
        errors = np.array([p[2] for p in points])
        blanked = np.array([p[3] for p in points])
        bins = min(SPECTRUM_BINS, max(len(points) // 2, 1))
        edges = np.linspace(waves.min(), waves.max() * 1.0001, bins + 1)
        which = np.clip(np.digitize(waves, edges) - 1, 0, bins - 1)
        binned: list[SpectrumPoint] = []
        for b in range(bins):
            inside = which == b
            count = int(inside.sum())
            if count == 0:
                continue
            flux = float(np.mean(fluxes[inside]))
            spread = float(np.std(fluxes[inside])) / math.sqrt(count) if count > 1 else float(np.mean(errors[inside]))
            share = float(np.mean(blanked[inside]))
            binned.append(
                SpectrumPoint(
                    wavelength_um=round(float(np.mean(waves[inside])), 4),
                    flux=round(flux, 5),
                    flux_err=round(max(spread, float(np.mean(errors[inside])) / math.sqrt(count)), 5),
                    frames=count,
                    blanked_fraction=round(share, 3),
                    suspect=share > SUSPECT_MASKED_FRACTION,
                )
            )
        mjds = [view.records[i].mjd for i in group]
        series.append(SpectrumSeries(pass_index=index, label=f"Pass {index} · {iso(float(np.mean(mjds)))}", points=binned))

    notes.append(
        "Wavelengths are taken from each frame's spectral WCS at the target, and vary by under one per cent across "
        "this field of view."
    )
    if any(p.suspect for s in series for p in s.points):
        notes.append(
            "Points marked suspect have part of the aperture blanked, usually a saturated core, so their flux is a "
            "lower limit."
        )
    sky = view.sky(x, y)
    return SpectrumResponse(
        target=view.target,
        detector=view.detector,
        ra=round(float(sky.ra.deg), 6),
        dec=round(float(sky.dec.deg), 6),
        px=(round(png_x if png_x is not None else x, 2), round(png_y if png_y is not None else view.shape[0] - 1 - y, 2)),
        aperture_arcsec=round(measure.APERTURE_PX * PIXEL_ARCSEC, 2),
        series=series,
        notes=notes,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


def light_curve(view: ViewData) -> LightCurveResponse:
    started = time.perf_counter()
    notes: list[str] = []
    waves = [w for w in view.wavelengths if w is not None]
    if not waves:
        return LightCurveResponse(
            target=view.target,
            detector=view.detector,
            bands=[],
            confidence="not-measurable",
            summary="No frame carries a usable wavelength, so no light curve can be built.",
            notes=["The spectral WCS could not be read for any frame."],
            elapsed_s=round(time.perf_counter() - started, 3),
        )

    pass_of = {i: index for index, group in enumerate(view.groups, start=1) for i in group}
    saturated = view.saturation_flags
    edges = np.linspace(min(waves), max(waves) * 1.0001, LIGHTCURVE_BANDS + 1)

    bands: list[LightCurveBand] = []
    for b in range(LIGHTCURVE_BANDS):
        low, high = float(edges[b]), float(edges[b + 1])
        points: list[LightCurvePoint] = []
        for i, wavelength in enumerate(view.wavelengths):
            if wavelength is None or not (low <= wavelength < high):
                continue
            index = pass_of.get(i, 1) - 1
            x, y = _target_px(view, index)
            flux, error, snr, masked = _point_flux(view.aligned[i], x, y)
            frame_flags = view.flags[i]
            suspect = masked > SUSPECT_MASKED_FRACTION or bool(set(frame_flags) & flags_module.SATURATION)
            points.append(
                LightCurvePoint(
                    date=iso(view.records[i].mjd),
                    mjd=round(view.records[i].mjd, 5),
                    wavelength_um=round(wavelength, 4),
                    flux=round(flux, 5),
                    flux_err=round(error, 5),
                    snr=round(snr, 2),
                    pass_index=index + 1,
                    flagged_fraction=round(masked, 3),
                    suspect=suspect,
                )
            )
        if len(points) < MIN_BAND_POINTS:
            continue
        points.sort(key=lambda p: p.mjd)
        change, sigma = _change(points)
        bands.append(
            LightCurveBand(
                label=f"{low:.2f}–{high:.2f} µm",
                wavelength_min_um=round(low, 4),
                wavelength_max_um=round(high, 4),
                points=points,
                change_percent=change,
                change_sigma=sigma,
            )
        )

    if not bands:
        return LightCurveResponse(
            target=view.target,
            detector=view.detector,
            bands=[],
            confidence="not-measurable",
            summary="Too few frames per wavelength band on this detector to compare brightness over time.",
            notes=notes,
            elapsed_s=round(time.perf_counter() - started, 3),
        )

    significant = [b for b in bands if b.change_sigma is not None and abs(b.change_sigma) >= 3 and b.change_percent is not None]
    blanked = float(np.median([p.flagged_fraction for b in bands for p in b.points])) if bands else 0.0

    if blanked > SUSPECT_MASKED_FRACTION:
        notes.append(
            "Each exposure has most of the aperture blanked: this source saturates the detector, so its core carries "
            "no usable values and no brightness can be measured from these frames. Positions are still reliable, "
            "because they come from the unsaturated wings."
        )
        if saturated:
            notes.append(f"Saturation flags on the source: {', '.join(saturated)}.")
        return LightCurveResponse(
            target=view.target,
            detector=view.detector,
            bands=bands,
            blanked_fraction=round(blanked, 3),
            confidence="not-measurable",
            summary=(
                f"{view.target.name} saturates SPHEREx on {view.detector}, so its brightness cannot be measured here. "
                "The numbers below come from the wings of a blanked source and should not be read as a light curve."
            ),
            notes=notes,
            elapsed_s=round(time.perf_counter() - started, 3),
        )

    if saturated:
        confidence = "indicative"
    elif significant:
        confidence = "measured"
    else:
        confidence = "marginal"

    if significant:
        strongest = max(significant, key=lambda b: abs(b.change_percent or 0))
        direction = "brightened" if (strongest.change_percent or 0) > 0 else "faded"
        first = strongest.points[0].date
        last = strongest.points[-1].date
        summary = (
            f"{view.target.name} {direction} by {abs(strongest.change_percent):.0f} per cent in the "
            f"{strongest.label} band between {first} and {last} ({abs(strongest.change_sigma):.1f}σ)."
        )
        if len(significant) > 1:
            summary += " The change differs between bands, so the spectrum itself is changing, not just the brightness."
    else:
        summary = (
            f"No brightness change above three sigma in any band of {view.detector} across the passes covered here."
        )

    if saturated:
        notes.append(
            f"Saturation flags on the source ({', '.join(saturated)}): the real change may be larger than measured, "
            "because the brightest pixels are clipped or blanked."
        )
    notes.append(
        "Each point is one exposure, measured inside a fixed aperture at the catalog position, with the modelled "
        "zodiacal light already removed. Points are grouped by wavelength because SPHEREx changes wavelength between "
        "exposures of the same pass."
    )
    if len(view.groups) < 8:
        notes.append(
            f"{len(view.groups)} passes cover this position, too few to fit a period; this shows change, not a cycle."
        )

    return LightCurveResponse(
        target=view.target,
        detector=view.detector,
        bands=bands,
        blanked_fraction=round(blanked, 3),
        confidence=confidence,
        summary=summary,
        notes=notes,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


def _change(points: list[LightCurvePoint]) -> tuple[float | None, float | None]:
    """Fractional change between the first and last pass present in a band, and its significance."""
    usable = [p for p in points if not p.suspect] or points
    first_pass = usable[0].pass_index
    last_pass = usable[-1].pass_index
    if first_pass == last_pass:
        return None, None
    start = [p for p in usable if p.pass_index == first_pass]
    end = [p for p in usable if p.pass_index == last_pass]
    if not start or not end:
        return None, None
    a = float(np.mean([p.flux for p in start]))
    b = float(np.mean([p.flux for p in end]))
    sa = float(np.std([p.flux for p in start])) / math.sqrt(len(start)) if len(start) > 1 else float(np.mean([p.flux_err for p in start]))
    sb = float(np.std([p.flux for p in end])) / math.sqrt(len(end)) if len(end) > 1 else float(np.mean([p.flux_err for p in end]))
    if a <= 0:
        return None, None
    change = (b - a) / a * 100
    spread = math.hypot(sa, sb)
    sigma = (b - a) / spread if spread > 0 else None
    return round(change, 1), round(sigma, 2) if sigma is not None else None
