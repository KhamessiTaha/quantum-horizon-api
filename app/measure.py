"""Astrometry and photometry on aligned frames.

Positions are measured on individual aligned frames, not only on the stack, so the scatter between frames gives an
empirical uncertainty. The spike on real Barnard's Star data recovered 10.06 +/- 0.63" against a catalog prediction of
10.28", and showed why this matters: a pass whose frames disagree by several pixels is fitting noise, not a source.

Everything here works in grid pixels with north up and east left. The caller converts to sky coordinates.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

BOX_HALF_PX = 3
APERTURE_PX = 1.6
ANNULUS_PX = (2.5, 4.5)
MIN_FRAME_SNR = 3.0
MEASURED_SNR = 5.0
MARGINAL_SNR = 2.0
MAX_SCATTER_PX = 2.0
MIN_FRAMES_MEASURED = 3
CLIP_SIGMA = 3.0
SIGNIFICANT = 3.0


@dataclass
class FrameFit:
    index: int
    x: float
    y: float
    flux: float
    noise: float
    snr: float
    rejected: bool = False
    reason: str | None = None


@dataclass
class PassFit:
    x: float | None
    y: float | None
    sigma_px: float | None
    scatter_px: float | None
    snr: float | None
    flux: float | None
    flux_err: float | None
    frames_fit: int
    frames_rejected: int
    confidence: str
    fits: list[FrameFit] = field(default_factory=list)


def _radii(shape: tuple[int, int], x: float, y: float) -> np.ndarray:
    ys, xs = np.indices(shape)
    return np.hypot(xs - x, ys - y)


def background(frame: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Robust background level and pixel noise from an annulus around (x, y)."""
    radius = _radii(frame.shape, x, y)
    ring = frame[(radius >= ANNULUS_PX[0]) & (radius <= ANNULUS_PX[1]) & np.isfinite(frame)]
    if ring.size < 8:
        ring = frame[np.isfinite(frame)]
    if ring.size == 0:
        return 0.0, 0.0
    level = float(np.median(ring))
    noise = 1.4826 * float(np.median(np.abs(ring - level)))
    if noise <= 0:
        noise = float(np.std(ring)) if ring.size > 1 else 0.0
    return level, noise


def photometry(frame: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """Aperture flux above the local background, its uncertainty, and the signal-to-noise ratio."""
    level, noise = background(frame, x, y)
    radius = _radii(frame.shape, x, y)
    inside = (radius <= APERTURE_PX) & np.isfinite(frame)
    count = int(inside.sum())
    if count == 0:
        return 0.0, 0.0, 0.0
    flux = float(np.sum(frame[inside] - level))
    error = noise * math.sqrt(count) if noise > 0 else 0.0
    snr = flux / error if error > 0 else 0.0
    return flux, error, snr


def centroid(frame: np.ndarray, x: float, y: float, half: int = BOX_HALF_PX) -> tuple[float, float] | None:
    """Flux-weighted centroid inside a small box, recentred twice. Returns None if the box holds no usable signal."""
    cx, cy = x, y
    for _ in range(2):
        x0, x1 = int(round(cx)) - half, int(round(cx)) + half + 1
        y0, y1 = int(round(cy)) - half, int(round(cy)) + half + 1
        if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
            return None
        window = frame[y0:y1, x0:x1]
        if not np.isfinite(window).any():
            return None
        level, _ = background(frame, cx, cy)
        weights = np.where(np.isfinite(window), window - level, 0.0)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            return None
        ys, xs = np.indices(window.shape)
        cx = x0 + float((weights * xs).sum() / total)
        cy = y0 + float((weights * ys).sum() / total)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            return None
    if math.hypot(cx - x, cy - y) > half + 1:
        return None
    return cx, cy


def stack_median(frames: list[np.ndarray]) -> np.ndarray:
    """Median of aligned frames with the sky level removed, ignoring masked pixels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stacked = np.nanmedian(np.stack(frames), axis=0)
    finite = stacked[np.isfinite(stacked)]
    level = float(np.median(finite)) if finite.size else 0.0
    return (stacked - level).astype(np.float32)


def fit_stack(frames: list[np.ndarray], x: float, y: float) -> tuple[float, float, float, float, float] | None:
    """Position, flux, error and S/N of a source measured on the stack of a pass."""
    stacked = stack_median(frames)
    found = centroid(stacked, x, y)
    if found is None:
        return None
    cx, cy = found
    flux, error, snr = photometry(stacked, cx, cy)
    return cx, cy, flux, error, snr


def jackknife(frames: list[np.ndarray], x: float, y: float) -> tuple[float, float] | None:
    """Leave-one-out uncertainty on the stack position, in pixels: sigma and the raw scatter it came from."""
    if len(frames) < 3:
        return None
    positions = []
    for drop in range(len(frames)):
        subset = [f for i, f in enumerate(frames) if i != drop]
        found = fit_stack(subset, x, y)
        if found is None:
            continue
        positions.append((found[0], found[1]))
    if len(positions) < 3:
        return None
    array = np.array(positions)
    mean = array.mean(axis=0)
    n = len(positions)
    variance = (n - 1) / n * float(np.sum((array - mean) ** 2))
    sigma = math.sqrt(max(variance, 0.0))
    spread = float(np.sqrt(np.mean(np.sum((array - mean) ** 2, axis=1))))
    return sigma, spread


def fit_frames(frames: list[np.ndarray], indices: list[int], x: float, y: float) -> PassFit:
    """Measure one source on every frame of a pass, reject outliers, and combine."""
    fits: list[FrameFit] = []
    for index, frame in zip(indices, frames):
        found = centroid(frame, x, y)
        if found is None:
            fits.append(FrameFit(index=index, x=x, y=y, flux=0.0, noise=0.0, snr=0.0, rejected=True, reason="no signal in the search box"))
            continue
        cx, cy = found
        flux, error, snr = photometry(frame, cx, cy)
        fit = FrameFit(index=index, x=cx, y=cy, flux=flux, noise=error, snr=snr)
        if snr < MIN_FRAME_SNR:
            fit.rejected = True
            fit.reason = f"signal-to-noise {snr:.1f} below {MIN_FRAME_SNR:.0f}"
        fits.append(fit)

    kept = [f for f in fits if not f.rejected]
    if kept:
        xs = np.array([f.x for f in kept])
        ys = np.array([f.y for f in kept])
        mx, my = float(np.median(xs)), float(np.median(ys))
        distance = np.hypot(xs - mx, ys - my)
        spread = 1.4826 * float(np.median(distance))
        if spread > 0 and len(kept) >= 4:
            for fit, far in zip(kept, distance > CLIP_SIGMA * spread):
                if far:
                    fit.rejected = True
                    fit.reason = "position disagrees with the other frames of this pass"
        kept = [f for f in fits if not f.rejected]

    rejected = sum(1 for f in fits if f.rejected)
    if not kept:
        return PassFit(None, None, None, None, None, None, None, 0, rejected, "not-measurable", fits)

    by_index = dict(zip(indices, frames))
    kept_frames = [by_index[f.index] for f in kept if f.index in by_index]
    xs = np.array([f.x for f in kept])
    ys = np.array([f.y for f in kept])
    fluxes = np.array([f.flux for f in kept])
    snrs = np.array([f.snr for f in kept])
    x_med, y_med = float(np.median(xs)), float(np.median(ys))
    offsets = np.hypot(xs - x_med, ys - y_med)
    scatter = 1.4826 * float(np.median(offsets)) if len(kept) > 1 else 0.0

    # The stack reaches a higher signal-to-noise than any single frame, and dithering fills in pixels blanked by
    # saturation, so the position comes from the stack and its uncertainty from leaving each frame out in turn.
    stacked = fit_stack(kept_frames, x_med, y_med) if kept_frames else None
    if stacked is not None:
        x_med, y_med, stack_flux, stack_err, stack_snr = stacked
    else:
        stack_flux, stack_err, stack_snr = float(np.median(fluxes)), 0.0, float(np.median(snrs))
    jack = jackknife(kept_frames, x_med, y_med) if kept_frames else None
    sigma = jack[0] if jack else (scatter / math.sqrt(len(kept)) if len(kept) > 1 else None)

    snr = max(stack_snr, float(np.median(snrs)))
    flux = float(np.median(fluxes))
    flux_err = float(1.4826 * np.median(np.abs(fluxes - flux)) / math.sqrt(len(kept))) if len(kept) > 1 else stack_err

    if len(kept) >= MIN_FRAMES_MEASURED and snr >= MEASURED_SNR and scatter <= MAX_SCATTER_PX:
        confidence = "measured"
    elif snr >= MARGINAL_SNR:
        confidence = "marginal"
    else:
        confidence = "not-measurable"

    return PassFit(x_med, y_med, sigma, scatter, snr, flux, flux_err, len(kept), rejected, confidence, fits)


def separation(first: PassFit, last: PassFit, pixel_arcsec: float) -> tuple[float, float, float] | None:
    """Motion between two passes in arcseconds, its uncertainty, and its position angle in degrees east of north."""
    if first.x is None or last.x is None:
        return None
    dx = (last.x - first.x) * pixel_arcsec
    dy = (last.y - first.y) * pixel_arcsec
    east = -dx
    north = dy
    distance = math.hypot(east, north)
    sigma_px = math.hypot(first.sigma_px or 0.0, last.sigma_px or 0.0)
    floor = 0.1 * pixel_arcsec
    sigma = max(sigma_px * pixel_arcsec, floor)
    angle = (math.degrees(math.atan2(east, north)) + 360) % 360
    return distance, sigma, angle


def grade_motion(measured: tuple[float, float, float] | None, first: PassFit, last: PassFit) -> str:
    if measured is None:
        return "not-measurable"
    distance, sigma, _ = measured
    order = ["measured", "marginal", "not-measurable"]
    weakest = order[max(order.index(first.confidence), order.index(last.confidence))]
    if weakest != "measured":
        return weakest
    if sigma > 0 and distance / sigma < SIGNIFICANT:
        return "below-noise"
    return "measured"
