import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from . import cache, discovery, processing, render
from .config import settings
from .models import DetectorId, FrameInfo, FramesResponse, Grid, ImageRecord, PassInfo, Resolved
from .pixels import OffDetector, Region, read_region
from .resolve import position_at

log = logging.getLogger(__name__)

PIXEL_ARCSEC = 6.15
REFERENCE_MJD = 61041.0
SATURATION_FLAGS = {"OVERFLOW", "BLOOM", "NONLINEAR"}
VIEW_VERSION = 1


class NoCoverage(Exception):
    pass


def _iso(mjd: float) -> str:
    return Time(mjd, format="mjd").iso[:10]


def _even_in_time(records: list[ImageRecord], limit: int) -> list[ImageRecord]:
    if limit <= 0 or len(records) <= limit:
        return records
    mjds = np.array([r.mjd for r in records])
    targets = np.linspace(mjds.min(), mjds.max(), limit)
    picks = np.unique(np.abs(mjds[None, :] - targets[:, None]).argmin(axis=1))
    return [records[i] for i in picks]


def _view_key(target: Resolved, detector: str, fov_deg: float, records: list[ImageRecord]) -> str:
    latest = max((r.mjd for r in records), default=0.0)
    return cache.key(
        "view", VIEW_VERSION, f"{target.ra:.6f}", f"{target.dec:.6f}", target.pmra, target.pmdec,
        detector, f"{fov_deg:.3f}", len(records), f"{latest:.5f}",
    )


def _png_px(grid_wcs, shape: tuple[int, int], coord: SkyCoord) -> tuple[float, float] | None:
    x, y = (float(v) for v in grid_wcs.world_to_pixel(coord))
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return round(x, 3), round(shape[0] - 1 - y, 3)


def build_view(target: Resolved, detector: DetectorId, fov_deg: float, refresh: bool = False) -> FramesResponse:
    started = time.perf_counter()
    notes: list[str] = []
    is_object = target.source != "user coordinates"

    search_center = position_at(target, REFERENCE_MJD)
    records, found_via = discovery.find_images(search_center.ra.deg, search_center.dec.deg, refresh=refresh)
    if found_via.startswith("stored (IRSA"):
        notes.append("IRSA is unavailable, so this view uses the stored image list; images released since may be missing.")
    selected = [r for r in records if r.detector == detector]
    if not selected:
        raise NoCoverage(f"SPHEREx has no {detector} images covering this position yet.")
    if len(selected) > settings.max_frames:
        notes.append(f"{len(selected)} images available; using {settings.max_frames} spread evenly in time.")
        selected = _even_in_time(selected, settings.max_frames)

    view_key = _view_key(target, detector, fov_deg, selected)
    cached = None if refresh else cache.read_json("views", view_key)
    if cached:
        response = FramesResponse(**cached)
        response.from_cache = True
        response.elapsed_s = round(time.perf_counter() - started, 3)
        return response

    mid_mjd = float(np.median([r.mjd for r in selected]))
    center = position_at(target, mid_mjd)
    grid_wcs, shape = processing.common_grid(center, PIXEL_ARCSEC, fov_deg)
    half_px = int(math.ceil(fov_deg * 3600 / PIXEL_ARCSEC / 2 * math.sqrt(2))) + 2

    regions: dict[str, Region] = {}
    infos: dict[str, FrameInfo] = {}

    def load(record: ImageRecord):
        return record, read_region(record, center, position_at(target, record.mjd), half_px, PIXEL_ARCSEC)

    with ThreadPoolExecutor(max_workers=max(settings.read_workers, 1)) as pool:
        futures = {pool.submit(load, r): r for r in selected}
        for future in as_completed(futures):
            record = futures[future]
            base = dict(plane_id=record.plane_id, mjd=record.mjd, date=_iso(record.mjd), release=record.release, uri=record.uri)
            try:
                _, region = future.result()
            except OffDetector as error:
                infos[record.plane_id] = FrameInfo(**base, wavelength_um=None, status="off-detector", source=None, flags_at_target=[], error=str(error))
                continue
            except Exception as error:
                infos[record.plane_id] = FrameInfo(**base, wavelength_um=None, status="failed", source=None, flags_at_target=[], error=str(error)[:300])
                continue
            regions[record.plane_id] = region
            infos[record.plane_id] = FrameInfo(
                **base, wavelength_um=region.wavelength_um, status="used", source=region.source, flags_at_target=region.flags_at_target
            )

    used = [r for r in selected if r.plane_id in regions]
    failed = sum(1 for i in infos.values() if i.status == "failed")
    if failed:
        notes.append(f"{failed} of {len(selected)} images could not be read and were skipped.")
    if not used:
        raise NoCoverage("None of the covering images could be read. Try again shortly.")

    aligned = {
        r.plane_id: processing.align(regions[r.plane_id].image, regions[r.plane_id].bad_mask(), regions[r.plane_id].wcs, grid_wcs, shape)
        for r in used
    }
    groups = processing.group_passes([r.mjd for r in used])
    stacks = [processing.stack([aligned[used[i].plane_id] for i in group]) for group in groups]
    low, high = render.shared_limits(stacks)

    passes: list[PassInfo] = []
    for index, (group, image) in enumerate(zip(groups, stacks), start=1):
        mjds = [used[i].mjd for i in group]
        waves = [regions[used[i].plane_id].wavelength_um for i in group if regions[used[i].plane_id].wavelength_um is not None]
        pass_mid = float(np.mean(mjds))
        name = f"{view_key}_pass{index}.png"
        cache.write_bytes("images", name, render.grayscale(image, low, high))
        passes.append(
            PassInfo(
                index=index,
                start=_iso(min(mjds)),
                end=_iso(max(mjds)),
                mid_mjd=round(pass_mid, 5),
                mid_date=_iso(pass_mid),
                frames_used=len(group),
                frames_total=sum(1 for r in selected if min(mjds) <= r.mjd <= max(mjds)),
                wavelength_min_um=round(min(waves), 4) if waves else None,
                wavelength_max_um=round(max(waves), 4) if waves else None,
                predicted_px=_png_px(grid_wcs, shape, position_at(target, pass_mid)) if is_object else None,
                image=f"/api/images/{name}",
            )
        )

    difference_image = None
    if len(stacks) >= 2:
        name = f"{view_key}_diff.png"
        cache.write_bytes("images", name, render.difference(stacks[-1], stacks[0]))
        difference_image = f"/api/images/{name}"
    else:
        notes.append("Only one pass covers this position so far, so there is nothing to compare yet.")

    saturated = sorted({flag for i in infos.values() for flag in i.flags_at_target if flag in SATURATION_FLAGS})
    if saturated:
        notes.append(f"Saturation flags at the target ({', '.join(saturated)}): brightness measurements here are indicative only.")
    if is_object and (target.pmra or target.pmdec):
        notes.append(f"Predicted positions use {target.source} proper motion, propagated from {target.epoch} to each pass.")

    response = FramesResponse(
        target=target,
        detector=detector,
        grid=Grid(ra=center.ra.deg, dec=center.dec.deg, width_px=shape[1], height_px=shape[0], pixel_arcsec=PIXEL_ARCSEC, fov_deg=fov_deg),
        passes=passes,
        difference_image=difference_image,
        frames=sorted(infos.values(), key=lambda i: i.mjd),
        releases=sorted({r.release for r in used}),
        notes=notes,
        elapsed_s=round(time.perf_counter() - started, 3),
        from_cache=False,
    )
    cache.write_json("views", view_key, response.model_dump())
    return response
