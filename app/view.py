"""Build the aligned frame cube for one position, detector and field of view.

Every endpoint works from this one object. Reading the pixels is the slow part (about 100 s for 61 frames from S3), so
the aligned cube is cached on disk: once a position has been seen, the blink, the measurements, the spectrum and the
light curve all come back without touching the network. Only the frame list needs IRSA, and that is stored too.
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

from . import cache, discovery, processing
from .config import settings
from .models import DetectorId, FrameInfo, ImageRecord, Resolved
from .pixels import OffDetector, Region, read_region
from .resolve import position_at

log = logging.getLogger(__name__)

PIXEL_ARCSEC = 6.15
REFERENCE_MJD = 61041.0
SATURATION_FLAGS = {"OVERFLOW", "BLOOM", "NONLINEAR"}
CUBE_VERSION = 2


class NoCoverage(Exception):
    pass


def iso(mjd: float) -> str:
    return Time(mjd, format="mjd").iso[:10]


def _even_in_time(records: list[ImageRecord], limit: int) -> list[ImageRecord]:
    if limit <= 0 or len(records) <= limit:
        return records
    mjds = np.array([r.mjd for r in records])
    targets = np.linspace(mjds.min(), mjds.max(), limit)
    picks = np.unique(np.abs(mjds[None, :] - targets[:, None]).argmin(axis=1))
    return [records[i] for i in picks]


@dataclass
class ViewData:
    key: str
    target: Resolved
    detector: DetectorId
    fov_deg: float
    center: SkyCoord
    grid: WCS
    shape: tuple[int, int]
    records: list[ImageRecord]
    aligned: np.ndarray
    wavelengths: list[float | None]
    flags: list[list[str]]
    infos: list[FrameInfo]
    groups: list[list[int]] = field(default_factory=list)
    stacks: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    found_via: str = ""
    from_cache: bool = False
    elapsed_s: float = 0.0

    @property
    def mjds(self) -> list[float]:
        return [r.mjd for r in self.records]

    @property
    def releases(self) -> list[str]:
        return sorted({r.release for r in self.records})

    @property
    def saturation_flags(self) -> list[str]:
        return sorted({f for frame in self.flags for f in frame if f in SATURATION_FLAGS})

    def png_px(self, coord: SkyCoord) -> tuple[float, float] | None:
        x, y = (float(v) for v in self.grid.world_to_pixel(coord))
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return round(x, 3), round(self.shape[0] - 1 - y, 3)

    def grid_px(self, png_x: float, png_y: float) -> tuple[float, float]:
        return png_x, self.shape[0] - 1 - png_y

    def sky(self, x: float, y: float) -> SkyCoord:
        return SkyCoord(*self.grid.pixel_to_world_values(x, y), unit="deg", frame="icrs")


def view_key(target: Resolved, detector: str, fov_deg: float, records: list[ImageRecord]) -> str:
    latest = max((r.mjd for r in records), default=0.0)
    return cache.key(
        "view", CUBE_VERSION, f"{target.ra:.6f}", f"{target.dec:.6f}", target.pmra, target.pmdec,
        detector, f"{fov_deg:.3f}", len(records), f"{latest:.5f}",
    )


def _geometry(target: Resolved, fov_deg: float, records: list[ImageRecord]) -> tuple[SkyCoord, WCS, tuple[int, int]]:
    mid_mjd = float(np.median([r.mjd for r in records]))
    center = position_at(target, mid_mjd)
    grid, shape = processing.common_grid(center, PIXEL_ARCSEC, fov_deg)
    return center, grid, shape


def _save_cube(key: str, view: ViewData) -> None:
    cache.write_arrays("cubes", key, aligned=view.aligned)
    cache.write_json(
        "cubes",
        key,
        {
            "records": [r.model_dump() for r in view.records],
            "wavelengths": view.wavelengths,
            "flags": view.flags,
            "infos": [i.model_dump() for i in view.infos],
            "notes": view.notes,
        },
    )


def _load_cube(key: str) -> tuple[np.ndarray, dict] | None:
    meta = cache.read_json("cubes", key)
    arrays = cache.read_arrays("cubes", key)
    if meta is None or arrays is None or "aligned" not in arrays:
        return None
    return arrays["aligned"], meta


def _read_all(target: Resolved, selected: list[ImageRecord], center: SkyCoord, fov_deg: float):
    half_px = int(math.ceil(fov_deg * 3600 / PIXEL_ARCSEC / 2 * math.sqrt(2))) + 2
    regions: dict[str, Region] = {}
    infos: dict[str, FrameInfo] = {}

    def load(record: ImageRecord):
        return record, read_region(record, center, position_at(target, record.mjd), half_px, PIXEL_ARCSEC)

    with ThreadPoolExecutor(max_workers=max(settings.read_workers, 1)) as pool:
        futures = {pool.submit(load, r): r for r in selected}
        for future in as_completed(futures):
            record = futures[future]
            base = dict(plane_id=record.plane_id, mjd=record.mjd, date=iso(record.mjd), release=record.release, uri=record.uri)
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
    return regions, infos


def prepare(target: Resolved, detector: DetectorId, fov_deg: float, refresh: bool = False) -> ViewData:
    started = time.perf_counter()
    notes: list[str] = []

    search_center = position_at(target, REFERENCE_MJD)
    records, found_via = discovery.find_images(search_center.ra.deg, search_center.dec.deg, refresh=refresh)
    if found_via.startswith("stored (IRSA"):
        notes.append("IRSA is unavailable, so this view uses the stored image list; images released since may be missing.")
    selected = [r for r in records if r.detector == detector]
    if not selected:
        raise NoCoverage(f"SPHEREx has no {detector} images covering this position yet.")
    limit = settings.max_frames if target.featured else settings.max_frames_new
    if len(selected) > limit:
        notes.append(f"{len(selected)} images available; using {limit} spread evenly in time.")
        selected = _even_in_time(selected, limit)

    key = view_key(target, detector, fov_deg, selected)
    center, grid, shape = _geometry(target, fov_deg, selected)

    stored = None if refresh else _load_cube(key)
    if stored is not None:
        aligned, meta = stored
        view = ViewData(
            key=key,
            target=target,
            detector=detector,
            fov_deg=fov_deg,
            center=center,
            grid=grid,
            shape=shape,
            records=[ImageRecord(**r) for r in meta["records"]],
            aligned=aligned,
            wavelengths=list(meta["wavelengths"]),
            flags=[list(f) for f in meta["flags"]],
            infos=[FrameInfo(**i) for i in meta["infos"]],
            notes=list(meta["notes"]) + [n for n in notes if n not in meta["notes"]],
            found_via=found_via,
            from_cache=True,
        )
    else:
        regions, infos = _read_all(target, selected, center, fov_deg)
        used = [r for r in selected if r.plane_id in regions]
        failed = sum(1 for i in infos.values() if i.status == "failed")
        if failed:
            notes.append(f"{failed} of {len(selected)} images could not be read and were skipped.")
        if not used:
            raise NoCoverage("None of the covering images could be read. Try again shortly.")
        stack_list = [
            processing.align(regions[r.plane_id].image, regions[r.plane_id].bad_mask(), regions[r.plane_id].wcs, grid, shape)
            for r in used
        ]
        view = ViewData(
            key=key,
            target=target,
            detector=detector,
            fov_deg=fov_deg,
            center=center,
            grid=grid,
            shape=shape,
            records=used,
            aligned=np.stack(stack_list),
            wavelengths=[regions[r.plane_id].wavelength_um for r in used],
            flags=[regions[r.plane_id].flags_at_target for r in used],
            infos=sorted(infos.values(), key=lambda i: i.mjd),
            notes=notes,
            found_via=found_via,
        )
        _save_cube(key, view)

    view.groups = processing.group_passes(view.mjds)
    view.stacks = [processing.stack([view.aligned[i] for i in group]) for group in view.groups]
    view.elapsed_s = round(time.perf_counter() - started, 3)
    return view


def frames_in(view: ViewData, group: list[int]) -> list[np.ndarray]:
    return [view.aligned[i] for i in group]


def meta_json(view: ViewData) -> str:
    return json.dumps({"key": view.key, "frames": len(view.records), "passes": len(view.groups)})
