import logging
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import analysis, cache, catalog, flags, movers, nearby, render, track
from .config import settings
from .discovery import DiscoveryError
from .models import (
    CatalogResponse,
    CatalogSource,
    DetectorId,
    FramesResponse,
    LightCurveResponse,
    MoversResponse,
    Resolved,
    SpectrumResponse,
)
from .pipeline import build_view
from .resolve import ResolveError, coordinates, resolve_name
from .view import NoCoverage, ViewData, prepare

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Quantum Horizon API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

IMAGE_NAME = re.compile(r"^[0-9a-f]{20}_(pass\d{1,3}|diff|diff_\d{1,3}_\d{1,3})\.png$")


def _target(target: str | None, ra: float | None, dec: float | None) -> Resolved:
    if ra is not None and dec is not None:
        return coordinates(ra, dec)
    if target:
        try:
            return resolve_name(target)
        except ResolveError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
    raise HTTPException(status_code=400, detail="Give a target name, or both ra and dec in degrees.")


def _predicted_for(view: ViewData, mjd: float):
    from .resolve import position_at

    return view.center if view.target.source == "user coordinates" else position_at(view.target, mjd)


def _view(resolved: Resolved, detector: DetectorId, fov_deg: float) -> ViewData:
    try:
        return prepare(resolved, detector, fov_deg)
    except NoCoverage as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DiscoveryError as error:
        raise HTTPException(status_code=503, detail=f"{error}. Try again shortly.") from error


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/resolve", response_model=Resolved)
def resolve(q: str = Query(min_length=1, max_length=120)) -> Resolved:
    try:
        return resolve_name(q)
    except ResolveError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/nearby")
def nearby_objects(
    ra: float = Query(ge=0, lt=360),
    dec: float = Query(ge=-90, le=90),
    radius: float = Query(default=0.25, gt=0, le=1.0),
) -> dict:
    """The best-known named objects within `radius` degrees, from SIMBAD, for labelling the sky map."""
    try:
        return nearby.named_objects(ra, dec, radius)
    except nearby.NearbyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/track/bodies")
def track_bodies() -> dict:
    return {"bodies": list(track.BODIES), "ready": track.available()}


@app.get("/api/track")
def track_body(
    body: str = Query(min_length=1, max_length=60),
    det: DetectorId = "D2",
    id: str | None = Query(default=None, max_length=40),
    kind: str = Query(default="asteroid", pattern="^(asteroid|comet)$"),
) -> dict:
    try:
        return track.track(body, det, id, kind)
    except track.TrackError as error:
        status = 503 if "not answering" in str(error) else 404
        raise HTTPException(status_code=status, detail=str(error)) from error


@app.get("/api/frames", response_model=FramesResponse)
def frames(
    target: str | None = Query(default=None, max_length=120),
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    det: DetectorId = "D2",
    fov: float = Query(default=0.06, ge=0.03, le=0.2),
    refresh: bool = False,
) -> FramesResponse:
    resolved = _target(target, ra, dec)
    try:
        return build_view(resolved, det, fov, refresh=refresh)
    except NoCoverage as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DiscoveryError as error:
        raise HTTPException(status_code=503, detail=f"{error}. Try again shortly.") from error


@app.get("/api/spectrum", response_model=SpectrumResponse)
def spectrum(
    target: str | None = Query(default=None, max_length=120),
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    det: DetectorId = "D2",
    fov: float = Query(default=0.06, ge=0.03, le=0.2),
    x: float | None = Query(default=None, description="pixel x in the returned PNG, left to right"),
    y: float | None = Query(default=None, description="pixel y in the returned PNG, top to bottom"),
) -> SpectrumResponse:
    view = _view(_target(target, ra, dec), det, fov)
    return analysis.spectrum(view, x, y)


@app.get("/api/lightcurve", response_model=LightCurveResponse)
def lightcurve(
    target: str | None = Query(default=None, max_length=120),
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    det: DetectorId = "D2",
    fov: float = Query(default=0.06, ge=0.03, le=0.2),
) -> LightCurveResponse:
    view = _view(_target(target, ra, dec), det, fov)
    return analysis.light_curve(view)


@app.get("/api/difference")
def difference(
    target: str | None = Query(default=None, max_length=120),
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    det: DetectorId = "D2",
    fov: float = Query(default=0.06, ge=0.03, le=0.2),
    a: int = Query(ge=1, le=200, description="earlier pass"),
    b: int = Query(ge=1, le=200, description="later pass"),
) -> dict:
    """The difference between any two passes, so epochs can be added and subtracted rather than only first and last."""
    view = _view(_target(target, ra, dec), det, fov)
    if a == b or max(a, b) > len(view.stacks):
        raise HTTPException(status_code=400, detail=f"This view has {len(view.stacks)} passes; pick two different ones.")
    name = f"{view.key}_diff_{a}_{b}.png"
    png = render.difference(view.stacks[b - 1], view.stacks[a - 1])
    cache.write_bytes("images", name, png)
    return {"image": render.data_uri(png), "earlier_pass": a, "later_pass": b}


@app.get("/api/catalog", response_model=CatalogResponse)
def catalogue(
    target: str | None = Query(default=None, max_length=120),
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    det: DetectorId = "D2",
    fov: float = Query(default=0.06, ge=0.03, le=0.2),
    limit_mag: float = Query(default=catalog.DEFAULT_LIMIT_MAG, ge=6, le=21),
) -> CatalogResponse:
    """Gaia DR3 sources over the field, each propagated to the date of the pass it is drawn on."""
    import time as _time

    started = _time.perf_counter()
    view = _view(_target(target, ra, dec), det, fov)
    radius = fov * 0.75
    try:
        rows, provider = catalog.sources(view.center.ra.deg, view.center.dec.deg, radius, limit_mag)
    except catalog.CatalogError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    mids = [float(sum(view.records[i].mjd for i in group) / len(group)) for group in view.groups]
    per_pass = {index + 1: catalog.at_epoch(rows, mid) for index, mid in enumerate(mids)}
    target_px = [view.png_px(_predicted_for(view, mid)) for mid in mids]

    found: list[CatalogSource] = []
    for order, row in enumerate(rows):
        positions: dict[str, tuple[float, float]] = {}
        for index, coords in per_pass.items():
            if not len(coords):
                continue
            place = view.png_px(coords[order])
            if place is None:
                continue
            if -2 <= place[0] <= view.shape[1] + 1 and -2 <= place[1] <= view.shape[0] + 1:
                positions[str(index)] = place
        if not positions:
            continue
        first = next(iter(positions.values()))
        near_target = any(
            spot is not None and abs(first[0] - spot[0]) < 2 and abs(first[1] - spot[1]) < 2 for spot in target_px
        )
        found.append(
            CatalogSource(
                source_id=row["source_id"],
                g_mag=row["g_mag"],
                pm_total_arcsec_per_year=round(float((row["pmra"] ** 2 + row["pmdec"] ** 2) ** 0.5) / 1000, 3),
                positions=positions,
                is_target=near_target,
            )
        )

    notes = [
        f"Gaia DR3 positions are for epoch J2016.0 and are propagated to each pass date before being drawn.",
        f"Showing sources brighter than G = {limit_mag:.1f}; fainter ones are below what SPHEREx resolves here.",
    ]
    if provider.startswith("cache"):
        notes.append("Served from the stored copy of this field." if provider == "cache" else provider)
    return CatalogResponse(
        provider=provider,
        limit_mag=limit_mag,
        sources=found,
        notes=notes,
        elapsed_s=round(_time.perf_counter() - started, 3),
    )


@app.get("/api/movers", response_model=MoversResponse)
def mover_search(
    target: str | None = Query(default=None, max_length=120),
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    det: DetectorId = "D2",
    fov: float = Query(default=0.06, ge=0.03, le=0.2),
) -> MoversResponse:
    """Everything in this field that moved between passes, each checked against Gaia DR3."""
    import time as _time

    started = _time.perf_counter()
    view = _view(_target(target, ra, dec), det, fov)
    found = movers.search(view)
    return MoversResponse(
        target=view.target,
        detector=view.detector,
        elapsed_s=round(_time.perf_counter() - started, 3),
        **found,
    )


@app.get("/api/flags")
def flag_reference() -> dict:
    """What every SPHEREx flag means, and how this tool treats it."""
    return {
        "bits": [
            {
                "bit": bit,
                "name": name,
                "meaning": flags.MEANINGS.get(name),
                "treatment": (
                    "pixel excluded"
                    if name in flags.UNUSABLE
                    else "kept, brightness indicative only"
                    if name in flags.SATURATION
                    else "kept, reported as possible contamination"
                    if name in flags.CONTAMINATION
                    else "informational"
                ),
            }
            for bit, name in sorted(flags.STANDARD_NAMES.items())
        ],
        "transient_rule": "TRANSIENT excludes a pixel only where SOURCE is not also set, so cosmic rays are removed while bright stars are kept.",
        "source": "Bit names come from each file's MSKN header cards; QR2 files carry none, so the standard table is used.",
    }


@app.get("/api/images/{name}")
def image(name: str) -> FileResponse:
    if not IMAGE_NAME.match(name):
        raise HTTPException(status_code=404, detail="Not found")
    target = cache.path("images", name)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
