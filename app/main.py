import logging
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import analysis, cache, flags
from .config import settings
from .discovery import DiscoveryError
from .models import DetectorId, FramesResponse, LightCurveResponse, Resolved, SpectrumResponse
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

IMAGE_NAME = re.compile(r"^[0-9a-f]{20}_(pass\d{1,3}|diff)\.png$")


def _target(target: str | None, ra: float | None, dec: float | None) -> Resolved:
    if ra is not None and dec is not None:
        return coordinates(ra, dec)
    if target:
        try:
            return resolve_name(target)
        except ResolveError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
    raise HTTPException(status_code=400, detail="Give a target name, or both ra and dec in degrees.")


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
