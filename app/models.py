from typing import Literal

from pydantic import BaseModel, Field

DetectorId = Literal["D1", "D2", "D3", "D4", "D5", "D6"]


class Resolved(BaseModel):
    query: str
    name: str
    simbad_id: str | None
    ra: float
    dec: float
    pmra: float = Field(description="mas/yr, includes cos(dec)")
    pmdec: float = Field(description="mas/yr")
    parallax: float | None = Field(description="mas")
    epoch: str
    source: str
    featured: str | None


class ImageRecord(BaseModel):
    plane_id: str
    mjd: float
    detector: DetectorId
    release: str
    uri: str


class FrameInfo(BaseModel):
    plane_id: str
    mjd: float
    date: str
    release: str
    uri: str
    wavelength_um: float | None
    status: Literal["used", "failed", "off-detector"]
    source: Literal["s3", "cutout", "cache"] | None
    flags_at_target: list[str]
    error: str | None = None


class PassInfo(BaseModel):
    index: int
    start: str
    end: str
    mid_mjd: float
    mid_date: str
    frames_used: int
    frames_total: int
    wavelength_min_um: float | None
    wavelength_max_um: float | None
    predicted_px: tuple[float, float] | None
    image: str


class Grid(BaseModel):
    ra: float
    dec: float
    width_px: int
    height_px: int
    pixel_arcsec: float
    fov_deg: float
    north_up: bool = True
    east_left: bool = True


class FramesResponse(BaseModel):
    target: Resolved
    detector: DetectorId
    grid: Grid
    passes: list[PassInfo]
    difference_image: str | None
    frames: list[FrameInfo]
    releases: list[str]
    notes: list[str]
    elapsed_s: float
    from_cache: bool
