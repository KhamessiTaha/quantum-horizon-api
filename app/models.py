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


Confidence = Literal["measured", "marginal", "not-measurable", "below-noise", "indicative"]


class PassMeasurement(BaseModel):
    index: int
    confidence: Confidence
    measured_px: tuple[float, float] | None
    measured_ra: float | None
    measured_dec: float | None
    sigma_arcsec: float | None
    scatter_arcsec: float | None
    snr: float | None
    flux: float | None = Field(description="aperture flux above background, MJy/sr summed over the aperture")
    flux_err: float | None
    frames_fit: int
    frames_rejected: int
    offset_from_predicted_arcsec: float | None


class Motion(BaseModel):
    confidence: Confidence
    baseline_years: float
    from_pass: int
    to_pass: int
    measured_arcsec: float | None
    sigma_arcsec: float | None
    measured_pa_deg: float | None
    measured_arcsec_per_year: float | None
    predicted_arcsec: float | None
    predicted_pa_deg: float | None
    predicted_arcsec_per_year: float | None
    difference_arcsec: float | None
    significance: float | None = Field(description="measured motion divided by its uncertainty")
    pixels: float | None
    summary: str


class Measurement(BaseModel):
    passes: list[PassMeasurement]
    motion: Motion | None
    photometry_confidence: Confidence
    photometry_note: str | None
    aperture_arcsec: float
    method: str


class FramesResponse(BaseModel):
    target: Resolved
    detector: DetectorId
    grid: Grid
    passes: list[PassInfo]
    difference_image: str | None
    frames: list[FrameInfo]
    releases: list[str]
    notes: list[str]
    measurement: Measurement | None = None
    elapsed_s: float
    from_cache: bool


class SpectrumPoint(BaseModel):
    wavelength_um: float
    flux: float
    flux_err: float
    frames: int
    blanked_fraction: float = Field(description="share of aperture pixels with no usable data, mostly saturated cores")
    suspect: bool


class SpectrumSeries(BaseModel):
    pass_index: int
    label: str
    points: list[SpectrumPoint]


class SpectrumResponse(BaseModel):
    target: Resolved
    detector: DetectorId
    ra: float
    dec: float
    px: tuple[float, float]
    aperture_arcsec: float
    series: list[SpectrumSeries]
    notes: list[str]
    elapsed_s: float


class LightCurvePoint(BaseModel):
    date: str
    mjd: float
    wavelength_um: float | None
    flux: float
    flux_err: float
    snr: float
    pass_index: int
    flagged_fraction: float
    suspect: bool


class LightCurveBand(BaseModel):
    label: str
    wavelength_min_um: float
    wavelength_max_um: float
    points: list[LightCurvePoint]
    change_percent: float | None
    change_sigma: float | None


class LightCurveResponse(BaseModel):
    target: Resolved
    detector: DetectorId
    bands: list[LightCurveBand]
    blanked_fraction: float = 0.0
    confidence: Confidence
    summary: str
    notes: list[str]
    elapsed_s: float
