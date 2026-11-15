"""PNG rendering for the browser.

All passes of one view share a single brightness scale, so a change in brightness between passes is real and not an
artifact of per-image stretching. The scale is anchored on the sky's own noise: the median sky sits just above black,
each noise step is visible, and faint sources rise without bright ones blowing out. Images are upsampled four times
with a Lanczos filter for display only; every measurement uses the original pixels.
"""

import base64
import io

import numpy as np
from PIL import Image

BACKGROUND = np.array([14, 17, 22], dtype=np.float64)
NO_DATA = np.array([32, 42, 54], dtype=np.float64)
ARRIVED = np.array([242, 165, 65], dtype=np.float64)
LEFT = np.array([92, 200, 255], dtype=np.float64)
UPSCALE = 4
SKY_SIGMAS = 4.0
SKY_LIFT = 1.6


def shared_limits(stacks: list[np.ndarray]) -> tuple[float, float, float]:
    """Black point, white point and asinh softening shared by every pass of a view."""
    values = np.concatenate([s[np.isfinite(s)] for s in stacks]) if stacks else np.array([])
    if values.size == 0:
        return 0.0, 1.0, 10.0
    median = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values - median)))
    high = float(np.percentile(values, 99.9))
    if sigma <= 0:
        low = float(np.percentile(values, 5))
        return low, max(high, low + 1.0), 10.0
    low = median - SKY_SIGMAS * sigma
    if high <= low:
        high = low + 1.0
    softening = float(np.clip(SKY_LIFT * (high - low) / (SKY_SIGMAS * sigma), 5.0, 50000.0))
    return low, high, softening


def _png(array: np.ndarray) -> bytes:
    image = Image.fromarray(array)
    image = image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def grayscale(image: np.ndarray, low: float, high: float, softening: float = 10.0) -> bytes:
    """Grey sky with blanked pixels in slate, so a saturated core reads as missing data rather than as black sky."""
    missing = ~np.isfinite(image)
    scaled = np.clip((np.nan_to_num(image, nan=low) - low) / (high - low), 0, 1)
    stretched = np.arcsinh(softening * scaled) / np.arcsinh(softening)
    grey = (stretched[..., None] * 255).repeat(3, axis=2)
    pixels = np.where(missing[..., None], NO_DATA, grey)
    return _png(np.flipud(pixels.round().astype(np.uint8)))


def difference(later: np.ndarray, earlier: np.ndarray) -> bytes:
    diff = later - earlier
    finite = np.isfinite(diff)
    limit = float(np.percentile(np.abs(diff[finite]), 99.5)) if finite.any() else 1.0
    limit = limit if limit > 0 else 1.0
    signed = np.where(finite, diff / limit, 0.0)
    strength = np.clip(np.abs(signed), 0, 1)[..., None]
    colour = np.where(signed[..., None] > 0, ARRIVED, LEFT)
    rgb = BACKGROUND + (colour - BACKGROUND) * strength
    pixels = np.flipud(rgb.round().astype(np.uint8))
    return _png(pixels)


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
