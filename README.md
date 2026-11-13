# Quantum Horizon API

FastAPI service that finds every SPHEREx image of a sky position, reads only the needed pixels, aligns them on one grid, stacks them per survey pass and serves PNGs plus metadata to the frontend.

## Run locally (Windows PowerShell, from `backend`)

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000/docs for the interactive API.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Liveness check |
| `GET /api/resolve?q=Barnard's Star` | Name to J2000 position, proper motion and parallax (SIMBAD, Harvard mirror, then built-in values for the five featured targets) |
| `GET /api/frames?target=...` or `?ra=&dec=` plus `det=D1..D6`, `fov=0.03..0.2`, `refresh=false` | One view: per-pass stacked PNGs, difference PNG, predicted positions, per-frame provenance and notes |
| `GET /api/images/{name}.png` | A rendered image from the cache |

`predicted_px` is in PNG pixel coordinates: origin at the top-left pixel's centre, north up, east left. For an overlay at x, the CSS position is `(x + 0.5) / width_px`.

## Data path

1. IRSA TAP finds the images (`spherex.plane` + `spherex.artifact`). The list is stored, so a position seen once still works when IRSA is down.
2. Pixels are read by byte range from `s3://nasa-irsa-spherex` (anonymous); the IRSA cutout service is the fallback.
3. ZODI is subtracted; pixels flagged TRANSIENT, SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, OUTLIER, STREAK or SNOWBALL are masked. Flag names come from each file's `MSKN*` cards.
4. Each frame is reprojected onto a common tangent-plane grid at 6.15″ per pixel, centred on the target's position at the median observation date.
5. Frames are grouped into passes (gap over 30 days, span at most 40 days), median-stacked, background-subtracted and rendered with one shared brightness scale.

## Settings (environment variables)

| Variable | Default |
| --- | --- |
| `QH_CACHE_DIR` | `cache` |
| `QH_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `QH_READ_WORKERS` | `8` |
| `QH_MAX_FRAMES` | `120` per view, spread evenly in time |
| `QH_SIMBAD_TAPS` | Strasbourg, then Harvard |

## Docker

```
docker build -t qh-api .
docker run -p 8080:8080 -v qh-cache:/data/cache qh-api
```
