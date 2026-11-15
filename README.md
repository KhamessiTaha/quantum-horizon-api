# Quantum Horizon API

The measurement service behind [Quantum Horizon](https://quantum-horizon-web.vercel.app), a blink comparator for NASA's
SPHEREx built for NASA Space Apps 2026. Given a target or a position, it finds every SPHEREx image covering it, reads
only the pixels it needs, aligns them on one grid, stacks them per survey pass, and measures motion, brightness and
spectrum — each with an uncertainty and a confidence label.

| | |
|---|---|
| **Live** | <https://quantum-horizon-api.vercel.app/api/health> |
| **Project, docs and history** | [KhamessiTaha/quantum-horizon](https://github.com/KhamessiTaha/quantum-horizon) |
| **Frontend** | [KhamessiTaha/quantum-horizon-web](https://github.com/KhamessiTaha/quantum-horizon-web) |

This repository is the `backend/` folder of the main project, split out with its history so Vercel can build it from a
root folder. Make changes in the main repository and bring them across with `git subtree split --prefix=backend`.

## Run locally

Python 3.12, Windows PowerShell, from this folder:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive API. Warm the five challenge targets with
`.venv\Scripts\python scripts/prewarm.py`.

## Endpoints

Every view takes a target as `target=<name>` or `ra=<deg>&dec=<deg>`, plus `det=D1…D6` and `fov=0.03…0.2` (degrees).

| Path | Returns |
| --- | --- |
| `GET /api/health` | Liveness check |
| `GET /api/resolve?q=…` | Name to J2000 position, proper motion and parallax, from SIMBAD (Strasbourg, then the Harvard mirror), with built-in values for the five featured targets |
| `GET /api/frames` | One view: a stacked image per pass and a difference image (PNG inside the JSON), predicted positions, the motion measurement, per-frame provenance, flags and notes |
| `GET /api/difference?a=&b=` | The difference between any two passes |
| `GET /api/spectrum` | Flux against wavelength per pass, at the target or at a clicked `x`, `y` |
| `GET /api/lightcurve` | Brightness over time inside narrow wavelength bands, or the reason it cannot be measured |
| `GET /api/catalog` | Gaia DR3 sources in the field, propagated to each pass date |
| `GET /api/movers` | Everything in the field that moved between passes, each checked against Gaia DR3 |
| `GET /api/nearby?ra=&dec=&radius=` | SIMBAD's best-known named objects around a position, for the sky map |
| `GET /api/flags` | The SPHEREx flag table and the policy applied to each flag |
| `GET /api/images/{name}.png` | A rendered image from the cache |

`predicted_px` is in PNG pixel coordinates: origin at the top-left pixel's centre, north up, east left. For an overlay
at x, the CSS position is `(x + 0.5) / width_px`.

## Data path

1. **Find.** IRSA TAP lists the images (`spherex.plane` joined to `spherex.artifact`). If TAP does not answer within
   its timeout, IRSA's SIA service is asked instead (QR3 and DR1), and TAP is skipped for five minutes. The list is
   stored, so a position seen once still works when IRSA is down.
2. **Read.** Pixels are read by byte range from `s3://nasa-irsa-spherex` (anonymous), with the IRSA cutout service as
   fallback. Every frame read is cached.
3. **Clean.** The modelled zodiacal light (ZODI) is subtracted. Flags are split four ways, by the reasoning in
   `app/flags.py`: unusable pixels are masked; the saturation family (OVERFLOW, NONLINEAR, BLOOM, SUR_ERROR, PERSIST)
   is kept and reported as a brightness caveat, because a bright star's own pixels carry it; contamination is kept
   and reported; informational flags are kept. TRANSIENT masks a pixel only where SOURCE is not also set. Flag names
   come from each file's `MSKN*` cards, or the standard table for QR2 files, which have none.
4. **Align.** Each frame is reprojected onto a common tangent-plane grid at 6.15″ per pixel, centred on the target's
   position at the median observation date, with every catalog position propagated to the frame's own date.
5. **Stack and measure.** Frames are grouped into passes (a gap over 30 days, or a pass longer than 40, starts a new
   one), median-stacked and
   rendered on one shared brightness scale. Positions are fitted per frame and on each stack, with a leave-one-out
   jackknife for the uncertainty.

Every request to IRSA, SIMBAD and Gaia carries a real network timeout (`app/http.py`): pyvo's own `timeout` argument
is sent to the server as a query parameter and does not stop a connection that never answers.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `QH_CACHE_DIR` | `cache` | Where image lists, pixels and views are cached |
| `QH_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Frontends allowed by CORS, comma separated |
| `QH_READ_WORKERS` | `8` | Frames read in parallel |
| `QH_READ_ATTEMPTS` | `3` | Attempts per frame before the cutout fallback |
| `QH_MAX_FRAMES` | `120` | Frames per view for the five featured targets, spread evenly in time |
| `QH_MAX_FRAMES_NEW` | `QH_MAX_FRAMES` | Frames per view for any other position |
| `QH_TAP_TIMEOUT_S` | `20` | Seconds to wait for IRSA TAP before using SIA |
| `QH_IRSA_TAP`, `QH_IRSA_SIA` | IRSA's services | Archive endpoints |
| `QH_SIMBAD_TAPS` | Strasbourg, then Harvard | SIMBAD TAP services, tried in order |
| `QH_S3_BUCKET` | `nasa-irsa-spherex` | The SPHEREx mirror on AWS Open Data |

## Deploying

**Vercel** (the live API). Import this repository into Vercel with the framework preset left as *Other*, and set
`QH_ALLOWED_ORIGINS` to the frontend's address. `pyproject.toml` names `api/index.py` as the entry point: it copies the
shipped `cache/` to `/tmp` on a cold start, gives astropy a writable home, and sets the Vercel limits (85 frames for a
new position, a 10 s TAP timeout) so a first look stays inside the 300 s that `vercel.json` allows.

**Docker or Fly.** For a host that keeps a disk between requests:

```
docker build -t qh-api .
docker run -p 8080:8080 -v qh-cache:/data/cache qh-api
```

`fly.toml` deploys the same image to Fly with a volume at `/data`; the entrypoint seeds it from the shipped cache on
first boot.

## Tests

`.venv\Scripts\python -m pytest` runs the pipeline end to end on synthetic SPHEREx-like files, the flag policy, the
epoch propagation, the archive fallbacks and the mover search. They make no network calls.

## Data and credits

SPHEREx images: NASA/JPL-Caltech/IPAC, via the NASA/IPAC Infrared Science Archive and the SPHEREx mirror on AWS Open
Data. Positions and proper motions: SIMBAD (CDS) and Gaia DR3 (ESA).
