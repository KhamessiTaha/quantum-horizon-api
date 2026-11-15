r"""Write the orbits of real asteroids, comets and distant objects for the Solar System view, from NASA/JPL's SBDB.

Run from backend/:

    .venv\Scripts\python scripts/small_bodies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
OUT = Path(__file__).resolve().parents[2] / "src" / "data" / "smallbodies.json"
ELEMENTS = "full_name,epoch_mjd,a,e,i,om,w,ma,diameter,class,H,pdes"
GAUSS_DEG_PER_DAY = 0.9856076686


def query(params: dict[str, str]) -> list[list[str | None]]:
    response = requests.get(API, params=params, timeout=120)
    response.raise_for_status()
    return response.json().get("data", [])


def number(value: str | None, digits: int = 5) -> float | None:
    if value in (None, ""):
        return None
    return float(f"{float(value):.{digits}g}")


def name_of(full: str) -> str:
    full = full.strip()
    if "(" in full and not full.startswith("("):
        head = full.split("(")[0].strip()
        parts = head.split(" ", 1)
        return parts[1] if len(parts) == 2 and parts[0].isdigit() else head
    return full.strip("()")


def asteroid(row: list[str | None], group: str) -> list:
    full, epoch, a, e, i, om, w, ma, diameter, cls, h, pdes = row
    return [name_of(full or ""), group, cls, number(epoch, 6), number(a), number(e), number(i), number(om), number(w),
            number(ma), number(diameter, 3), number(h, 3), pdes]


def comet(row: list[str | None]) -> list | None:
    full, epoch, q, e, i, om, w, tp, cls, pdes = row
    q_au, ecc = float(q or 0), float(e or 1)
    if ecc >= 0.99 or q_au <= 0 or float(epoch or 0) < 40000 or "D/" in (full or ""):
        return None
    a = q_au / (1 - ecc)
    mean_motion = GAUSS_DEG_PER_DAY / a**1.5
    epoch_jd = float(epoch) + 2400000.5
    ma = (mean_motion * (epoch_jd - float(tp))) % 360
    return [(full or "").strip(), "comet", cls, number(epoch, 6), number(a), number(ecc), number(i), number(om),
            number(w), number(ma), None, None, pdes]


FAMOUS = ["Bennu", "Apophis", "Ryugu", "Didymos", "Itokawa", "Eros", "Arrokoth", "Psyche"]


def named(name: str) -> list[str | None] | None:
    response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api", params={"sstr": name, "phys-par": "1", "full-prec": "1"}, timeout=60)
    if response.status_code != 200:
        return None
    data = response.json()
    if "orbit" not in data:
        return None
    elements = {el["name"]: el["value"] for el in data["orbit"]["elements"]}
    physical = {p["name"]: p["value"] for p in data.get("phys_par", [])}
    epoch_mjd = str(float(data["orbit"]["epoch"]) - 2400000.5)
    cls = data["object"].get("orbit_class", {}).get("code")
    return [data["object"]["fullname"], epoch_mjd, elements.get("a"), elements.get("e"), elements.get("i"),
            elements.get("om"), elements.get("w"), elements.get("ma"), physical.get("diameter"), cls, physical.get("H"),
            data["object"].get("des")]


def main() -> int:
    bodies: dict[str, list] = {}

    def add(rows: list[list], group: str) -> None:
        for row in rows:
            entry = asteroid(row, group)
            if entry[4] is not None and entry[5] is not None and entry[5] < 1:
                bodies.setdefault(entry[0], entry)

    add(query({"fields": ELEMENTS, "sb-kind": "a", "sb-cdata": json.dumps({"AND": ["a|GT|150", "q|GT|30"]})}), "extreme")
    add(query({"fields": ELEMENTS, "sb-kind": "a", "sb-class": "TNO", "sort": "H", "limit": "400"}), "distant")
    add(query({"fields": ELEMENTS, "sb-kind": "a", "sb-group": "neo", "sort": "H", "limit": "700"}), "neo")
    add([row for row in (named(n) for n in FAMOUS) if row], "famous")
    add(query({"fields": ELEMENTS, "sb-kind": "a", "sb-ns": "n", "sort": "H", "limit": "6000"}), "asteroid")

    comets = query({"fields": "full_name,epoch_mjd,q,e,i,om,w,tp,class,pdes", "sb-kind": "c", "sb-ns": "n", "limit": "600"})
    for row in comets:
        entry = comet(row)
        if entry:
            bodies.setdefault(entry[0], entry)

    rows = sorted(bodies.values(), key=lambda b: (b[11] if b[11] is not None else 99))
    payload = {
        "source": "NASA/JPL Small-Body Database (SBDB) query API",
        "fields": ["name", "group", "class", "epoch_mjd", "a", "e", "i", "om", "w", "ma", "diameter_km", "H", "designation"],
        "bodies": rows,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    groups: dict[str, int] = {}
    for b in rows:
        groups[b[1]] = groups.get(b[1], 0) + 1
    print(f"{len(rows)} bodies {groups}; {OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
