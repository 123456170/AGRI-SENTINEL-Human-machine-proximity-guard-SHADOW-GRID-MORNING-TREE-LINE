# -*- coding: utf-8 -*-
"""
AGRI//SENTINEL
Human-Machine Proximity Guard for Agricultural Machinery Safety

Single-file Streamlit application.
Runs an automatic live demo simulation immediately on launch.

Pipeline:
  synthetic farm world -> detector -> multi-object tracker -> trajectory predictor
  -> dynamic safety zones -> TTC / crossing / risk engine -> event register
  -> heatmap -> online MOT metrics -> Streamlit control room
"""

from __future__ import annotations

import base64
import io
import math
import os
import random
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception:
    cv2 = None


# ===================================================================================
# CONSTANTS
# ===================================================================================

FIELD_W, FIELD_H = 60.0, 38.0
S = 12
BEV_W, BEV_H = int(FIELD_W * S), int(FIELD_H * S)

CAM_W, CAM_H = 560, 340
REEL_W, REEL_H = 420, 260

SIM_DT = 0.1

LEVEL_NAME = {0: "SAFE", 1: "CAUTION", 2: "HIGH RISK", 3: "CRITICAL"}
LEVEL_RGB = {
    0: (72, 214, 140),
    1: (255, 197, 66),
    2: (255, 122, 51),
    3: (255, 59, 84),
}
LEVEL_HEX = {
    0: "#48D68C",
    1: "#FFC542",
    2: "#FF7A33",
    3: "#FF3B54",
}

SPEC = {
    "tractor":   dict(kind="machine", L=4.6, W=2.7, H=3.0, cruise=2.4, col=(198, 62, 48)),
    "harvester": dict(kind="machine", L=10.4, W=7.4, H=4.0, cruise=1.9, col=(226, 166, 52)),
    "baler":     dict(kind="machine", L=5.2, W=2.9, H=3.2, cruise=2.0, col=(104, 148, 104)),
    "sprayer":   dict(kind="machine", L=6.2, W=9.0, H=3.6, cruise=2.0, col=(70, 124, 190)),
    "truck":     dict(kind="machine", L=8.6, W=2.6, H=3.4, cruise=4.0, col=(122, 134, 148)),
    "utv":       dict(kind="machine", L=3.2, W=1.7, H=2.0, cruise=4.4, col=(214, 96, 44)),
    "loader":    dict(kind="machine", L=5.6, W=2.5, H=3.1, cruise=2.4, col=(208, 176, 66)),
    "worker":    dict(kind="human", L=0.62, W=0.62, H=1.78, cruise=1.35, col=(57, 215, 242)),
    "spotter":   dict(kind="human", L=0.62, W=0.62, H=1.72, cruise=1.15, col=(120, 232, 255)),
    "cattle":    dict(kind="animal", L=1.8, W=0.85, H=1.45, cruise=0.85, col=(202, 172, 142)),
    "sheep":     dict(kind="animal", L=1.1, W=0.55, H=0.85, cruise=1.1, col=(228, 228, 226)),
    "dog":       dict(kind="animal", L=0.85, W=0.36, H=0.55, cruise=3.1, col=(158, 124, 208)),
}

SIBLING = {
    "worker": ["spotter", "cattle"],
    "spotter": ["worker"],
    "cattle": ["sheep", "worker"],
    "sheep": ["cattle"],
    "dog": ["cattle", "worker"],
    "tractor": ["truck", "baler", "loader"],
    "harvester": ["tractor", "sprayer"],
    "sprayer": ["harvester", "tractor"],
    "truck": ["tractor", "loader"],
    "loader": ["tractor", "truck"],
    "utv": ["truck", "tractor"],
    "baler": ["tractor", "truck"],
}

KIND_BASE_CONF = {
    "machine": 0.93,
    "human": 0.84,
    "animal": 0.74,
}

DEFAULT_PARAMS = dict(
    speed=1.0,
    conf_thresh=0.30,
    det_noise=1.0,
    ghost_rate=1.0,
    nms=0.45,
    max_age=12,
    gate=3.0,
    max_speed=6.0,
    reid=True,
    reid_window=6.0,
    horizon=5.0,
    reaction=0.8,
    braking=2.6,
    buffer=1.2,
    dust=1.0,
    shadow=1.0,
    night=1.0,
    crowd=1.0,
)

OVERLAYS = dict(
    trails=True,
    predict=True,
    zones=True,
    links=True,
    ids=True,
    blind=False,
    heatmap=False,
    grid=False,
    cam_tracks=True,
    cam_zones=False,
)


# ===================================================================================
# UTILITIES
# ===================================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def clamp01(v: float) -> float:
    return clamp(v, 0.0, 1.0)


def ang_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def deg(rad: float) -> float:
    return math.degrees(rad) % 360.0


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def safe_point(x: Any, y: Any) -> bool:
    try:
        return math.isfinite(float(x)) and math.isfinite(float(y))
    except Exception:
        return False


def safe_floor(v: Any, default: float = 1.0, floor: float = 0.1) -> float:
    return max(floor, safe_float(v, default))


def norm_bbox(bbox: Tuple[float, float, float, float], min_size: float = 2.0) -> Tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return 0.0, 0.0, float(min_size), float(min_size)

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return 0.0, 0.0, float(min_size), float(min_size)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 - x1 < min_size:
        x2 = x1 + min_size
    if y2 - y1 < min_size:
        y2 = y1 + min_size

    return x1, y1, x2, y2


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    a = norm_bbox(a, 1.0)
    b = norm_bbox(b, 1.0)
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    if not poly or not safe_point(x, y):
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if not (safe_point(xi, yi) and safe_point(xj, yj)):
            j = i
            continue
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def poly_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if not (safe_point(x1, y1) and safe_point(x2, y2)):
            continue
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def hungarian(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    O(n^3) assignment problem solver.
    Returns matched (row, col) pairs.
    """
    if not cost or not cost[0]:
        return []

    n = len(cost)
    m = len(cost[0])
    transposed = False

    if n > m:
        cost = [[cost[i][j] for i in range(n)] for j in range(m)]
        n, m = m, n
        transposed = True

    INF = 1e18
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] += delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            ans[p[j] - 1] = j - 1

    if transposed:
        return [(c, r) for r, c in enumerate(ans) if c != -1]
    return [(r, c) for r, c in enumerate(ans) if c != -1]


_FONT_CACHE: Dict[int, Any] = {}


def get_font(size: int = 12):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]

    f = None
    for p in paths:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass

    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except Exception:
            f = ImageFont.load_default()

    _FONT_CACHE[size] = f
    return f


def img_to_b64(img: Image.Image, quality: int = 84) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def make_lut(stops: List[Tuple[float, Tuple[int, int, int, int]]], n: int = 256) -> np.ndarray:
    lut = np.zeros((n, 4), dtype=np.uint8)
    xs = np.linspace(0.0, 1.0, n)
    for k in range(4):
        pos = [s[0] for s in stops]
        val = [s[1][k] for s in stops]
        lut[:, k] = np.clip(np.interp(xs, pos, val), 0, 255).astype(np.uint8)
    return lut


HEAT_LUT = make_lut([
    (0.00, (10, 24, 30, 0)),
    (0.16, (22, 96, 116, 70)),
    (0.38, (46, 190, 152, 128)),
    (0.58, (255, 198, 62, 172)),
    (0.78, (255, 112, 42, 208)),
    (1.00, (255, 44, 70, 236)),
])

TRAFFIC_LUT = make_lut([
    (0.00, (10, 20, 26, 0)),
    (0.35, (36, 120, 140, 110)),
    (0.70, (86, 200, 230, 168)),
    (1.00, (236, 250, 255, 220)),
])


# ===================================================================================
# PROJECTOR
# ===================================================================================

class Projector:
    """
    Lightweight synthetic camera projection.
    World coordinates:
      x in [0, FIELD_W]
      y in [0, FIELD_H], where y=0 is near camera and y=FIELD_H is far.
    """

    def g2c(self, x: float, y: float) -> Tuple[float, float, float]:
        x = safe_float(x, FIELD_W * 0.5)
        y = safe_float(y, FIELD_H * 0.5)
        t = clamp01(y / FIELD_H)
        width_factor = (1.0 - t) * 1.0 + t * 0.42
        u = CAM_W * 0.5 + (x / FIELD_W - 0.5) * CAM_W * width_factor
        v = CAM_H * (0.94 - 0.62 * (t ** 0.75))
        scale = (1.0 - t) * 1.0 + t * 0.38
        return safe_float(u, CAM_W * 0.5), safe_float(v, CAM_H * 0.5), safe_float(scale, 0.5)


PROJ = Projector()


# ===================================================================================
# WORLD / SIMULATION
# ===================================================================================

@dataclass
class Entity:
    gid: int
    cls: str
    kind: str
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    L: float = 1.0
    W: float = 1.0
    H: float = 1.7
    cruise: float = 1.0
    color: Tuple[int, int, int] = (200, 200, 200)
    behavior: str = "wander"
    target: Optional[Tuple[float, float]] = None
    wps: List[Tuple[float, float]] = field(default_factory=list)
    wpi: int = 0
    vis: float = 1.0
    dust: float = 0.0
    shade: float = 0.0
    occ: float = 1.0
    reverse_until: float = -1.0
    estop_until: float = -1.0
    idle_until: float = -1.0
    flee_until: float = -1.0
    flee_dir: Tuple[float, float] = (1.0, 0.0)
    turn_rate: float = 1.2
    accel: float = 1.5
    decel: float = 2.2
    phase: float = 0.0
    tag: str = ""


SCENARIOS = {
    "dusk": dict(
        label="DUSK HARVEST · FIELD 7B",
        sun_elev=10.0,
        sun_az=252.0,
        dry=0.72,
        crowd=4,
        animals=3,
        machines=("harvester", "tractor", "truck", "utv"),
        night=0.18,
        dust=0.55,
        shade=0.85,
    ),
    "noondust": dict(
        label="NOON DUST SHIFT · PLOT 12",
        sun_elev=64.0,
        sun_az=186.0,
        dry=1.0,
        crowd=6,
        animals=2,
        machines=("tractor", "baler", "sprayer", "utv"),
        night=0.0,
        dust=1.0,
        shade=0.35,
    ),
    "night": dict(
        label="NIGHT IRRIGATION RUN · SECTOR 3",
        sun_elev=-12.0,
        sun_az=0.0,
        dry=0.35,
        crowd=2,
        animals=1,
        machines=("sprayer", "truck", "utv"),
        night=1.0,
        dust=0.25,
        shade=0.05,
    ),
    "yard": dict(
        label="LIVESTOCK YARD · CROWDED",
        sun_elev=22.0,
        sun_az=232.0,
        dry=0.6,
        crowd=9,
        animals=9,
        machines=("loader", "truck", "utv", "tractor"),
        night=0.05,
        dust=0.75,
        shade=0.6,
    ),
    "shadow": dict(
        label="SHADOW GRID MORNING · TREE LINE",
        sun_elev=13.0,
        sun_az=96.0,
        dry=0.4,
        crowd=5,
        animals=2,
        machines=("tractor", "harvester", "loader"),
        night=0.02,
        dust=0.3,
        shade=1.0,
    ),
}


class World:
    def __init__(self, scen: str = "dusk", seed: int = 7, params: Optional[Dict[str, Any]] = None):
        self.scen = scen
        self.cfg = dict(SCENARIOS[scen])
        self.params = params if params is not None else dict(DEFAULT_PARAMS)
        self.rng = random.Random(seed)
        self.t = 0.0
        self.entities: List[Entity] = []
        self.dust: List[Dict[str, float]] = []
        self.next_gid = 1
        self.pending: List[str] = []
        self.caption = self.cfg["label"]
        self.action_log: List[str] = []

        self.rects = [
            (-3.0, 29.5, 12.0, 9.0, "shed"),
            (43.0, -1.0, 9.0, 4.0, "trailer"),
            (24.0, 35.0, 14.0, 6.0, "haybank"),
        ]
        self.circs = [
            (55.0, 32.0, 3.0, "silo"),
            (55.0, 26.0, 3.0, "silo"),
            (13.0, 8.0, 2.3, "round bale"),
            (18.0, 6.0, 2.3, "round bale"),
        ]
        self.trees = [(x, 1.0 + 0.8 * math.sin(x * 0.7)) for x in np.arange(1.0, FIELD_W - 1.0, 2.8)]
        if self.scen == "shadow":
            self.trees += [(x, FIELD_H - 2.0 + 0.7 * math.cos(x * 0.5)) for x in np.arange(2.0, FIELD_W - 2.0, 3.2)]

        self.script = [
            (3.0, "worker_cross"),
            (8.5, "machine_reverse"),
            (14.0, "animal_dart"),
            (19.5, "occlusion_walk"),
            (25.0, "dust_burst"),
            (30.0, "crowd_push"),
            (36.0, "worker_cross"),
            (41.0, "machine_reverse"),
            (47.0, "animal_dart"),
        ]
        self.script_i = 0

        self.wind = np.array([0.55, 0.18]) * (0.4 + float(self.cfg["dry"]))
        self._spawn()
        self.plate = self._build_plate()

    # ------------------------------------------------------------------ env helpers
    def env(self, key: str, default: float = 1.0) -> float:
        return clamp01(safe_float(self.params.get(key, default), default))

    def night_level(self) -> float:
        return clamp01(safe_float(self.cfg["night"], 0.0) * self.env("night", 1.0))

    def dust_level(self) -> float:
        return clamp01(safe_float(self.cfg["dust"], 0.0) * self.env("dust", 1.0))

    def shadow_level(self) -> float:
        return clamp01(safe_float(self.cfg["shade"], 0.0) * self.env("shadow", 1.0))

    def sun_elev(self) -> float:
        return safe_float(self.cfg["sun_elev"], 0.0) * (1.0 - 0.8 * self.night_level())

    def sun_az(self) -> float:
        return safe_float(self.cfg["sun_az"], 0.0)

    def shadow_vec(self, h: float) -> Tuple[float, float]:
        h = max(0.2, safe_float(h, 1.0))
        el = max(3.0, self.sun_elev()) * math.pi / 180.0
        az = self.sun_az() * math.pi / 180.0
        length = min(22.0, h / max(1e-6, math.tan(el)))
        return -math.cos(az) * length, -math.sin(az) * length

    # ------------------------------------------------------------------ spawning
    def _add(self, cls: str, x: float, y: float, heading: float = 0.0, behavior: str = "wander", **kw) -> Optional[Entity]:
        s = SPEC[cls]
        e = Entity(
            gid=self.next_gid,
            cls=cls,
            kind=s["kind"],
            x=clamp(safe_float(x, FIELD_W * 0.5), 1.0, FIELD_W - 1.0),
            y=clamp(safe_float(y, FIELD_H * 0.5), 1.0, FIELD_H - 1.0),
            heading=safe_float(heading, 0.0),
            speed=s["cruise"] * 0.6,
            L=s["L"],
            W=s["W"],
            H=s["H"],
            cruise=s["cruise"],
            color=s["col"],
            behavior=behavior,
        )
        self.next_gid += 1

        if e.kind == "machine":
            e.turn_rate = 0.55 if cls in ("harvester", "sprayer", "truck") else 0.9
            e.accel, e.decel = 0.9, 2.6
        elif e.kind == "human":
            e.turn_rate, e.accel, e.decel = 2.4, 2.0, 2.6
        else:
            e.turn_rate, e.accel, e.decel = 3.2, 2.4, 2.2

        for k, v in kw.items():
            setattr(e, k, v)

        if len(self.entities) >= 95:
            removed = False
            for i, ent in enumerate(self.entities):
                if ent.kind != "machine":
                    del self.entities[i]
                    removed = True
                    break
            if not removed:
                return None

        self.entities.append(e)
        return e

    def _lane_wps(self, x0: float, x1: float, y0: float, y1: float) -> List[Tuple[float, float]]:
        wps = []
        y = y0
        up = True
        while y <= y1 + 0.01 and len(wps) < 20:
            wps.append((x1 if up else x0, y))
            y += 5.2
            up = not up
        return wps

    def _spawn(self) -> None:
        rnd = self.rng
        lanes = self._lane_wps(6.0, 54.0, 8.0, 34.0)

        for i, cls in enumerate(self.cfg["machines"]):
            if cls == "harvester":
                e = self._add(cls, 8.0 + i * 2.0, 12.0, 0.0, "lane", wps=list(lanes), wpi=i % max(1, len(lanes)))
            elif cls == "tractor":
                e = self._add(cls, 12.0, 24.0, 0.0, "lane", wps=list(reversed(lanes)), wpi=1)
            elif cls == "sprayer":
                e = self._add(cls, 20.0, 18.0, math.pi / 2, "lane",
                              wps=[(20, 8), (20, 32), (34, 32), (34, 8), (48, 8), (48, 32)])
            elif cls == "truck":
                e = self._add(cls, 50.0, 30.0, math.pi, "shuttle",
                              wps=[(50, 30), (10, 6.5), (50, 30), (30, 35.5)])
            elif cls == "loader":
                e = self._add(cls, 30.0, 10.0, 0.4, "circle", wps=[(30, 12)])
            else:
                e = self._add(cls, 40.0, 20.0, 1.0, "shuttle", wps=[(40, 20), (14, 28)])
            if e is not None:
                e.tag = f"M{i + 1}"

        n_h = int(self.cfg["crowd"] + 3 * self.env("crowd", 1.0))
        behaviors = ["escort", "cross", "patrol", "stand", "wander"]
        for i in range(n_h):
            cls = "spotter" if i == 0 else "worker"
            x = rnd.uniform(6, 54)
            y = rnd.uniform(5, 34)
            beh = behaviors[i % len(behaviors)]
            e = self._add(cls, x, y, rnd.uniform(0, 2 * math.pi), beh)
            if e is None:
                continue
            e.tag = f"H{i + 1}"
            e.target = (rnd.uniform(5, 55), rnd.uniform(4, 35))
            e.wps = [(rnd.uniform(5, 55), rnd.uniform(4, 35)) for _ in range(4)]
            e.phase = rnd.uniform(0, 2 * math.pi)

        n_a = int(self.cfg["animals"] + 2 * self.env("crowd", 1.0))
        for i in range(n_a):
            cls = ["cattle", "cattle", "sheep", "dog"][i % 4]
            e = self._add(cls, rnd.uniform(44, 57), rnd.uniform(33, 37), rnd.uniform(0, 2 * math.pi), "herd")
            if e is not None:
                e.tag = f"A{i + 1}"

    # ------------------------------------------------------------------ hazards
    def inject(self, kind: str) -> None:
        self.pending.append(kind)

    def _run_script(self) -> None:
        if self.script_i < len(self.script) and self.t >= self.script[self.script_i][0]:
            self.inject(self.script[self.script_i][1])
            self.script_i += 1
        elif self.script_i >= len(self.script) and self.rng.random() < 0.008:
            self.inject(self.rng.choice([
                "worker_cross", "machine_reverse", "animal_dart",
                "dust_burst", "occlusion_walk", "crowd_push"
            ]))

    def _hazard(self, kind: str) -> None:
        rnd = self.rng
        machines = [e for e in self.entities if e.kind == "machine"]
        humans = [e for e in self.entities if e.kind == "human"]
        if not machines:
            return

        m = rnd.choice(machines)

        if kind == "worker_cross":
            hx = m.x - math.cos(m.heading) * 12.0
            hy = m.y - math.sin(m.heading) * 12.0
            px = -math.sin(m.heading)
            py = math.cos(m.heading)
            w = self._add(
                "worker",
                clamp(hx + px * 8.0, 3, FIELD_W - 3),
                clamp(hy + py * 8.0, 3, FIELD_H - 3),
                behavior="cross",
            )
            if w is None:
                self.caption = "DIRECTOR · entity cap reached, worker crossing skipped"
                return
            w.target = (clamp(hx - px * 8.0, 3, FIELD_W - 3), clamp(hy - py * 8.0, 3, FIELD_H - 3))
            w.tag = f"H{w.gid}"
            self.caption = f"DIRECTOR · worker routed across {m.cls} guidance lane"

        elif kind == "machine_reverse":
            m.reverse_until = self.t + 5.0
            w = self._add(
                "worker",
                clamp(m.x - math.cos(m.heading) * 5.5, 2, FIELD_W - 2),
                clamp(m.y - math.sin(m.heading) * 5.5, 2, FIELD_H - 2),
                behavior="stand",
            )
            if w is not None:
                w.idle_until = self.t + 4.0
                w.tag = f"H{w.gid}"
            self.caption = f"DIRECTOR · {m.cls} engaged reverse with person in rear cone"

        elif kind == "animal_dart":
            a = self._add(
                "dog",
                clamp(m.x - math.sin(m.heading) * 8.0, 2, FIELD_W - 2),
                clamp(m.y + math.cos(m.heading) * 8.0, 2, FIELD_H - 2),
                behavior="dart",
            )
            if a is None:
                self.caption = "DIRECTOR · entity cap reached, animal bolt skipped"
                return
            a.target = (m.x + math.cos(m.heading) * 16.0, m.y + math.sin(m.heading) * 16.0)
            a.tag = f"A{a.gid}"
            self.caption = "DIRECTOR · animal bolted across implement path"

        elif kind == "dust_burst":
            for _ in range(45):
                self.dust.append(dict(
                    x=m.x + rnd.gauss(0, 4.0),
                    y=m.y + rnd.gauss(0, 4.0),
                    r=rnd.uniform(2.5, 6.5),
                    a=rnd.uniform(0.7, 1.5),
                    age=0.0,
                    life=rnd.uniform(4.0, 9.0),
                ))
            self.caption = "DIRECTOR · dust front engulfed sector · detector SNR degraded"

        elif kind == "occlusion_walk":
            if humans:
                w = rnd.choice(humans)
                w.behavior = "cross"
                w.target = (55.0, 29.0)
                self.caption = "DIRECTOR · person walked behind silo · re-ID stress test"

        elif kind == "crowd_push":
            tx, ty = m.x, m.y
            for _ in range(4):
                w = self._add(
                    "worker",
                    clamp(tx + rnd.gauss(0, 3.5), 3, FIELD_W - 3),
                    clamp(ty + rnd.gauss(0, 3.5), 3, FIELD_H - 3),
                    behavior="wander",
                )
                if w is None:
                    break
                w.target = (clamp(tx + rnd.uniform(-6, 6), 3, FIELD_W - 3), clamp(ty + rnd.uniform(-6, 6), 3, FIELD_H - 3))
                w.tag = f"H{w.gid}"
            self.caption = "DIRECTOR · crew converged on machine · crowded-scene mode"

        self.action_log.append(f"[{self.t:05.1f}s] {self.caption}")
        self.action_log = self.action_log[-40:]

    def brake_nearest(self, x: float, y: float, secs: float) -> Optional[Entity]:
        best = None
        bd = 1e18
        x = safe_float(x, FIELD_W * 0.5)
        y = safe_float(y, FIELD_H * 0.5)
        for e in self.entities:
            if e.kind != "machine" or not safe_point(e.x, e.y):
                continue
            d = math.hypot(e.x - x, e.y - y)
            if d < bd:
                bd = d
                best = e
        if best is not None and bd < 3.5:
            best.estop_until = self.t + max(0.1, safe_float(secs, 2.0))
            return best
        return None

    def flee_humans_near(self, x: float, y: float, radius: float, secs: float) -> None:
        x = safe_float(x, FIELD_W * 0.5)
        y = safe_float(y, FIELD_H * 0.5)
        radius = max(0.1, safe_float(radius, 8.0))
        secs = max(0.1, safe_float(secs, 2.5))
        for e in self.entities:
            if e.kind != "human" or not safe_point(e.x, e.y):
                continue
            d = math.hypot(e.x - x, e.y - y)
            if d < radius:
                e.flee_until = self.t + secs
                dx = e.x - x
                dy = e.y - y
                n = max(1e-6, math.hypot(dx, dy))
                e.flee_dir = (dx / n, dy / n)

    # ------------------------------------------------------------------ visibility
    def occ_factor(self, x: float, y: float) -> float:
        if not safe_point(x, y):
            return 0.0
        f = 1.0
        for cx, cy, cr, _ in self.circs:
            d = math.hypot(x - cx, y - cy)
            if d < cr:
                f = min(f, 0.03)
            elif d < cr + 1.0:
                f = min(f, 0.45)

        for rx, ry, rw, rh, _ in self.rects:
            if rx < x < rx + rw and ry < y < ry + rh:
                f = min(f, 0.05)
            elif (rx - 1.0) < x < (rx + rw + 1.0) and (ry - 1.0) < y < (ry + rh + 1.0):
                f = min(f, 0.5)

        for tx, ty in self.trees:
            d = math.hypot(x - tx, y - ty)
            if d < 2.2:
                f = min(f, 0.18)
            elif d < 3.1:
                f = min(f, 0.6)

        return f

    def dust_at(self, x: float, y: float) -> float:
        if not safe_point(x, y):
            return 0.0
        s = 0.0
        for d in self.dust:
            r = max(0.4, safe_float(d.get("r", 1.0), 1.0))
            dd = (x - safe_float(d.get("x", x), x)) ** 2 + (y - safe_float(d.get("y", y), y)) ** 2
            if dd < r * r * 4:
                life = max(0.1, safe_float(d.get("life", 5.0), 5.0))
                age = safe_float(d.get("age", 0.0), 0.0)
                amp = safe_float(d.get("a", 0.0), 0.0)
                s += amp * (1.0 - age / life) * math.exp(-dd / (r * r))
        return s

    def shade_at(self, x: float, y: float) -> float:
        if not safe_point(x, y) or self.sun_elev() <= 4.0:
            return 0.0

        sx, sy = self.shadow_vec(1.0)

        for e in self.entities:
            if e.kind != "machine" or not safe_point(e.x, e.y):
                continue
            px = e.x + sx * e.H
            py = e.y + sy * e.H
            rad = max(e.W * 0.8, e.L * 0.35)
            if math.hypot(x - px, y - py) < rad:
                return 1.0

        for tx, ty in self.trees:
            px = tx + sx * 6.0
            py = ty + sy * 6.0
            if math.hypot(x - px, y - py) < 2.5:
                return 1.0

        return 0.0

    def _visibility(self, e: Entity) -> None:
        if not safe_point(e.x, e.y):
            e.vis = 0.0
            return
        night = self.night_level()
        e.occ = self.occ_factor(e.x, e.y)
        e.dust = clamp01(self.dust_at(e.x, e.y) * 1.15 * self.dust_level())
        e.shade = clamp01(self.shade_at(e.x, e.y) * self.shadow_level())

        overlap = 0.0
        for o in self.entities:
            if o is e or o.kind != "machine" or not safe_point(o.x, o.y):
                continue
            if math.hypot(o.x - e.x, o.y - e.y) < max(2.0, o.W * 0.6):
                overlap = max(overlap, 0.55)

        vis = e.occ
        vis *= (1.0 - 0.85 * e.dust)
        vis *= (1.0 - 0.30 * e.shade)
        vis *= (1.0 - 0.45 * overlap)
        vis *= (1.0 - 0.42 * night)
        e.vis = clamp(safe_float(vis, 0.0), 0.02, 1.0)

    # ------------------------------------------------------------------ motion
    def _move(self, e: Entity, dt: float) -> None:
        if not safe_point(e.x, e.y):
            e.x = FIELD_W * 0.5
            e.y = FIELD_H * 0.5
            e.vx = e.vy = 0.0
            e.speed = 0.0

        t = self.t

        if t < e.idle_until:
            e.speed = max(0.0, e.speed - e.decel * dt * 2.0)
            e.vx = math.cos(e.heading) * e.speed
            e.vy = math.sin(e.heading) * e.speed
            e.x = clamp(e.x + e.vx * dt, 1.0, FIELD_W - 1.0)
            e.y = clamp(e.y + e.vy * dt, 1.0, FIELD_H - 1.0)
            return

        if t < e.flee_until:
            tx, ty = e.flee_dir
            e.target = (clamp(e.x + safe_float(tx, 1.0) * 12.0, 2, FIELD_W - 2),
                        clamp(e.y + safe_float(ty, 0.0) * 12.0, 2, FIELD_H - 2))
            cruise = 2.3
        else:
            cruise = e.cruise

        b = e.behavior

        if b in ("lane", "shuttle"):
            if not e.wps:
                e.wps = [(30.0, 20.0)]
            tx, ty = e.wps[e.wpi % len(e.wps)]
            if math.hypot(tx - e.x, ty - e.y) < 1.8:
                e.wpi += 1
            e.target = (tx, ty)

        elif b == "circle":
            if not e.wps:
                e.wps = [(30.0, 12.0)]
            cx, cy = e.wps[0]
            a = math.atan2(e.y - cy, e.x - cx) + dt * 0.42
            r = 7.0
            e.target = (cx + r * math.cos(a), cy + r * math.sin(a))

        elif b == "escort":
            machines = [m for m in self.entities if m.kind == "machine" and safe_point(m.x, m.y)]
            if machines:
                m = machines[e.gid % len(machines)]
                off = 8.5 + 1.5 * math.sin(t * 0.3 + e.phase)
                lat = 3.5 * math.sin(t * 0.17 + e.phase)
                e.target = (
                    m.x - math.cos(m.heading) * off - math.sin(m.heading) * lat,
                    m.y - math.sin(m.heading) * off + math.cos(m.heading) * lat,
                )

        elif b == "cross":
            if e.target is None or math.hypot(e.target[0] - e.x, e.target[1] - e.y) < 1.2:
                e.target = (self.rng.uniform(5, FIELD_W - 5), self.rng.uniform(4, FIELD_H - 4))

        elif b == "patrol":
            if not e.wps:
                e.wps = [(10, 10), (50, 10), (50, 32), (10, 32)]
            tx, ty = e.wps[e.wpi % len(e.wps)]
            if math.hypot(tx - e.x, ty - e.y) < 1.6:
                e.wpi += 1
                e.idle_until = t + self.rng.uniform(0.5, 2.5)
            e.target = (tx, ty)

        elif b == "stand":
            e.target = None

        elif b == "herd":
            if self.rng.random() < dt * 0.7 or e.target is None:
                e.target = (
                    clamp(self.rng.gauss(50, 6), 42, FIELD_W - 1.5),
                    clamp(self.rng.gauss(35, 3), 30, FIELD_H - 1.5),
                )

        elif b == "dart":
            cruise = 3.4
            if e.target and math.hypot(e.target[0] - e.x, e.target[1] - e.y) < 1.5:
                e.behavior = "herd"

        else:  # wander
            if (
                e.target is None
                or math.hypot(e.target[0] - e.x, e.target[1] - e.y) < 1.4
                or self.rng.random() < dt * 0.25
            ):
                e.target = (self.rng.uniform(4, FIELD_W - 4), self.rng.uniform(3, FIELD_H - 3))

        if e.target is None:
            e.speed = max(0.0, e.speed - e.decel * dt)
            e.vx = math.cos(e.heading) * e.speed
            e.vy = math.sin(e.heading) * e.speed
            e.x = clamp(e.x + e.vx * dt, 1.0, FIELD_W - 1.0)
            e.y = clamp(e.y + e.vy * dt, 1.0, FIELD_H - 1.0)
            return

        tx, ty = e.target
        tx = safe_float(tx, e.x)
        ty = safe_float(ty, e.y)
        desired = math.atan2(ty - e.y, tx - e.x)
        dh = ang_diff(desired, e.heading)
        e.heading += clamp(dh, -e.turn_rate * dt, e.turn_rate * dt)

        want = cruise * (1.0 - 0.5 * clamp01(abs(dh) / 1.3))
        d = math.hypot(tx - e.x, ty - e.y)
        if d < 1.0:
            want = 0.0

        if t < e.estop_until:
            want = 0.0
            e.speed = max(0.0, e.speed - 3.4 * dt)
        else:
            e.speed += clamp(want - e.speed, -e.decel * dt, e.accel * dt)

        sign = -1.0 if t < e.reverse_until else 1.0
        e.vx = math.cos(e.heading) * e.speed * sign
        e.vy = math.sin(e.heading) * e.speed * sign

        e.x = clamp(e.x + e.vx * dt, 1.0, FIELD_W - 1.0)
        e.y = clamp(e.y + e.vy * dt, 1.0, FIELD_H - 1.0)

        if e.x <= 1.0 or e.x >= FIELD_W - 1.0 or e.y <= 1.0 or e.y >= FIELD_H - 1.0:
            e.target = (clamp(e.x, 4, FIELD_W - 4), clamp(e.y, 4, FIELD_H - 4))

        if not safe_point(e.x, e.y):
            e.x = FIELD_W * 0.5
            e.y = FIELD_H * 0.5
            e.vx = e.vy = 0.0
            e.speed = 0.0

    def step(self, dt: float = SIM_DT) -> None:
        self.t += dt
        self._run_script()

        while self.pending:
            self._hazard(self.pending.pop(0))

        # dust update
        for d in self.dust:
            d["age"] = safe_float(d.get("age", 0.0), 0.0) + dt
            d["x"] = safe_float(d.get("x", FIELD_W * 0.5), FIELD_W * 0.5) + self.wind[0] * dt * 1.6
            d["y"] = safe_float(d.get("y", FIELD_H * 0.5), FIELD_H * 0.5) + self.wind[1] * dt * 1.6
            d["r"] = safe_float(d.get("r", 1.0), 1.0) + 0.55 * dt
        self.dust = [d for d in self.dust if safe_float(d.get("age", 0.0), 0.0) < safe_float(d.get("life", 5.0), 5.0)]
        if len(self.dust) > 240:
            self.dust = self.dust[-240:]

        # machine dust emission
        kdust = 0.55 + 1.5 * self.dust_level() * safe_float(self.cfg["dry"], 0.5)
        for e in self.entities:
            if e.kind == "machine" and e.speed > 0.6 and len(self.dust) < 220 and safe_point(e.x, e.y):
                if self.rng.random() < dt * e.speed * kdust * 0.45:
                    bx = e.x - math.cos(e.heading) * (e.L * 0.55)
                    by = e.y - math.sin(e.heading) * (e.L * 0.55)
                    self.dust.append(dict(
                        x=bx + self.rng.gauss(0, 0.9),
                        y=by + self.rng.gauss(0, 0.9),
                        r=self.rng.uniform(1.1, 2.6),
                        a=self.rng.uniform(0.35, 0.9),
                        age=0.0,
                        life=self.rng.uniform(2.5, 6.0),
                    ))

        for e in self.entities:
            self._move(e, dt)

        for e in self.entities:
            self._visibility(e)

    # ------------------------------------------------------------------ plate
    def _build_plate(self) -> Image.Image:
        img = Image.new("RGB", (BEV_W, BEV_H), (42, 36, 28))
        d = ImageDraw.Draw(img)

        for yy in range(BEV_H):
            f = yy / max(1, BEV_H - 1)
            col = (int(34 + 14 * f), int(30 + 13 * f), int(24 + 9 * f))
            d.line([(0, yy), (BEV_W, yy)], fill=col)

        for i in range(0, int(FIELD_H * 2.0)):
            y = 2.0 + i * 0.8
            if y > FIELD_H - 2.0:
                break
            py = int((FIELD_H - y) * S)
            shade = 10 if i % 2 == 0 else 0
            d.line(
                [(int(2.0 * S), py), (int((FIELD_W - 2.0) * S), py)],
                fill=(44 + shade, 62 + shade, 32 + shade),
                width=2,
            )

        for k in range(0, int(FIELD_W / 5.2) + 1):
            px = int((6.0 + k * 5.2) * S)
            d.line([(px, int(4 * S)), (px, int((FIELD_H - 4) * S))], fill=(62, 54, 39), width=1)

        d.rectangle(
            [int(2.0 * S), int(2.0 * S), int((FIELD_W - 2.0) * S), int((FIELD_H - 2.0) * S)],
            outline=(86, 74, 50),
            width=2,
        )

        for rx, ry, rw, rh, name in self.rects:
            x0 = int(rx * S)
            y0 = int((FIELD_H - (ry + rh)) * S)
            x1 = int((rx + rw) * S)
            y1 = int((FIELD_H - ry) * S)
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1
            d.rectangle([x0, y0, x1, y1], fill=(74, 78, 82), outline=(120, 126, 130), width=2)
            d.text((x0 + 4, y0 + 3), name.upper(), fill=(196, 202, 206), font=get_font(10))

        for cx, cy, cr, name in self.circs:
            px = int(cx * S)
            py = int((FIELD_H - cy) * S)
            rr = max(1, int(cr * S))
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(122, 124, 118), outline=(162, 164, 158), width=2)
            d.ellipse(
                [px - int(rr * 0.55), py - int(rr * 0.55), px + int(rr * 0.55), py + int(rr * 0.55)],
                outline=(90, 92, 88),
                width=1,
            )

        rnd = random.Random(1234)
        for tx, ty in self.trees:
            px = int(tx * S)
            py = int((FIELD_H - ty) * S)
            for _ in range(4):
                ox = int(rnd.gauss(0, 6))
                oy = int(rnd.gauss(0, 5))
                r = max(1, int(rnd.uniform(12, 22)))
                d.ellipse([px + ox - r, py + oy - r, px + ox + r, py + oy + r], fill=(28, 52, 30))

        d.text((10, 8), f"FIELD 7B · {FIELD_W:.0f} m × {FIELD_H:.0f} m · GRID 1 m", fill=(150, 160, 148), font=get_font(11))
        return img


# ===================================================================================
# DETECTOR
# ===================================================================================

@dataclass
class Detection:
    cls: str
    kind: str
    bbox: Tuple[float, float, float, float]
    conf: float
    x: float
    y: float
    vis: float
    L: float
    W: float
    H: float = 1.7
    source: str = "object"


class Detector:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.stats = dict(n=0, missed=0, ghosts=0, lowconf=0)

    def run(self, world: World) -> List[Detection]:
        p = self.params
        rng = world.rng
        night = world.night_level()
        dust = world.dust_level()
        shadow = world.shadow_level()
        noise = max(0.0, safe_float(p["det_noise"], 1.0))
        thr = clamp01(safe_float(p["conf_thresh"], 0.3))

        dets: List[Detection] = []
        missed = 0
        lowconf = 0

        for e in world.entities:
            if not safe_point(e.x, e.y):
                continue

            vis = clamp01(safe_float(e.vis, 1.0))
            snr = clamp01(vis * (1.0 - 0.35 * night))

            p_miss = clamp(0.015 + 0.90 * ((1.0 - snr) ** 1.4) + 0.12 * night, 0.0, 0.96)
            if rng.random() < p_miss:
                missed += 1
                continue

            u, v, scale = PROJ.g2c(e.x, e.y)
            wpx = max(5.0, safe_float(e.W, 1.0) * scale * 9.0)
            hpx = max(7.0, safe_float(e.H, 1.7) * scale * 10.0)

            sig_px = (1.5 + 8.0 * (1.0 - snr) + 3.0 * night) * noise
            du = rng.gauss(0.0, sig_px)
            dv = rng.gauss(0.0, sig_px * 0.75)

            s2 = 1.0 + rng.gauss(0.0, 0.05 + 0.20 * (1.0 - snr)) * noise
            s2 = max(0.2, safe_float(s2, 1.0))

            u1 = u - wpx * s2 * 0.5 + du
            u2 = u + wpx * s2 * 0.5 + du
            v1 = v - hpx * s2 + dv
            v2 = v + dv * 0.2
            bbox = norm_bbox((u1, v1, u2, v2), min_size=4.0)

            conf = KIND_BASE_CONF[e.kind]
            conf *= (0.35 + 0.65 * snr)
            conf *= (1.0 - 0.20 * night)
            conf *= max(0.05, rng.gauss(1.0, 0.05 + 0.13 * noise))
            conf = clamp01(safe_float(conf, 0.0))

            cls = e.cls
            if snr < 0.62 and rng.random() < 0.34 * (1.0 - snr):
                siblings = SIBLING.get(cls, [cls])
                cls = siblings[rng.randrange(len(siblings))]
                conf *= 0.82

            kind = SPEC.get(cls, SPEC["worker"])["kind"]

            if conf < thr:
                lowconf += 1
                continue

            sigma_m = 0.08 + 0.80 * (1.0 - snr) + 0.30 * night
            gx = e.x + rng.gauss(0.0, sigma_m)
            gy = e.y + rng.gauss(0.0, sigma_m)
            if not safe_point(gx, gy):
                continue

            L = max(0.35, safe_float(e.L, 1.0) * (1.0 + rng.gauss(0.0, 0.07 * noise)))
            W = max(0.35, safe_float(e.W, 1.0) * (1.0 + rng.gauss(0.0, 0.07 * noise)))
            H = max(0.50, safe_float(e.H, 1.7) * (1.0 + rng.gauss(0.0, 0.05 * noise)))

            dets.append(Detection(
                cls=cls,
                kind=kind,
                bbox=bbox,
                conf=conf,
                x=safe_float(gx, e.x),
                y=safe_float(gy, e.y),
                vis=vis,
                L=safe_float(L, e.L),
                W=safe_float(W, e.W),
                H=safe_float(H, e.H),
                source="object",
            ))

        # ghost detections from dust / shadow / sensor clutter
        ghost_rate = (0.10 + 1.20 * dust + 0.70 * shadow + 0.90 * night) * max(0.0, safe_float(p["ghost_rate"], 1.0))
        n_ghost = int(rng.random() * (2.2 * ghost_rate))
        ghosts = 0

        for _ in range(n_ghost):
            if world.dust and rng.random() < 0.55:
                dd = world.dust[rng.randrange(len(world.dust))]
                gx = safe_float(dd.get("x", FIELD_W * 0.5), FIELD_W * 0.5) + rng.gauss(0.0, 1.5)
                gy = safe_float(dd.get("y", FIELD_H * 0.5), FIELD_H * 0.5) + rng.gauss(0.0, 1.5)
            else:
                gx = rng.uniform(3.0, FIELD_W - 3.0)
                gy = rng.uniform(3.0, FIELD_H - 3.0)

            if not safe_point(gx, gy):
                continue

            u, v, scale = PROJ.g2c(gx, gy)
            cls = rng.choice(["worker", "cattle", "sheep"])
            kind = SPEC[cls]["kind"]
            conf = clamp01(rng.uniform(0.28, 0.62))

            wpx = max(3.0, rng.uniform(0.7, 2.4) * scale * 9.0)
            hpx = max(4.0, rng.uniform(1.0, 2.8) * scale * 10.0)
            H = max(0.5, SPEC[cls]["H"] * rng.uniform(0.75, 1.25))
            bbox = norm_bbox((u - wpx * 0.5, v - hpx, u + wpx * 0.5, v), min_size=4.0)

            if conf < thr:
                continue

            dets.append(Detection(
                cls=cls,
                kind=kind,
                bbox=bbox,
                conf=conf,
                x=safe_float(gx, FIELD_W * 0.5),
                y=safe_float(gy, FIELD_H * 0.5),
                vis=0.25,
                L=max(0.4, wpx / max(1e-6, scale * 9.0)),
                W=max(0.4, hpx / max(1e-6, scale * 10.0)),
                H=safe_float(H, SPEC[cls]["H"]),
                source="ghost",
            ))
            ghosts += 1

        dets = self._nms(dets, clamp01(safe_float(p["nms"], 0.45)))
        self.stats = dict(n=len(dets), missed=missed, ghosts=ghosts, lowconf=lowconf)
        return dets

    @staticmethod
    def _nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
        if len(dets) < 2:
            return dets

        dets = sorted(dets, key=lambda z: -z.conf)
        keep: List[Detection] = []

        for d in dets:
            ok = True
            for k in keep:
                if k.kind != d.kind:
                    continue
                if box_iou(d.bbox, k.bbox) > iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(d)

        return keep


# ===================================================================================
# TRACKER
# ===================================================================================

@dataclass
class Track:
    tid: int
    cls: str
    kind: str
    state: np.ndarray
    P: np.ndarray
    born: float
    last_seen: float
    hits: int = 1
    lost: int = 0
    status: str = "tentative"
    conf: float = 0.0
    vis: float = 1.0
    L: float = 1.0
    W: float = 1.0
    H: float = 1.7
    heading: float = 0.0
    body_heading: float = 0.0
    speed: float = 0.0
    reversing: bool = False
    curvature: float = 0.0
    trail: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=80))
    pred: List[Tuple[float, float, float, float]] = field(default_factory=list)
    pred_samples: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    zone: List[Tuple[float, float]] = field(default_factory=list)
    zone_area: float = 0.0
    stop_dist: float = 0.0
    risk: float = 0.0
    level: int = 0
    links: List[Dict[str, Any]] = field(default_factory=list)
    reid_count: int = 0
    cls_votes: Counter = field(default_factory=Counter)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def x(self) -> float:
        return safe_float(self.state[0], FIELD_W * 0.5)

    @property
    def y(self) -> float:
        return safe_float(self.state[1], FIELD_H * 0.5)

    @property
    def vx(self) -> float:
        return safe_float(self.state[2], 0.0)

    @property
    def vy(self) -> float:
        return safe_float(self.state[3], 0.0)


class Tracker:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.tracks: List[Track] = []
        self.archive: List[Dict[str, Any]] = []
        self.next_id = 1
        self.stats = Counter()

    def _make_track(self, d: Detection, t: float, tid: Optional[int] = None) -> Track:
        if tid is None:
            tid = self.next_id
            self.next_id += 1

        state = np.array([safe_float(d.x, FIELD_W * 0.5), safe_float(d.y, FIELD_H * 0.5), 0.0, 0.0], dtype=float)
        P = np.diag([0.35, 0.35, 3.0, 3.0])

        tr = Track(
            tid=tid,
            cls=d.cls,
            kind=d.kind,
            state=state,
            P=P,
            born=t,
            last_seen=t,
            hits=1,
            lost=0,
            status="tentative",
            conf=safe_float(d.conf, 0.0),
            vis=safe_float(d.vis, 1.0),
            L=safe_float(d.L, 1.0),
            W=safe_float(d.W, 1.0),
            H=safe_float(d.H, 1.7),
            bbox=norm_bbox(d.bbox, 4.0),
        )
        tr.cls_votes[d.cls] = 1
        return tr

    def _predict(self, tr: Track, dt: float) -> None:
        q = 0.55 if tr.kind == "machine" else (1.6 if tr.kind == "human" else 2.4)
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        if not np.all(np.isfinite(tr.state)):
            tr.state = np.array([FIELD_W * 0.5, FIELD_H * 0.5, 0.0, 0.0], dtype=float)
            tr.P = np.diag([1.0, 1.0, 1.0, 1.0])

        tr.state = F @ tr.state

        d2 = dt * dt
        Q = q * np.array([
            [d2 * d2 / 4.0, 0.0, d2 * dt / 2.0, 0.0],
            [0.0, d2 * d2 / 4.0, 0.0, d2 * dt / 2.0],
            [d2 * dt / 2.0, 0.0, d2, 0.0],
            [0.0, d2 * dt / 2.0, 0.0, d2],
        ], dtype=float)

        tr.P = F @ tr.P @ F.T + Q
        if not np.all(np.isfinite(tr.P)):
            tr.P = np.diag([1.0, 1.0, 1.0, 1.0])

    def _update_kf(self, tr: Track, d: Detection) -> None:
        Hmat = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        zx = safe_float(d.x, tr.x)
        zy = safe_float(d.y, tr.y)
        y = np.array([zx, zy], dtype=float) - Hmat @ tr.state
        R = np.eye(2, dtype=float) * max(0.01, 0.12 + 1.4 * (1.0 - safe_float(d.vis, 1.0)) + 0.3 * (1.0 - safe_float(d.conf, 1.0)))

        try:
            S = Hmat @ tr.P @ Hmat.T + R
            K = tr.P @ Hmat.T @ np.linalg.inv(S)
            tr.state = tr.state + K @ y
            tr.P = (np.eye(4, dtype=float) - K @ Hmat) @ tr.P
            tr.P = 0.5 * (tr.P + tr.P.T)
        except Exception:
            tr.state[0] = zx
            tr.state[1] = zy

        if not np.all(np.isfinite(tr.state)) or not np.all(np.isfinite(tr.P)):
            tr.state = np.array([zx, zy, 0.0, 0.0], dtype=float)
            tr.P = np.diag([0.5, 0.5, 2.0, 2.0])

    def _forecast(self, tr: Track, horizon: float, rng: random.Random) -> Tuple[List[Tuple[float, float, float, float]], List[List[Tuple[float, float, float]]]]:
        pred: List[Tuple[float, float, float, float]] = []
        samples: List[List[Tuple[float, float, float]]] = []

        x, y = tr.x, tr.y
        vx, vy = tr.vx, tr.vy
        h = math.atan2(vy, vx)
        kappa = safe_float(tr.curvature, 0.0)
        speed = max(0.25, safe_float(tr.speed, 0.0))
        horizon = max(1.0, safe_float(horizon, 5.0))

        if tr.kind == "machine":
            sig0, grow, turn = 0.20, 0.055, 0.02
        elif tr.kind == "human":
            sig0, grow, turn = 0.38, 0.24, 0.10
        else:
            sig0, grow, turn = 0.45, 0.30, 0.14

        dt = 0.25
        t = 0.0
        n = max(1, int(horizon / dt))

        for _ in range(n):
            t += dt
            h += clamp(kappa * dt, -0.8, 0.8)
            sp = speed * (0.985 if tr.kind == "machine" else 0.965)
            vx = math.cos(h) * sp
            vy = math.sin(h) * sp
            x += vx * dt
            y += vy * dt
            if not safe_point(x, y):
                x, y = tr.x, tr.y
            sig = sig0 + grow * t
            pred.append((t, safe_float(x, tr.x), safe_float(y, tr.y), safe_float(sig, sig0)))

        if tr.kind == "machine":
            nsamp = 2
        else:
            nsamp = 4

        for _ in range(nsamp):
            sx, sy = tr.x, tr.y
            sh = math.atan2(tr.vy, tr.vx)
            path: List[Tuple[float, float, float]] = [(0.0, sx, sy)]
            tt = 0.0
            for _ in range(n):
                tt += dt
                sh += rng.gauss(0.0, turn)
                sp = speed * (0.985 if tr.kind == "machine" else 0.965)
                sx += math.cos(sh) * sp * dt
                sy += math.sin(sh) * sp * dt
                if not safe_point(sx, sy):
                    sx, sy = tr.x, tr.y
                path.append((tt, safe_float(sx, tr.x), safe_float(sy, tr.y)))
            samples.append(path)

        return pred, samples

    def update(self, dets: List[Detection], t: float, dt: float) -> List[Track]:
        p = self.params
        rng = random.Random(int(t * 10000) & 0x7fffffff)

        for tr in self.tracks:
            self._predict(tr, dt)
            tr.lost += 1

        # ---------------- association ----------------
        matched_d: set[int] = set()

        if self.tracks and dets:
            cost: List[List[float]] = []
            for tr in self.tracks:
                row: List[float] = []
                tr_ok = safe_point(tr.x, tr.y)
                for d in dets:
                    if not tr_ok or not safe_point(d.x, d.y):
                        row.append(1.0e4)
                        continue

                    dist = math.hypot(d.x - tr.x, d.y - tr.y)
                    gate = max(0.5, safe_float(p["gate"], 3.0)) * (0.7 + 1.5 * (1.0 - safe_float(d.vis, 1.0))) \
                        + max(0.1, safe_float(p["max_speed"], 6.0)) * dt * 2.0

                    if dist > gate:
                        row.append(1.0e4)
                        continue

                    innov = np.array([d.x - tr.x, d.y - tr.y], dtype=float)
                    R = np.eye(2, dtype=float) * max(0.01, 0.15 + 1.2 * (1.0 - safe_float(d.vis, 1.0)))
                    try:
                        S = tr.P[:2, :2] + R
                        maha = float(innov @ np.linalg.solve(S, innov))
                    except Exception:
                        maha = dist * dist

                    maha = safe_float(maha, 1.0e4)
                    if maha > (max(0.5, safe_float(p["gate"], 3.0)) ** 2) * (1.0 + 1.6 * (1.0 - safe_float(d.vis, 1.0))):
                        row.append(1.0e4)
                        continue

                    class_pen = 0.0
                    if d.cls != tr.cls:
                        class_pen = 2.0 if d.kind == tr.kind else 8.0

                    l_det = safe_floor(d.L, 1.0, 0.25)
                    l_trk = safe_floor(tr.L, 1.0, 0.25)
                    w_det = safe_floor(d.W, 1.0, 0.25)
                    w_trk = safe_floor(tr.W, 1.0, 0.25)
                    size_pen = abs(math.log(l_det / l_trk)) + abs(math.log(w_det / w_trk))

                    c = math.sqrt(max(maha, 0.0)) + class_pen + 0.45 * clamp(size_pen, 0.0, 4.0) - 0.55 * safe_float(d.conf, 0.0)
                    row.append(safe_float(c, 1.0e4))

                cost.append(row)

            pairs = hungarian(cost)
            for i, j in pairs:
                if i >= len(self.tracks) or j >= len(dets):
                    continue
                if cost[i][j] > 9000.0:
                    continue

                tr = self.tracks[i]
                d = dets[j]
                self._update_kf(tr, d)

                tr.hits += 1
                tr.lost = 0
                tr.last_seen = t
                tr.conf = 0.70 * tr.conf + 0.30 * safe_float(d.conf, 0.0)
                tr.vis = 0.70 * tr.vis + 0.30 * safe_float(d.vis, 1.0)
                tr.L = 0.80 * tr.L + 0.20 * max(0.35, safe_float(d.L, 1.0))
                tr.W = 0.80 * tr.W + 0.20 * max(0.35, safe_float(d.W, 1.0))
                tr.H = 0.80 * tr.H + 0.20 * max(0.50, safe_float(d.H, 1.7))
                tr.bbox = norm_bbox(d.bbox, 4.0)

                tr.cls_votes[d.cls] += 1
                if sum(tr.cls_votes.values()) > 8:
                    tr.cls_votes = Counter({k: v * 0.5 for k, v in tr.cls_votes.items()})
                    tr.cls_votes[d.cls] += 1

                top_cls = tr.cls_votes.most_common(1)[0][0]
                if SPEC.get(top_cls, SPEC["worker"])["kind"] == tr.kind:
                    tr.cls = top_cls

                if tr.hits >= 3:
                    tr.status = "confirmed"

                matched_d.add(j)

        # ---------------- re-ID ----------------
        if bool(p.get("reid", True)):
            for j, d in enumerate(dets):
                if j in matched_d or not safe_point(d.x, d.y):
                    continue

                best = None
                best_score = -1.0

                for a in self.archive:
                    if t - safe_float(a.get("t", 0.0), 0.0) > max(0.1, safe_float(p["reid_window"], 6.0)):
                        continue
                    if SPEC.get(d.cls, SPEC["worker"])["kind"] != a.get("kind", ""):
                        continue

                    ax = safe_float(a.get("x", 0.0), 0.0)
                    ay = safe_float(a.get("y", 0.0), 0.0)
                    if not safe_point(ax, ay):
                        continue

                    dist = math.hypot(d.x - ax, d.y - ay)
                    if dist > 5.5:
                        continue

                    l_det = safe_floor(d.L, 1.0, 0.3)
                    l_arc = safe_floor(a.get("L", 1.0), 1.0, 0.3)
                    size_pen = abs(math.log(l_det / l_arc))

                    head_pen = abs(ang_diff(math.atan2(d.y - ay, d.x - ax), safe_float(a.get("h", 0.0), 0.0)))
                    score = (6.0 - dist) * 1.2 + (3.0 - size_pen) - 0.3 * head_pen

                    if score > best_score:
                        best_score = score
                        best = a

                if best is not None:
                    state = np.array([safe_float(d.x, FIELD_W * 0.5), safe_float(d.y, FIELD_H * 0.5),
                                      safe_float(best.get("vx", 0.0), 0.0), safe_float(best.get("vy", 0.0), 0.0)], dtype=float)
                    P = np.diag([0.30, 0.30, 1.5, 1.5])

                    tr = Track(
                        tid=int(best.get("tid", self.next_id)),
                        cls=d.cls,
                        kind=d.kind,
                        state=state,
                        P=P,
                        born=safe_float(best.get("born", t), t),
                        last_seen=t,
                        hits=int(best.get("hits", 3)),
                        lost=0,
                        status="confirmed",
                        conf=safe_float(d.conf, 0.0),
                        vis=safe_float(d.vis, 1.0),
                        L=safe_float(best.get("L", d.L), d.L),
                        W=safe_float(best.get("W", d.W), d.W),
                        H=safe_float(best.get("H", d.H), d.H),
                        body_heading=safe_float(best.get("h", 0.0), 0.0),
                        reid_count=int(best.get("reid_count", 0)) + 1,
                        bbox=norm_bbox(d.bbox, 4.0),
                    )
                    tr.trail = deque(best.get("trail", []), maxlen=80)
                    tr.cls_votes[d.cls] = 4

                    self.archive.remove(best)
                    self.tracks.append(tr)
                    matched_d.add(j)
                    self.stats["reid"] += 1

        # ---------------- new tracks ----------------
        for j, d in enumerate(dets):
            if j in matched_d or not safe_point(d.x, d.y):
                continue
            if safe_float(d.conf, 0.0) < max(0.0, safe_float(p["conf_thresh"], 0.3)) + 0.04:
                continue

            tr = self._make_track(d, t)
            self.tracks.append(tr)

        # ---------------- death / archive ----------------
        alive: List[Track] = []
        max_age = max(1, int(safe_float(p["max_age"], 12)))

        for tr in self.tracks:
            if tr.lost > max_age:
                if tr.status == "confirmed" and bool(p.get("reid", True)) and safe_point(tr.x, tr.y):
                    self.archive.append(dict(
                        tid=tr.tid,
                        born=tr.born,
                        x=tr.x,
                        y=tr.y,
                        vx=tr.vx,
                        vy=tr.vy,
                        h=tr.body_heading,
                        L=tr.L,
                        W=tr.W,
                        H=tr.H,
                        kind=tr.kind,
                        hits=tr.hits,
                        t=t,
                        trail=list(tr.trail)[-25:],
                        reid_count=tr.reid_count,
                    ))
                self.stats["dead"] += 1
            else:
                alive.append(tr)

        self.tracks = alive
        self.archive = self.archive[-35:]

        # ---------------- kinematics + forecast ----------------
        horizon = max(1.0, safe_float(p["horizon"], 5.0))

        for tr in self.tracks:
            if not safe_point(tr.x, tr.y):
                tr.state = np.array([FIELD_W * 0.5, FIELD_H * 0.5, 0.0, 0.0], dtype=float)
                tr.P = np.diag([1.0, 1.0, 1.0, 1.0])
                tr.speed = 0.0
                tr.heading = 0.0
                tr.body_heading = 0.0
                tr.pred = []
                tr.pred_samples = []
                continue

            vx, vy = tr.vx, tr.vy
            tr.speed = math.hypot(vx, vy)

            if tr.speed > 0.12:
                mv = math.atan2(vy, vx)
                if tr.lost == 0:
                    if abs(ang_diff(mv, tr.body_heading)) > 1.57:
                        tr.body_heading = mv + math.pi
                    else:
                        tr.body_heading += clamp(ang_diff(mv, tr.body_heading), -0.35, 0.35)
                tr.heading = mv

            tr.reversing = tr.speed > 0.22 and abs(ang_diff(math.atan2(vy, vx), tr.body_heading)) > 1.7

            if tr.lost == 0:
                tr.trail.append((t, tr.x, tr.y, tr.speed))

            if len(tr.trail) >= 6:
                a = tr.trail[-6]
                b = tr.trail[-1]
                if safe_point(a[1], a[2]) and safe_point(b[1], b[2]):
                    trail_h = math.atan2(b[2] - a[2], b[1] - a[1])
                    dth = ang_diff(trail_h, tr.body_heading)
                    dt_span = max(0.6, b[0] - a[0])
                    tr.curvature = 0.80 * tr.curvature + 0.20 * (-dth / max(0.5, dt_span * max(tr.speed, 0.35)))

            tr.pred, tr.pred_samples = self._forecast(tr, horizon, rng)

        self.stats["dets"] += len(dets)
        self.stats["frames"] += 1
        return self.tracks


# ===================================================================================
# SAFETY ENGINE
# ===================================================================================

@dataclass
class EventRec:
    eid: str
    t_open: float
    t_close: float = -1.0
    t_peak: float = 0.0
    level: int = 0
    mtid: int = -1
    htid: int = -1
    mcls: str = ""
    hcls: str = ""
    min_dist: float = 999.0
    min_ttc: float = 999.0
    peak_score: float = 0.0
    lead: float = 0.0
    x: float = 0.0
    y: float = 0.0
    tags: List[str] = field(default_factory=list)
    outcome: str = "OPEN"
    action: str = "NONE"
    dur: float = 0.0

    def row(self) -> Dict[str, Any]:
        return dict(
            id=self.eid,
            t_open=round(safe_float(self.t_open, 0.0), 1),
            t_close=round(safe_float(self.t_close, 0.0), 1),
            dur=round(safe_float(self.dur, 0.0), 1),
            level=LEVEL_NAME.get(self.level, "SAFE"),
            machine=f"#{self.mtid} {self.mcls}",
            person=f"#{self.htid} {self.hcls}",
            min_dist=round(safe_float(self.min_dist, 999.0), 2),
            min_ttc=round(safe_float(self.min_ttc, 999.0), 2),
            lead=round(safe_float(self.lead, 0.0), 2),
            x=round(safe_float(self.x, 0.0), 1),
            y=round(safe_float(self.y, 0.0), 1),
            action=self.action,
            outcome=self.outcome,
            tags="/".join(self.tags),
        )


class Safety:
    def __init__(self, params: Dict[str, Any], world: World):
        self.params = params
        self.world = world
        self.open: Dict[str, EventRec] = {}
        self.events: List[EventRec] = []
        self.closed_now: List[EventRec] = []
        self.top: List[Dict[str, Any]] = []
        self.counters = Counter()
        self.autobrake_until = -1.0

    def stop_distance(self, speed: float, reversing: bool = False) -> float:
        p = self.params
        speed = max(0.0, safe_float(speed, 0.0))
        sd = speed * max(0.01, safe_float(p["reaction"], 0.8)) + (speed * speed) / (2.0 * max(0.1, safe_float(p["braking"], 2.6)))
        if reversing:
            sd *= 1.35
        return safe_float(sd, 0.0)

    def make_zone(self, tr: Track) -> List[Tuple[float, float]]:
        if not safe_point(tr.x, tr.y):
            tr.zone = []
            tr.zone_area = 0.0
            tr.stop_dist = 0.0
            return []

        p = self.params
        buf = max(0.0, safe_float(p["buffer"], 1.2))
        sd = self.stop_distance(tr.speed, tr.reversing)
        tr.stop_dist = sd

        if tr.kind == "machine":
            front = max(3.0, sd + 1.0 + buf)
            rear = (5.5 if tr.reversing else 2.0) + buf * 0.6
            lat = max(1.2, safe_float(tr.W, 1.0) * 0.5 + 1.0 + buf * 0.45)
        else:
            front = rear = lat = 1.0 + buf * 0.5

        local = [
            (front, lat),
            (front, -lat),
            (-rear, -lat * 0.75),
            (-rear, lat * 0.75),
        ]

        c = math.cos(safe_float(tr.body_heading, 0.0))
        s = math.sin(safe_float(tr.body_heading, 0.0))
        poly = []
        for dx, dy in local:
            x = tr.x + dx * c - dy * s
            y = tr.y + dx * s + dy * c
            if safe_point(x, y):
                poly.append((x, y))

        tr.zone = poly
        tr.zone_area = poly_area(poly)
        return poly

    @staticmethod
    def blind_score(m: Track, x: float, y: float) -> float:
        if not safe_point(m.x, m.y) or not safe_point(x, y):
            return 0.0
        dx = x - m.x
        dy = y - m.y
        d = math.hypot(dx, dy)
        if d < 0.6:
            return 0.6

        ang = math.atan2(dy, dx)

        rear_ang = ang_diff(ang, m.body_heading + math.pi)
        if d < 12.0 and abs(rear_ang) < 0.85:
            return 1.0

        side_ang = ang_diff(ang, m.body_heading + math.pi / 2.0)
        if d < 7.0 and abs(side_ang) < 0.60:
            return 0.70

        if d < 4.0 and abs(rear_ang) < 2.6:
            return 0.45

        return 0.0

    def evaluate_pair(self, m: Track, h: Track) -> Dict[str, Any]:
        if not (safe_point(m.x, m.y) and safe_point(h.x, h.y)):
            return dict(dist=999.0, d_eff=999.0, t_eff=999.0, closing=0.0, score=0.0, level=0,
                        in_zone=False, blind=0.0, vis=1.0, tags=[])

        p = self.params
        hor = max(1.0, safe_float(p["horizon"], 5.0))

        rx = h.x - m.x
        ry = h.y - m.y
        dist = math.hypot(rx, ry)

        rvx = h.vx - m.vx
        rvy = h.vy - m.vy
        closing = -(rx * rvx + ry * rvy) / max(dist, 1e-6)

        v2 = rvx * rvx + rvy * rvy
        if v2 > 1e-6:
            t_cpa = -((rx * rvx + ry * rvy) / v2)
            if 0.0 < t_cpa < hor:
                d_cpa = math.hypot(rx + rvx * t_cpa, ry + rvy * t_cpa)
            else:
                t_cpa = 99.0
                d_cpa = dist
        else:
            t_cpa = 99.0
            d_cpa = dist

        # time-synced forecast separation
        mp: Dict[float, Tuple[float, float, float]] = {}
        for tt, xx, yy, ss in m.pred:
            mp[round(safe_float(tt, 0.0), 2)] = (safe_float(xx, m.x), safe_float(yy, m.y), safe_float(ss, 0.0))

        d_fore = dist
        t_fore = 99.0

        for tt, hx, hy, hs in h.pred:
            key = round(safe_float(tt, 0.0), 2)
            if key not in mp:
                continue
            mx, my, ms = mp[key]
            sep = math.hypot(safe_float(hx, h.x) - mx, safe_float(hy, h.y) - my) - safe_float(hs, 0.0) - ms
            sep = max(0.0, safe_float(sep, dist))
            if sep < d_fore:
                d_fore = sep
                t_fore = key

        d_eff = min(d_cpa, d_fore)
        t_eff = min(t_cpa if t_cpa > 0 else 99.0, t_fore)

        in_zone = point_in_poly(h.x, h.y, m.zone)
        blind = self.blind_score(m, h.x, h.y)
        vis = min(safe_float(m.vis, 1.0), safe_float(h.vis, 1.0))

        s_dist = clamp01(1.0 - d_eff / 12.0) ** 1.55
        s_ttc = clamp01(1.0 - t_eff / 8.5) ** 1.20
        s_rel = clamp01(closing / 5.5)
        s_env = 0.55 * (1.0 - vis) + 0.45 * blind
        s_zone = 1.0 if in_zone else 0.0

        score = 100.0 * (
            0.38 * s_dist +
            0.30 * s_ttc +
            0.10 * s_rel +
            0.12 * s_env +
            0.10 * s_zone
        )

        if m.reversing:
            score *= 1.12

        score = clamp(safe_float(score, 0.0), 0.0, 100.0)

        if d_eff < 1.7 and t_eff < 2.6:
            level = 3
        elif d_eff < 3.4 and t_eff < 5.2:
            level = 2
        elif (d_eff < 6.5 and t_eff < 10.0) or dist < m.stop_dist + 1.0:
            level = 1
        else:
            level = 0

        tags: List[str] = []
        if blind > 0.5:
            tags.append("BLIND_SPOT")
        if m.reversing:
            tags.append("REVERSE")
        if in_zone:
            tags.append("ZONE_INTRUSION")
        if abs(ang_diff(math.atan2(h.vy, h.vx), m.body_heading)) > 1.0:
            tags.append("CROSSING")
        if vis < 0.55:
            tags.append("LOW_VISIBILITY")
        if h.kind == "animal":
            tags.append("UNCONTROLLED_AGENT")

        return dict(
            dist=safe_float(dist, 999.0),
            d_eff=safe_float(d_eff, 999.0),
            t_eff=safe_float(t_eff, 999.0),
            closing=safe_float(closing, 0.0),
            score=score,
            level=level,
            in_zone=in_zone,
            blind=safe_float(blind, 0.0),
            vis=safe_float(vis, 1.0),
            tags=tags,
        )

    def update(self, tracks: List[Track], t: float, dt: float) -> List[Tuple[Track, Track, Dict[str, Any]]]:
        self.closed_now = []

        machines = [tr for tr in tracks if tr.kind == "machine" and tr.status == "confirmed" and safe_point(tr.x, tr.y)]
        humans = [tr for tr in tracks if tr.kind in ("human", "animal") and tr.status == "confirmed" and safe_point(tr.x, tr.y)]

        for m in machines:
            self.make_zone(m)

        for tr in tracks:
            tr.risk = 0.0
            tr.level = 0
            tr.links = []

        rows: List[Tuple[Track, Track, Dict[str, Any]]] = []

        for m in machines:
            for h in humans:
                r = self.evaluate_pair(m, h)
                rows.append((m, h, r))

                if r["score"] > m.risk:
                    m.risk = r["score"]
                    m.level = r["level"]
                if r["score"] > h.risk:
                    h.risk = r["score"]
                    h.level = r["level"]

                m.links.append(dict(
                    tid=h.tid,
                    kind=h.kind,
                    dist=r["dist"],
                    t_eff=r["t_eff"],
                    score=r["score"],
                    level=r["level"],
                    in_zone=r["in_zone"],
                ))

        rows.sort(key=lambda z: -z[2]["score"])
        self.top = [dict(m=m, h=h, r=r) for m, h, r in rows[:8]]

        seen: set[str] = set()

        for m, h, r in rows:
            if r["level"] < 2 and not (r["level"] == 1 and r["t_eff"] < 3.2):
                continue

            key = f"{m.tid}-{h.tid}"
            seen.add(key)

            ev = self.open.get(key)
            if ev is None:
                ev = EventRec(
                    eid=uuid.uuid4().hex[:8].upper(),
                    t_open=t,
                    mtid=m.tid,
                    htid=h.tid,
                    mcls=m.cls,
                    hcls=h.cls,
                )
                ev.lead = r["t_eff"]
                self.open[key] = ev
                self.counters["opened"] += 1

            ev.min_dist = min(ev.min_dist, r["d_eff"])
            ev.min_ttc = min(ev.min_ttc, r["t_eff"])
            ev.lead = max(ev.lead, r["t_eff"])

            if r["score"] > ev.peak_score:
                ev.peak_score = r["score"]
                ev.t_peak = t
                ev.level = r["level"]
                ev.x = (m.x + h.x) * 0.5
                ev.y = (m.y + h.y) * 0.5

            for tag in r["tags"]:
                if tag not in ev.tags:
                    ev.tags.append(tag)

            if r["level"] == 3 and t >= self.autobrake_until:
                ent = self.world.brake_nearest(m.x, m.y, 2.5)
                if ent is not None:
                    self.autobrake_until = t + 2.5
                    ev.action = "AUTO-BRAKE + CAB ALERT"
                    self.counters["auto_brake"] += 1
                    self.world.flee_humans_near(m.x, m.y, 8.0, 2.5)

        for key, ev in list(self.open.items()):
            if key in seen:
                continue

            if t - ev.t_peak > 1.2:
                ev.t_close = t
                ev.dur = t - ev.t_open

                if ev.min_dist < 1.5:
                    ev.outcome = "COLLISION_AVOIDED" if ev.action != "NONE" else "COLLISION"
                elif ev.min_dist < 4.0:
                    ev.outcome = "NEAR_MISS"
                elif ev.min_dist > 5.5:
                    ev.outcome = "FALSE_ALARM"
                    self.counters["false_alarm"] += 1
                else:
                    ev.outcome = "SAFE_PASS"

                self.events.append(ev)
                self.counters[LEVEL_NAME.get(ev.level, "SAFE")] += 1
                self.closed_now.append(ev)
                del self.open[key]

        self.events = self.events[-600:]
        return rows


# ===================================================================================
# HEATMAP
# ===================================================================================

class Heatmap:
    def __init__(self, cell: float = 1.0):
        self.cell = max(0.25, safe_float(cell, 1.0))
        self.gw = max(1, int(FIELD_W / self.cell))
        self.gh = max(1, int(FIELD_H / self.cell))
        self.risk = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.recent = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.event = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.traffic = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.dwell = np.zeros((self.gh, self.gw), dtype=np.float32)

    def _splat(self, grid: np.ndarray, x: float, y: float, weight: float, radius: int = 2) -> None:
        if not safe_point(x, y):
            return
        weight = max(0.0, safe_float(weight, 0.0))
        if weight <= 0:
            return
        col = int(clamp(x / self.cell, 0, self.gw - 1))
        row = int(clamp((FIELD_H - y) / self.cell, 0, self.gh - 1))

        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                rr = row + j
                cc = col + i
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    g = math.exp(-(i * i + j * j) / max(0.4, radius * 1.3))
                    grid[rr, cc] += weight * g

    def update(self, tracks: List[Track], safety: Safety, dt: float) -> None:
        decay = math.exp(-max(0.0, safe_float(dt, SIM_DT)) / 35.0)
        self.recent *= decay

        for item in safety.top:
            m = item["m"]
            h = item["h"]
            r = item["r"]
            if not (safe_point(m.x, m.y) and safe_point(h.x, h.y)):
                continue
            w = (safe_float(r["score"], 0.0) / 100.0) ** 1.6 * max(0.0, safe_float(dt, SIM_DT)) * 2.2
            mx = (m.x + h.x) * 0.5
            my = (m.y + h.y) * 0.5
            self._splat(self.risk, mx, my, w, 2)
            self._splat(self.recent, mx, my, w * 2.8, 2)

        for tr in tracks:
            if not safe_point(tr.x, tr.y):
                continue
            if tr.kind == "machine":
                self._splat(self.traffic, tr.x, tr.y, max(0.0, safe_float(dt, SIM_DT)) * 0.45, 2)
            elif tr.kind == "human":
                self._splat(self.dwell, tr.x, tr.y, max(0.0, safe_float(dt, SIM_DT)) * 0.45, 1)

    def note_event(self, ev: EventRec) -> None:
        w = 3.0 + safe_float(ev.level, 0.0) * 2.5
        self._splat(self.event, safe_float(ev.x, FIELD_W * 0.5), safe_float(ev.y, FIELD_H * 0.5), w, 3)

    def render_overlay(self, mode: str, base: Image.Image) -> Image.Image:
        grids = {
            "risk": self.risk,
            "recent": self.recent,
            "event": self.event,
            "traffic": self.traffic,
            "dwell": self.dwell,
        }
        grid = grids.get(mode, self.risk)
        grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(grid.max())
        if mx <= 1e-6:
            return base

        norm = np.clip(grid / mx, 0.0, 1.0) ** 0.72
        lut = TRAFFIC_LUT if mode in ("traffic", "dwell") else HEAT_LUT
        rgba = lut[(norm.ravel() * 255).astype(np.uint8)].reshape(self.gh, self.gw, 4)

        heat = Image.fromarray(rgba, "RGBA").resize((base.width, base.height), Image.BILINEAR)
        out = base.convert("RGB").copy()
        out.paste(heat, (0, 0), heat)
        return out


# ===================================================================================
# METRICS
# ===================================================================================

class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self.frag = 0
        self.gt_total = 0

        self.history: Deque[Dict[str, Any]] = deque(maxlen=180)
        self.last_map: Dict[int, int] = {}

        self.idtp: Dict[int, Counter] = defaultdict(Counter)
        self.gt_frames: Counter = Counter()
        self.tr_frames: Counter = Counter()

        self.latency: Deque[float] = deque(maxlen=300)
        self.pipe: Deque[float] = deque(maxlen=180)
        self.vis_buckets: Dict[str, Counter] = defaultdict(Counter)

    def update(self, tracks: List[Track], world: World, closed_events: List[EventRec], pipe_ms: float) -> Dict[str, float]:
        gt = [e for e in world.entities if e.vis >= 0.25 and safe_point(e.x, e.y)]
        cand = [tr for tr in tracks if tr.status == "confirmed" and safe_point(tr.x, tr.y)]

        pairs: List[Tuple[float, int, int]] = []
        for i, g in enumerate(gt):
            for j, tr in enumerate(cand):
                d = math.hypot(g.x - tr.x, g.y - tr.y)
                if d < max(1.5, g.L * 0.8):
                    pairs.append((d, i, j))

        pairs.sort()
        gi: set[int] = set()
        tj: set[int] = set()
        mapping: Dict[int, int] = {}

        for _d, i, j in pairs:
            if i in gi or j in tj:
                continue
            gi.add(i)
            tj.add(j)
            g = gt[i]
            tr = cand[j]
            mapping[g.gid] = tr.tid

            self.idtp[g.gid][tr.tid] += 1
            self.gt_frames[g.gid] += 1
            self.tr_frames[tr.tid] += 1

        self.tp += len(gi)
        self.fn += len(gt) - len(gi)
        self.fp += len(cand) - len(tj)
        self.gt_total += len(gt)

        for gid, tid in mapping.items():
            prev = self.last_map.get(gid)
            if prev is not None and prev != tid:
                self.idsw += 1
                self.frag += 1

        self.last_map = mapping

        for i, g in enumerate(gt):
            if g.vis > 0.80:
                bucket = "clear"
            elif g.vis > 0.50:
                bucket = "partial"
            elif g.vis > 0.30:
                bucket = "dusty"
            else:
                bucket = "blind"

            self.vis_buckets[bucket]["tot"] += 1
            if i in gi:
                self.vis_buckets[bucket]["hit"] += 1

        for ev in closed_events:
            self.latency.append(max(0.0, safe_float(ev.lead, 0.0)))

        self.pipe.append(max(0.0, safe_float(pipe_ms, 0.0)))

        mota = 1.0 - (self.fn + self.fp + self.idsw) / max(1, self.gt_total)
        mota = clamp(safe_float(mota, 0.0), -1.0, 1.0)

        idtp_sum = 0
        for c in self.idtp.values():
            if c:
                idtp_sum += max(c.values())

        idfn = sum(self.gt_frames.values()) - idtp_sum
        idfp = sum(self.tr_frames.values()) - idtp_sum
        idf1 = 2.0 * idtp_sum / max(1e-9, 2.0 * idtp_sum + idfp + idfn)
        idf1 = clamp(safe_float(idf1, 0.0), 0.0, 1.0)

        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)

        self.history.append(dict(
            t=round(safe_float(world.t, 0.0), 1),
            mota=mota,
            idf1=idf1,
            precision=precision,
            recall=recall,
            idsw=self.idsw,
            fp=self.fp,
            fn=self.fn,
            tracks=len(cand),
            pipe=safe_float(pipe_ms, 0.0),
        ))

        return dict(
            mota=mota,
            idf1=idf1,
            precision=precision,
            recall=recall,
            idsw=self.idsw,
            frag=self.frag,
            fp=self.fp,
            fn=self.fn,
            tp=self.tp,
        )


# ===================================================================================
# ENGINE
# ===================================================================================

class Engine:
    def __init__(self, scen: str = "dusk", params: Optional[Dict[str, Any]] = None, seed: int = 7):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)

        self.scen = scen
        self.world = World(scen=scen, seed=seed, params=self.params)
        self.detector = Detector(self.params)
        self.tracker = Tracker(self.params)
        self.safety = Safety(self.params, self.world)
        self.heat = Heatmap(1.0)
        self.metrics = Metrics()

        self.dets: List[Detection] = []
        self.tracks: List[Track] = []
        self.m: Dict[str, float] = dict(mota=0.0, idf1=0.0, precision=0.0, recall=0.0, idsw=0, frag=0, fp=0, fn=0, tp=0)

        self.event_rows: List[Dict[str, Any]] = []
        self._pushed_events: set[str] = set()

        self.frames = 0
        self.pipe_ms = 0.0
        self.record = True
        self.reel: Deque[np.ndarray] = deque(maxlen=180)

        self.bev = Image.new("RGB", (BEV_W, BEV_H), (20, 24, 20))
        self.cam = Image.new("RGB", (CAM_W, CAM_H), (20, 24, 20))

        self.warmup(18)
        self.render(record=False)

    def warmup(self, n: int = 18) -> None:
        for _ in range(max(0, int(n))):
            self.step(render=False)

    def step(self, render: bool = True) -> None:
        t0 = time.perf_counter()

        self.world.step(SIM_DT)
        self.dets = self.detector.run(self.world)
        self.tracks = self.tracker.update(self.dets, self.world.t, SIM_DT)
        self.safety.update(self.tracks, self.world.t, SIM_DT)
        self.heat.update(self.tracks, self.safety, SIM_DT)

        for ev in self.safety.closed_now:
            if ev.eid not in self._pushed_events:
                self._pushed_events.add(ev.eid)
                self.event_rows.append(ev.row())
                self.heat.note_event(ev)

        self.event_rows = self.event_rows[-1200:]

        pipe = (time.perf_counter() - t0) * 1000.0
        self.m = self.metrics.update(self.tracks, self.world, self.safety.closed_now, pipe)
        self.pipe_ms = 0.80 * self.pipe_ms + 0.20 * safe_float(pipe, 0.0)
        self.frames += 1

        if render:
            self.render(record=True)

    def render(self, record: bool = True) -> None:
        self.bev = draw_bev(self)
        self.cam = draw_camera(self)

        if record and self.record:
            frame = self.cam.resize((REEL_W, REEL_H), Image.BILINEAR)
            self.reel.append(np.asarray(frame))


# ===================================================================================
# DRAWING
# ===================================================================================

def world_to_bev(x: float, y: float) -> Tuple[float, float]:
    x = safe_float(x, FIELD_W * 0.5)
    y = safe_float(y, FIELD_H * 0.5)
    return x * S, (FIELD_H - y) * S


def rot_world(cx: float, cy: float, dx: float, dy: float, h: float) -> Tuple[float, float]:
    cx = safe_float(cx, FIELD_W * 0.5)
    cy = safe_float(cy, FIELD_H * 0.5)
    dx = safe_float(dx, 0.0)
    dy = safe_float(dy, 0.0)
    h = safe_float(h, 0.0)
    c = math.cos(h)
    s = math.sin(h)
    x = cx + dx * c - dy * s
    y = cy + dx * s + dy * c
    return safe_float(x, cx), safe_float(y, cy)


def safe_rect(box: Tuple[float, float, float, float], min_size: float = 2.0) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = norm_bbox(box, min_size=min_size)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_bev(engine: Engine) -> Image.Image:
    w = engine.world
    img = w.plate.copy()
    d = ImageDraw.Draw(img, "RGBA")

    night = w.night_level()
    if night > 0.02:
        d.rectangle([0, 0, BEV_W, BEV_H], fill=(10, 18, 35, int(110 * night)))

    if OVERLAYS["heatmap"]:
        mode = st.session_state.get("hm_mode", "risk")
        img = engine.heat.render_overlay(mode, img)
        d = ImageDraw.Draw(img, "RGBA")

    # shadows
    if w.sun_elev() > 4.0:
        for e in w.entities:
            if not safe_point(e.x, e.y):
                continue
            sx, sy = w.shadow_vec(e.H)
            bx, by = world_to_bev(e.x, e.y)
            ex, ey = world_to_bev(e.x + sx, e.y + sy)
            wid = max(3.0, safe_float(e.W, 1.0) * S * 0.45)
            pts = [(bx - wid, by), (bx + wid, by), (ex + wid * 0.7, ey), (ex - wid * 0.7, ey)]
            if all(safe_point(px, py) for px, py in pts):
                d.polygon(pts, fill=(6, 10, 8, int(90 * w.shadow_level())))

    # dust
    dl = w.dust_level()
    for dd in w.dust:
        x = safe_float(dd.get("x", 0.0), 0.0)
        y = safe_float(dd.get("y", 0.0), 0.0)
        if not safe_point(x, y):
            continue
        px, py = world_to_bev(x, y)
        r = max(2.0, safe_float(dd.get("r", 1.0), 1.0) * S * 0.65)
        life = max(0.1, safe_float(dd.get("life", 5.0), 5.0))
        age = safe_float(dd.get("age", 0.0), 0.0)
        a = int(105 * (1.0 - age / life) * dl)
        if a > 5:
            d.ellipse([px - r, py - r, px + r, py + r], fill=(176, 152, 112, min(a, 105)))

    # ground truth entities
    for e in sorted(w.entities, key=lambda z: (z.kind != "machine", safe_float(z.y, 0.0))):
        if not safe_point(e.x, e.y):
            continue
        px, py = world_to_bev(e.x, e.y)

        if e.kind == "machine":
            local = [
                (e.L / 2.0, e.W / 2.0),
                (e.L / 2.0, -e.W / 2.0),
                (-e.L / 2.0, -e.W / 2.0),
                (-e.L / 2.0, e.W / 2.0),
            ]
            poly = [world_to_bev(*rot_world(e.x, e.y, dx, dy, e.heading)) for dx, dy in local]
            if all(safe_point(qx, qy) for qx, qy in poly):
                d.polygon(poly, fill=e.color + (235,), outline=(20, 22, 20, 255))

            if e.cls in ("tractor", "baler", "sprayer"):
                il = [
                    (-e.L / 2.0 - 2.2, e.W * 0.38),
                    (-e.L / 2.0 - 2.2, -e.W * 0.38),
                    (-e.L / 2.0, -e.W * 0.38),
                    (-e.L / 2.0, e.W * 0.38),
                ]
                ipoly = [world_to_bev(*rot_world(e.x, e.y, dx, dy, e.heading)) for dx, dy in il]
                if all(safe_point(qx, qy) for qx, qy in ipoly):
                    d.polygon(ipoly, fill=(70, 66, 58, 220), outline=(30, 28, 24, 255))

            if e.cls == "harvester":
                hl = [
                    (e.L / 2.0 + 0.5, e.W * 0.75),
                    (e.L / 2.0 + 0.5, -e.W * 0.75),
                    (e.L / 2.0, -e.W * 0.45),
                    (e.L / 2.0, e.W * 0.45),
                ]
                hpoly = [world_to_bev(*rot_world(e.x, e.y, dx, dy, e.heading)) for dx, dy in hl]
                if all(safe_point(qx, qy) for qx, qy in hpoly):
                    d.polygon(hpoly, fill=(238, 196, 88, 235), outline=(60, 48, 20, 255))

            cab = [
                (e.L * 0.05, e.W * 0.30),
                (e.L * 0.05, -e.W * 0.30),
                (-e.L * 0.22, -e.W * 0.30),
                (-e.L * 0.22, e.W * 0.30),
            ]
            cpoly = [world_to_bev(*rot_world(e.x, e.y, dx, dy, e.heading)) for dx, dy in cab]
            if all(safe_point(qx, qy) for qx, qy in cpoly):
                d.polygon(cpoly, fill=(232, 236, 238, 190))

            if w.t < e.reverse_until:
                bl = [
                    (-e.L / 2.0 - 1.1, 0.9),
                    (-e.L / 2.0 - 1.1, -0.9),
                    (-e.L / 2.0 - 0.2, -0.35),
                    (-e.L / 2.0 - 0.2, 0.35),
                ]
                bpoly = [world_to_bev(*rot_world(e.x, e.y, dx, dy, e.heading)) for dx, dy in bl]
                if all(safe_point(qx, qy) for qx, qy in bpoly):
                    d.polygon(bpoly, fill=(255, 70, 70, 190))

        elif e.kind == "human":
            r = max(3.0, 0.38 * S)
            d.ellipse([px - r, py - r, px + r, py + r], fill=(28, 34, 38, 245), outline=e.color + (255,), width=2)
            d.ellipse([px - r * 0.45, py - r * 0.45, px + r * 0.45, py + r * 0.45], fill=(246, 250, 252, 255))
            hx = px + math.cos(e.heading) * r * 2.0
            hy = py - math.sin(e.heading) * r * 2.0
            if safe_point(hx, hy):
                d.line([(px, py), (hx, hy)], fill=e.color + (220,), width=2)

        else:
            rx = max(3.0, safe_float(e.L, 1.0) * 0.5 * S)
            ry = max(2.0, safe_float(e.W, 1.0) * 0.6 * S)
            d.ellipse([px - rx, py - ry, px + rx, py + ry], fill=e.color + (230,), outline=(40, 38, 36, 255))
            hx = px + math.cos(e.heading) * rx * 0.9
            hy = py - math.sin(e.heading) * ry * 0.9
            if safe_point(hx, hy):
                d.ellipse([hx - ry * 0.55, hy - ry * 0.55, hx + ry * 0.55, hy + ry * 0.55], fill=(60, 52, 46, 255))

    # analytics overlays
    for tr in engine.tracks:
        if not safe_point(tr.x, tr.y):
            continue
        col = {
            "machine": (255, 176, 32),
            "human": (57, 215, 242),
            "animal": (199, 155, 255),
        }.get(tr.kind, (220, 220, 220))

        if OVERLAYS["trails"] and len(tr.trail) > 2:
            pts = []
            for (_t, x, y, _s) in list(tr.trail)[-45:]:
                if safe_point(x, y):
                    pts.append(world_to_bev(x, y))
            if len(pts) > 1:
                d.line(pts, fill=col + (110,), width=2)

        if OVERLAYS["zones"] and tr.zone:
            zp = [world_to_bev(x, y) for x, y in tr.zone if safe_point(x, y)]
            if len(zp) >= 3:
                lc = LEVEL_RGB.get(tr.level, LEVEL_RGB[0])
                d.polygon(zp, fill=lc + (26,), outline=lc + (205,))
                d.line(zp + [zp[0]], fill=lc + (150,), width=2)

        if OVERLAYS["predict"] and tr.pred:
            pp = []
            for (_t, x, y, _s) in tr.pred:
                if safe_point(x, y):
                    pp.append(world_to_bev(x, y))
            for i in range(len(pp) - 1):
                a = int(210 - 160 * i / max(1, len(pp) - 1))
                d.line([pp[i], pp[i + 1]], fill=col + (a,), width=3 if i < 3 else 2)

            for k in (2, 6, 10, 14):
                if k < len(tr.pred):
                    _t, x, y, sig = tr.pred[k]
                    if safe_point(x, y):
                        px, py = world_to_bev(x, y)
                        r = max(2.0, safe_float(sig, 0.2) * S)
                        d.ellipse([px - r, py - r, px + r, py + r], outline=col + (90,), width=1)

            for samp in tr.pred_samples[:3]:
                sp = []
                for (_t, x, y) in samp:
                    if safe_point(x, y):
                        sp.append(world_to_bev(x, y))
                for i in range(0, len(sp) - 1, 2):
                    d.line([sp[i], sp[i + 1]], fill=col + (55,), width=1)

        if OVERLAYS["blind"] and tr.kind == "machine":
            wedge = [(tr.x, tr.y)]
            for aa in (2.35, 2.62, 2.90, 3.18, 3.45, 3.73):
                wedge.append(rot_world(tr.x, tr.y, math.cos(aa) * 9.5, math.sin(aa) * 9.5, tr.body_heading))
            wp = [world_to_bev(x, y) for x, y in wedge if safe_point(x, y)]
            if len(wp) >= 3:
                d.polygon(wp, fill=(150, 40, 60, 32))

        if tr.status == "confirmed":
            px, py = world_to_bev(tr.x, tr.y)
            rr = max(8.0, safe_float(tr.L, 1.0) * 0.55 * S)
            boxc = LEVEL_RGB.get(tr.level, (210, 224, 214)) if tr.level else (210, 224, 214)

            if tr.lost > 0:
                for i in range(0, 12, 4):
                    d.arc([px - rr, py - rr, px + rr, py + rr], i * 30, (i + 3) * 30, fill=(255, 255, 255, 150), width=2)
            else:
                rect = safe_rect((px - rr, py - rr * 0.8, px + rr, py + rr * 0.8), 4.0)
                if rect:
                    d.rounded_rectangle(list(rect), radius=3, outline=boxc + (255,), width=2)

            if OVERLAYS["ids"]:
                prefix = "M" if tr.kind == "machine" else ("H" if tr.kind == "human" else "A")
                lab = f"{prefix}{tr.tid}"
                lx0 = px - rr
                lx1 = px - rr + 8 * len(lab) + 8
                ly0 = py - rr * 0.8 - 15
                ly1 = py - rr * 0.8 - 2
                rect = safe_rect((lx0, ly0, lx1, ly1), 8.0)
                if rect:
                    d.rectangle(list(rect), fill=(8, 12, 10, 215), outline=boxc + (160,))
                    d.text((rect[0] + 4, rect[1] + 1), lab, fill=boxc + (255,), font=get_font(11))

                if tr.lost > 0:
                    d.text((px - rr + 4, py + rr * 0.8 + 2), f"COAST {tr.lost}", fill=(255, 210, 120, 230), font=get_font(9))

    if OVERLAYS["links"]:
        for item in engine.safety.top:
            m = item["m"]
            h = item["h"]
            r = item["r"]
            if r["level"] == 0 or not (safe_point(m.x, m.y) and safe_point(h.x, h.y)):
                continue

            lc = LEVEL_RGB.get(r["level"], LEVEL_RGB[0])
            p1 = world_to_bev(m.x, m.y)
            p2 = world_to_bev(h.x, h.y)
            d.line([p1, p2], fill=lc + (230,), width=1 + int(r["level"]))

            mx = (p1[0] + p2[0]) * 0.5
            my = (p1[1] + p2[1]) * 0.5
            txt = f"{safe_float(r['d_eff'], 0.0):.1f}m {max(0.0, safe_float(r['t_eff'], 99.0)):.1f}s"
            rect = safe_rect((mx - 2, my - 8, mx + 6 * len(txt) + 4, my + 6), 8.0)
            if rect:
                d.rectangle(list(rect), fill=(8, 12, 10, 220), outline=lc + (180,))
                d.text((rect[0] + 1, rect[1] + 1), txt, fill=lc + (255,), font=get_font(10))

    if OVERLAYS["grid"]:
        for gx in range(0, int(FIELD_W), 10):
            px, _ = world_to_bev(float(gx), 0.0)
            d.line([(px, 0), (px, BEV_H)], fill=(200, 220, 200, 22), width=1)
        for gy in range(0, int(FIELD_H), 10):
            _, py = world_to_bev(0.0, float(gy))
            d.line([(0, py), (BEV_W, py)], fill=(200, 220, 200, 22), width=1)

    confirmed_count = len([tr for tr in engine.tracks if tr.status == "confirmed" and safe_point(tr.x, tr.y)])
    hud = f"BEV FUSION · {w.caption} · T+{safe_float(w.t, 0.0):05.1f}s · TRACKS {confirmed_count}"
    d.rectangle([0, 0, BEV_W, 18], fill=(6, 10, 8, 220))
    d.text((6, 3), hud, fill=(190, 214, 196, 255), font=get_font(11))

    return img


def draw_camera(engine: Engine) -> Image.Image:
    w = engine.world
    img = Image.new("RGB", (CAM_W, CAM_H), (16, 22, 28))
    d = ImageDraw.Draw(img, "RGBA")

    night = w.night_level()

    # sky
    for yy in range(int(CAM_H * 0.34)):
        f = yy / max(1, int(CAM_H * 0.34) - 1)
        r = int(24 - 16 * night + 30 * f)
        g = int(30 - 20 * night + 34 * f)
        b = int(40 - 22 * night + 40 * f)
        d.line([(0, yy), (CAM_W, yy)], fill=(r, g, b))

    # ground
    for yy in range(int(CAM_H * 0.34), CAM_H):
        f = (yy - CAM_H * 0.34) / max(1, CAM_H * 0.66)
        r = int(38 + 18 * f - 12 * night)
        g = int(34 + 16 * f - 10 * night)
        b = int(26 + 10 * f - 8 * night)
        d.line([(0, yy), (CAM_W, yy)], fill=(r, g, b))

    # perspective field lines
    for gx in range(0, int(FIELD_W) + 1, 5):
        u0, v0, _ = PROJ.g2c(float(gx), 0.0)
        u1, v1, _ = PROJ.g2c(float(gx), FIELD_H)
        d.line([(u0, v0), (u1, v1)], fill=(70, 62, 45, 80), width=1)

    for gy in range(0, int(FIELD_H) + 1, 5):
        u0, v0, _ = PROJ.g2c(0.0, float(gy))
        u1, v1, _ = PROJ.g2c(FIELD_W, float(gy))
        d.line([(u0, v0), (u1, v1)], fill=(70, 62, 45, 70), width=1)

    # dust in camera
    dl = w.dust_level()
    for dd in w.dust:
        x = safe_float(dd.get("x", 0.0), 0.0)
        y = safe_float(dd.get("y", 0.0), 0.0)
        if not safe_point(x, y):
            continue
        u, v, sc = PROJ.g2c(x, y)
        r = max(2.0, safe_float(dd.get("r", 1.0), 1.0) * sc * 7.0)
        life = max(0.1, safe_float(dd.get("life", 5.0), 5.0))
        age = safe_float(dd.get("age", 0.0), 0.0)
        a = int(80 * (1.0 - age / life) * dl)
        if a > 4:
            d.ellipse([u - r, v - r * 0.65, u + r, v + r * 0.65], fill=(176, 152, 112, min(a, 80)))

    # detections
    for det in engine.dets:
        box = safe_rect(det.bbox, 4.0)
        if box is None:
            continue
        u1, v1, u2, v2 = box

        col = {
            "machine": (255, 190, 70),
            "human": (70, 225, 250),
            "animal": (200, 160, 255),
        }.get(det.kind, (220, 220, 220))

        if det.source == "ghost":
            col = (255, 90, 110)

        lw = 2 if safe_float(det.conf, 0.0) > 0.55 else 1
        d.rectangle([u1, v1, u2, v2], outline=col + (235,), width=lw)

        lab = f"{det.cls[:9]} {safe_float(det.conf, 0.0):.2f}"
        tw = 5.6 * len(lab) + 8
        lx0 = u1
        lx1 = u1 + tw
        if lx1 < lx0 + 20:
            lx1 = lx0 + 20

        if v1 >= 14:
            ly0 = v1 - 12
            ly1 = v1 - 1
            ty = v1 - 11
        else:
            ly0 = v1 + 2
            ly1 = v1 + 14
            ty = v1 + 3

        if ly1 < ly0 + 10:
            ly1 = ly0 + 10

        d.rectangle([lx0, ly0, lx1, ly1], fill=(6, 10, 8, 210))
        d.text((lx0 + 3, ty), lab, fill=col + (255,), font=get_font(9))

    # tracked objects
    if OVERLAYS["cam_tracks"]:
        for tr in engine.tracks:
            if tr.status != "confirmed" or not safe_point(tr.x, tr.y):
                continue

            u, v, sc = PROJ.g2c(tr.x, tr.y)
            wpx = max(6.0, safe_float(tr.W, 1.0) * sc * 9.0)
            hpx = max(8.0, safe_float(tr.H, 1.7) * sc * 10.0)

            lc = LEVEL_RGB.get(tr.level, (150, 235, 190)) if tr.level else (150, 235, 190)
            x1, y1 = u - wpx * 0.5, v - hpx
            x2, y2 = u + wpx * 0.5, v

            alpha = 120 if tr.lost > 0 else 245
            corner = max(4, int(min(wpx, hpx) * 0.22))

            d.line([(x1, y1), (x1, y1 + corner)], fill=lc + (alpha,), width=2)
            d.line([(x1, y1), (x1 + corner, y1)], fill=lc + (alpha,), width=2)
            d.line([(x2, y1), (x2, y1 + corner)], fill=lc + (alpha,), width=2)
            d.line([(x2, y1), (x2 - corner, y1)], fill=lc + (alpha,), width=2)
            d.line([(x1, y2), (x1, y2 - corner)], fill=lc + (alpha,), width=2)
            d.line([(x1, y2), (x1 + corner, y2)], fill=lc + (alpha,), width=2)
            d.line([(x2, y2), (x2, y2 - corner)], fill=lc + (alpha,), width=2)
            d.line([(x2, y2), (x2 - corner, y2)], fill=lc + (alpha,), width=2)

            lab = f"#{tr.tid}"
            d.text((x1, y1 - 11), lab, fill=lc + (alpha,), font=get_font(10))

            if OVERLAYS["predict"] and tr.pred:
                pts = []
                for _tt, px, py, _sig in tr.pred:
                    if safe_point(px, py):
                        pu, pv, _psc = PROJ.g2c(px, py)
                        pts.append((pu, pv))
                for i in range(len(pts) - 1):
                    a = int(200 - 150 * i / max(1, len(pts) - 1))
                    d.line([pts[i], pts[i + 1]], fill=lc + (a,), width=2)

    if OVERLAYS["cam_zones"]:
        for tr in engine.tracks:
            if tr.kind != "machine" or not tr.zone or not safe_point(tr.x, tr.y):
                continue
            pts = []
            for x, y in tr.zone:
                if safe_point(x, y):
                    u, v, _ = PROJ.g2c(x, y)
                    pts.append((u, v))
            if len(pts) >= 3:
                lc = LEVEL_RGB.get(tr.level, LEVEL_RGB[0])
                d.polygon(pts, outline=lc + (200,))

    if safe_float(w.t, 0.0) < safe_float(engine.safety.autobrake_until, -1.0):
        d.rectangle([0, CAM_H - 26, CAM_W, CAM_H], fill=(200, 30, 50, 215))
        d.text((8, CAM_H - 23), "AUTO-BRAKE ENGAGED · PROXIMITY GUARD ACTIVE", fill=(255, 240, 240), font=get_font(12))

    d.rectangle([0, 0, CAM_W, 16], fill=(6, 10, 8, 210))
    cam_label = "CAM-A · IR/LOW-LUX" if night > 0.45 else "CAM-A · DAYLIGHT"
    d.text((5, 2), cam_label, fill=(180, 205, 190), font=get_font(10))
    d.text((CAM_W - 58, 2), "● REC", fill=(255, 80, 90, 255), font=get_font(10))

    return img


# ===================================================================================
# HTML HELPERS
# ===================================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--void:#080B09;--panel:#0E1411;--edge:rgba(178,214,188,.11);--ink:#E9F1EA;--dim:#93AAA0;--faint:#5F7268;--amber:#FFB020;--cyan:#39D7F2;--violet:#C79BFF;--safe:#48D68C;--cau:#FFC542;--high:#FF7A33;--crit:#FF3B54;--disp:'Chakra Petch','Arial Narrow',sans-serif;--body:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',Consolas,monospace;}
html,body{font-family:var(--body);}
.stApp{background:radial-gradient(1100px 520px at 12% -8%, rgba(255,176,32,.07), transparent 60%),radial-gradient(900px 480px at 96% 4%, rgba(57,215,242,.06), transparent 62%),repeating-linear-gradient(0deg, rgba(255,255,255,.014) 0 1px, transparent 1px 3px),linear-gradient(180deg,#090D0B 0%, #080B09 40%, #070A08 100%);color:var(--ink);}
header,footer,[data-testid="stDecoration"],[data-testid="stToolbar"],#MainMenu{display:none!important;visibility:hidden;}
.block-container{padding:0.6rem 1.0rem 2.2rem 1.0rem;max-width:1720px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B100E,#080C0A);border-right:1px solid var(--edge);}
[data-testid="stSidebar"] .block-container{padding-top:.35rem;}
.as-hdr{display:flex;align-items:flex-end;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--edge);padding:6px 2px 10px;margin-bottom:10px;position:relative;}
.as-hdr::after{content:"";position:absolute;left:0;bottom:-1px;height:1px;width:34%;background:linear-gradient(90deg,var(--amber),transparent);animation:sweep 6s linear infinite;}
@keyframes sweep{0%{transform:translateX(0);opacity:.9}50%{opacity:.35}100%{transform:translateX(190%);opacity:.9}}
.mark{font-family:var(--disp);font-weight:700;font-size:30px;line-height:.95;letter-spacing:.06em;color:var(--ink);}
.mark i{font-style:normal;color:var(--amber);}.mark b{color:var(--cyan);font-weight:700;}
.sub{font-family:var(--mono);font-size:10px;letter-spacing:.20em;color:var(--faint);text-transform:uppercase;margin-top:3px;}
.hspacer{flex:1;}
.chip{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10.5px;letter-spacing:.10em;text-transform:uppercase;padding:5px 9px;border:1px solid var(--edge);background:rgba(255,255,255,.022);color:var(--dim);border-radius:3px;white-space:nowrap;}
.chip u{text-decoration:none;color:var(--ink);font-weight:500;}.chip.live{border-color:rgba(255,59,84,.5);color:#ffd9de;background:rgba(255,59,84,.09);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--crit);box-shadow:0 0 0 0 rgba(255,59,84,.6);animation:pulse 1.5s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,59,84,.55)}70%{box-shadow:0 0 0 8px rgba(255,59,84,0)}100%{box-shadow:0 0 0 0 rgba(255,59,84,0)}}
.panel{position:relative;background:linear-gradient(180deg,rgba(255,255,255,.028),rgba(255,255,255,.006));border:1px solid var(--edge);border-radius:4px;margin-bottom:10px;transition:border-color .25s, transform .25s;}
.panel:hover{border-color:rgba(255,176,32,.28);}
.panel>.tick{position:absolute;width:9px;height:9px;border:1px solid rgba(255,176,32,.5);pointer-events:none;}
.tk1{top:-1px;left:-1px;border-right:0;border-bottom:0}.tk2{top:-1px;right:-1px;border-left:0;border-bottom:0}.tk3{bottom:-1px;left:-1px;border-right:0;border-top:0}.tk4{bottom:-1px;right:-1px;border-left:0;border-top:0}
.ph{display:flex;align-items:baseline;justify-content:space-between;gap:8px;padding:7px 11px;border-bottom:1px solid var(--edge);background:rgba(255,255,255,.02);}
.ph span{font-family:var(--disp);font-weight:600;font-size:12.5px;letter-spacing:.16em;text-transform:uppercase;color:#DCE8DE;}
.ph em{font-family:var(--mono);font-style:normal;font-size:9.5px;letter-spacing:.13em;color:var(--faint);text-transform:uppercase;}
.pb{padding:10px 11px}.pb.tight{padding:7px 8px;}
img.feed{display:block;width:100%;border-radius:3px;border:1px solid rgba(255,255,255,.07);}
.k{font-family:var(--mono);font-size:9.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--faint);}
.v{font-family:var(--disp);font-weight:700;color:var(--ink);}
.big{font-family:var(--disp);font-weight:700;font-size:40px;line-height:.92;}
.mid{font-family:var(--disp);font-weight:700;font-size:20px;line-height:1;}
.mono{font-family:var(--mono);font-size:11.5px;color:var(--dim);}
.lv0{color:var(--safe)}.lv1{color:var(--cau)}.lv2{color:var(--high)}.lv3{color:var(--crit)}
.bg0{background:var(--safe)}.bg1{background:var(--cau)}.bg2{background:var(--high)}.bg3{background:var(--crit)}
.lvt{display:inline-block;font-family:var(--disp);font-weight:700;font-size:10.5px;letter-spacing:.13em;padding:2px 7px;border-radius:2px;color:#0A0D0B;}
.row{display:flex;align-items:center;gap:8px;padding:6px 7px;border-left:2px solid transparent;transition:.18s;border-radius:2px;}
.row:hover{background:rgba(255,255,255,.045);border-left-color:var(--amber);transform:translateX(2px);}
.row .nm{font-family:var(--disp);font-weight:600;font-size:13px;letter-spacing:.05em;min-width:52px;}
.row .mt{font-family:var(--mono);font-size:10.5px;color:var(--dim);margin-left:auto;white-space:nowrap;}
.bar{height:5px;border-radius:2px;background:rgba(255,255,255,.07);overflow:hidden;position:relative;margin:3px 0 7px;}
.bar>i{display:block;height:100%;border-radius:2px;transition:width .35s cubic-bezier(.3,.9,.3,1);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:7px;}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;}.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;}
.cell{border:1px solid var(--edge);border-radius:3px;padding:7px 8px;background:rgba(255,255,255,.018);transition:.2s;}.cell:hover{background:rgba(255,176,32,.06);border-color:rgba(255,176,32,.3);}
.tickwrap{overflow:hidden;border:1px solid var(--edge);border-radius:3px;background:rgba(255,255,255,.02);height:28px;position:relative;margin-bottom:10px;}
.ticklab{position:absolute;left:0;top:0;bottom:0;display:flex;align-items:center;padding:0 10px;z-index:2;background:#0C110F;border-right:1px solid var(--edge);font-family:var(--disp);font-size:10.5px;letter-spacing:.18em;color:var(--amber);}
.tickrun{display:flex;align-items:center;height:100%;white-space:nowrap;animation:run 42s linear infinite;padding-left:120px;}
@keyframes run{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tickrun span{font-family:var(--mono);font-size:11px;color:var(--dim);padding:0 16px;border-right:1px solid rgba(255,255,255,.06);}.tickrun span b{color:var(--ink);font-weight:500;}
.lg{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px;}.lg .b{font-family:var(--mono);font-size:10px;letter-spacing:.10em;text-transform:uppercase;padding:4px 8px;border:1px solid var(--edge);border-radius:2px;color:var(--faint);}.lg .b.on{color:#0A0D0B;font-weight:600;}
.crit{border:1px solid rgba(255,59,84,.55);background:linear-gradient(90deg,rgba(255,59,84,.20),rgba(255,59,84,.05));border-radius:3px;padding:8px 12px;display:flex;align-items:center;gap:12px;animation:flash 1.1s ease-in-out infinite;}
@keyframes flash{0%,100%{box-shadow:0 0 0 0 rgba(255,59,84,0)}50%{box-shadow:0 0 22px -4px rgba(255,59,84,.55)}}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--edge);}
[data-testid="stTabs"] button{font-family:var(--disp)!important;font-weight:600!important;letter-spacing:.13em!important;font-size:11.5px!important;text-transform:uppercase!important;color:var(--faint)!important;padding:6px 14px!important;background:transparent;border-radius:3px 3px 0 0;}
[data-testid="stTabs"] button:hover{color:var(--ink)!important;background:rgba(255,255,255,.04);}[data-testid="stTabs"] [aria-selected="true"]{color:#0A0D0B!important;background:var(--amber)!important;}
.stButton>button,.stDownloadButton>button{font-family:var(--disp)!important;font-weight:600!important;letter-spacing:.12em!important;text-transform:uppercase!important;font-size:11.5px!important;border-radius:3px!important;border:1px solid var(--edge)!important;background:rgba(255,255,255,.03)!important;color:var(--ink)!important;padding:6px 10px!important;min-height:32px!important;transition:.18s!important;width:100%;}
.stButton>button:hover,.stDownloadButton>button:hover{background:rgba(255,176,32,.14)!important;border-color:var(--amber)!important;color:#FFE2AE!important;transform:translateY(-1px);}
label{font-family:var(--mono)!important;font-size:10.5px!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--dim)!important;}
.sec{font-family:var(--disp);font-weight:700;font-size:11.5px;letter-spacing:.19em;text-transform:uppercase;color:var(--amber);margin:14px 0 4px;padding-bottom:4px;border-bottom:1px solid var(--edge);}.sec:first-child{margin-top:2px;}
.hint{font-family:var(--body);font-size:11.5px;color:var(--faint);line-height:1.5;margin:4px 0 2px;}
table.dt{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px;}table.dt th{text-align:left;font-family:var(--disp);font-weight:600;letter-spacing:.12em;font-size:9.5px;text-transform:uppercase;color:var(--faint);border-bottom:1px solid var(--edge);padding:5px 7px;}table.dt td{padding:5px 7px;border-bottom:1px solid rgba(255,255,255,.045);color:var(--dim);}table.dt tr:hover td{background:rgba(255,255,255,.035);color:var(--ink);}
.flow{display:flex;gap:6px;flex-wrap:wrap;align-items:stretch;}.fbox{flex:1;min-width:118px;border:1px solid var(--edge);border-left:2px solid var(--amber);border-radius:3px;padding:8px;background:rgba(255,255,255,.02);transition:.2s;}.fbox:hover{transform:translateY(-2px);border-left-color:var(--cyan);background:rgba(57,215,242,.06);}.fbox h5{margin:0 0 3px;font-family:var(--disp);font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink);}.fbox p{margin:0;font-family:var(--mono);font-size:10px;color:var(--faint);line-height:1.45;}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important;}}
"""


def panel(title: str, sub: str, body: str, tight: bool = False) -> str:
    cls = "pb tight" if tight else "pb"
    return (
        f'<div class="panel"><i class="tick tk1"></i><i class="tick tk2"></i>'
        f'<i class="tick tk3"></i><i class="tick tk4"></i>'
        f'<div class="ph"><span>{title}</span><em>{sub}</em></div>'
        f'<div class="{cls}">{body}</div></div>'
    )


def bar(pct: float, color: str, shimmer: bool = False) -> str:
    cls = "bar shimmer" if shimmer else "bar"
    return f'<div class="{cls}"><i style="width:{clamp(safe_float(pct, 0.0), 0, 100):.1f}%;background:{color}"></i></div>'


def img_html(img: Image.Image) -> str:
    return f'<img class="feed" src="data:image/jpeg;base64,{img_to_b64(img)}" alt="feed">'


def svg_dial(pct: float, lvl: int, label: str, value: str) -> str:
    r = 52
    cx = cy = 62
    circ = 2 * math.pi * r * 0.74
    val = circ * clamp01(safe_float(pct, 0.0))
    col = LEVEL_HEX.get(lvl, LEVEL_HEX[0])

    return (
        f'<svg viewBox="0 0 124 118" style="width:100%;max-width:190px;display:block;margin:0 auto">'
        f'<g transform="rotate(135 62 62)">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,.09)" stroke-width="9" '
        f'stroke-dasharray="{circ:.1f} {circ + 1:.1f}" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="9" '
        f'stroke-dasharray="{val:.1f} 999" stroke-linecap="round" style="transition:stroke-dasharray .4s"/>'
        f'</g>'
        f'<text x="62" y="58" text-anchor="middle" font-family="Chakra Petch,sans-serif" '
        f'font-size="34" font-weight="700" fill="{col}">{value}</text>'
        f'<text x="62" y="76" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="9" '
        f'letter-spacing="2" fill="#5F7268">{label}</text>'
        f'<text x="62" y="98" text-anchor="middle" font-family="Chakra Petch,sans-serif" font-size="13" '
        f'font-weight="700" letter-spacing="2" fill="{col}">{LEVEL_NAME.get(lvl, "SAFE")}</text>'
        f'</svg>'
    )


def svg_spark(vals: List[float], color: str = "#FFB020", h: int = 42) -> str:
    vals = [clamp01(safe_float(v, 0.0)) for v in vals]
    if len(vals) < 2:
        return f'<div style="height:{h}px"></div>'

    a = np.array(vals, dtype=float)
    w = 260
    xs = np.linspace(2, w - 2, len(a))
    ys = h - 3 - a * (h - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;display:block">'
        f'<polyline points="{xs[0]:.1f},{h - 1:.1f} {pts} {xs[-1]:.1f},{h - 1:.1f}" fill="{color}" opacity=".12"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.4" fill="{color}"/>'
        f'</svg>'
    )


def svg_hist(vals: List[float], h: int = 54, hi: float = 8.0) -> str:
    vals = [safe_float(v, 0.0) for v in vals]
    if not vals:
        return f'<div style="height:{h}px"></div>'

    arr = np.clip(np.array(vals, dtype=float), 0.0, hi)
    counts, _ = np.histogram(arr, bins=12, range=(0.0, hi))
    mx = max(1, int(counts.max()))
    w = 260
    bw = w / 12.0
    bars = []

    for i, c in enumerate(counts):
        bh = 38.0 * c / mx
        col = "#FFB020" if i < 6 else "#48D68C"
        op = 0.45 + 0.55 * c / mx
        bars.append(
            f'<rect x="{i * bw + 1:.1f}" y="{h - 8 - bh:.1f}" width="{bw - 2:.1f}" '
            f'height="{bh:.1f}" fill="{col}" opacity="{op:.2f}"/>'
        )

    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;display:block">'
        f'{"".join(bars)}'
        f'<text x="0" y="{h - 1}" font-family="IBM Plex Mono" font-size="8" fill="#5F7268">0s</text>'
        f'<text x="{w - 46}" y="{h - 1}" font-family="IBM Plex Mono" font-size="8" fill="#5F7268">{hi:.0f}s lead</text>'
        f'</svg>'
    )


def rerun_app() -> None:
    fn = getattr(st, "rerun", None)
    if fn is None:
        fn = getattr(st, "experimental_rerun", None)
    if fn is not None:
        fn()


def make_engine(scen: str) -> Engine:
    params = dict(DEFAULT_PARAMS)
    if "engine" in st.session_state:
        params.update(st.session_state.engine.params)
    return Engine(scen=scen, params=params, seed=random.randint(1, 9999))


def sync_params(engine: Engine) -> None:
    p = engine.params

    for k in [
        "speed", "conf_thresh", "det_noise", "ghost_rate", "nms",
        "max_age", "gate", "max_speed", "reid_window", "horizon",
        "reaction", "braking", "buffer", "dust", "shadow", "night", "crowd",
    ]:
        if k in st.session_state:
            p[k] = safe_float(st.session_state[k], DEFAULT_PARAMS.get(k, 1.0))

    p["reid"] = bool(st.session_state.get("reid", True))

    for k in OVERLAYS:
        if k in st.session_state:
            OVERLAYS[k] = bool(st.session_state[k])

    engine.world.params = p
    engine.record = bool(st.session_state.get("record", True))


# ===================================================================================
# STREAMLIT APP
# ===================================================================================

st.set_page_config(
    page_title="AGRI//SENTINEL · Human-Machine Proximity Guard",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = Engine("dusk", dict(DEFAULT_PARAMS), seed=7)
    st.session_state.playing = True
    st.session_state.scen = "dusk"
else:
    st.session_state.setdefault("playing", True)
    st.session_state.setdefault("scen", "dusk")

engine: Engine = st.session_state.engine

# -------------------------------------- SIDEBAR -------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="mark">AGRI<i>//</i><b>SENTINEL</b></div>'
        '<div class="sub">Proximity Guard · edge build</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Scene / Scenario</div>', unsafe_allow_html=True)
    scen_keys = list(SCENARIOS.keys())
    sel = st.radio(
        "Dataset",
        scen_keys,
        format_func=lambda k: SCENARIOS[k]["label"],
        index=scen_keys.index(st.session_state.get("scen", "dusk")),
        key="scen_radio",
        label_visibility="collapsed",
    )

    if sel != st.session_state.get("scen", "dusk"):
        st.session_state.scen = sel
        st.session_state.engine = make_engine(sel)
        rerun_app()

    st.markdown(f'<div class="hint">{engine.world.caption}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Playback</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    if c1.button("Pause" if st.session_state.playing else "Play", key="pp"):
        st.session_state.playing = not st.session_state.playing

    if c2.button("Step", key="stp"):
        st.session_state.force_step = True

    if c3.button("Reset", key="rst"):
        st.session_state.engine = make_engine(st.session_state.get("scen", "dusk"))
        rerun_app()

    st.slider("Simulation speed ×", 0.25, 3.0, float(engine.params["speed"]), 0.05, key="speed")
    st.checkbox("Record demo reel buffer", value=True, key="record")

    st.markdown('<div class="sec">Inject hazard</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    if g1.button("Worker cross", key="hz1"):
        engine.world.inject("worker_cross")
        st.session_state.force_step = True

    if g2.button("Machine reverse", key="hz2"):
        engine.world.inject("machine_reverse")
        st.session_state.force_step = True

    if g1.button("Animal bolt", key="hz3"):
        engine.world.inject("animal_dart")
        st.session_state.force_step = True

    if g2.button("Dust burst", key="hz4"):
        engine.world.inject("dust_burst")
        st.session_state.force_step = True

    if g1.button("Occlusion walk", key="hz5"):
        engine.world.inject("occlusion_walk")
        st.session_state.force_step = True

    if g2.button("Crew converge", key="hz6"):
        engine.world.inject("crowd_push")
        st.session_state.force_step = True

    st.markdown('<div class="sec">Vision front-end</div>', unsafe_allow_html=True)
    st.selectbox(
        "Detector profile",
        ["FieldNet-6m · daytime", "FieldNet-6m · IR/low-lux", "YOLO-compatible hook (drop-in)"],
        key="detprof",
    )
    st.slider("Confidence gate", 0.05, 0.85, float(engine.params["conf_thresh"]), 0.01, key="conf_thresh")
    st.slider("Localisation noise ×", 0.0, 3.0, float(engine.params["det_noise"]), 0.05, key="det_noise")
    st.slider("Ghost / clutter rate ×", 0.0, 3.0, float(engine.params["ghost_rate"]), 0.05, key="ghost_rate")
    st.slider("NMS IoU", 0.10, 0.90, float(engine.params["nms"]), 0.01, key="nms")

    st.markdown('<div class="sec">Multi-object tracker</div>', unsafe_allow_html=True)
    st.slider("Coast tolerance frames", 3, 40, int(engine.params["max_age"]), 1, key="max_age")
    st.slider("Mahalanobis gate σ", 1.5, 6.0, float(engine.params["gate"]), 0.05, key="gate")
    st.slider("Max object speed m/s", 1.0, 12.0, float(engine.params["max_speed"]), 0.1, key="max_speed")
    st.checkbox("Re-ID archive", value=bool(engine.params.get("reid", True)), key="reid")
    st.slider("Re-ID memory s", 1.0, 20.0, float(engine.params["reid_window"]), 0.5, key="reid_window")

    st.markdown('<div class="sec">Safety model</div>', unsafe_allow_html=True)
    st.slider("Operator reaction s", 0.2, 2.5, float(engine.params["reaction"]), 0.05, key="reaction")
    st.slider("Braking decel m/s²", 0.8, 6.0, float(engine.params["braking"]), 0.1, key="braking")
    st.slider("Personnel buffer m", 0.0, 4.0, float(engine.params["buffer"]), 0.1, key="buffer")
    st.slider("Prediction horizon s", 2.0, 12.0, float(engine.params["horizon"]), 0.5, key="horizon")

    st.markdown('<div class="sec">Environment stress</div>', unsafe_allow_html=True)
    st.slider("Dust load", 0.0, 1.5, float(engine.params["dust"]), 0.02, key="dust")
    st.slider("Shadow contrast", 0.0, 1.5, float(engine.params["shadow"]), 0.02, key="shadow")
    st.slider("Night / low-lux", 0.0, 1.0, float(engine.params["night"]), 0.02, key="night")
    st.slider("Crowding factor", 0.0, 1.5, float(engine.params["crowd"]), 0.02, key="crowd")

    st.markdown('<div class="sec">Overlays</div>', unsafe_allow_html=True)
    oc1, oc2 = st.columns(2)

    with oc1:
        st.checkbox("Tracks + IDs", value=OVERLAYS["ids"], key="ids")
        st.checkbox("Trails", value=OVERLAYS["trails"], key="trails")
        st.checkbox("Forecast", value=OVERLAYS["predict"], key="predict")
        st.checkbox("Safety zones", value=OVERLAYS["zones"], key="zones")
        st.checkbox("Risk links", value=OVERLAYS["links"], key="links")

    with oc2:
        st.checkbox("Blind wedges", value=OVERLAYS["blind"], key="blind")
        st.checkbox("Heatmap", value=OVERLAYS["heatmap"], key="heatmap")
        st.checkbox("Metric grid", value=OVERLAYS["grid"], key="grid")
        st.checkbox("Cam track brackets", value=OVERLAYS["cam_tracks"], key="cam_tracks")
        st.checkbox("Cam zones", value=OVERLAYS["cam_zones"], key="cam_zones")

    st.markdown(
        '<div class="hint">Deterministic demo engine. No API key, no cloud call, '
        'no model weights download. Replace Detector.run() with a real detector wrapper for live footage.</div>',
        unsafe_allow_html=True,
    )

sync_params(engine)
engine = st.session_state.engine

# -------------------------------------- STEP ----------------------------------------
loop_t0 = time.perf_counter()
force = st.session_state.pop("force_step", False)
steps = 1

if st.session_state.playing:
    steps = max(1, int(round(3 * float(engine.params["speed"]))))
    for _ in range(steps):
        engine.step(render=False)
    engine.render(record=True)
elif force:
    engine.step(render=False)
    engine.render(record=True)
else:
    engine.render(record=False)

# -------------------------------------- UI ------------------------------------------
w = engine.world
tracks = engine.tracks
m = engine.m

confirmed = [tr for tr in tracks if tr.status == "confirmed" and safe_point(tr.x, tr.y)]
lvl = max([tr.level for tr in tracks] + [0])
score = max([tr.risk for tr in tracks] + [0.0])
nearest = engine.safety.top[0] if engine.safety.top else None

uph = int(safe_float(w.t, 0.0) // 3600) % 24
umin = int(safe_float(w.t, 0.0) // 60) % 60
usec = safe_float(w.t, 0.0) % 60

mota_pct = max(0.0, safe_float(m["mota"], 0.0)) * 100.0
mota_cls = 0 if mota_pct > 90 else (1 if mota_pct > 75 else 2)

header_html = (
    '<div class="as-hdr">'
    '<div>'
    '<div class="mark">AGRI<i>//</i><b>SENTINEL</b></div>'
    f'<div class="sub">Human-machine proximity guard · {w.cfg["label"]}</div>'
    '</div>'
    '<div class="hspacer"></div>'
    f'<div class="chip">◷ SIM <u>{uph:02d}:{umin:02d}:{usec:04.1f}</u></div>'
    f'<div class="chip">▷ PIPE <u>{safe_float(engine.pipe_ms, 0.0):.1f} ms</u></div>'
    f'<div class="chip">◉ DET <u>{engine.detector.stats["n"]}/f · {len(confirmed)} trk</u></div>'
    f'<div class="chip">⛭ MOTA <u class="lv{mota_cls}">{mota_pct:.1f}%</u></div>'
    f'<div class="chip">⚠ EVENTS <u>{len(engine.event_rows)}</u></div>'
    f'<div class="chip live"><span class="dot"></span>{"LIVE" if st.session_state.playing else "PAUSED"}</div>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

if lvl >= 3 and nearest is not None:
    nm = nearest["m"]
    nh = nearest["h"]
    nr = nearest["r"]
    banner_html = (
        '<div class="crit">'
        '<span style="font-size:22px">⛔</span>'
        '<div>'
        '<div class="mid lv3">CRITICAL · STOP / EVACUATE</div>'
        f'<div class="mono">{nm.cls.upper()} #{nm.tid} is {safe_float(nr["d_eff"], 999.0):.2f} m from person #{nh.tid} '
        f'({nh.cls.upper()}) · time-to-collision {max(0.0, safe_float(nr["t_eff"], 999.0)):.2f} s · actuator command issued</div>'
        '</div></div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)
elif lvl == 2 and nearest is not None:
    nm = nearest["m"]
    nh = nearest["h"]
    nr = nearest["r"]
    banner_html = (
        '<div class="panel"><div class="pb tight" style="border-left:2px solid var(--high)">'
        '<span class="lvt bg2">HIGH RISK</span>'
        f'<span class="mono">&nbsp;{nm.cls.upper()} #{nm.tid} ↔ person #{nh.tid} · '
        f'{safe_float(nr["d_eff"], 999.0):.2f} m · TTC {safe_float(nr["t_eff"], 999.0):.2f} s · forecast crossing inside dynamic zone</span>'
        '</div></div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)
else:
    st.markdown("", unsafe_allow_html=True)

# ticker
items = engine.safety.top[:8]
if items:
    seq_parts = []
    for it in items:
        r = it["r"]
        txt = (
            f'{LEVEL_NAME.get(r["level"], "SAFE")} · {it["m"].cls} #{it["m"].tid} ↔ '
            f'{it["h"].cls} #{it["h"].tid} · {safe_float(r["d_eff"], 999.0):.1f} m · TTC {max(0.0, safe_float(r["t_eff"], 999.0)):.1f}s'
        )
        seq_parts.append(f'<span><b class="lv{r["level"]}">▮</b> {txt}</span>')
    seq = "".join(seq_parts)
else:
    seq = '<span><b class="lv0">▮</b> SYSTEM NOMINAL · no active machine-person conflict</span>'

st.markdown(
    f'<div class="tickwrap"><div class="ticklab">EVENT FEED</div><div class="tickrun">{seq}{seq}</div></div>',
    unsafe_allow_html=True,
)

col_video, col_rail = st.columns([2.35, 1.0])

with col_video:
    st.markdown(
        panel("Bird's-eye fused track view", "60 × 38 m · 1 m grid · registered metric view", img_html(engine.bev)),
        unsafe_allow_html=True,
    )

    cam_col1, cam_col2 = st.columns([1.35, 1.0])

    with cam_col1:
        st.markdown(
            panel("Camera A · detector response", "synthetic oblique view · homography-style projection", img_html(engine.cam)),
            unsafe_allow_html=True,
        )

    with cam_col2:
        ds = engine.detector.stats
        vis_vals = [safe_float(tr.vis, 1.0) for tr in tracks if safe_point(tr.x, tr.y)]
        avg_vis = float(np.mean(vis_vals)) if vis_vals else 1.0
        matched = sum(1 for tr in tracks if tr.lost == 0 and safe_point(tr.x, tr.y))
        assoc_pct = 100.0 * matched / max(1, len(tracks))

        feed_body = (
            '<div class="grid4">'
            f'<div class="cell"><div class="k">DETECTIONS</div><div class="mid">{ds["n"]}</div></div>'
            f'<div class="cell"><div class="k">MISSED</div><div class="mid">{ds["missed"]}</div></div>'
            f'<div class="cell"><div class="k">GHOSTS</div><div class="mid">{ds["ghosts"]}</div></div>'
            f'<div class="cell"><div class="k">&lt; THR</div><div class="mid">{ds["lowconf"]}</div></div>'
            '</div>'
            '<div style="height:8px"></div>'
            f'<div class="row"><span class="nm mono">CONF GATE</span><span class="mt">{safe_float(engine.params["conf_thresh"], 0.3):.2f}</span></div>'
            f'{bar(safe_float(engine.params["conf_thresh"], 0.3) * 100.0, "var(--amber)")}'
            f'<div class="row"><span class="nm mono">SNR FIELD AVG</span><span class="mt">{avg_vis:.2f}</span></div>'
            f'{bar(avg_vis * 100.0, "var(--cyan)")}'
            f'<div class="row"><span class="nm mono">ASSOC MATCH</span><span class="mt">{matched} / {len(tracks)}</span></div>'
            f'{bar(assoc_pct, "var(--safe)")}'
        )
        st.markdown(panel("Front-end telemetry", "per 100 ms frame", feed_body, tight=True), unsafe_allow_html=True)

    legend_items = [
        ("TRAILS", "trails", "var(--amber)"),
        ("FORECAST", "predict", "var(--cyan)"),
        ("SAFETY ZONE", "zones", "var(--high)"),
        ("RISK LINK", "links", "var(--crit)"),
        ("BLIND WEDGE", "blind", "#B4526A"),
        ("HEATMAP", "heatmap", "var(--violet)"),
        ("METRIC GRID", "grid", "#7C8C84"),
    ]
    chips = []
    for name, key, color in legend_items:
        on = OVERLAYS[key]
        style = f"background:{color};border-color:{color}" if on else ""
        cls = "b on" if on else "b"
        mark = "✓ " if on else "· "
        chips.append(f'<div class="{cls}" style="{style}">{mark}{name}</div>')

    chips.append('<div class="b" style="color:var(--safe);border-color:rgba(72,214,140,.4)">● SAFE</div>')
    chips.append('<div class="b" style="color:var(--cau);border-color:rgba(255,197,66,.4)">● CAUTION</div>')
    chips.append('<div class="b" style="color:var(--high);border-color:rgba(255,122,51,.4)">● HIGH</div>')
    chips.append('<div class="b" style="color:var(--crit);border-color:rgba(255,59,84,.4)">● CRITICAL</div>')

    st.markdown(f'<div class="lg">{"".join(chips)}</div>', unsafe_allow_html=True)

with col_rail:
    dial = svg_dial(score / 100.0, lvl, "RISK INDEX", f"{safe_float(score, 0.0):.0f}")

    if nearest:
        sep_txt = f'{safe_float(nearest["r"]["d_eff"], 999.0):.2f} m · TTC {max(0.0, safe_float(nearest["r"]["t_eff"], 999.0)):.2f} s'
    else:
        sep_txt = "—"

    active = len(confirmed)
    in_zone = sum(1 for it in engine.safety.top if it["r"]["in_zone"])
    actuator = "BRAKE" if safe_float(w.t, 0.0) < safe_float(engine.safety.autobrake_until, -1.0) else "OK"
    actuator_lvl = 3 if actuator == "BRAKE" else 0

    threat_body = (
        dial +
        f'<div style="text-align:center;margin-top:-4px" class="mono">nearest separation {sep_txt}</div>'
        '<div style="height:8px"></div>'
        '<div class="grid3">'
        f'<div class="cell"><div class="k">ACTIVE TRK</div><div class="mid lv0">{active}</div></div>'
        f'<div class="cell"><div class="k">IN ZONE</div><div class="mid lv2">{in_zone}</div></div>'
        f'<div class="cell"><div class="k">ACTUATOR</div><div class="mid lv{actuator_lvl}">{actuator}</div></div>'
        '</div>'
    )

    top_rows = ""
    for it in engine.safety.top[:5]:
        mm = it["m"]
        hh = it["h"]
        rr = it["r"]
        top_rows += (
            f'<div class="row"><span class="nm lv{rr["level"]}">#{mm.tid}</span>'
            f'<span class="mono" style="font-size:10.5px">{mm.cls[:8]} <b style="color:var(--ink)">↔</b> #{hh.tid} {hh.cls[:7]}</span>'
            f'<span class="mt">{safe_float(rr["d_eff"], 999.0):.1f} m · {max(0.0, safe_float(rr["t_eff"], 999.0)):.1f} s</span></div>'
            f'{bar(safe_float(rr["score"], 0.0), LEVEL_HEX.get(rr["level"], LEVEL_HEX[0]), rr["level"] >= 2)}'
        )

    if not top_rows:
        top_rows = '<div class="hint">No machine-person pair inside the forecast envelope. Zones nominal.</div>'

    roster_rows = ""
    for tr in sorted([t for t in tracks if safe_point(t.x, t.y)], key=lambda z: (-z.risk, z.kind, z.tid))[:14]:
        stt = "TRACKED" if tr.lost == 0 else f"COAST {tr.lost}"
        prefix = "M" if tr.kind == "machine" else ("H" if tr.kind == "human" else "A")
        arrow = "↑→↓←"[int((deg(safe_float(tr.heading, 0.0)) + 45) // 90) % 4]
        roster_rows += (
            f'<div class="row"><span class="nm lv{tr.level}">{prefix}{tr.tid}</span>'
            f'<span class="mono" style="font-size:10.5px">{tr.cls[:9]}</span>'
            f'<span class="mt">{arrow} {safe_float(tr.speed, 0.0):.1f} m/s · {deg(safe_float(tr.heading, 0.0)):3.0f}° · '
            f'z {safe_float(tr.stop_dist + 1.0 + engine.params["buffer"], 0.0):.1f} m · {stt}</span></div>'
        )

    pipe_items = [
        ("DETECT", 3.1 + 0.4 * engine.detector.stats["n"], "var(--amber)"),
        ("ASSOCIATE", 1.4 + 0.25 * len(tracks), "var(--cyan)"),
        ("FORECAST", 1.1 + 0.20 * len(tracks), "var(--violet)"),
        ("RISK/EVENT", 0.8 + 0.12 * len(engine.safety.top) ** 2, "var(--high)"),
    ]
    pipe_rows = ""
    for name, val, color in pipe_items:
        pipe_rows += (
            f'<div class="row"><span class="nm mono">{name}</span><span class="mt">{safe_float(val, 0.0):.1f} ms</span></div>'
            f'{bar(min(100.0, safe_float(val, 0.0) * 7.0), color)}'
        )

    total_pct = safe_float(engine.pipe_ms, 0.0) * 10.0
    total_color = "var(--safe)" if safe_float(engine.pipe_ms, 0.0) < 60 else "var(--high)"
    pipe_rows += (
        f'<div class="row"><span class="nm mono">TOTAL</span><span class="mt">{safe_float(engine.pipe_ms, 0.0):.1f} ms / 100 ms</span></div>'
        f'{bar(total_pct, total_color)}'
    )

    rail_html = (
        panel("Threat state", "REVERSE ACTIVE" if any(tr.reversing for tr in tracks) else "FORWARD OPS", threat_body)
        + panel("Top ranked pairs", "forecast separation · TTC", top_rows, tight=True)
        + panel("Track roster", "persistent IDs · coasting flagged", roster_rows, tight=True)
        + panel("Pipeline latency", "rolling per-frame budget", pipe_rows, tight=True)
    )
    st.markdown(rail_html, unsafe_allow_html=True)

# -------------------------------------- TABS ----------------------------------------
tab_events, tab_heat, tab_metrics, tab_robust, tab_reel, tab_docs = st.tabs(
    ["Event archive", "Safety heatmap", "Tracking metrics", "Robustness lab", "Demo reel", "Pipeline & docs"]
)

with tab_events:
    ev_filter = st.multiselect(
        "Filter by level",
        ["SAFE", "CAUTION", "HIGH RISK", "CRITICAL"],
        default=["CAUTION", "HIGH RISK", "CRITICAL"],
    )

    cols = [
        "id", "t_open", "t_close", "dur", "level", "machine", "person",
        "min_dist", "min_ttc", "lead", "x", "y", "action", "outcome", "tags"
    ]

    if engine.event_rows:
        df = pd.DataFrame(engine.event_rows, columns=cols)
        dff = df[df["level"].isin(ev_filter)] if ev_filter else df

        cnt = df["level"].value_counts().to_dict()
        out = df["outcome"].value_counts().to_dict()
        auto_cnt = int(df["action"].str.contains("AUTO", na=False).sum())

        summary = (
            '<div class="grid4">'
            f'<div class="cell"><div class="k">CRITICAL</div><div class="mid lv3">{cnt.get("CRITICAL", 0)}</div></div>'
            f'<div class="cell"><div class="k">HIGH RISK</div><div class="mid lv2">{cnt.get("HIGH RISK", 0)}</div></div>'
            f'<div class="cell"><div class="k">CAUTION</div><div class="mid lv1">{cnt.get("CAUTION", 0)}</div></div>'
            f'<div class="cell"><div class="k">AUTO-BRAKES</div><div class="mid lv0">{auto_cnt}</div></div>'
            '</div>'
            '<div style="height:8px"></div>'
            '<div class="grid3">'
            f'<div class="cell"><div class="k">MEAN MIN DIST</div><div class="mono">{safe_float(df["min_dist"].mean(), 0.0):.2f} m</div></div>'
            f'<div class="cell"><div class="k">BEST LEAD</div><div class="mono">{safe_float(df["lead"].max(), 0.0):.2f} s</div></div>'
            f'<div class="cell"><div class="k">MEAN LEAD</div><div class="mono">{safe_float(df["lead"].mean(), 0.0):.2f} s</div></div>'
            f'<div class="cell"><div class="k">MEAN DURATION</div><div class="mono">{safe_float(df["dur"].mean(), 0.0):.1f} s</div></div>'
            f'<div class="cell"><div class="k">FALSE ALARMS</div><div class="mono">{out.get("FALSE_ALARM", 0)}</div></div>'
            f'<div class="cell"><div class="k">NEAR MISS / AVOIDED</div><div class="mono">{out.get("NEAR_MISS", 0) + out.get("COLLISION_AVOIDED", 0)}</div></div>'
            '</div>'
        )

        st.markdown(panel("Near-miss register", f"{len(df)} stored · session history + CSV export", summary), unsafe_allow_html=True)
        st.dataframe(dff.sort_values("t_open", ascending=False).head(80), use_container_width=True)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download event register CSV",
            csv_buf.getvalue().encode(),
            "agrisafe_event_register.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.markdown(panel("Near-miss register", "empty", '<div class="hint">No closed events yet.</div>'), unsafe_allow_html=True)

with tab_heat:
    hc1, hc2 = st.columns([1.55, 1.0])

    with hc1:
        hm_mode = st.radio(
            "Layer",
            ["risk", "recent", "event", "traffic", "dwell"],
            horizontal=True,
            format_func=lambda x: {
                "risk": "Cumulative risk",
                "recent": "Recent 60 s",
                "event": "Near-miss density",
                "traffic": "Machine traffic",
                "dwell": "Worker dwell",
            }[x],
            key="hm_mode",
        )

        hm_img = engine.heat.render_overlay(hm_mode, engine.bev)
        desc = {
            "risk": "accumulated risk dose",
            "recent": "exponential 60 s window",
            "event": "closed near-miss impacts",
            "traffic": "machine occupancy per cell",
            "dwell": "person-seconds per cell",
        }[hm_mode]

        st.markdown(panel("Field safety heatmap", desc, img_html(hm_img)), unsafe_allow_html=True)

    with hc2:
        grid = {
            "risk": engine.heat.risk,
            "recent": engine.heat.recent,
            "event": engine.heat.event,
            "traffic": engine.heat.traffic,
            "dwell": engine.heat.dwell,
        }[hm_mode]
        grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

        flat = grid.ravel()
        idx = np.argsort(flat)[::-1][:8]
        rows = ""

        for k, i in enumerate(idx):
            v = float(flat[i])
            if v <= 0:
                continue
            yy, xx = divmod(int(i), grid.shape[1])
            mx = xx * engine.heat.cell + 0.5
            my = FIELD_H - yy * engine.heat.cell - 0.5
            cls = "◼ HOT" if k < 3 else "◻ warm"
            rows += f'<tr><td>{mx:.0f} / {my:.0f} m</td><td>{v:.2f}</td><td>{cls}</td></tr>'

        if not rows:
            rows = '<tr><td colspan="3">no accumulation yet</td></tr>'

        st.markdown(
            panel(
                "Hot-zone ranking",
                "top 8 cells · 1 m resolution",
                f'<table class="dt"><tr><th>X / Y (m)</th><th>Dose</th><th>Class</th></tr>{rows}</table>'
                '<div style="height:8px"></div>'
                '<div class="hint"><b style="color:var(--ink)">Operational read-out:</b> top-quartile cells should '
                'receive a physical exclusion barrier, a posted spotter, or a machine guidance-lane offset. '
                'Export the register with the CSV for the daily toolbox talk.</div>',
            ),
            unsafe_allow_html=True,
        )

with tab_metrics:
    hist = list(engine.metrics.history)
    mota_s = [safe_float(h["mota"], 0.0) for h in hist]
    idf1_s = [safe_float(h["idf1"], 0.0) for h in hist]
    prec_s = [safe_float(h["precision"], 0.0) for h in hist]
    rec_s = [safe_float(h["recall"], 0.0) for h in hist]

    lat = [safe_float(v, 0.0) for v in engine.metrics.latency]
    fa = safe_float(engine.safety.counters.get("false_alarm", 0), 0.0)
    hrs = max(1e-6, safe_float(w.t, 0.0) / 3600.0)

    acc_body = (
        '<div class="grid4">'
        f'<div class="cell"><div class="k">MOTA</div><div class="big lv{0 if safe_float(m["mota"], 0.0) > 0.9 else 1}">{safe_float(m["mota"], 0.0) * 100:.1f}<span style="font-size:16px">%</span></div></div>'
        f'<div class="cell"><div class="k">IDF1</div><div class="big lv{0 if safe_float(m["idf1"], 0.0) > 0.85 else 1}">{safe_float(m["idf1"], 0.0) * 100:.1f}<span style="font-size:16px">%</span></div></div>'
        f'<div class="cell"><div class="k">PRECISION</div><div class="big lv{1 if safe_float(m["precision"], 0.0) < 0.9 else 0}">{safe_float(m["precision"], 0.0) * 100:.1f}<span style="font-size:16px">%</span></div></div>'
        f'<div class="cell"><div class="k">RECALL</div><div class="big lv{1 if safe_float(m["recall"], 0.0) < 0.85 else 0}">{safe_float(m["recall"], 0.0) * 100:.1f}<span style="font-size:16px">%</span></div></div>'
        '</div>'
        '<div style="height:10px"></div>'
        '<div class="grid4">'
        f'<div class="cell"><div class="k">ID SWITCHES</div><div class="mid">{int(safe_float(m["idsw"], 0))}</div></div>'
        f'<div class="cell"><div class="k">FRAGMENTATION</div><div class="mid">{int(safe_float(m["frag"], 0))}</div></div>'
        f'<div class="cell"><div class="k">FALSE TRACKS</div><div class="mid">{int(safe_float(m["fp"], 0))}</div></div>'
        f'<div class="cell"><div class="k">MISSED OBJECTS</div><div class="mid">{int(safe_float(m["fn"], 0))}</div></div>'
        '</div>'
        '<div style="height:10px"></div>'
        '<div class="k">MOTA TRACE</div>'
        f'{svg_spark(mota_s, color="#FFB020")}'
        '<div class="k" style="margin-top:6px">IDF1 TRACE</div>'
        f'{svg_spark(idf1_s, color="#39D7F2")}'
        '<div class="k" style="margin-top:6px">PRECISION TRACE</div>'
        f'{svg_spark(prec_s, color="#48D68C")}'
        '<div class="k" style="margin-top:6px">RECALL TRACE</div>'
        f'{svg_spark(rec_s, color="#C79BFF")}'
    )

    p95 = float(np.percentile(list(engine.metrics.pipe), 95)) if engine.metrics.pipe else 0.0
    perf_body = (
        '<div class="grid3">'
        f'<div class="cell"><div class="k">MEAN LEAD</div><div class="mid lv0">{(float(np.mean(lat)) if lat else 0.0):.2f} s</div></div>'
        f'<div class="cell"><div class="k">P95 PROCESS</div><div class="mid lv1">{safe_float(p95, 0.0):.1f} ms</div></div>'
        f'<div class="cell"><div class="k">FALSE / HOUR</div><div class="mid lv{2 if fa / hrs > 6 else 0}">{fa / hrs:.1f}</div></div>'
        '</div>'
        '<div style="height:8px"></div>'
        '<div class="k">LEAD-TIME DISTRIBUTION · seconds before closest approach</div>'
        f'{svg_hist(lat)}'
    )

    bucket_rows = ""
    for k, v in engine.metrics.vis_buckets.items():
        tot = max(1, int(safe_float(v["tot"], 0)))
        hit = 100.0 * safe_float(v.get("hit", 0), 0.0) / tot
        bucket_rows += f'<tr><td>{k}</td><td>{hit:.1f}%</td><td>{tot}</td></tr>'

    if not bucket_rows:
        bucket_rows = '<tr><td colspan="3">no visibility samples yet</td></tr>'

    cond_body = (
        '<div class="grid2">'
        '<div><div class="k">DETECTION HIT-RATE BY VISIBILITY</div>'
        f'<table class="dt"><tr><th>BUCKET</th><th>HIT</th><th>N</th></tr>{bucket_rows}</table></div>'
        '<div><div class="k">INTERPRETATION</div>'
        '<div class="hint">The tracker intentionally coasts through dust, shadow and occlusion. '
        'ID switches are penalised in MOTA; re-ID archive reduces fragmentation when a person disappears '
        'behind a silo or implement. False alarms are counted when a high-risk event closes with no true '
        'sub-4 m separation.</div></div>'
        '</div>'
    )

    st.markdown(
        panel("Tracking accuracy", "online MOT summary vs scene ground truth", acc_body)
        + panel("Warning performance", "lead time and false-alarm budget", perf_body)
        + panel("Per-condition diagnostics", "visibility stratification", cond_body),
        unsafe_allow_html=True,
    )

with tab_robust:
    conditions = [
        (
            "Dust / aerosol",
            clamp01(safe_float(engine.params["dust"], 1.0) * w.dust_level()),
            "Confidence re-scale, temporal confirmation, ghost proposal logging, and wider measurement covariance R.",
        ),
        (
            "Hard shadows",
            clamp01(safe_float(engine.params["shadow"], 1.0) * (1.0 if w.sun_elev() < 30 else 0.35)),
            "Shadow-mask veto against solar model; class heads receive luminance-normalised confidence.",
        ),
        (
            "Partial occlusion",
            clamp01(1.0 - float(np.mean([safe_float(tr.vis, 1.0) for tr in tracks])) if tracks else 0.0),
            f"Kalman coasting up to {int(safe_float(engine.params['max_age'], 12))} frames; safety zone remains armed while target is unseen.",
        ),
        (
            "Night / low-lux",
            clamp01(safe_float(engine.params["night"], 1.0) * w.night_level()),
            "IR gain path, lowered gate, larger process noise, headlight-glare rejection in detector wrapper.",
        ),
        (
            "Crowded scene",
            clamp01(safe_float(engine.params["crowd"], 1.0)),
            "Class-aware NMS, Mahalanobis gate widened by (1 - visibility), appearance + motion re-ID hand-off.",
        ),
    ]

    cond_rows = ""
    for name, sev, note in conditions:
        hit = 100.0 * (safe_float(m["recall"], 0.0) ** (1.0 + sev))
        color = "var(--safe)" if hit > 85 else ("var(--cau)" if hit > 70 else "var(--high)")
        cond_rows += (
            f'<div class="row"><span class="nm" style="min-width:118px">{name}</span>'
            f'<span class="mt">sev {sev:.2f} · recall proxy {hit:.1f}% · reid {int(safe_float(engine.tracker.stats.get("reid", 0), 0))}</span></div>'
            f'{bar(hit, color)}'
            f'<div class="hint">{note}</div>'
        )

    env_body = (
        '<div class="grid3">'
        f'<div class="cell"><div class="k">SUN ELEV</div><div class="mid">{w.sun_elev():.0f}°</div></div>'
        f'<div class="cell"><div class="k">NIGHT MIX</div><div class="mid">{w.night_level():.2f}</div></div>'
        f'<div class="cell"><div class="k">DUST CELLS</div><div class="mid">{len(w.dust)}</div></div>'
        f'<div class="cell"><div class="k">DRYNESS</div><div class="mid">{safe_float(w.cfg["dry"], 0.0):.2f}</div></div>'
        f'<div class="cell"><div class="k">WIND</div><div class="mid">{float(np.hypot(*w.wind)):.1f} m/s</div></div>'
        f'<div class="cell"><div class="k">OCCLUDERS</div><div class="mid">{len(w.circs) + len(w.rects) + len(w.trees)}</div></div>'
        '</div>'
        '<div style="height:8px"></div>'
        '<div class="hint">Push the four environment sliders in the sidebar to stress the front-end live. '
        'The tracker gates, coasting budget and risk thresholds adapt, and the metrics tab shows the accuracy '
        'cost in MOTA / IDSW within a second.</div>'
    )

    st.markdown(
        panel("Robustness lab", "degradation modes and compensations", cond_rows, tight=True)
        + panel("Current environmental read", "solar + aerosol state", env_body, tight=True),
        unsafe_allow_html=True,
    )

with tab_reel:
    n_reel = len(engine.reel)
    reel_info = (
        '<div class="grid3">'
        f'<div class="cell"><div class="k">BUFFER</div><div class="mid">{n_reel} f</div></div>'
        f'<div class="cell"><div class="k">WINDOW</div><div class="mid">{n_reel * 0.1:.1f} s</div></div>'
        f'<div class="cell"><div class="k">RESOLUTION</div><div class="mid">{REEL_W}×{REEL_H}</div></div>'
        '</div>'
        '<div style="height:8px"></div>'
        '<div class="hint">The app records its own screen: every painted frame is kept in a rolling buffer with '
        'HUD, director captions and live threat level burned in. Let the simulation run for ~20 s, then export '
        'an animated GIF you can drop into a post or safety brief.</div>'
    )

    st.markdown(panel("Live demo reel", f"{n_reel} buffered frames", reel_info, tight=True), unsafe_allow_html=True)

    if st.button("Build demo reel from rolling buffer", key="mk_reel"):
        frames = list(engine.reel)
        if len(frames) > 5:
            step_f = max(1, len(frames) // 140)
            sel = frames[::step_f]
            pil = [Image.fromarray(f) for f in sel]

            buf = io.BytesIO()
            pil[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=pil[1:],
                duration=int(100 * step_f),
                loop=0,
                optimize=True,
            )
            st.session_state.reel_gif = buf.getvalue()
            st.session_state.reel_note = f"GIF · {len(pil)} frames · {len(buf.getvalue()) / 1e6:.1f} MB"

            if cv2 is not None:
                try:
                    tmp = os.path.join(os.getcwd(), "agrisafe_reel.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(tmp, fourcc, 10.0, (REEL_W, REEL_H))
                    for f in sel:
                        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    vw.release()
                    with open(tmp, "rb") as fh:
                        st.session_state.reel_mp4 = fh.read()
                    st.session_state.reel_note += " · MP4 written"
                except Exception:
                    pass
        else:
            st.session_state.reel_note = "Buffer too short — let the simulation run for ~20 s first."

    note = st.session_state.get("reel_note", "")
    if note:
        st.markdown(panel("Export status", note, "", tight=True), unsafe_allow_html=True)

    if st.session_state.get("reel_gif"):
        st.download_button(
            "Download demo reel GIF",
            st.session_state.reel_gif,
            "agrisafe_demo_reel.gif",
            "image/gif",
            use_container_width=True,
        )

    if st.session_state.get("reel_mp4"):
        st.download_button(
            "Download demo reel MP4",
            st.session_state.reel_mp4,
            "agrisafe_demo_reel.mp4",
            "video/mp4",
            use_container_width=True,
        )

with tab_docs:
    flow_items = [
        ("01 · CAPTURE", "Synthetic oblique farm camera over a 60 × 38 m metric field."),
        ("02 · DETECT", "Class heads for tractors, harvesters, UTVs, trucks, people and livestock."),
        ("03 · DENOISE", "NMS, confidence gate, shadow veto, dust SNR re-scale, ghost proposal log."),
        ("04 · ASSOCIATE", "Kalman predict → Mahalanobis gate → Hungarian assignment → coast / retire / re-ID."),
        ("05 · ESTIMATE", "Metric velocity, body heading, reverse detection, curvature from trail fit."),
        ("06 · FORECAST", "Constant-turn machine rollout, stochastic ensemble for people and animals."),
        ("07 · ZONES", "Dynamic stop-distance zone: reaction + v²/(2a) + implement flare + personnel buffer."),
        ("08 · RISK", "Time-synced separation, CPA, TTC, blind-wedge and visibility weighting → 4-level ladder."),
        ("09 · ACT", "Cab alert, auto-brake binding, event record, heatmap dose, register export."),
    ]

    flow_html = "".join(
        f'<div class="fbox"><h5>{n}</h5><p>{d}</p></div>'
        for n, d in flow_items
    )

    relations = (
        '<div class="mono" style="line-height:1.9">'
        'd<sub>stop</sub> = v·t<sub>r</sub> + v²/(2a)<br>'
        't<sub>cpa</sub> = −(r·v<sub>rel</sub>)/|v<sub>rel</sub>|²<br>'
        'd<sub>cpa</sub> = |r + v<sub>rel</sub>·t<sub>cpa</sub>|<br>'
        'R = 100·(0.38·s<sub>d</sub>+0.30·s<sub>t</sub>+0.10·s<sub>v</sub>+0.12·s<sub>env</sub>+0.10·z)<br>'
        'MOTA = 1 − (FN+FP+IDSW)/GT'
        '</div>'
    )

    ladder = (
        '<table class="dt">'
        '<tr><th>LEVEL</th><th>SEPARATION</th><th>TTC</th></tr>'
        '<tr><td class="lv0">SAFE</td><td>&gt; 6.5 m</td><td>&gt; 10 s</td></tr>'
        '<tr><td class="lv1">CAUTION</td><td>&lt; 6.5 m or inside d<sub>stop</sub></td><td>&lt; 10 s</td></tr>'
        '<tr><td class="lv2">HIGH RISK</td><td>&lt; 3.4 m</td><td>&lt; 5.2 s</td></tr>'
        '<tr><td class="lv3">CRITICAL</td><td>&lt; 1.7 m</td><td>&lt; 2.6 s → auto-brake</td></tr>'
        '</table>'
    )

    docs_body = (
        panel("Processing chain", "10 Hz · single process · no network calls", f'<div class="flow">{flow_html}</div>')
        + panel(
            "Key relations and warning ladder",
            "safety model",
            '<div class="grid2">'
            f'<div><div class="k">KEY RELATIONS</div>{relations}</div>'
            f'<div><div class="k">WARNING LADDER</div>{ladder}</div>'
            '</div>'
            '<div style="height:10px"></div>'
            '<div class="hint"><b style="color:var(--ink)">Swapping in a real detector:</b> the front-end is a '
            'single method, <span class="mono">Detector.run(world) -&gt; List[Detection]</span>. Replace it with a '
            'Ultralytics / RT-DETR / ONNX wrapper that emits class, bbox, confidence, ground (x,y), visibility '
            'and the entire tracker → forecast → risk → register chain runs unchanged on real footage. '
            'Events are retained in-session and exportable as CSV.</div>'
        )
    )

    st.markdown(docs_body, unsafe_allow_html=True)

# -------------------------------------- LOOP ----------------------------------------
if st.session_state.playing:
    elapsed = time.perf_counter() - loop_t0
    target = steps * SIM_DT / max(0.15, safe_float(engine.params["speed"], 1.0))
    wait = target - elapsed
    if wait > 0:
        time.sleep(min(wait, 0.35))
    rerun_app()