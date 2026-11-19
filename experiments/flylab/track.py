"""Closed race tracks for the driving fly: geometry (lateral offset, heading error, progress) and a flybrain World."""
from __future__ import annotations

import numpy as np

from .core import World, Sphere, Plane, Box, add_material


def stadium(L=0.6, R=0.3, n=720):
    s = np.linspace(0, 1, n, endpoint=False)
    P = 2 * L + 2 * np.pi * R
    pts = []
    for u in s * P:                                   # counter-clockwise, starting bottom-left going +x
        if u < L:
            pts.append((-L / 2 + u, -R))
        elif u < L + np.pi * R:
            a = -np.pi / 2 + (u - L) / R
            pts.append((L / 2 + R * np.cos(a), R * np.sin(a)))
        elif u < 2 * L + np.pi * R:
            pts.append((L / 2 - (u - L - np.pi * R), R))
        else:
            a = np.pi / 2 + (u - 2 * L - np.pi * R) / R
            pts.append((-L / 2 + R * np.cos(a), R * np.sin(a)))
    return np.array(pts)


def kidney(n=900, scale=0.42):
    """An unseen test track: a bean with one inward (right-hand) bend and tighter turns."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.28 * np.cos(2 * t) - 0.22 * np.sin(3 * t)
    x, y = r * np.cos(t) * 1.25, r * np.sin(t) * 0.85
    return np.c_[x, y] * scale


class Track:
    def __init__(self, center: np.ndarray, width=0.16, post_gap=0.045, post_h=0.07, post_r=0.008):
        self.c = center.astype(np.float64)
        d = np.roll(self.c, -1, 0) - self.c
        self.seg = np.linalg.norm(d, axis=1)
        self.tan = d / self.seg[:, None]
        self.normal = np.c_[-self.tan[:, 1], self.tan[:, 0]]           # left of travel
        self.cum = np.r_[0, np.cumsum(self.seg)[:-1]]
        self.P = float(self.seg.sum())
        self.width, self.post_gap, self.post_h, self.post_r = width, post_gap, post_h, post_r
        heading = np.arctan2(self.tan[:, 1], self.tan[:, 0])
        self.heading = heading
        dh = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))
        self.curv = dh / self.seg

    def locate(self, xy: np.ndarray):
        """xy (B,2) -> index of nearest centre point, signed lateral offset (+ = left), tangent heading, curvature."""
        diff = xy[:, None, :] - self.c[None]
        i = np.argmin((diff ** 2).sum(-1), axis=1)
        off = (diff[np.arange(len(xy)), i] * self.normal[i]).sum(-1)
        return i, off, self.heading[i], self.curv[i]

    def posts(self):
        out = []
        acc = 0.0
        for k in range(len(self.c)):
            acc += self.seg[k]
            if acc >= self.post_gap:
                acc = 0.0
                for side in (+1, -1):
                    p = self.c[k] + side * self.normal[k] * self.width / 2
                    out.append((p, side))
        return out

    def world(self) -> World:
        add_material("asphalt", (0.025, 0.06, 0.08, 0.09), pattern=1, refl2=(0.012, 0.03, 0.04, 0.05), scale=0.04)
        add_material("cone_in", (0.01, 0.02, 0.02, 0.02))                          # black poles on the inside
        add_material("cone_out", (0.75, 1.0, 1.0, 1.0))                            # white (UV-bright) poles outside
        add_material("stripe_line", (0.5, 0.8, 0.8, 0.8))
        w = World(detail=0.6, light_pos=(0.0, 0.0, 1.6), light_color=(1.2, 2.2, 2.2, 2.2), ambient=(0.25, 0.3, 0.3, 0.3))
        w.planes.append(Plane((0, 0, 0), (0, 0, 1), "asphalt"))
        # featureless room: plain walls and ceiling, so the only usable cues are the track itself (with striped walls
        # a readout can memorise "where am I in the room" instead of reading the road -- v1 did exactly that)
        add_material("plain", (0.3, 0.55, 0.55, 0.55))
        w.planes.append(Plane((0, 0, 1.8), (0, 0, -1), "plain"))
        for a, n in [((2.5, 0, 0), (-1, 0, 0)), ((-2.5, 0, 0), (1, 0, 0)), ((0, 2.5, 0), (0, -1, 0)), ((0, -2.5, 0), (0, 1, 0))]:
            w.planes.append(Plane(a, n, "plain"))
        x, y = self.c[:, 0], self.c[:, 1]
        ccw = (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0          # loop orientation: inside is left if CCW
        for p, side in self.posts():
            inside = (side > 0) == ccw
            w.spheres.append(Sphere((p[0], p[1], self.post_h / 2), (self.post_r, self.post_r, self.post_h / 2),
                                    "cone_in" if inside else "cone_out"))
        return w


def wavy(n=900, scale=0.36):
    """Training-only loop with alternating left/right bends (the oval only ever turns one way)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.22 * np.sin(4 * t)
    return np.c_[r * np.cos(t) * 1.35, r * np.sin(t) * 0.9] * scale
